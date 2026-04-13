"""Single polyphonic voice — reads a sample with pitch and applies per-voice DSP."""
from __future__ import annotations

import enum
import time

import numpy as np

from vynth.config import SAMPLE_RATE
from vynth.dsp.adsr import ADSREnvelope
from vynth.dsp.filter import BiquadFilter
from vynth.dsp.formant import FormantPreserver
from vynth.dsp.granular import GranularEngine
from vynth.dsp.pitch_shift import PitchShifter
from vynth.sampler.sample import Sample
from vynth.utils.audio_utils import note_to_freq


class PlaybackMode(enum.Enum):
    SAMPLER = enum.auto()
    GRANULAR = enum.auto()
    SLICE = enum.auto()


class Voice:
    """One voice — reads a sample with pitch and applies per-voice DSP.

    Per-voice DSP chain: ADSR → PitchShifter → FormantPreserver → BiquadFilter.
    """

    __slots__ = (
        "voice_id",
        "_sample_rate",
        "_sample",
        "_note",
        "_velocity",
        "_playing",
        "_position",
        "_pitch_ratio",
        "_start_time",
        "_mode",
        "_adsr",
        "_pitch_shifter",
        "_formant",
        "_filter",
        "_granular",
        # Slice mode config
        "_slice_num_slices",
        "_slice_start_note",
        "_slice_start_frame",
        "_slice_end_frame",
    )

    def __init__(self, voice_id: int, sample_rate: int = SAMPLE_RATE) -> None:
        self.voice_id = voice_id
        self._sample_rate = sample_rate

        self._sample: Sample | None = None
        self._note: int = 0
        self._velocity: int = 0
        self._playing: bool = False
        self._position: float = 0.0
        self._pitch_ratio: float = 1.0
        self._start_time: float = 0.0
        self._mode: PlaybackMode = PlaybackMode.SAMPLER

        # Slice mode config
        self._slice_num_slices: int = 16
        self._slice_start_note: int = 36  # C2
        self._slice_start_frame: int = 0
        self._slice_end_frame: int = 0

        # Per-voice DSP
        self._adsr = ADSREnvelope(sample_rate)
        self._pitch_shifter = PitchShifter(sample_rate=sample_rate)
        self._formant = FormantPreserver(sample_rate=sample_rate)
        self._filter = BiquadFilter(sample_rate)
        self._granular = GranularEngine(sample_rate)

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def note(self) -> int:
        return self._note

    @property
    def velocity(self) -> int:
        return self._velocity

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def mode(self) -> PlaybackMode:
        return self._mode

    @mode.setter
    def mode(self, value: PlaybackMode) -> None:
        self._mode = value

    @property
    def is_active(self) -> bool:
        return self._adsr.is_active or self._playing

    # ── Note control ─────────────────────────────────────────────────────

    def note_on(self, note: int, velocity: int, sample: Sample, start_frame: int = 0) -> None:
        """Start playing *sample* at the pitch implied by *note*."""
        self._sample = sample
        self._note = note
        self._velocity = velocity
        self._playing = True
        self._start_time = time.monotonic()

        self._pitch_ratio = note_to_freq(note) / note_to_freq(sample.root_note)

        # Reset DSP state for clean start
        self._adsr.reset()
        self._pitch_shifter.reset()
        self._formant.reset()
        self._filter.reset()

        self._adsr.gate_on(velocity / 127.0)

        if self._mode == PlaybackMode.SLICE:
            # Calculate which slice this note maps to
            slice_idx = note - self._slice_start_note
            if slice_idx < 0 or slice_idx >= self._slice_num_slices:
                # Note outside slice range — don't play
                self._playing = False
                self._adsr.reset()
                return
            slice_len = max(1, sample.length // self._slice_num_slices)
            self._slice_start_frame = slice_idx * slice_len
            self._slice_end_frame = min((slice_idx + 1) * slice_len, sample.length)
            self._position = float(self._slice_start_frame)
            # In slice mode, pitch ratio is 1.0 (each slice plays at original pitch)
            self._pitch_ratio = 1.0
        elif self._mode == PlaybackMode.GRANULAR:
            self._position = float(start_frame)
            self._granular.set_source(sample.mono, sample.sample_rate)
            self._granular.set_param("pitch_ratio", self._pitch_ratio)
            self._granular.reset()
        else:
            # SAMPLER mode
            self._position = float(start_frame)

    def note_off(self) -> None:
        """Trigger ADSR release; the voice stays active until the envelope reaches IDLE."""
        self._adsr.gate_off()

    # ── ADSR convenience ─────────────────────────────────────────────────

    def set_adsr_params(
        self, attack_ms: float, decay_ms: float, sustain: float, release_ms: float
    ) -> None:
        self._adsr.set_param("attack_ms", attack_ms)
        self._adsr.set_param("decay_ms", decay_ms)
        self._adsr.set_param("sustain", sustain)
        self._adsr.set_param("release_ms", release_ms)

    # ── Processing ───────────────────────────────────────────────────────

    def process(self, n_frames: int) -> np.ndarray:
        """Render *n_frames* of stereo audio ``(n_frames, 2)`` float32."""
        silence = np.zeros((n_frames, 2), dtype=np.float32)

        if not self.is_active or self._sample is None:
            self._playing = False
            return silence

        # If ADSR finished (release done), stop the voice entirely
        if not self._adsr.is_active:
            self._playing = False
            return silence

        if self._mode == PlaybackMode.GRANULAR:
            return self._process_granular(n_frames)

        if self._mode == PlaybackMode.SLICE:
            return self._process_slice(n_frames)

        return self._process_sampler(n_frames)

    # ── Sampler playback ─────────────────────────────────────────────────

    def _process_sampler(self, n_frames: int) -> np.ndarray:
        sample = self._sample
        assert sample is not None

        stereo = sample.get_stereo()  # (length, 2)
        length = sample.length
        loop = sample.loop

        # Read with linear interpolation at pitch_ratio speed
        out = np.zeros((n_frames, 2), dtype=np.float32)
        pos = self._position

        for i in range(n_frames):
            if not self._playing:
                break

            idx0 = int(pos)
            frac = pos - idx0

            if loop.enabled:
                # Wrap position inside loop region
                loop_len = loop.end - loop.start
                if loop_len > 0 and pos >= loop.end:
                    pos = loop.start + (pos - loop.end) % loop_len
                    idx0 = int(pos)
                    frac = pos - idx0
            else:
                if idx0 >= length - 1:
                    self._playing = False
                    break

            idx1 = idx0 + 1

            # Clamp indices
            if idx0 >= length:
                self._playing = False
                break
            if idx1 >= length:
                idx1 = idx0

            # Linear interpolation
            out[i] = stereo[idx0] * (1.0 - frac) + stereo[idx1] * frac

            pos += self._pitch_ratio

        self._position = pos

        # Apply ADSR envelope
        out = self._adsr.process(out)

        # Apply per-voice pitch shifter (for additional shifting beyond sample pitch)
        out = self._pitch_shifter.process_maybe_bypass(out)

        # Apply formant preservation
        out = self._formant.process_maybe_bypass(out)

        # Apply per-voice filter
        out = self._filter.process_maybe_bypass(out)

        return out.astype(np.float32)

    # ── Granular playback ────────────────────────────────────────────────

    def _process_granular(self, n_frames: int) -> np.ndarray:
        out = self._granular.process(n_frames)

        # Apply ADSR envelope
        out = self._adsr.process(out)

        # Apply per-voice filter
        out = self._filter.process_maybe_bypass(out)

        return out.astype(np.float32)

    # ── Slice playback ───────────────────────────────────────────────────

    def _process_slice(self, n_frames: int) -> np.ndarray:
        """Play a specific slice of the sample (one-shot, no loop)."""
        sample = self._sample
        assert sample is not None

        stereo = sample.get_stereo()
        slice_end = self._slice_end_frame

        out = np.zeros((n_frames, 2), dtype=np.float32)
        pos = self._position

        for i in range(n_frames):
            if not self._playing:
                break

            idx0 = int(pos)
            frac = pos - idx0

            if idx0 >= slice_end - 1:
                self._playing = False
                break

            idx1 = min(idx0 + 1, slice_end - 1)

            # Linear interpolation
            out[i] = stereo[idx0] * (1.0 - frac) + stereo[idx1] * frac

            pos += self._pitch_ratio

        self._position = pos

        # Apply ADSR envelope
        out = self._adsr.process(out)

        # Apply per-voice filter
        out = self._filter.process_maybe_bypass(out)

        return out.astype(np.float32)

    # ── Slice configuration ──────────────────────────────────────────────

    def set_slice_config(self, num_slices: int, start_note: int) -> None:
        """Configure slice mode parameters."""
        self._slice_num_slices = max(1, min(num_slices, 128))
        self._slice_start_note = max(0, min(start_note, 127))

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all voice state."""
        self._sample = None
        self._note = 0
        self._velocity = 0
        self._playing = False
        self._position = 0.0
        self._pitch_ratio = 1.0
        self._start_time = 0.0
        self._slice_start_frame = 0
        self._slice_end_frame = 0
        self._adsr.reset()
        self._pitch_shifter.reset()
        self._formant.reset()
        self._filter.reset()
        self._granular.reset()
