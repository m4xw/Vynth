"""64-voice polyphonic voice allocator with voice stealing and master DSP."""
from __future__ import annotations

import numpy as np

from vynth.config import MAX_VOICES, SAMPLE_RATE, VOICE_STOP_LAST_ON_RETRIGGER_DEFAULT
from vynth.dsp.chorus import Chorus
from vynth.dsp.delay import Delay
from vynth.dsp.gain import GainEffect
from vynth.dsp.limiter import Limiter
from vynth.dsp.reverb import Reverb
from vynth.engine.voice import PlaybackMode, Voice
from vynth.sampler.sample import Sample


# Parameter name prefix → DSP target
_PARAM_PREFIXES = ("adsr_", "filter_", "gain_", "chorus_", "delay_", "reverb_", "limiter_")


class VoiceAllocator:
    """Manages :data:`MAX_VOICES` Voice instances with voice stealing.

    Master DSP chain (applied after voice mix): Chorus → Delay → Reverb → Limiter.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self.voices: list[Voice] = [Voice(i, sample_rate) for i in range(MAX_VOICES)]
        self._current_sample: Sample | None = None
        self._stop_last_on_retrigger: bool = VOICE_STOP_LAST_ON_RETRIGGER_DEFAULT

        # Master DSP chain
        self._gain = GainEffect(sample_rate)
        self._chorus = Chorus(sample_rate)
        self._delay = Delay(sample_rate)
        self._reverb = Reverb(sample_rate)
        self._limiter = Limiter(sample_rate)

    # ── Sample management ────────────────────────────────────────────────

    def set_sample(self, sample: Sample) -> None:
        """Set the sample all voices will play."""
        self._current_sample = sample

    # ── Note events ──────────────────────────────────────────────────────

    def note_on(self, note: int, velocity: int, start_frame: int = 0, end_frame: int = 0) -> None:
        """Allocate a voice for *note* (steal the oldest if all are busy)."""
        if self._current_sample is None:
            return

        if self._stop_last_on_retrigger:
            self._stop_active_voices_for_note(note)

        voice = self._find_free_voice()
        if voice is None:
            voice = self._steal_oldest_voice()

        voice.note_on(note, velocity, self._current_sample, start_frame, end_frame)

    def note_off(self, note: int) -> None:
        """Release all voices currently playing *note*."""
        for v in self.voices:
            if v.is_active and v.note == note:
                v.note_off()

    def all_notes_off(self) -> None:
        """Release all active voices."""
        for v in self.voices:
            if v.is_active:
                v.note_off()

    # ── Processing ───────────────────────────────────────────────────────

    def process(self, n_frames: int) -> np.ndarray:
        """Mix all active voices, apply master DSP, return ``(n_frames, 2)`` float32."""
        mix = np.zeros((n_frames, 2), dtype=np.float32)

        active = 0
        for v in self.voices:
            if v.is_active:
                mix += v.process(n_frames)
                active += 1

        # Scale down by sqrt of active voices to prevent accumulation clipping
        if active > 1:
            mix *= 1.0 / np.sqrt(active)

        # Master DSP chain
        mix = self._gain.process_maybe_bypass(mix)
        mix = self._chorus.process_maybe_bypass(mix)
        mix = self._delay.process_maybe_bypass(mix)
        mix = self._reverb.process_maybe_bypass(mix)
        mix = self._limiter.process_maybe_bypass(mix)

        return mix

    # ── Voice allocation helpers ─────────────────────────────────────────

    def _find_free_voice(self) -> Voice | None:
        for v in self.voices:
            if not v.is_active:
                return v
        return None

    def _steal_oldest_voice(self) -> Voice:
        """Return the voice that has been active the longest."""
        return min(self.voices, key=lambda v: v.start_time)

    def _stop_active_voices_for_note(self, note: int) -> None:
        """Immediately stop active voices that are already playing *note*."""
        for voice in self.voices:
            if voice.is_active and voice.note == note:
                voice.reset()

    # ── Parameters ───────────────────────────────────────────────────────

    def set_param(self, name: str, value: float) -> None:
        """Route a parameter to the correct DSP module based on prefix.

        Prefixes: ``adsr_``, ``filter_``, ``chorus_``, ``delay_``,
        ``reverb_``, ``limiter_``.  Suffix ``_bypass`` toggles bypass.
        """
        # Handle bypass commands: e.g. "chorus_bypass" → bypass chorus
        if name.endswith("_bypass"):
            prefix = name[:-7]  # strip "_bypass"
            bypassed = value >= 0.5
            self._set_bypass(prefix, bypassed)
            return

        if name.startswith("adsr_"):
            param = name[5:]
            for v in self.voices:
                v._adsr.set_param(param, value)
        elif name.startswith("filter_"):
            param = name[7:]
            for v in self.voices:
                v._filter.set_param(param, value)
        elif name.startswith("gain_"):
            self._gain.set_param(name[5:], value)
        elif name.startswith("chorus_"):
            self._chorus.set_param(name[7:], value)
        elif name.startswith("delay_"):
            self._delay.set_param(name[6:], value)
        elif name.startswith("reverb_"):
            self._reverb.set_param(name[7:], value)
        elif name.startswith("limiter_"):
            self._limiter.set_param(name[8:], value)
        elif name.startswith("slice_"):
            param = name[6:]
            if param == "num_slices":
                self.set_slice_config(int(value), self.voices[0]._slice_start_note)
            elif param == "start_note":
                self.set_slice_config(self.voices[0]._slice_num_slices, int(value))
            elif param == "region_start":
                self.set_slice_region(int(value), self.voices[0]._slice_region_end)
            elif param == "region_end":
                self.set_slice_region(self.voices[0]._slice_region_start, int(value))
        elif name == "voice_stop_last_on_retrigger":
            self._stop_last_on_retrigger = value >= 0.5

    def get_param(self, name: str) -> float:
        """Read back a parameter value."""
        if name.startswith("adsr_"):
            return self.voices[0]._adsr.get_param(name[5:])
        if name.startswith("filter_"):
            return self.voices[0]._filter.get_param(name[7:])
        if name.startswith("gain_"):
            return self._gain.get_param(name[5:])
        if name.startswith("chorus_"):
            return self._chorus.get_param(name[7:])
        if name.startswith("delay_"):
            return self._delay.get_param(name[6:])
        if name.startswith("reverb_"):
            return self._reverb.get_param(name[7:])
        if name.startswith("limiter_"):
            return self._limiter.get_param(name[8:])
        if name == "voice_stop_last_on_retrigger":
            return 1.0 if self._stop_last_on_retrigger else 0.0
        return 0.0

    # ── Bypass control ─────────────────────────────────────────────────────

    def _set_bypass(self, prefix: str, bypassed: bool) -> None:
        """Toggle bypass on the DSP module identified by *prefix*."""
        if prefix == "adsr":
            for v in self.voices:
                v._adsr.bypassed = bypassed
        elif prefix == "filter":
            for v in self.voices:
                v._filter.bypassed = bypassed
        elif prefix == "gain":
            self._gain.bypassed = bypassed
        elif prefix == "pitch_shift":
            for v in self.voices:
                v._pitch_shifter.bypassed = bypassed
        elif prefix == "chorus":
            self._chorus.bypassed = bypassed
        elif prefix == "delay":
            self._delay.bypassed = bypassed
        elif prefix == "reverb":
            self._reverb.bypassed = bypassed
        elif prefix == "limiter":
            self._limiter.bypassed = bypassed
        elif prefix == "granular":
            for v in self.voices:
                v._granular.bypassed = bypassed

    # ── Mode ─────────────────────────────────────────────────────────────

    def set_playback_mode(self, mode: PlaybackMode) -> None:
        """Set the playback mode for all voices."""
        for v in self.voices:
            v.mode = mode

    def set_slice_config(self, num_slices: int, start_note: int) -> None:
        """Configure slice mode parameters for all voices."""
        for v in self.voices:
            v.set_slice_config(num_slices, start_note)

    def set_slice_region(self, start_frame: int, end_frame: int) -> None:
        """Configure slice mode source region for all voices."""
        for v in self.voices:
            v.set_slice_region(start_frame, end_frame)

    # ── State ────────────────────────────────────────────────────────────

    @property
    def active_voice_count(self) -> int:
        return sum(1 for v in self.voices if v.is_active)

    def reset(self) -> None:
        """Reset all voices and master DSP."""
        for v in self.voices:
            v.reset()
        self._chorus.reset()
        self._delay.reset()
        self._reverb.reset()
        self._limiter.reset()
