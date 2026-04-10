"""64-voice polyphonic voice allocator with voice stealing and master DSP."""
from __future__ import annotations

import numpy as np

from vynth.config import MAX_VOICES, SAMPLE_RATE
from vynth.dsp.chorus import Chorus
from vynth.dsp.delay import Delay
from vynth.dsp.limiter import Limiter
from vynth.dsp.reverb import Reverb
from vynth.engine.voice import PlaybackMode, Voice
from vynth.sampler.sample import Sample


# Parameter name prefix → DSP target
_PARAM_PREFIXES = ("adsr_", "filter_", "chorus_", "delay_", "reverb_", "limiter_")


class VoiceAllocator:
    """Manages :data:`MAX_VOICES` Voice instances with voice stealing.

    Master DSP chain (applied after voice mix): Chorus → Delay → Reverb → Limiter.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self.voices: list[Voice] = [Voice(i, sample_rate) for i in range(MAX_VOICES)]
        self._current_sample: Sample | None = None

        # Master DSP chain
        self._chorus = Chorus(sample_rate)
        self._delay = Delay(sample_rate)
        self._reverb = Reverb(sample_rate)
        self._limiter = Limiter(sample_rate)

    # ── Sample management ────────────────────────────────────────────────

    def set_sample(self, sample: Sample) -> None:
        """Set the sample all voices will play."""
        self._current_sample = sample

    # ── Note events ──────────────────────────────────────────────────────

    def note_on(self, note: int, velocity: int) -> None:
        """Allocate a voice for *note* (steal the oldest if all are busy)."""
        if self._current_sample is None:
            return

        voice = self._find_free_voice()
        if voice is None:
            voice = self._steal_oldest_voice()

        voice.note_on(note, velocity, self._current_sample)

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

        for v in self.voices:
            if v.is_active:
                mix += v.process(n_frames)

        # Master DSP chain
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

    # ── Parameters ───────────────────────────────────────────────────────

    def set_param(self, name: str, value: float) -> None:
        """Route a parameter to the correct DSP module based on prefix.

        Prefixes: ``adsr_``, ``filter_``, ``chorus_``, ``delay_``,
        ``reverb_``, ``limiter_``.
        """
        if name.startswith("adsr_"):
            param = name[5:]
            for v in self.voices:
                v._adsr.set_param(param, value)
        elif name.startswith("filter_"):
            param = name[7:]
            for v in self.voices:
                v._filter.set_param(param, value)
        elif name.startswith("chorus_"):
            self._chorus.set_param(name[7:], value)
        elif name.startswith("delay_"):
            self._delay.set_param(name[6:], value)
        elif name.startswith("reverb_"):
            self._reverb.set_param(name[7:], value)
        elif name.startswith("limiter_"):
            self._limiter.set_param(name[8:], value)

    def get_param(self, name: str) -> float:
        """Read back a parameter value."""
        if name.startswith("adsr_"):
            return self.voices[0]._adsr.get_param(name[5:])
        if name.startswith("filter_"):
            return self.voices[0]._filter.get_param(name[7:])
        if name.startswith("chorus_"):
            return self._chorus.get_param(name[7:])
        if name.startswith("delay_"):
            return self._delay.get_param(name[6:])
        if name.startswith("reverb_"):
            return self._reverb.get_param(name[7:])
        if name.startswith("limiter_"):
            return self._limiter.get_param(name[8:])
        return 0.0

    # ── Mode ─────────────────────────────────────────────────────────────

    def set_playback_mode(self, mode: PlaybackMode) -> None:
        """Set the playback mode for all voices."""
        for v in self.voices:
            v.mode = mode

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
