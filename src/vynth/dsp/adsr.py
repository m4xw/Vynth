"""Sample-accurate ADSR envelope generator."""
from __future__ import annotations

import enum
import math

import numpy as np

from vynth.config import (
    ADSR_DEFAULT_ATTACK_MS,
    ADSR_DEFAULT_DECAY_MS,
    ADSR_DEFAULT_RELEASE_MS,
    ADSR_DEFAULT_SUSTAIN,
)
from vynth.dsp.base import DSPEffect


class ADSRState(enum.IntEnum):
    IDLE = 0
    ATTACK = 1
    DECAY = 2
    SUSTAIN = 3
    RELEASE = 4


class ADSREnvelope(DSPEffect):
    """Exponential ADSR envelope with sample-accurate processing."""

    _SILENCE_THRESHOLD = 1e-4

    def __init__(self, sample_rate: int = 48_000) -> None:
        super().__init__(sample_rate)
        self._params.update(
            {
                "attack_ms": ADSR_DEFAULT_ATTACK_MS,
                "decay_ms": ADSR_DEFAULT_DECAY_MS,
                "sustain": ADSR_DEFAULT_SUSTAIN,
                "release_ms": ADSR_DEFAULT_RELEASE_MS,
            }
        )
        self._state = ADSRState.IDLE
        self._level: float = 0.0
        self._peak: float = 1.0

        # Pre-computed coefficients
        self._attack_coeff: float = 0.0
        self._decay_coeff: float = 0.0
        self._release_coeff: float = 0.0
        self._recalc_coefficients()

    # ── Coefficient helpers ──────────────────────────────────────────────

    @staticmethod
    def _ms_to_coeff(ms: float, sr: int) -> float:
        """Convert a time in ms to an exponential coefficient."""
        samples = max((ms / 1000.0) * sr, 1.0)
        return math.exp(-1.0 / samples)

    def _recalc_coefficients(self) -> None:
        sr = self._sample_rate
        self._attack_coeff = self._ms_to_coeff(self._params["attack_ms"], sr)
        self._decay_coeff = self._ms_to_coeff(self._params["decay_ms"], sr)
        self._release_coeff = self._ms_to_coeff(self._params["release_ms"], sr)

    def _on_param_changed(self, name: str, value: float) -> None:
        if name in {"attack_ms", "decay_ms", "release_ms"}:
            self._recalc_coefficients()

    # ── Gate control ─────────────────────────────────────────────────────

    def gate_on(self, velocity: float = 1.0) -> None:
        """Trigger the attack stage from the current level."""
        self._peak = float(np.clip(velocity, 0.0, 1.0))
        self._state = ADSRState.ATTACK

    def gate_off(self) -> None:
        """Trigger the release stage from the current level."""
        if self._state != ADSRState.IDLE:
            self._state = ADSRState.RELEASE

    # ── State query ──────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._state != ADSRState.IDLE

    @property
    def state(self) -> ADSRState:
        return self._state

    @property
    def level(self) -> float:
        return self._level

    # ── Processing ───────────────────────────────────────────────────────

    def generate(self, n_frames: int) -> np.ndarray:
        """Generate *n_frames* of envelope values (no audio input)."""
        out = np.empty(n_frames, dtype=np.float32)

        sustain = self._params["sustain"]
        level = self._level
        state = self._state
        peak = self._peak
        a_coeff = self._attack_coeff
        d_coeff = self._decay_coeff
        r_coeff = self._release_coeff

        for i in range(n_frames):
            if state == ADSRState.ATTACK:
                level = peak + a_coeff * (level - peak)
                if level >= peak - self._SILENCE_THRESHOLD:
                    level = peak
                    state = ADSRState.DECAY
            elif state == ADSRState.DECAY:
                target = sustain * peak
                level = target + d_coeff * (level - target)
                if level <= target + self._SILENCE_THRESHOLD:
                    level = target
                    state = ADSRState.SUSTAIN
            elif state == ADSRState.SUSTAIN:
                level = sustain * peak
            elif state == ADSRState.RELEASE:
                level = r_coeff * level
                if level < self._SILENCE_THRESHOLD:
                    level = 0.0
                    state = ADSRState.IDLE
            else:  # IDLE
                level = 0.0

            out[i] = level

        self._level = level
        self._state = state
        return out

    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply the envelope to an audio buffer.

        Handles mono ``(N,)`` and stereo ``(N, 2)`` arrays.
        """
        if self._bypassed:
            return data

        n_frames = data.shape[0]
        env = self.generate(n_frames)

        if data.ndim == 2:
            env = env[:, np.newaxis]

        return data * env

    def reset(self) -> None:
        self._state = ADSRState.IDLE
        self._level = 0.0
        self._peak = 1.0
