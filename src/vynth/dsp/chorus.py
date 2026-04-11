"""Chorus / Unison effect with LFO-modulated delay."""
from __future__ import annotations

import math

import numpy as np

from vynth.dsp.base import DSPEffect

_TWO_PI = 2.0 * math.pi
_MAX_DELAY_S = 0.05  # 50 ms


class Chorus(DSPEffect):
    """Multi-voice chorus with per-voice LFO, detune, and stereo spread."""

    def __init__(self, sample_rate: int = 48_000) -> None:
        super().__init__(sample_rate)
        self._params.update(
            num_voices=4.0,
            detune_cents=12.0,
            rate_hz=1.5,
            depth=0.5,
            mix=0.5,
            spread=0.7,
        )
        self._max_delay = int(_MAX_DELAY_S * sample_rate)
        self._buf_l = np.zeros(self._max_delay, dtype=np.float32)
        self._buf_r = np.zeros(self._max_delay, dtype=np.float32)
        self._write_idx = 0
        self._lfo_phases: list[float] = [0.0] * 8  # pre-allocate for max voices

    # ------------------------------------------------------------------
    @staticmethod
    def _lerp(buf: np.ndarray, idx: float, length: int) -> float:
        i0 = int(idx) % length
        i1 = (i0 + 1) % length
        frac = idx - int(idx)
        return float(buf[i0] + (buf[i1] - buf[i0]) * frac)

    # ------------------------------------------------------------------
    def process(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            left = data
            right = data.copy()
        else:
            left = data[:, 0]
            right = data[:, 1] if data.shape[1] > 1 else data[:, 0].copy()

        num_voices = max(1, min(8, int(self._params["num_voices"])))
        detune_cents = float(self._params["detune_cents"])
        rate = float(self._params["rate_hz"])
        depth = float(self._params["depth"])
        mix = float(self._params["mix"])
        spread = float(self._params["spread"])

        sr = self._sample_rate
        max_del = self._max_delay
        buf_l = self._buf_l
        buf_r = self._buf_r
        w = self._write_idx
        lfo_inc = rate / sr

        # Centre delay (half of max) — LFO swings around this point.
        centre_delay = max_del * 0.5 * depth
        if centre_delay < 1.0:
            centre_delay = 1.0

        out_l = np.empty_like(left)
        out_r = np.empty_like(right)

        phases = self._lfo_phases
        lerp = self._lerp

        for i in range(len(left)):
            # Write dry sample into circular buffer.
            buf_l[w] = left[i]
            buf_r[w] = right[i]

            wet_l = 0.0
            wet_r = 0.0

            for v in range(num_voices):
                # Each voice has evenly-distributed LFO phase offset.
                phase = phases[v]
                lfo = math.sin(_TWO_PI * phase)

                # Delay modulation (in samples).
                delay_samples = centre_delay + centre_delay * lfo

                # Pitch detune — slight speed deviation per voice.
                voice_detune = detune_cents * ((v / (num_voices - 1)) - 0.5) * 2.0 if num_voices > 1 else 0.0
                speed_ratio = 2.0 ** (voice_detune / 1200.0)
                delay_samples *= speed_ratio

                delay_samples = max(1.0, min(delay_samples, max_del - 1.0))
                read_pos = (w - delay_samples) % max_del

                samp_l = lerp(buf_l, read_pos, max_del)
                samp_r = lerp(buf_r, read_pos, max_del)

                # Stereo spread: pan voice across field.
                pan = 0.5 + spread * ((v / max(num_voices - 1, 1)) - 0.5)
                gain_l = math.cos(pan * math.pi * 0.5)
                gain_r = math.sin(pan * math.pi * 0.5)

                wet_l += samp_l * gain_l
                wet_r += samp_r * gain_r

                phases[v] = (phase + lfo_inc + v * lfo_inc * 0.01) % 1.0

            inv_voices = 1.0 / num_voices
            wet_l *= inv_voices
            wet_r *= inv_voices

            out_l[i] = left[i] * (1.0 - mix) + wet_l * mix
            out_r[i] = right[i] * (1.0 - mix) + wet_r * mix

            w = (w + 1) % max_del

        self._write_idx = w
        return np.column_stack((out_l, out_r))

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._buf_l[:] = 0.0
        self._buf_r[:] = 0.0
        self._write_idx = 0
        num_voices = max(1, min(8, int(self._params["num_voices"])))
        for v in range(len(self._lfo_phases)):
            self._lfo_phases[v] = v / num_voices if v < num_voices else 0.0
