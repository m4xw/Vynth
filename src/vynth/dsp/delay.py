"""Delay line with feedback and optional ping-pong mode."""
from __future__ import annotations

import numpy as np

from vynth.dsp.base import DSPEffect


class Delay(DSPEffect):
    """Stereo delay with feedback saturation and ping-pong option."""

    _MAX_TIME_S = 2.0

    def __init__(self, sample_rate: int = 48_000) -> None:
        super().__init__(sample_rate)
        self._params.update(
            time_ms=250.0,
            feedback=0.4,
            mix=0.3,
            ping_pong=0.0,
        )
        max_samples = int(self._MAX_TIME_S * sample_rate)
        self._buf_l = np.zeros(max_samples, dtype=np.float32)
        self._buf_r = np.zeros(max_samples, dtype=np.float32)
        self._buf_len = max_samples
        self._write_idx = 0

    # ------------------------------------------------------------------
    def _delay_samples(self) -> int:
        samples = int(self._params["time_ms"] * 0.001 * self._sample_rate)
        return max(1, min(samples, self._buf_len - 1))

    # ------------------------------------------------------------------
    def process(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            left = data
            right = data.copy()
        else:
            left = data[:, 0]
            right = data[:, 1] if data.shape[1] > 1 else data[:, 0].copy()

        delay_len = self._delay_samples()
        fb = float(self._params["feedback"])
        mix = float(self._params["mix"])
        ping_pong = self._params["ping_pong"] >= 0.5

        out_l = np.empty_like(left)
        out_r = np.empty_like(right)

        buf_l = self._buf_l
        buf_r = self._buf_r
        buf_len = self._buf_len
        w = self._write_idx

        for i in range(len(left)):
            read_idx = (w - delay_len) % buf_len
            dl = buf_l[read_idx]
            dr = buf_r[read_idx]

            if ping_pong:
                # L delay feeds R, R feeds L
                fb_l = np.float32(np.tanh(dr * fb))
                fb_r = np.float32(np.tanh(dl * fb))
            else:
                fb_l = np.float32(np.tanh(dl * fb))
                fb_r = np.float32(np.tanh(dr * fb))

            buf_l[w] = left[i] + fb_l
            buf_r[w] = right[i] + fb_r

            out_l[i] = left[i] * (1.0 - mix) + dl * mix
            out_r[i] = right[i] * (1.0 - mix) + dr * mix

            w = (w + 1) % buf_len

        self._write_idx = w
        return np.column_stack((out_l, out_r))

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._buf_l[:] = 0.0
        self._buf_r[:] = 0.0
        self._write_idx = 0
