"""Algorithmic Freeverb implementation."""
from __future__ import annotations

import numpy as np

from vynth.dsp.base import DSPEffect

# Comb filter delay lengths at 48 kHz reference rate.
_COMB_DELAYS = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
_ALLPASS_DELAYS = [556, 441, 341, 225]
_STEREO_SPREAD = 23  # extra samples for right channel


class _CombFilter:
    __slots__ = ("_buffer", "_size", "_idx", "_filter_state")

    def __init__(self, delay_length: int) -> None:
        self._size = delay_length
        self._buffer = np.zeros(delay_length, dtype=np.float32)
        self._idx = 0
        self._filter_state: float = 0.0

    def process(self, inp: float, room_size: float, damp1: float, damp2: float) -> float:
        out = self._buffer[self._idx]
        self._filter_state = out * damp2 + self._filter_state * damp1
        self._buffer[self._idx] = inp + self._filter_state * room_size
        self._idx += 1
        if self._idx >= self._size:
            self._idx = 0
        return out

    def reset(self) -> None:
        self._buffer[:] = 0.0
        self._idx = 0
        self._filter_state = 0.0


class _AllpassFilter:
    __slots__ = ("_buffer", "_size", "_idx")

    def __init__(self, delay_length: int) -> None:
        self._size = delay_length
        self._buffer = np.zeros(delay_length, dtype=np.float32)
        self._idx = 0

    def process(self, inp: float) -> float:
        buf_out = self._buffer[self._idx]
        out = -inp + buf_out
        self._buffer[self._idx] = inp + buf_out * 0.5
        self._idx += 1
        if self._idx >= self._size:
            self._idx = 0
        return out

    def reset(self) -> None:
        self._buffer[:] = 0.0
        self._idx = 0


class Reverb(DSPEffect):
    """Freeverb — 8 parallel comb filters into 4 series allpass filters, per channel."""

    def __init__(self, sample_rate: int = 48_000) -> None:
        super().__init__(sample_rate)
        self._params.update(
            room_size=0.5,
            damping=0.5,
            wet=0.3,
            dry=0.7,
            width=1.0,
        )
        scale = sample_rate / 48_000
        self._combs_l = [_CombFilter(int(d * scale)) for d in _COMB_DELAYS]
        self._combs_r = [_CombFilter(int((d + _STEREO_SPREAD) * scale)) for d in _COMB_DELAYS]
        self._allpasses_l = [_AllpassFilter(int(d * scale)) for d in _ALLPASS_DELAYS]
        self._allpasses_r = [_AllpassFilter(int(d * scale)) for d in _ALLPASS_DELAYS]

    # ------------------------------------------------------------------
    def process(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            left = data
            right = data.copy()
        else:
            left = data[:, 0]
            right = data[:, 1] if data.shape[1] > 1 else data[:, 0].copy()

        room = float(self._params["room_size"])
        damp = float(self._params["damping"])
        wet = float(self._params["wet"])
        dry = float(self._params["dry"])
        width = float(self._params["width"])

        damp1 = damp
        damp2 = 1.0 - damp

        out_l = np.empty_like(left)
        out_r = np.empty_like(right)

        combs_l = self._combs_l
        combs_r = self._combs_r
        aps_l = self._allpasses_l
        aps_r = self._allpasses_r

        for i in range(len(left)):
            inp = (left[i] + right[i]) * 0.5
            sum_l = 0.0
            sum_r = 0.0
            for c in combs_l:
                sum_l += c.process(inp, room, damp1, damp2)
            for c in combs_r:
                sum_r += c.process(inp, room, damp1, damp2)
            for ap in aps_l:
                sum_l = ap.process(sum_l)
            for ap in aps_r:
                sum_r = ap.process(sum_r)

            wet1 = wet * (1.0 + width) * 0.5
            wet2 = wet * (1.0 - width) * 0.5
            out_l[i] = sum_l * wet1 + sum_r * wet2 + left[i] * dry
            out_r[i] = sum_r * wet1 + sum_l * wet2 + right[i] * dry

        return np.column_stack((out_l, out_r))

    # ------------------------------------------------------------------
    def reset(self) -> None:
        for c in self._combs_l + self._combs_r:
            c.reset()
        for ap in self._allpasses_l + self._allpasses_r:
            ap.reset()
