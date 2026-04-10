"""Biquad filter with multiple modes (Audio EQ Cookbook)."""
from __future__ import annotations

import math

import numpy as np

from vynth.dsp.base import DSPEffect


class BiquadFilter(DSPEffect):
    """Multi-mode biquad filter based on Robert Bristow-Johnson's Audio EQ Cookbook.

    Modes (integer):
        0 = LowPass, 1 = HighPass, 2 = BandPass, 3 = Notch,
        4 = Peak, 5 = LowShelf, 6 = HighShelf
    """

    PARAM_DEFAULTS: dict[str, tuple[float, float, float]] = {
        # name: (min, max, default)
        "frequency": (20.0, 20_000.0, 1000.0),
        "q": (0.1, 20.0, 0.707),
        "gain_db": (-24.0, 24.0, 0.0),
        "mode": (0.0, 6.0, 0.0),
    }

    def __init__(self, sample_rate: int = 48_000):
        super().__init__(sample_rate)
        for name, (lo, hi, default) in self.PARAM_DEFAULTS.items():
            self._params[name] = default

        # Coefficients
        self._b = np.zeros(3, dtype=np.float64)
        self._a = np.zeros(3, dtype=np.float64)
        self._a[0] = 1.0

        # Per-channel state: up to 2 channels, (x[n-1], x[n-2], y[n-1], y[n-2])
        self._state: np.ndarray = np.zeros((2, 4), dtype=np.float64)

        self._recalculate_coefficients()

    # ------------------------------------------------------------------
    def _on_param_changed(self, name: str, value: float):
        lo, hi, _ = self.PARAM_DEFAULTS[name]
        self._params[name] = float(np.clip(value, lo, hi))
        if name == "mode":
            self._params["mode"] = float(int(self._params["mode"]))
        self._recalculate_coefficients()

    # ------------------------------------------------------------------
    def _recalculate_coefficients(self) -> None:
        freq = self._params["frequency"]
        q = self._params["q"]
        gain_db = self._params["gain_db"]
        mode = int(self._params["mode"])
        sr = self._sample_rate

        A = 10.0 ** (gain_db / 40.0)  # sqrt of linear gain
        w0 = 2.0 * math.pi * freq / sr
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)

        if mode == 0:  # LowPass
            b0 = (1.0 - cos_w0) / 2.0
            b1 = 1.0 - cos_w0
            b2 = (1.0 - cos_w0) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif mode == 1:  # HighPass
            b0 = (1.0 + cos_w0) / 2.0
            b1 = -(1.0 + cos_w0)
            b2 = (1.0 + cos_w0) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif mode == 2:  # BandPass (constant skirt gain, peak gain = Q)
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif mode == 3:  # Notch
            b0 = 1.0
            b1 = -2.0 * cos_w0
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif mode == 4:  # Peak (parametric EQ)
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / A

        elif mode == 5:  # LowShelf
            sqrt_A = math.sqrt(A)
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
            a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha

        elif mode == 6:  # HighShelf
            sqrt_A = math.sqrt(A)
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
            a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha

        else:
            # Fallback: pass-through
            b0, b1, b2 = 1.0, 0.0, 0.0
            a0, a1, a2 = 1.0, 0.0, 0.0

        # Normalise by a0
        inv_a0 = 1.0 / a0
        self._b[:] = [b0 * inv_a0, b1 * inv_a0, b2 * inv_a0]
        self._a[:] = [1.0, a1 * inv_a0, a2 * inv_a0]

    # ------------------------------------------------------------------
    def process(self, data: np.ndarray) -> np.ndarray:
        """Process audio through the biquad filter.

        Parameters
        ----------
        data : np.ndarray
            Mono ``(frames,)`` or stereo ``(frames, 2)`` float32 audio.

        Returns
        -------
        np.ndarray
            Filtered audio, same shape and dtype as *data*.
        """
        if data.size == 0:
            return data

        mono = data.ndim == 1
        work = data.astype(np.float64)
        if mono:
            work = work[:, np.newaxis]

        num_channels = work.shape[1]
        out = np.empty_like(work)
        b0, b1, b2 = self._b
        a1, a2 = self._a[1], self._a[2]

        for ch in range(num_channels):
            x1, x2, y1, y2 = self._state[ch]
            x = work[:, ch]
            y = out[:, ch]
            for n in range(len(x)):
                xn = x[n]
                yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                y[n] = yn
                x2, x1 = x1, xn
                y2, y1 = y1, yn
            self._state[ch] = [x1, x2, y1, y2]

        if mono:
            out = out[:, 0]
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._state[:] = 0.0
