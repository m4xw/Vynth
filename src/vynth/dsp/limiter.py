"""Lookahead soft limiter for the master bus."""
from __future__ import annotations

import numpy as np

from vynth.dsp.base import DSPEffect


class Limiter(DSPEffect):
    """Stereo lookahead limiter with instant attack and configurable release.

    Parameters
    ----------
    threshold_db : float
        Ceiling in dB (−20 … 0, default −1).
    release_ms : float
        Release time in milliseconds (10 … 500, default 100).
    lookahead_ms : float
        Lookahead in milliseconds (0 … 10, default 5).
    """

    PARAM_DEFAULTS: dict[str, tuple[float, float, float]] = {
        "threshold_db": (-20.0, 0.0, -1.0),
        "release_ms": (10.0, 500.0, 100.0),
        "lookahead_ms": (0.0, 10.0, 5.0),
    }

    def __init__(self, sample_rate: int = 48_000):
        super().__init__(sample_rate)
        for name, (lo, hi, default) in self.PARAM_DEFAULTS.items():
            self._params[name] = default

        # Derived state — built on first access / param change
        self._threshold_lin: float = 0.0
        self._release_coeff: float = 0.0
        self._lookahead_samples: int = 0
        self._delay_buf: np.ndarray = np.empty(0, dtype=np.float64)
        self._delay_pos: int = 0
        self._envelope: float = 0.0

        self._rebuild_internals()

    # ------------------------------------------------------------------
    def _on_param_changed(self, name: str, value: float) -> None:
        lo, hi, _ = self.PARAM_DEFAULTS[name]
        self._params[name] = float(np.clip(value, lo, hi))
        self._rebuild_internals()

    # ------------------------------------------------------------------
    def _rebuild_internals(self) -> None:
        threshold_db = self._params["threshold_db"]
        release_ms = self._params["release_ms"]
        lookahead_ms = self._params["lookahead_ms"]
        sr = self._sample_rate

        self._threshold_lin = 10.0 ** (threshold_db / 20.0)

        # Release coefficient — one-pole smoother
        release_samples = max(1.0, release_ms * 0.001 * sr)
        self._release_coeff = np.exp(-1.0 / release_samples)

        # Lookahead delay line (stereo, circular)
        self._lookahead_samples = max(1, int(round(lookahead_ms * 0.001 * sr)))
        buf_len = self._lookahead_samples
        self._delay_buf = np.zeros((buf_len, 2), dtype=np.float64)
        self._delay_pos = 0
        self._envelope = 0.0

    # ------------------------------------------------------------------
    def process(self, data: np.ndarray) -> np.ndarray:
        """Limit a block of audio.

        Parameters
        ----------
        data : np.ndarray
            Mono ``(frames,)`` or stereo ``(frames, 2)`` float32 audio.

        Returns
        -------
        np.ndarray
            Limited audio, same shape and dtype as *data*.
        """
        if data.size == 0:
            return data

        mono = data.ndim == 1
        work = data.astype(np.float64)
        if mono:
            work = work[:, np.newaxis]
        if work.shape[1] == 1:
            work = np.column_stack([work[:, 0], work[:, 0]])
            was_mono = True
        else:
            was_mono = False

        frames = work.shape[0]
        out = np.empty_like(work)
        threshold = self._threshold_lin
        release_coeff = self._release_coeff
        buf = self._delay_buf
        buf_len = buf.shape[0]
        pos = self._delay_pos
        env = self._envelope

        for n in range(frames):
            # Read delayed sample
            delayed = buf[pos].copy()

            # Write current sample into delay buffer
            buf[pos, 0] = work[n, 0]
            buf[pos, 1] = work[n, 1]
            pos = (pos + 1) % buf_len

            # Peak detect over lookahead window
            peak = max(abs(work[n, 0]), abs(work[n, 1]))

            # Target gain
            if peak > threshold:
                target_env = peak
            else:
                target_env = threshold

            # Envelope follower: instant attack, smooth release
            if target_env > env:
                env = target_env  # instant attack
            else:
                env = release_coeff * env + (1.0 - release_coeff) * target_env

            # Compute gain reduction
            if env > threshold:
                gain = threshold / env
            else:
                gain = 1.0

            out[n, 0] = delayed[0] * gain
            out[n, 1] = delayed[1] * gain

        self._delay_pos = pos
        self._envelope = env

        if mono or was_mono:
            out = out[:, 0]
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._delay_buf[:] = 0.0
        self._delay_pos = 0
        self._envelope = 0.0
