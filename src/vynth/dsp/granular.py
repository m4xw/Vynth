"""Granular synthesis engine for voice sampler."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal.windows import gaussian, hann, triang

from vynth.dsp.base import DSPEffect

_MAX_GRAINS = 128


def _build_window(window_type: int, length: int) -> np.ndarray:
    """Return a normalised amplitude envelope of *length* samples."""
    if length < 2:
        return np.ones(1, dtype=np.float32)
    if window_type == 1:
        # scipy gaussian needs a std-dev param; 0.4*length gives a nice taper
        win = gaussian(length, std=0.4 * length).astype(np.float32)
    elif window_type == 2:
        win = triang(length).astype(np.float32)
    else:
        win = hann(length).astype(np.float32)
    peak = win.max()
    if peak > 0.0:
        win /= peak
    return win


@dataclass
class _Grain:
    """State for a single active grain."""

    source: np.ndarray  # reference to mono source buffer
    read_pos: float  # fractional sample position into source
    read_inc: float  # per-output-sample increment (pitch_ratio)
    envelope: np.ndarray  # pre-computed window
    env_idx: int = 0  # current index into envelope
    pan: float = 0.0  # -1 … +1

    @property
    def alive(self) -> bool:
        return self.env_idx < len(self.envelope)


class GranularEngine(DSPEffect):
    """Granular synthesis processor.

    Call :meth:`set_source` to load an audio buffer, then repeatedly call
    :meth:`process` to render stereo output.
    """

    def __init__(self, sample_rate: int = 48_000) -> None:
        super().__init__(sample_rate)

        # --- default parameters ---
        self._params.update(
            {
                "grain_size_ms": 50.0,
                "overlap": 0.5,
                "scatter": 0.1,
                "density": 10.0,
                "pitch_ratio": 1.0,
                "position": 0.5,
                "spread": 0.5,
                "window_type": 0.0,
            }
        )

        # --- internal state ---
        self._source: np.ndarray | None = None  # mono float32
        self._source_sr: int = sample_rate
        self._grains: list[_Grain] = []
        self._rng = np.random.default_rng()

        # scheduler accumulator: counts samples since last grain spawn
        self._sched_acc: float = 0.0

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def set_source(self, buffer: np.ndarray, sample_rate: int) -> None:
        """Set the audio source buffer.

        *buffer* can be mono ``(N,)`` or stereo ``(N, 2)``; stereo is mixed
        down to mono internally.  The reference is stored — the caller must
        not mutate the array while it is in use.
        """
        buf = np.asarray(buffer, dtype=np.float32)
        if buf.ndim == 2:
            buf = buf.mean(axis=1)
        # atomic reference swap — safe from the audio thread
        self._source = buf
        self._source_sr = sample_rate

    # ------------------------------------------------------------------
    # DSPEffect interface
    # ------------------------------------------------------------------

    def process(self, data: np.ndarray | int) -> np.ndarray:
        """Generate stereo granular output.

        *data* is either an integer frame count **or** an ndarray whose first
        dimension is used as the frame count (conforming to the DSPEffect
        contract).
        """
        if isinstance(data, np.ndarray):
            n_frames = data.shape[0]
        else:
            n_frames = int(data)

        out = np.zeros((n_frames, 2), dtype=np.float32)

        if self._source is None or len(self._source) < 2:
            return out

        # cache parameters once per block
        grain_size_ms: float = np.clip(self._params["grain_size_ms"], 5.0, 500.0)
        scatter: float = np.clip(self._params["scatter"], 0.0, 1.0)
        density: float = np.clip(self._params["density"], 1.0, 50.0)
        pitch_ratio: float = np.clip(self._params["pitch_ratio"], 0.25, 4.0)
        position: float = np.clip(self._params["position"], 0.0, 1.0)
        spread: float = np.clip(self._params["spread"], 0.0, 1.0)
        window_type: int = int(np.clip(self._params["window_type"], 0, 2))

        grain_size_samples = int(grain_size_ms * 0.001 * self._sample_rate)
        if grain_size_samples < 2:
            grain_size_samples = 2

        spawn_interval: float = self._sample_rate / density  # samples between spawns

        source = self._source
        source_len = len(source)

        for i in range(n_frames):
            # --- scheduler ---
            self._sched_acc += 1.0
            if self._sched_acc >= spawn_interval:
                self._sched_acc -= spawn_interval
                if len(self._grains) < _MAX_GRAINS:
                    self._spawn_grain(
                        source,
                        source_len,
                        grain_size_samples,
                        position,
                        scatter,
                        pitch_ratio,
                        spread,
                        window_type,
                    )

            # --- mix active grains ---
            left: float = 0.0
            right: float = 0.0
            for g in self._grains:
                amp = g.envelope[g.env_idx]
                sample = _interp_sample(source, g.read_pos, source_len)
                val = sample * amp

                # constant-power-ish pan
                r_gain = (g.pan + 1.0) * 0.5  # 0…1
                l_gain = 1.0 - r_gain
                left += val * l_gain
                right += val * r_gain

                g.read_pos += g.read_inc
                g.env_idx += 1

            out[i, 0] = left
            out[i, 1] = right

            # --- reap finished grains ---
            self._grains = [g for g in self._grains if g.alive]

        return out

    def reset(self) -> None:
        """Clear all active grains and the scheduler accumulator."""
        self._grains.clear()
        self._sched_acc = 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn_grain(
        self,
        source: np.ndarray,
        source_len: int,
        grain_size_samples: int,
        position: float,
        scatter: float,
        pitch_ratio: float,
        spread: float,
        window_type: int,
    ) -> None:
        center = position * source_len
        offset = scatter * grain_size_samples * (self._rng.random() * 2.0 - 1.0)
        start = center + offset
        # clamp into valid range
        start = max(0.0, min(float(source_len - 1), start))

        envelope = _build_window(window_type, grain_size_samples)
        pan = spread * (self._rng.random() * 2.0 - 1.0)

        self._grains.append(
            _Grain(
                source=source,
                read_pos=start,
                read_inc=pitch_ratio,
                envelope=envelope,
                env_idx=0,
                pan=float(pan),
            )
        )


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def _interp_sample(buf: np.ndarray, pos: float, length: int) -> float:
    """Linear interpolation read with wrapping."""
    idx0 = int(pos) % length
    idx1 = (idx0 + 1) % length
    frac = pos - int(pos)
    return float(buf[idx0] + (buf[idx1] - buf[idx0]) * frac)
