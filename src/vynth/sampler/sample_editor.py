"""Non-destructive sample editing operations."""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly
from math import gcd

from vynth.sampler.sample import LoopRegion, Sample
from vynth.utils.audio_utils import crossfade, normalize, mono_to_stereo, stereo_to_mono


class SampleEditor:
    """Operations on Sample objects — all return new Sample instances."""

    @staticmethod
    def _resolve_region(
        sample: Sample,
        start_frame: int | None,
        end_frame: int | None,
    ) -> tuple[int, int]:
        if start_frame is None or end_frame is None:
            return (0, sample.length)

        start = max(0, int(start_frame))
        end = min(sample.length, int(end_frame))
        if start >= end:
            raise ValueError("start_frame must be less than end_frame.")
        return (start, end)

    @staticmethod
    def trim(sample: Sample, start_frame: int, end_frame: int) -> Sample:
        """Return a new Sample trimmed to [start_frame, end_frame)."""
        start_frame = max(0, start_frame)
        end_frame = min(sample.length, end_frame)
        if start_frame >= end_frame:
            raise ValueError("start_frame must be less than end_frame.")
        data = sample.data[start_frame:end_frame].copy()
        return Sample.from_buffer(data, sample.sample_rate, sample.name)

    @staticmethod
    def normalize(
        sample: Sample,
        peak: float = 0.95,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> Sample:
        """Return a copy with peak-normalized region (or whole sample)."""
        start, end = SampleEditor._resolve_region(sample, start_frame, end_frame)
        data = sample.data.copy()
        data[start:end] = normalize(data[start:end], peak)
        return Sample.from_buffer(data, sample.sample_rate, sample.name)

    @staticmethod
    def reverse(
        sample: Sample,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> Sample:
        """Return a copy with reversed region (or whole sample)."""
        start, end = SampleEditor._resolve_region(sample, start_frame, end_frame)
        data = sample.data.copy()
        data[start:end] = data[start:end][::-1]
        return Sample.from_buffer(data, sample.sample_rate, sample.name)

    @staticmethod
    def fade_in(
        sample: Sample,
        duration_ms: float,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> Sample:
        """Apply a linear fade-in to region start (or whole sample)."""
        start, end = SampleEditor._resolve_region(sample, start_frame, end_frame)
        region_len = end - start
        n_frames = int(sample.sample_rate * duration_ms / 1000.0)
        n_frames = min(n_frames, region_len)
        if n_frames <= 0:
            return Sample.from_buffer(sample.data.copy(), sample.sample_rate, sample.name)

        data = sample.data.copy()
        ramp = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
        if data.ndim == 1:
            data[start:start + n_frames] *= ramp
        else:
            data[start:start + n_frames] *= ramp[:, None]
        return Sample.from_buffer(data, sample.sample_rate, sample.name)

    @staticmethod
    def fade_out(
        sample: Sample,
        duration_ms: float,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> Sample:
        """Apply a linear fade-out to region end (or whole sample)."""
        start, end = SampleEditor._resolve_region(sample, start_frame, end_frame)
        region_len = end - start
        n_frames = int(sample.sample_rate * duration_ms / 1000.0)
        n_frames = min(n_frames, region_len)
        if n_frames <= 0:
            return Sample.from_buffer(sample.data.copy(), sample.sample_rate, sample.name)

        data = sample.data.copy()
        ramp = np.linspace(1.0, 0.0, n_frames, dtype=np.float32)
        if data.ndim == 1:
            data[end - n_frames:end] *= ramp
        else:
            data[end - n_frames:end] *= ramp[:, None]
        return Sample.from_buffer(data, sample.sample_rate, sample.name)

    @staticmethod
    def set_loop_points(sample: Sample, start: int, end: int,
                        crossfade_len: int = 256) -> Sample:
        """Return a copy with loop region set (validated)."""
        start = max(0, start)
        end = min(sample.length, end)
        if start >= end:
            raise ValueError("Loop start must be before loop end.")
        crossfade_len = min(crossfade_len, (end - start) // 2)

        data = sample.data.copy()
        new = Sample.from_buffer(data, sample.sample_rate, sample.name)
        new.loop = LoopRegion(
            start=start, end=end, crossfade=crossfade_len, enabled=True,
        )
        return new

    @staticmethod
    def apply_crossfade_loop(sample: Sample) -> Sample:
        """Crossfade the loop region boundaries for seamless looping."""
        loop = sample.loop
        if not loop.enabled or loop.end <= loop.start:
            return Sample.from_buffer(sample.data.copy(), sample.sample_rate, sample.name)

        xfade_len = min(loop.crossfade, (loop.end - loop.start) // 2)
        if xfade_len <= 0:
            return Sample.from_buffer(sample.data.copy(), sample.sample_rate, sample.name)

        data = sample.data.copy()

        # Crossfade: blend the end of the loop region with its start
        loop_end_region = data[loop.end - xfade_len:loop.end].copy()
        loop_start_region = data[loop.start:loop.start + xfade_len].copy()

        fade_out = np.linspace(1.0, 0.0, xfade_len, dtype=np.float32)
        fade_in = 1.0 - fade_out

        if data.ndim == 1:
            blended = loop_end_region * fade_out + loop_start_region * fade_in
        else:
            blended = (loop_end_region * fade_out[:, None]
                       + loop_start_region * fade_in[:, None])

        data[loop.end - xfade_len:loop.end] = blended

        new = Sample.from_buffer(data, sample.sample_rate, sample.name)
        new.loop = LoopRegion(
            start=loop.start, end=loop.end,
            crossfade=loop.crossfade, enabled=True,
        )
        return new

    @staticmethod
    def resample(sample: Sample, target_sr: int) -> Sample:
        """Return a copy resampled to *target_sr*."""
        if target_sr == sample.sample_rate:
            return Sample.from_buffer(sample.data.copy(), sample.sample_rate, sample.name)
        if target_sr <= 0:
            raise ValueError("Target sample rate must be positive.")

        divisor = gcd(sample.sample_rate, target_sr)
        up = target_sr // divisor
        down = sample.sample_rate // divisor

        if sample.data.ndim == 1:
            resampled = resample_poly(sample.data, up, down).astype(np.float32)
        else:
            channels = [
                resample_poly(sample.data[:, ch], up, down).astype(np.float32)
                for ch in range(sample.data.shape[1])
            ]
            resampled = np.column_stack(channels)

        return Sample.from_buffer(resampled, target_sr, sample.name)

    @staticmethod
    def to_mono(sample: Sample) -> Sample:
        """Mix down to mono."""
        data = stereo_to_mono(sample.data)
        return Sample.from_buffer(data, sample.sample_rate, sample.name)

    @staticmethod
    def to_stereo(sample: Sample) -> Sample:
        """Duplicate mono to stereo."""
        data = mono_to_stereo(sample.data)
        return Sample.from_buffer(data, sample.sample_rate, sample.name)
