"""Exhaustive tests for SampleEditor operations."""
import numpy as np
import pytest

from vynth.sampler.sample import Sample, LoopRegion
from vynth.sampler.sample_editor import SampleEditor


def _make_sample(sr=48000, duration_s=0.5, name="test"):
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    data = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    return Sample.from_buffer(data, sr, name)


def _make_stereo_sample(sr=48000, duration_s=0.5, name="stereo"):
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    mono = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    data = np.column_stack([mono, mono * 0.5])
    return Sample.from_buffer(data, sr, name)


class TestTrim:
    def test_trim_basic(self):
        s = _make_sample()
        trimmed = SampleEditor.trim(s, 1000, 5000)
        assert trimmed.length == 4000

    def test_trim_start_zero(self):
        s = _make_sample()
        trimmed = SampleEditor.trim(s, 0, 1000)
        assert trimmed.length == 1000

    def test_trim_to_end(self):
        s = _make_sample()
        trimmed = SampleEditor.trim(s, s.length - 1000, s.length)
        assert trimmed.length == 1000

    def test_trim_clamps_bounds(self):
        s = _make_sample()
        trimmed = SampleEditor.trim(s, -100, s.length + 100)
        assert trimmed.length == s.length

    def test_trim_invalid_range_raises(self):
        s = _make_sample()
        with pytest.raises(ValueError):
            SampleEditor.trim(s, 5000, 1000)

    def test_trim_preserves_sample_rate(self):
        s = _make_sample()
        trimmed = SampleEditor.trim(s, 0, 1000)
        assert trimmed.sample_rate == s.sample_rate


class TestNormalize:
    def test_normalize_peak(self):
        s = _make_sample()
        normalized = SampleEditor.normalize(s)
        assert np.max(np.abs(normalized.data)) == pytest.approx(0.95, abs=0.01)

    def test_normalize_custom_peak(self):
        s = _make_sample()
        normalized = SampleEditor.normalize(s, peak=0.5)
        assert np.max(np.abs(normalized.data)) == pytest.approx(0.5, abs=0.01)

    def test_normalize_returns_new_sample(self):
        s = _make_sample()
        normalized = SampleEditor.normalize(s)
        assert normalized is not s

    def test_normalize_preserves_shape(self):
        s = _make_stereo_sample()
        normalized = SampleEditor.normalize(s)
        assert normalized.data.shape == s.data.shape

    def test_normalize_selection_only(self):
        s = _make_sample(duration_s=0.2)
        start, end = 1000, 3000
        normalized = SampleEditor.normalize(s, start_frame=start, end_frame=end)
        np.testing.assert_allclose(normalized.data[:start], s.data[:start], atol=1e-6)
        np.testing.assert_allclose(normalized.data[end:], s.data[end:], atol=1e-6)
        assert np.max(np.abs(normalized.data[start:end])) == pytest.approx(0.95, abs=0.01)


class TestReverse:
    def test_reverse_basic(self):
        s = _make_sample()
        rev = SampleEditor.reverse(s)
        np.testing.assert_array_equal(rev.data, s.data[::-1])

    def test_reverse_twice_original(self):
        s = _make_sample()
        rev = SampleEditor.reverse(SampleEditor.reverse(s))
        np.testing.assert_allclose(rev.data, s.data, atol=1e-7)

    def test_reverse_preserves_length(self):
        s = _make_sample()
        rev = SampleEditor.reverse(s)
        assert rev.length == s.length

    def test_reverse_selection_only(self):
        s = _make_sample(duration_s=0.2)
        start, end = 900, 2400
        rev = SampleEditor.reverse(s, start_frame=start, end_frame=end)
        np.testing.assert_allclose(rev.data[:start], s.data[:start], atol=1e-7)
        np.testing.assert_allclose(rev.data[end:], s.data[end:], atol=1e-7)
        np.testing.assert_allclose(rev.data[start:end], s.data[start:end][::-1], atol=1e-7)


class TestFadeIn:
    def test_fade_in_starts_zero(self):
        s = _make_sample()
        faded = SampleEditor.fade_in(s, 100.0)
        assert abs(faded.data[0]) < 0.01

    def test_fade_in_end_unaffected(self):
        s = _make_sample()
        faded = SampleEditor.fade_in(s, 10.0)
        # Far from the fade region, data should be unchanged
        np.testing.assert_allclose(faded.data[-100:], s.data[-100:], atol=1e-6)

    def test_fade_in_zero_ms(self):
        s = _make_sample()
        faded = SampleEditor.fade_in(s, 0.0)
        np.testing.assert_allclose(faded.data, s.data, atol=1e-6)

    def test_fade_in_stereo(self):
        s = _make_stereo_sample()
        faded = SampleEditor.fade_in(s, 50.0)
        assert abs(faded.data[0, 0]) < 0.01
        assert abs(faded.data[0, 1]) < 0.01

    def test_fade_in_selection_only(self):
        s = _make_sample(duration_s=0.3)
        start, end = 2000, 5000
        faded = SampleEditor.fade_in(s, 20.0, start_frame=start, end_frame=end)
        np.testing.assert_allclose(faded.data[:start], s.data[:start], atol=1e-6)
        np.testing.assert_allclose(faded.data[end:], s.data[end:], atol=1e-6)
        assert abs(faded.data[start]) <= abs(s.data[start])


class TestFadeOut:
    def test_fade_out_ends_zero(self):
        s = _make_sample()
        faded = SampleEditor.fade_out(s, 100.0)
        assert abs(faded.data[-1]) < 0.01

    def test_fade_out_start_unaffected(self):
        s = _make_sample()
        faded = SampleEditor.fade_out(s, 10.0)
        np.testing.assert_allclose(faded.data[:100], s.data[:100], atol=1e-6)

    def test_fade_out_zero_ms(self):
        s = _make_sample()
        faded = SampleEditor.fade_out(s, 0.0)
        np.testing.assert_allclose(faded.data, s.data, atol=1e-6)

    def test_fade_out_selection_only(self):
        s = _make_sample(duration_s=0.3)
        start, end = 2000, 5000
        faded = SampleEditor.fade_out(s, 20.0, start_frame=start, end_frame=end)
        np.testing.assert_allclose(faded.data[:start], s.data[:start], atol=1e-6)
        np.testing.assert_allclose(faded.data[end:], s.data[end:], atol=1e-6)
        assert abs(faded.data[end - 1]) <= abs(s.data[end - 1])


class TestSetLoopPoints:
    def test_valid_loop(self):
        s = _make_sample()
        looped = SampleEditor.set_loop_points(s, 1000, 5000)
        assert looped.loop.enabled is True
        assert looped.loop.start == 1000
        assert looped.loop.end == 5000

    def test_invalid_loop_raises(self):
        s = _make_sample()
        with pytest.raises(ValueError):
            SampleEditor.set_loop_points(s, 5000, 1000)

    def test_crossfade_clamped(self):
        s = _make_sample()
        looped = SampleEditor.set_loop_points(s, 1000, 1100, crossfade_len=10000)
        assert looped.loop.crossfade <= (1100 - 1000) // 2


class TestApplyCrossfadeLoop:
    def test_crossfade_loop_basic(self):
        s = _make_sample(duration_s=1.0)
        s.loop = LoopRegion(start=1000, end=10000, crossfade=256, enabled=True)
        result = SampleEditor.apply_crossfade_loop(s)
        assert result.loop.enabled is True

    def test_no_loop_returns_copy(self):
        s = _make_sample()
        result = SampleEditor.apply_crossfade_loop(s)
        np.testing.assert_array_equal(result.data, s.data)


class TestResample:
    def test_resample_same_rate(self):
        s = _make_sample(sr=48000)
        resampled = SampleEditor.resample(s, 48000)
        np.testing.assert_array_equal(resampled.data, s.data)

    def test_resample_to_44100(self):
        s = _make_sample(sr=48000)
        resampled = SampleEditor.resample(s, 44100)
        assert resampled.sample_rate == 44100
        expected_len = int(s.length * 44100 / 48000)
        assert abs(resampled.length - expected_len) < 10

    def test_resample_stereo(self):
        s = _make_stereo_sample(sr=48000)
        resampled = SampleEditor.resample(s, 44100)
        assert resampled.data.ndim == 2
        assert resampled.data.shape[1] == 2

    def test_resample_invalid_rate_raises(self):
        s = _make_sample()
        with pytest.raises(ValueError):
            SampleEditor.resample(s, 0)


class TestToMono:
    def test_stereo_to_mono(self):
        s = _make_stereo_sample()
        mono = SampleEditor.to_mono(s)
        assert mono.data.ndim == 1
        assert mono.channels == 1

    def test_mono_stays_mono(self):
        s = _make_sample()
        mono = SampleEditor.to_mono(s)
        assert mono.data.ndim == 1


class TestToStereo:
    def test_mono_to_stereo(self):
        s = _make_sample()
        stereo = SampleEditor.to_stereo(s)
        assert stereo.data.ndim == 2
        assert stereo.data.shape[1] == 2
        assert stereo.channels == 2

    def test_stereo_stays_stereo(self):
        s = _make_stereo_sample()
        stereo = SampleEditor.to_stereo(s)
        assert stereo.data.shape[1] == 2


class TestEditorReturnNewSample:
    """All operations should return NEW Sample instances, not modify in-place."""

    def test_trim_new_instance(self):
        s = _make_sample()
        trimmed = SampleEditor.trim(s, 0, 1000)
        assert trimmed is not s

    def test_normalize_new_instance(self):
        s = _make_sample()
        result = SampleEditor.normalize(s)
        assert result is not s

    def test_reverse_new_instance(self):
        s = _make_sample()
        result = SampleEditor.reverse(s)
        assert result is not s

    def test_fade_in_new_instance(self):
        s = _make_sample()
        result = SampleEditor.fade_in(s, 10.0)
        assert result is not s

    def test_fade_out_new_instance(self):
        s = _make_sample()
        result = SampleEditor.fade_out(s, 10.0)
        assert result is not s

    def test_resample_new_instance(self):
        s = _make_sample()
        result = SampleEditor.resample(s, 44100)
        assert result is not s
