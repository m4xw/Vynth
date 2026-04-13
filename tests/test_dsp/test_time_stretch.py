"""Exhaustive tests for WSOLA time stretcher."""
import numpy as np
import pytest

from vynth.dsp.time_stretch import TimeStretch


@pytest.fixture
def ts(sr):
    return TimeStretch(sr)


def _sine(sr, freq=440, duration_s=0.5):
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


class TestTimeStretchInit:
    def test_default_ratio(self, sr):
        ts = TimeStretch(sr)
        assert ts.get_param("stretch_ratio") == pytest.approx(1.0)

    def test_window_size(self, sr):
        ts = TimeStretch(sr)
        assert ts.WINDOW_SIZE == 2048


class TestTimeStretchProcess:
    def test_ratio_1_returns_copy(self, sr, ts):
        sig = _sine(sr)
        out = ts.process(sig)
        np.testing.assert_allclose(out, sig, atol=1e-6)

    def test_stretch_makes_longer(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 2.0)
        sig = _sine(sr, duration_s=0.5)
        out = ts.process(sig)
        # Output should be roughly twice as long
        assert len(out) > len(sig) * 1.5

    def test_compress_makes_shorter(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 0.5)
        sig = _sine(sr, duration_s=0.5)
        out = ts.process(sig)
        assert len(out) < len(sig) * 0.8

    def test_mono_stays_mono(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 1.5)
        sig = _sine(sr)
        out = ts.process(sig)
        assert out.ndim == 1

    def test_stereo_stays_stereo(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 1.5)
        sig = _sine(sr)
        stereo = np.column_stack([sig, sig])
        out = ts.process(stereo)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_dtype_float32(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 1.5)
        out = ts.process(_sine(sr))
        assert out.dtype == np.float32

    def test_no_nan(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 2.0)
        out = ts.process(_sine(sr))
        assert not np.any(np.isnan(out))

    def test_no_inf(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 0.5)
        out = ts.process(_sine(sr))
        assert not np.any(np.isinf(out))

    def test_empty_input(self, sr, ts):
        empty = np.array([], dtype=np.float32)
        out = ts.process(empty)
        assert out.size == 0


class TestTimeStretchRatios:
    @pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0])
    def test_output_length_proportional(self, sr, ratio):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", ratio)
        sig = _sine(sr, duration_s=0.5)
        out = ts.process(sig)
        expected_len = len(sig) * ratio
        # Allow ~30% tolerance due to WSOLA windowing
        assert abs(len(out) - expected_len) / expected_len < 0.3


class TestTimeStretchParamClamping:
    def test_ratio_clamped_low(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 0.01)
        assert ts.get_param("stretch_ratio") == pytest.approx(0.25)

    def test_ratio_clamped_high(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 100.0)
        assert ts.get_param("stretch_ratio") == pytest.approx(4.0)


class TestTimeStretchBestOverlap:
    def test_identical_signals_offset_zero(self):
        signal = np.sin(np.linspace(0, 2 * np.pi * 10, 2048))
        offset = TimeStretch._best_overlap_offset(signal, signal[:512], 50)
        assert offset == 0

    def test_returns_valid_offset(self):
        rng = np.random.default_rng(42)
        src = rng.standard_normal(2048)
        target = src[100:612]
        offset = TimeStretch._best_overlap_offset(src, target, 200)
        assert 0 <= offset <= 400


class TestTimeStretchBypass:
    def test_bypass_passthrough(self, sr, sine_440):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 2.0)
        ts.bypassed = True
        out = ts.process_maybe_bypass(sine_440[:512])
        np.testing.assert_array_equal(out, sine_440[:512])


class TestTimeStretchReset:
    def test_reset_no_error(self, sr, ts):
        ts.process(_sine(sr))
        ts.reset()  # Should not raise


class TestTimeStretchEdgeCases:
    def test_very_short_signal(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 2.0)
        short = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = ts.process(short)
        # Too short for WSOLA, may return empty or very short
        assert not np.any(np.isnan(out))

    def test_dc_signal(self, sr):
        ts = TimeStretch(sr)
        ts.set_param("stretch_ratio", 1.5)
        dc = np.ones(sr, dtype=np.float32) * 0.5
        out = ts.process(dc)
        assert not np.any(np.isnan(out))
