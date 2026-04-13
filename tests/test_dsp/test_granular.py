"""Exhaustive tests for granular synthesis engine."""
import numpy as np
import pytest

from vynth.dsp.granular import GranularEngine, _build_window, _interp_sample


@pytest.fixture
def gran(sr):
    return GranularEngine(sr)


@pytest.fixture
def source_sine(sr):
    t = np.arange(sr, dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)


class TestGranularInit:
    def test_default_params(self, sr):
        g = GranularEngine(sr)
        assert g.get_param("grain_size_ms") == pytest.approx(50.0)
        assert g.get_param("overlap") == pytest.approx(0.5)
        assert g.get_param("scatter") == pytest.approx(0.1)
        assert g.get_param("density") == pytest.approx(10.0)
        assert g.get_param("pitch_ratio") == pytest.approx(1.0)
        assert g.get_param("position") == pytest.approx(0.5)
        assert g.get_param("spread") == pytest.approx(0.5)
        assert g.get_param("window_type") == pytest.approx(0.0)

    def test_no_source_initially(self, sr):
        g = GranularEngine(sr)
        assert g._source is None


class TestGranularNoSource:
    def test_produces_silence_with_ndarray(self, gran):
        out = gran.process(np.zeros(512, dtype=np.float32))
        assert np.abs(out).max() < 0.001
        assert out.shape == (512, 2)

    def test_produces_silence_with_int(self, gran):
        out = gran.process(512)
        assert np.abs(out).max() < 0.001
        assert out.shape == (512, 2)


class TestGranularSetSource:
    def test_mono_source(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        assert gran._source is not None
        assert gran._source.ndim == 1

    def test_stereo_source_mixdown(self, gran, source_sine, sr):
        stereo = np.column_stack([source_sine, source_sine])
        gran.set_source(stereo, sr)
        assert gran._source.ndim == 1

    def test_source_is_float32(self, gran, sr):
        data = np.random.randn(sr).astype(np.float64)
        gran.set_source(data, sr)
        assert gran._source.dtype == np.float32


class TestGranularProcess:
    def test_with_source_produces_audio(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("density", 20.0)
        out = gran.process(sr)
        assert np.abs(out).max() > 0.01

    def test_output_is_stereo(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        out = gran.process(1024)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_output_dtype(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        out = gran.process(512)
        assert out.dtype == np.float32

    def test_no_nan(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("density", 30.0)
        out = gran.process(sr)
        assert not np.any(np.isnan(out))

    def test_no_inf(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        out = gran.process(sr)
        assert not np.any(np.isinf(out))

    def test_consecutive_blocks(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("density", 20.0)
        for _ in range(5):
            out = gran.process(512)
            assert out.shape == (512, 2)
            assert not np.any(np.isnan(out))

    def test_ndarray_input_uses_shape(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        data = np.zeros((256, 2), dtype=np.float32)
        out = gran.process(data)
        assert out.shape[0] == 256


class TestGranularParams:
    def test_grain_size_small(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("grain_size_ms", 5.0)
        out = gran.process(sr)
        assert not np.any(np.isnan(out))

    def test_grain_size_large(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("grain_size_ms", 500.0)
        out = gran.process(sr)
        assert not np.any(np.isnan(out))

    def test_high_density(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("density", 50.0)
        out = gran.process(sr // 4)
        assert np.abs(out).max() > 0.001

    def test_low_density(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("density", 1.0)
        out = gran.process(sr // 2)
        assert out.shape == (sr // 2, 2)

    def test_position_start(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("position", 0.0)
        out = gran.process(sr // 4)
        assert not np.any(np.isnan(out))

    def test_position_end(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("position", 1.0)
        out = gran.process(sr // 4)
        assert not np.any(np.isnan(out))

    def test_scatter_zero(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("scatter", 0.0)
        out = gran.process(1024)
        assert not np.any(np.isnan(out))

    def test_scatter_max(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("scatter", 1.0)
        out = gran.process(1024)
        assert not np.any(np.isnan(out))

    def test_pitch_ratio_low(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("pitch_ratio", 0.25)
        out = gran.process(1024)
        assert not np.any(np.isnan(out))

    def test_pitch_ratio_high(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("pitch_ratio", 4.0)
        out = gran.process(1024)
        assert not np.any(np.isnan(out))

    def test_spread_stereo(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("spread", 1.0)
        gran.set_param("density", 20.0)
        out = gran.process(sr)
        # L and R should differ somewhat
        diff = np.max(np.abs(out[:, 0] - out[:, 1]))
        assert diff > 0.0


class TestGranularWindowTypes:
    @pytest.mark.parametrize("wtype", [0, 1, 2])
    def test_window_type(self, gran, source_sine, sr, wtype):
        gran.set_source(source_sine, sr)
        gran.set_param("window_type", float(wtype))
        out = gran.process(sr // 4)
        assert not np.any(np.isnan(out))


class TestBuildWindow:
    def test_hann_window(self):
        w = _build_window(0, 256)
        assert w.shape == (256,)
        assert w.dtype == np.float32
        assert np.isclose(w.max(), 1.0)

    def test_gaussian_window(self):
        w = _build_window(1, 256)
        assert w.shape == (256,)
        assert np.isclose(w.max(), 1.0)

    def test_triangle_window(self):
        w = _build_window(2, 256)
        assert w.shape == (256,)
        assert np.isclose(w.max(), 1.0)

    def test_length_one(self):
        w = _build_window(0, 1)
        assert len(w) == 1
        assert w[0] == pytest.approx(1.0)

    def test_length_zero(self):
        w = _build_window(0, 0)
        assert len(w) == 1


class TestInterpSample:
    def test_integer_position(self):
        buf = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert _interp_sample(buf, 0.0, 4) == pytest.approx(1.0)
        assert _interp_sample(buf, 2.0, 4) == pytest.approx(3.0)

    def test_fractional_position(self):
        buf = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert _interp_sample(buf, 0.5, 3) == pytest.approx(0.5)

    def test_wrapping(self):
        buf = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        val = _interp_sample(buf, 3.0, 3)
        assert val == pytest.approx(1.0)


class TestGranularReset:
    def test_reset_clears_grains(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.set_param("density", 50.0)
        gran.process(sr)
        gran.reset()
        assert len(gran._grains) == 0
        assert gran._sched_acc == 0.0

    def test_reset_and_reprocess(self, gran, source_sine, sr):
        gran.set_source(source_sine, sr)
        gran.process(1024)
        gran.reset()
        out = gran.process(1024)
        assert out.shape == (1024, 2)


class TestGranularBypass:
    def test_bypass(self, sr):
        g = GranularEngine(sr)
        g.bypassed = True
        data = np.zeros(512, dtype=np.float32)
        out = g.process_maybe_bypass(data)
        np.testing.assert_array_equal(out, data)
