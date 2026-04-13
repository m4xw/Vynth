"""Exhaustive tests for the GainEffect."""
import numpy as np
import pytest

from vynth.dsp.gain import GainEffect


@pytest.fixture
def gain(sr):
    return GainEffect(sr)


class TestGainInit:
    def test_default_gain(self, sr):
        g = GainEffect(sr)
        assert g.get_param("gain_db") == pytest.approx(0.0)

    def test_default_linear_is_unity(self, sr):
        g = GainEffect(sr)
        assert g._linear == pytest.approx(1.0)


class TestGainProcess:
    def test_zero_db_passthrough(self, sr, sine_440):
        g = GainEffect(sr)
        out = g.process(sine_440[:512].copy())
        np.testing.assert_allclose(out, sine_440[:512], atol=1e-7)

    def test_positive_gain_amplifies(self, sr, sine_440):
        g = GainEffect(sr)
        g.set_param("gain_db", 6.0)
        block = sine_440[:512].copy()
        out = g.process(block)
        expected_ratio = 10.0 ** (6.0 / 20.0)
        np.testing.assert_allclose(out, sine_440[:512] * expected_ratio, atol=1e-5)

    def test_negative_gain_attenuates(self, sr, sine_440):
        g = GainEffect(sr)
        g.set_param("gain_db", -6.0)
        block = sine_440[:512].copy()
        out = g.process(block)
        expected_ratio = 10.0 ** (-6.0 / 20.0)
        np.testing.assert_allclose(out, sine_440[:512] * expected_ratio, atol=1e-5)

    @pytest.mark.parametrize("db", [-24, -12, -6, -3, 0, 3, 6, 12, 24])
    def test_various_gain_values(self, sr, db):
        g = GainEffect(sr)
        g.set_param("gain_db", float(db))
        sig = np.ones(100, dtype=np.float32) * 0.5
        out = g.process(sig)
        expected = 0.5 * (10.0 ** (db / 20.0))
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_stereo_input(self, sr, sine_440_stereo):
        g = GainEffect(sr)
        g.set_param("gain_db", 6.0)
        block = sine_440_stereo[:512].copy()
        out = g.process(block)
        ratio = 10.0 ** (6.0 / 20.0)
        np.testing.assert_allclose(out, sine_440_stereo[:512] * ratio, atol=1e-5)

    def test_mono_input(self, sr, sine_440):
        g = GainEffect(sr)
        g.set_param("gain_db", -12.0)
        out = g.process(sine_440[:256].copy())
        assert out.ndim == 1

    def test_silence_stays_silent(self, sr, silence):
        g = GainEffect(sr)
        g.set_param("gain_db", 24.0)
        out = g.process(silence[:512].copy())
        np.testing.assert_allclose(out, 0.0, atol=1e-10)

    def test_dtype_preserved(self, sr, sine_440):
        g = GainEffect(sr)
        out = g.process(sine_440[:512].copy())
        assert out.dtype == np.float32


class TestGainParamChange:
    def test_param_updates_linear(self, sr):
        g = GainEffect(sr)
        g.set_param("gain_db", 20.0)
        assert g._linear == pytest.approx(10.0, rel=1e-3)

    def test_param_minus_infinity_approach(self, sr):
        g = GainEffect(sr)
        g.set_param("gain_db", -120.0)
        assert g._linear < 1e-5


class TestGainBypass:
    def test_bypass_passthrough(self, sr, sine_440):
        g = GainEffect(sr)
        g.set_param("gain_db", 24.0)
        g.bypassed = True
        block = sine_440[:512].copy()
        out = g.process_maybe_bypass(block)
        np.testing.assert_array_equal(out, block)


class TestGainEdgeCases:
    def test_empty_array(self, sr):
        g = GainEffect(sr)
        empty = np.array([], dtype=np.float32)
        out = g.process(empty)
        assert len(out) == 0

    def test_single_sample(self, sr):
        g = GainEffect(sr)
        g.set_param("gain_db", 6.0)
        out = g.process(np.array([0.5], dtype=np.float32))
        expected = 0.5 * (10.0 ** (6.0 / 20.0))
        assert out[0] == pytest.approx(expected, rel=1e-4)
