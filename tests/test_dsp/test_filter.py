"""Exhaustive tests for biquad filter."""
import math

import numpy as np
import pytest

from vynth.dsp.filter import BiquadFilter


@pytest.fixture
def filt(sr):
    return BiquadFilter(sr)


def _make_sine(sr, freq, duration_s=1.0):
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


class TestBiquadInit:
    def test_default_params(self, sr):
        f = BiquadFilter(sr)
        assert f.get_param("frequency") == pytest.approx(1000.0)
        assert f.get_param("q") == pytest.approx(0.707)
        assert f.get_param("gain_db") == pytest.approx(0.0)
        assert f.get_param("mode") == pytest.approx(0.0)

    def test_custom_sample_rate(self):
        f = BiquadFilter(96000)
        assert f.sample_rate == 96000

    def test_coefficients_initialized(self, sr):
        f = BiquadFilter(sr)
        assert f._a[0] == pytest.approx(1.0)


class TestBiquadLowPass:
    def test_passes_low_freq(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 0)
        f.set_param("frequency", 500.0)
        f.set_param("q", 0.707)
        sig = _make_sine(sr, 100)
        out = f.process(sig)
        # Steady-state (skip transient)
        assert _rms(out[sr // 2:]) > 0.5

    def test_attenuates_high_freq(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 0)
        f.set_param("frequency", 500.0)
        f.set_param("q", 0.707)
        sig = _make_sine(sr, 10000)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) < 0.1

    def test_cutoff_at_resonant_freq(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 0)
        f.set_param("frequency", 1000.0)
        f.set_param("q", 0.707)
        sig = _make_sine(sr, 1000)
        out = f.process(sig)
        # At cutoff, -3dB → rms ≈ 0.707 * input rms
        rms_out = _rms(out[sr // 2:])
        assert 0.3 < rms_out < 1.0

    def test_higher_q_resonance(self, sr):
        f_low_q = BiquadFilter(sr)
        f_low_q.set_param("mode", 0)
        f_low_q.set_param("frequency", 1000.0)
        f_low_q.set_param("q", 0.5)

        f_high_q = BiquadFilter(sr)
        f_high_q.set_param("mode", 0)
        f_high_q.set_param("frequency", 1000.0)
        f_high_q.set_param("q", 10.0)

        sig = _make_sine(sr, 1000)
        out_lo = f_low_q.process(sig.copy())
        out_hi = f_high_q.process(sig.copy())
        # Higher Q should have higher peak at resonance
        assert np.max(np.abs(out_hi)) >= np.max(np.abs(out_lo)) * 0.9


class TestBiquadHighPass:
    def test_attenuates_low_freq(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 1)
        f.set_param("frequency", 5000.0)
        sig = _make_sine(sr, 50)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) < 0.05

    def test_passes_high_freq(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 1)
        f.set_param("frequency", 500.0)
        sig = _make_sine(sr, 10000)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) > 0.5


class TestBiquadBandPass:
    def test_passes_center_freq(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 2)
        f.set_param("frequency", 1000.0)
        f.set_param("q", 5.0)
        sig = _make_sine(sr, 1000)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) > 0.1

    def test_rejects_far_freq(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 2)
        f.set_param("frequency", 1000.0)
        f.set_param("q", 5.0)
        sig = _make_sine(sr, 10000)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) < 0.3


class TestBiquadNotch:
    def test_rejects_center_freq(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 3)
        f.set_param("frequency", 1000.0)
        f.set_param("q", 10.0)
        sig = _make_sine(sr, 1000)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) < 0.3

    def test_passes_far_freq(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 3)
        f.set_param("frequency", 1000.0)
        f.set_param("q", 5.0)
        sig = _make_sine(sr, 5000)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) > 0.3


class TestBiquadPeak:
    def test_boost_at_center(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 4)
        f.set_param("frequency", 1000.0)
        f.set_param("q", 1.0)
        f.set_param("gain_db", 12.0)
        sig = _make_sine(sr, 1000)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) > _rms(sig[sr // 2:])

    def test_cut_at_center(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 4)
        f.set_param("frequency", 1000.0)
        f.set_param("q", 1.0)
        f.set_param("gain_db", -12.0)
        sig = _make_sine(sr, 1000)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) < _rms(sig[sr // 2:])


class TestBiquadShelves:
    def test_lowshelf_boost(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 5)
        f.set_param("frequency", 500.0)
        f.set_param("gain_db", 12.0)
        sig = _make_sine(sr, 100)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) > _rms(sig[sr // 2:]) * 1.2

    def test_highshelf_boost(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 6)
        f.set_param("frequency", 5000.0)
        f.set_param("gain_db", 12.0)
        sig = _make_sine(sr, 10000)
        out = f.process(sig)
        assert _rms(out[sr // 2:]) > _rms(sig[sr // 2:]) * 1.2


class TestBiquadAllModes:
    @pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, 5, 6])
    def test_mode_produces_no_nan(self, sr, mode):
        f = BiquadFilter(sr)
        f.set_param("mode", float(mode))
        sig = _make_sine(sr, 440)
        out = f.process(sig)
        assert not np.any(np.isnan(out))

    @pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, 5, 6])
    def test_mode_stereo(self, sr, mode):
        f = BiquadFilter(sr)
        f.set_param("mode", float(mode))
        mono = _make_sine(sr, 440, 0.1)
        stereo = np.column_stack([mono, mono])
        out = f.process(stereo)
        assert out.shape == stereo.shape

    @pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, 5, 6])
    def test_mode_preserves_silence(self, sr, mode):
        f = BiquadFilter(sr)
        f.set_param("mode", float(mode))
        f.reset()
        silence = np.zeros(512, dtype=np.float32)
        out = f.process(silence)
        assert np.max(np.abs(out)) < 0.001


class TestBiquadInputFormats:
    def test_mono_input(self, sr):
        f = BiquadFilter(sr)
        mono = _make_sine(sr, 440, 0.1)
        out = f.process(mono)
        assert out.ndim == 1
        assert out.dtype == np.float32

    def test_stereo_input(self, sr):
        f = BiquadFilter(sr)
        mono = _make_sine(sr, 440, 0.1)
        stereo = np.column_stack([mono, mono])
        out = f.process(stereo)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_empty_input(self, sr):
        f = BiquadFilter(sr)
        empty = np.array([], dtype=np.float32)
        out = f.process(empty)
        assert out.size == 0


class TestBiquadParamClamping:
    def test_frequency_clamped_low(self, sr):
        f = BiquadFilter(sr)
        f.set_param("frequency", 1.0)
        assert f.get_param("frequency") >= 20.0

    def test_frequency_clamped_high(self, sr):
        f = BiquadFilter(sr)
        f.set_param("frequency", 30000.0)
        assert f.get_param("frequency") <= 20000.0

    def test_q_clamped_low(self, sr):
        f = BiquadFilter(sr)
        f.set_param("q", 0.0)
        assert f.get_param("q") >= 0.1

    def test_q_clamped_high(self, sr):
        f = BiquadFilter(sr)
        f.set_param("q", 100.0)
        assert f.get_param("q") <= 20.0

    def test_gain_db_clamped(self, sr):
        f = BiquadFilter(sr)
        f.set_param("gain_db", -50.0)
        assert f.get_param("gain_db") >= -24.0
        f.set_param("gain_db", 50.0)
        assert f.get_param("gain_db") <= 24.0

    def test_mode_clamped_to_int(self, sr):
        f = BiquadFilter(sr)
        f.set_param("mode", 2.7)
        assert f.get_param("mode") == pytest.approx(2.0)


class TestBiquadReset:
    def test_reset_clears_state(self, sr):
        f = BiquadFilter(sr)
        f.process(np.random.randn(1024).astype(np.float32))
        f.reset()
        out = f.process(np.zeros(512, dtype=np.float32))
        assert np.abs(out).max() < 0.001

    def test_reset_preserves_params(self, sr):
        f = BiquadFilter(sr)
        f.set_param("frequency", 2000.0)
        f.set_param("q", 5.0)
        f.reset()
        assert f.get_param("frequency") == pytest.approx(2000.0)
        assert f.get_param("q") == pytest.approx(5.0)


class TestBiquadBypass:
    def test_bypass_passthrough_mono(self, sr):
        f = BiquadFilter(sr)
        f.bypassed = True
        sig = _make_sine(sr, 440, 0.1)
        out = f.process_maybe_bypass(sig.copy())
        np.testing.assert_array_equal(out, sig)

    def test_bypass_passthrough_stereo(self, sr):
        f = BiquadFilter(sr)
        f.bypassed = True
        mono = _make_sine(sr, 440, 0.1)
        stereo = np.column_stack([mono, mono])
        out = f.process_maybe_bypass(stereo.copy())
        np.testing.assert_array_equal(out, stereo)


class TestBiquadStability:
    """Filter stability under extreme parameters."""

    def test_extreme_frequency_no_crash(self, sr):
        f = BiquadFilter(sr)
        f.set_param("frequency", 20.0)
        sig = _make_sine(sr, 440, 0.1)
        out = f.process(sig)
        assert not np.any(np.isnan(out))

    def test_extreme_q_no_crash(self, sr):
        f = BiquadFilter(sr)
        f.set_param("q", 20.0)
        sig = _make_sine(sr, 440, 0.1)
        out = f.process(sig)
        assert not np.any(np.isnan(out))

    def test_long_processing_no_instability(self, sr):
        f = BiquadFilter(sr)
        f.set_param("q", 10.0)
        for _ in range(10):
            sig = _make_sine(sr, 440)
            out = f.process(sig)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_rapid_param_changes_no_crash(self, sr):
        f = BiquadFilter(sr)
        rng = np.random.default_rng(42)
        sig = _make_sine(sr, 440, 0.1)
        for _ in range(50):
            f.set_param("frequency", float(rng.uniform(20, 20000)))
            f.set_param("q", float(rng.uniform(0.1, 20)))
            f.set_param("mode", float(rng.integers(0, 7)))
            out = f.process(sig[:128])
            assert not np.any(np.isnan(out))
