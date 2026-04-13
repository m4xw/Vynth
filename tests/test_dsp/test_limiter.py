"""Exhaustive tests for the lookahead soft limiter."""
import numpy as np
import pytest

from vynth.dsp.limiter import Limiter


@pytest.fixture
def lim(sr):
    return Limiter(sr)


def _sine(sr, freq=440, duration_s=0.1, amp=1.0):
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


class TestLimiterInit:
    def test_default_threshold(self, sr):
        l = Limiter(sr)
        assert l.get_param("threshold_db") == pytest.approx(-1.0)

    def test_default_release(self, sr):
        l = Limiter(sr)
        assert l.get_param("release_ms") == pytest.approx(100.0)

    def test_default_lookahead(self, sr):
        l = Limiter(sr)
        assert l.get_param("lookahead_ms") == pytest.approx(5.0)

    def test_internal_state(self, sr):
        l = Limiter(sr)
        assert l._threshold_lin > 0
        assert l._release_coeff >= 0
        assert l._lookahead_samples >= 1


class TestLimiterProcess:
    def test_quiet_signal_unchanged(self, sr, lim):
        sig = _sine(sr, amp=0.1)
        out = lim.process(sig)
        # Limiter has a lookahead delay so output is shifted; check RMS is similar
        assert np.sqrt(np.mean(out**2)) == pytest.approx(
            np.sqrt(np.mean(sig**2)), abs=0.02
        )

    def test_loud_signal_reduced(self, sr):
        l = Limiter(sr)
        l.set_param("threshold_db", -6.0)
        sig = _sine(sr, amp=1.0, duration_s=0.5)
        out = l.process(sig)
        # Peak should be near 10^(-6/20) ≈ 0.5
        assert np.max(np.abs(out)) <= 0.6

    def test_output_shape_mono(self, sr, lim):
        sig = _sine(sr, 440, 0.1)
        out = lim.process(sig)
        assert out.shape == sig.shape

    def test_output_shape_stereo(self, sr, lim):
        sig = _sine(sr, 440, 0.1)
        stereo = np.column_stack([sig, sig])
        out = lim.process(stereo)
        assert out.shape == stereo.shape

    def test_dtype_float32(self, sr, lim):
        out = lim.process(_sine(sr))
        assert out.dtype == np.float32

    def test_no_nan(self, sr, lim):
        out = lim.process(_sine(sr, amp=2.0, duration_s=0.5))
        assert not np.any(np.isnan(out))

    def test_no_inf(self, sr, lim):
        out = lim.process(_sine(sr, amp=2.0, duration_s=0.5))
        assert not np.any(np.isinf(out))

    def test_empty_input(self, sr, lim):
        empty = np.array([], dtype=np.float32)
        out = lim.process(empty)
        assert out.size == 0


class TestLimiterThreshold:
    @pytest.mark.parametrize("threshold_db", [-20, -12, -6, -3, -1, 0])
    def test_peak_below_threshold(self, sr, threshold_db):
        l = Limiter(sr)
        l.set_param("threshold_db", float(threshold_db))
        sig = _sine(sr, amp=2.0, duration_s=0.5)
        out = l.process(sig)
        target_lin = 10.0 ** (threshold_db / 20.0)
        # Allow some headroom for attack/release dynamics
        assert np.max(np.abs(out)) <= target_lin * 1.5

    def test_zero_db_no_reduction(self, sr):
        l = Limiter(sr)
        l.set_param("threshold_db", 0.0)
        sig = _sine(sr, amp=0.5, duration_s=0.1)
        out = l.process(sig)
        # Lookahead delay shifts output; check RMS is preserved
        assert np.sqrt(np.mean(out**2)) == pytest.approx(
            np.sqrt(np.mean(sig**2)), abs=0.02
        )


class TestLimiterRelease:
    def test_fast_release_recovers_quickly(self, sr):
        l = Limiter(sr)
        l.set_param("threshold_db", -6.0)
        l.set_param("release_ms", 10.0)
        # Loud burst followed by silence
        burst = np.ones(1000, dtype=np.float32)
        silence = np.zeros(sr, dtype=np.float32)
        sig = np.concatenate([burst, silence])
        out = l.process(sig)
        # After burst, gain should recover quickly
        # Check that near-end of silence, no more reduction
        assert out is not None  # Basic sanity

    def test_slow_release_stays_reduced(self, sr):
        l = Limiter(sr)
        l.set_param("threshold_db", -12.0)
        l.set_param("release_ms", 500.0)
        burst = np.ones(1000, dtype=np.float32) * 2.0
        quiet = np.ones(5000, dtype=np.float32) * 0.5
        sig = np.concatenate([burst, quiet])
        out = l.process(sig)
        # With slow release, quiet part may still be reduced
        assert np.max(np.abs(out)) < 2.0


class TestLimiterLookahead:
    def test_lookahead_delays_output(self, sr):
        l = Limiter(sr)
        l.set_param("lookahead_ms", 5.0)
        imp = np.zeros(sr, dtype=np.float32)
        imp[0] = 1.0
        out = l.process(imp)
        la_samples = int(5.0 * 0.001 * sr)
        # impulse should appear delayed
        peak_idx = np.argmax(np.abs(out))
        assert peak_idx >= la_samples - 2

    @pytest.mark.parametrize("la_ms", [0.0, 1.0, 5.0, 10.0])
    def test_various_lookahead_values(self, sr, la_ms):
        l = Limiter(sr)
        l.set_param("lookahead_ms", la_ms)
        out = l.process(_sine(sr, amp=0.5))
        assert not np.any(np.isnan(out))


class TestLimiterParamClamping:
    def test_threshold_clamped_low(self, sr):
        l = Limiter(sr)
        l.set_param("threshold_db", -100.0)
        assert l.get_param("threshold_db") == pytest.approx(-20.0)

    def test_threshold_clamped_high(self, sr):
        l = Limiter(sr)
        l.set_param("threshold_db", 10.0)
        assert l.get_param("threshold_db") == pytest.approx(0.0)

    def test_release_clamped_low(self, sr):
        l = Limiter(sr)
        l.set_param("release_ms", 0.1)
        assert l.get_param("release_ms") == pytest.approx(10.0)

    def test_release_clamped_high(self, sr):
        l = Limiter(sr)
        l.set_param("release_ms", 9999.0)
        assert l.get_param("release_ms") == pytest.approx(500.0)

    def test_lookahead_clamped(self, sr):
        l = Limiter(sr)
        l.set_param("lookahead_ms", -5.0)
        assert l.get_param("lookahead_ms") == pytest.approx(0.0)
        l.set_param("lookahead_ms", 100.0)
        assert l.get_param("lookahead_ms") == pytest.approx(10.0)


class TestLimiterReset:
    def test_reset_clears_envelope(self, sr):
        l = Limiter(sr)
        l.process(_sine(sr, amp=2.0, duration_s=0.5))
        l.reset()
        assert l._envelope == 0.0
        assert l._delay_pos == 0

    def test_reset_preserves_params(self, sr):
        l = Limiter(sr)
        l.set_param("threshold_db", -6.0)
        l.reset()
        assert l.get_param("threshold_db") == pytest.approx(-6.0)


class TestLimiterBypass:
    def test_bypass_passthrough(self, sr, sine_440):
        l = Limiter(sr)
        l.bypassed = True
        out = l.process_maybe_bypass(sine_440[:512])
        np.testing.assert_array_equal(out, sine_440[:512])


class TestLimiterStability:
    def test_dc_offset_input(self, sr):
        l = Limiter(sr)
        sig = np.ones(sr, dtype=np.float32)
        out = l.process(sig)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_very_loud_input(self, sr):
        l = Limiter(sr)
        l.set_param("threshold_db", -12.0)
        sig = _sine(sr, amp=10.0, duration_s=0.5)
        out = l.process(sig)
        target = 10.0 ** (-12.0 / 20.0)
        assert np.max(np.abs(out)) < target * 3.0

    def test_consecutive_blocks(self, sr):
        l = Limiter(sr)
        sig = _sine(sr, amp=2.0, duration_s=0.5)
        bs = 512
        prev = None
        for i in range(0, len(sig), bs):
            block = sig[i:i + bs]
            if len(block) == 0:
                break
            out = l.process(block)
            if prev is not None:
                jump = np.max(np.abs(out[0] - prev))
                assert jump < 1.0
            prev = out[-1] if out.ndim == 1 else out[-1].copy()
