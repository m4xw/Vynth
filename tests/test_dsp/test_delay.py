"""Exhaustive tests for stereo delay with feedback and ping-pong."""
import numpy as np
import pytest

from vynth.dsp.delay import Delay


@pytest.fixture
def delay(sr):
    return Delay(sr)


def _sine(sr, freq=440, duration_s=0.1):
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


class TestDelayInit:
    def test_default_params(self, sr):
        d = Delay(sr)
        assert d.get_param("time_ms") == pytest.approx(250.0)
        assert d.get_param("feedback") == pytest.approx(0.4)
        assert d.get_param("mix") == pytest.approx(0.3)
        assert d.get_param("ping_pong") == pytest.approx(0.0)

    def test_buffer_allocated(self, sr):
        d = Delay(sr)
        max_samples = int(2.0 * sr)
        assert len(d._buf_l) == max_samples
        assert len(d._buf_r) == max_samples

    def test_write_idx_starts_zero(self, sr):
        d = Delay(sr)
        assert d._write_idx == 0


class TestDelayProcess:
    def test_output_always_stereo(self, sr, delay):
        mono = _sine(sr, 440, 0.05)
        out = delay.process(mono)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_output_shape_matches_input(self, sr, delay):
        mono = _sine(sr, 440, 0.05)
        out = delay.process(mono)
        assert out.shape[0] == len(mono)

    def test_stereo_input_accepted(self, sr, delay):
        mono = _sine(sr, 440, 0.05)
        stereo = np.column_stack([mono, mono])
        out = delay.process(stereo)
        assert out.shape == stereo.shape

    def test_dtype_float32(self, sr, delay):
        out = delay.process(_sine(sr, 440, 0.05))
        assert out.dtype == np.float32

    def test_no_nan(self, sr, delay):
        out = delay.process(_sine(sr, 440, 0.5))
        assert not np.any(np.isnan(out))

    def test_no_inf(self, sr, delay):
        out = delay.process(_sine(sr, 440, 0.5))
        assert not np.any(np.isinf(out))


class TestDelayTime:
    def test_echo_appears_at_correct_delay(self, sr):
        d = Delay(sr)
        d.set_param("time_ms", 100.0)
        d.set_param("feedback", 0.0)
        d.set_param("mix", 1.0)
        n = sr  # 1 second
        imp = np.zeros(n, dtype=np.float32)
        imp[0] = 1.0
        out = d.process(imp)
        delay_samples = int(100.0 * 0.001 * sr)
        # Expect echo peak around delay_samples
        peak_idx = np.argmax(np.abs(out[delay_samples - 5:delay_samples + 5, 0]))
        assert peak_idx == pytest.approx(5, abs=2)

    @pytest.mark.parametrize("ms", [10, 50, 100, 500, 1000])
    def test_various_delay_times(self, sr, ms):
        d = Delay(sr)
        d.set_param("time_ms", float(ms))
        d.set_param("mix", 0.5)
        out = d.process(_sine(sr, 440, 0.1))
        assert out.shape[0] == int(sr * 0.1)


class TestDelayFeedback:
    def test_zero_feedback_single_echo(self, sr):
        d = Delay(sr)
        d.set_param("feedback", 0.0)
        d.set_param("mix", 1.0)
        d.set_param("time_ms", 50.0)
        n = sr
        imp = np.zeros(n, dtype=np.float32)
        imp[0] = 1.0
        out = d.process(imp)
        delay_s = int(50 * 0.001 * sr)
        # After first echo, should be essentially zero
        post_echo = out[delay_s + delay_s:, 0]
        assert np.max(np.abs(post_echo)) < 0.01

    def test_high_feedback_multiple_echoes(self, sr):
        d = Delay(sr)
        d.set_param("feedback", 0.8)
        d.set_param("mix", 1.0)
        d.set_param("time_ms", 50.0)
        n = sr
        imp = np.zeros(n, dtype=np.float32)
        imp[0] = 1.0
        out = d.process(imp)
        delay_s = int(50 * 0.001 * sr)
        # Should still have energy much later
        late = out[delay_s * 5:delay_s * 6, 0]
        assert np.max(np.abs(late)) > 0.001

    def test_feedback_uses_tanh_saturation(self, sr):
        d = Delay(sr)
        d.set_param("feedback", 0.99)
        d.set_param("mix", 1.0)
        d.set_param("time_ms", 10.0)
        loud = np.ones(sr, dtype=np.float32)
        out = d.process(loud)
        # tanh saturation should prevent infinite growth
        assert np.max(np.abs(out)) < 100.0


class TestDelayMix:
    def test_dry_only(self, sr):
        d = Delay(sr)
        d.set_param("mix", 0.0)
        sig = _sine(sr, 440, 0.1)
        out = d.process(sig)
        np.testing.assert_allclose(out[:, 0], sig, atol=1e-6)

    def test_wet_only(self, sr):
        d = Delay(sr)
        d.set_param("mix", 1.0)
        d.set_param("feedback", 0.0)
        sig = _sine(sr, 440, 0.1)
        out = d.process(sig)
        # Output should differ from input (delayed)
        diff = np.max(np.abs(out[:, 0] - sig))
        assert diff > 0.01


class TestDelayPingPong:
    def test_ping_pong_off_channels_equal(self, sr):
        d = Delay(sr)
        d.set_param("ping_pong", 0.0)
        d.set_param("feedback", 0.5)
        d.set_param("mix", 1.0)
        imp = np.zeros(sr, dtype=np.float32)
        imp[0] = 1.0
        out = d.process(imp)
        # Both channels should be the same (mono input)
        np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-6)

    def test_ping_pong_on_alternates(self, sr):
        d = Delay(sr)
        d.set_param("ping_pong", 1.0)
        d.set_param("feedback", 0.6)
        d.set_param("mix", 1.0)
        d.set_param("time_ms", 50.0)
        # Use stereo input with different channels to break symmetry
        n = sr
        left = np.zeros(n, dtype=np.float32)
        left[0] = 1.0
        right = np.zeros(n, dtype=np.float32)
        stereo_in = np.column_stack([left, right])
        out = d.process(stereo_in)
        delay_s = int(50 * 0.001 * sr)
        # With asymmetric input and ping-pong, the channels should differ
        late = out[delay_s:delay_s * 4]
        ch_diff = np.max(np.abs(late[:, 0] - late[:, 1]))
        assert ch_diff > 0.001


class TestDelayReset:
    def test_reset_clears_buffers(self, sr):
        d = Delay(sr)
        d.process(_sine(sr, 440, 0.5))
        d.reset()
        assert np.all(d._buf_l == 0.0)
        assert np.all(d._buf_r == 0.0)
        assert d._write_idx == 0

    def test_after_reset_no_tail(self, sr):
        d = Delay(sr)
        d.set_param("mix", 1.0)
        d.set_param("feedback", 0.5)
        d.process(_sine(sr, 440, 0.5))
        d.reset()
        silence = np.zeros(512, dtype=np.float32)
        out = d.process(silence)
        assert np.max(np.abs(out)) < 0.001


class TestDelayBypass:
    def test_bypass_passthrough(self, sr, sine_440):
        d = Delay(sr)
        d.bypassed = True
        out = d.process_maybe_bypass(sine_440[:512])
        np.testing.assert_array_equal(out, sine_440[:512])


class TestDelayEdgeCases:
    def test_single_sample(self, sr):
        d = Delay(sr)
        out = d.process(np.array([0.5], dtype=np.float32))
        assert out.shape == (1, 2)

    def test_very_short_delay(self, sr):
        d = Delay(sr)
        d.set_param("time_ms", 0.01)
        out = d.process(_sine(sr, 440, 0.05))
        assert not np.any(np.isnan(out))

    def test_max_delay_time(self, sr):
        d = Delay(sr)
        d.set_param("time_ms", 2000.0)
        out = d.process(_sine(sr, 440, 0.1))
        assert out.shape[0] == int(sr * 0.1)


class TestDelayConsecutiveBlocks:
    def test_no_clicks_at_boundaries(self, sr):
        d = Delay(sr)
        d.set_param("mix", 0.5)
        sig = _sine(sr, 440, 0.5)
        bs = 512
        prev = None
        for i in range(0, len(sig), bs):
            block = sig[i:i + bs]
            if len(block) == 0:
                break
            out = d.process(block)
            if prev is not None:
                jump = np.max(np.abs(out[0] - prev))
                assert jump < 0.5
            prev = out[-1].copy()

    def test_state_persists_across_blocks(self, sr):
        d = Delay(sr)
        d.set_param("feedback", 0.5)
        d.set_param("mix", 1.0)
        d.set_param("time_ms", 10.0)
        imp = np.zeros(512, dtype=np.float32)
        imp[0] = 1.0
        d.process(imp)
        silence = np.zeros(512, dtype=np.float32)
        out = d.process(silence)
        # Should still have echo energy from previous block
        assert np.max(np.abs(out)) > 0.001
