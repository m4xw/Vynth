"""Exhaustive tests for ADSR envelope generator."""
import numpy as np
import pytest
from vynth.dsp.adsr import ADSREnvelope, ADSRState


class TestADSRInitialization:
    """Construction and default state."""

    def test_initial_state_is_idle(self, sr):
        env = ADSREnvelope(sr)
        assert env.state == ADSRState.IDLE

    def test_initial_level_is_zero(self, sr):
        env = ADSREnvelope(sr)
        assert env.level == 0.0

    def test_not_active_initially(self, sr):
        env = ADSREnvelope(sr)
        assert not env.is_active

    def test_default_attack_ms(self, sr):
        env = ADSREnvelope(sr)
        assert env.get_param("attack_ms") == pytest.approx(10.0)

    def test_default_decay_ms(self, sr):
        env = ADSREnvelope(sr)
        assert env.get_param("decay_ms") == pytest.approx(100.0)

    def test_default_sustain(self, sr):
        env = ADSREnvelope(sr)
        assert env.get_param("sustain") == pytest.approx(0.7)

    def test_default_release_ms(self, sr):
        env = ADSREnvelope(sr)
        assert env.get_param("release_ms") == pytest.approx(200.0)

    def test_custom_sample_rate(self):
        env = ADSREnvelope(44100)
        assert env.sample_rate == 44100

    def test_default_sample_rate(self):
        env = ADSREnvelope()
        assert env.sample_rate == 48000


class TestADSRGateControl:
    """Gate on/off behavior."""

    def test_gate_on_transitions_to_attack(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        assert env.state == ADSRState.ATTACK
        assert env.is_active

    def test_gate_on_with_velocity(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(0.5)
        assert env.is_active

    def test_gate_on_zero_velocity(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(0.0)
        assert env.is_active
        assert env.state == ADSRState.ATTACK

    def test_gate_on_clips_velocity_high(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(2.0)
        env.set_param("attack_ms", 1.0)
        out = env.generate(sr)
        assert out.max() <= 1.0 + 1e-4

    def test_gate_on_clips_velocity_low(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(-0.5)
        env.set_param("attack_ms", 1.0)
        out = env.generate(sr // 10)
        assert out.min() >= -1e-4

    def test_gate_off_from_attack(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        env.gate_off()
        assert env.state == ADSRState.RELEASE

    def test_gate_off_from_sustain(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 1.0)
        env.gate_on(1.0)
        env.generate(sr // 2)
        assert env.state == ADSRState.SUSTAIN
        env.gate_off()
        assert env.state == ADSRState.RELEASE

    def test_gate_off_from_idle_stays_idle(self, sr):
        env = ADSREnvelope(sr)
        env.gate_off()
        assert env.state == ADSRState.IDLE

    def test_retrigger_from_release(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        env.generate(100)
        env.gate_off()
        env.generate(50)
        env.gate_on(0.8)
        assert env.state == ADSRState.ATTACK

    def test_retrigger_from_sustain(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 1.0)
        env.gate_on(1.0)
        env.generate(sr // 2)
        env.gate_on(0.6)
        assert env.state == ADSRState.ATTACK


class TestADSRStageTransitions:
    """Verify correct state progression."""

    def test_attack_reaches_peak(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 10.0)
        env.gate_on(1.0)
        out = env.generate(sr // 10)
        assert out.max() >= 0.99

    def test_attack_to_decay_transition(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 5.0)
        env.gate_on(1.0)
        env.generate(sr // 10)
        assert env.state in (ADSRState.DECAY, ADSRState.SUSTAIN)

    def test_decay_to_sustain_transition(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 10.0)
        env.set_param("sustain", 0.5)
        env.gate_on(1.0)
        env.generate(sr // 2)
        assert env.state == ADSRState.SUSTAIN

    def test_sustain_level_holds(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 1.0)
        env.set_param("sustain", 0.6)
        env.gate_on(1.0)
        env.generate(sr // 2)
        out = env.generate(1024)
        np.testing.assert_allclose(out, 0.6, atol=1e-3)

    def test_release_to_idle(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 1.0)
        env.set_param("release_ms", 10.0)
        env.gate_on(1.0)
        env.generate(sr // 10)
        env.gate_off()
        env.generate(sr)
        assert env.state == ADSRState.IDLE
        assert not env.is_active

    def test_full_cycle_idle_to_idle(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 10)
        env.set_param("decay_ms", 10)
        env.set_param("sustain", 0.5)
        env.set_param("release_ms", 10)
        env.gate_on(1.0)
        env.generate(sr // 10)
        env.gate_off()
        env.generate(sr)
        assert not env.is_active

    def test_very_fast_adsr(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 1.0)
        env.set_param("sustain", 0.0)
        env.set_param("release_ms", 1.0)
        env.gate_on(1.0)
        env.generate(sr // 10)
        env.gate_off()
        env.generate(sr)
        assert not env.is_active

    def test_very_slow_attack(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 5000.0)
        env.gate_on(1.0)
        out = env.generate(sr // 10)
        assert env.state == ADSRState.ATTACK
        assert out[-1] > out[0]
        assert out[-1] < 0.5


class TestADSRVelocity:
    """Velocity-dependent peak scaling."""

    def test_velocity_half(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.gate_on(0.5)
        out = env.generate(sr // 10)
        assert 0.45 < out.max() < 0.55

    def test_velocity_full(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.gate_on(1.0)
        out = env.generate(sr // 10)
        assert out.max() >= 0.99

    def test_velocity_scales_sustain(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 10.0)
        env.set_param("sustain", 0.5)
        env.gate_on(0.8)
        env.generate(sr // 2)
        out = env.generate(256)
        np.testing.assert_allclose(out, 0.4, atol=0.02)

    def test_different_velocities_monotonic(self, sr):
        peaks = []
        for vel in [0.25, 0.5, 0.75, 1.0]:
            env = ADSREnvelope(sr)
            env.set_param("attack_ms", 1.0)
            env.gate_on(vel)
            out = env.generate(sr // 10)
            peaks.append(out.max())
        for i in range(len(peaks) - 1):
            assert peaks[i] < peaks[i + 1]


class TestADSRGenerate:
    """generate() output shape and correctness."""

    def test_output_shape(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        out = env.generate(1024)
        assert out.shape == (1024,)
        assert out.dtype == np.float32

    def test_idle_generates_zeros(self, sr):
        env = ADSREnvelope(sr)
        out = env.generate(512)
        np.testing.assert_array_equal(out, np.zeros(512, dtype=np.float32))

    def test_attack_monotonically_increasing(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 100.0)
        env.gate_on(1.0)
        out = env.generate(int(sr * 0.05))
        diffs = np.diff(out)
        assert np.all(diffs >= -1e-6)

    def test_release_monotonically_decreasing(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 1.0)
        env.set_param("release_ms", 200.0)
        env.gate_on(1.0)
        env.generate(sr // 10)
        env.gate_off()
        out = env.generate(int(sr * 0.1))
        diffs = np.diff(out)
        assert np.all(diffs <= 1e-6)

    def test_no_negative_values(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        out = env.generate(sr)
        env.gate_off()
        out2 = env.generate(sr)
        assert out.min() >= -1e-6
        assert out2.min() >= -1e-6

    def test_consecutive_generate_equals_single(self, sr):
        env1 = ADSREnvelope(sr)
        env1.set_param("attack_ms", 50.0)
        env1.gate_on(1.0)
        all_at_once = env1.generate(2048)

        env2 = ADSREnvelope(sr)
        env2.set_param("attack_ms", 50.0)
        env2.gate_on(1.0)
        chunked = np.concatenate([env2.generate(512) for _ in range(4)])

        np.testing.assert_allclose(all_at_once, chunked, atol=1e-6)

    def test_single_sample_generate(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        out = env.generate(1)
        assert out.shape == (1,)

    def test_zero_frame_generate(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        out = env.generate(0)
        assert out.shape == (0,)


class TestADSRProcess:
    """process() applies envelope to audio."""

    def test_process_mono(self, sr, sine_440):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        out = env.process(sine_440[:1024])
        assert out.shape == (1024,)
        assert out.dtype == np.float32

    def test_process_stereo(self, sr, sine_440_stereo):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        out = env.process(sine_440_stereo[:1024])
        assert out.shape == (1024, 2)

    def test_process_multiplies_audio(self, sr, sine_440):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        out = env.process(sine_440[:1024])
        assert np.abs(out[0]) < np.abs(sine_440[1]) + 0.01

    def test_process_idle_returns_zeros(self, sr, sine_440):
        env = ADSREnvelope(sr)
        out = env.process(sine_440[:512])
        np.testing.assert_array_equal(out, np.zeros(512, dtype=np.float32))

    def test_process_sustain_level(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 1.0)
        env.set_param("sustain", 0.5)
        env.gate_on(1.0)
        env.generate(sr // 10)
        ones = np.ones(256, dtype=np.float32)
        out = env.process(ones)
        np.testing.assert_allclose(out, 0.5, atol=0.01)


class TestADSRBypass:
    """Bypass mode behavior."""

    def test_bypassed_sustain_passes_audio(self, sr):
        env = ADSREnvelope(sr)
        env.bypassed = True
        env.gate_on(1.0)
        data = np.ones(512, dtype=np.float32)
        out = env.process(data)
        np.testing.assert_allclose(out, data, atol=1e-6)

    def test_bypassed_attack_jumps_to_sustain(self, sr):
        env = ADSREnvelope(sr)
        env.bypassed = True
        env.gate_on(1.0)
        env.process(np.ones(512, dtype=np.float32))
        assert env.state == ADSRState.SUSTAIN

    def test_bypassed_release_still_fades(self, sr):
        env = ADSREnvelope(sr)
        env.bypassed = True
        env.set_param("release_ms", 20)
        env.gate_on(1.0)
        env.process(np.ones(512, dtype=np.float32))
        env.gate_off()
        out = env.process(np.ones(512, dtype=np.float32))
        assert out[0] > out[-1]
        assert out[-1] < 0.99


class TestADSRReset:
    """reset() clears state."""

    def test_reset_clears_state(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        env.generate(100)
        env.reset()
        assert not env.is_active
        assert env.state == ADSRState.IDLE
        assert env.level == 0.0

    def test_reset_allows_reuse(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        env.generate(sr)
        env.reset()
        env.gate_on(0.7)
        out = env.generate(sr // 10)
        assert env.is_active
        assert out.max() > 0.0


class TestADSRParamChanges:
    """Parameter changes and coefficient recalculation."""

    def test_set_attack_ms(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 50.0)
        assert env.get_param("attack_ms") == pytest.approx(50.0)

    def test_set_decay_ms(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("decay_ms", 200.0)
        assert env.get_param("decay_ms") == pytest.approx(200.0)

    def test_set_sustain(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("sustain", 0.3)
        assert env.get_param("sustain") == pytest.approx(0.3)

    def test_set_release_ms(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("release_ms", 500.0)
        assert env.get_param("release_ms") == pytest.approx(500.0)

    def test_param_change_during_sustain(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 1.0)
        env.set_param("sustain", 0.8)
        env.gate_on(1.0)
        env.generate(sr // 2)
        env.set_param("sustain", 0.3)
        out = env.generate(512)
        np.testing.assert_allclose(out, 0.3, atol=0.02)

    def test_get_params_returns_all(self, sr):
        env = ADSREnvelope(sr)
        params = env.get_params()
        assert "attack_ms" in params
        assert "decay_ms" in params
        assert "sustain" in params
        assert "release_ms" in params

    def test_set_params_bulk(self, sr):
        env = ADSREnvelope(sr)
        env.set_params({"attack_ms": 25.0, "decay_ms": 50.0, "sustain": 0.4, "release_ms": 300.0})
        assert env.get_param("attack_ms") == pytest.approx(25.0)
        assert env.get_param("sustain") == pytest.approx(0.4)


class TestADSRCoefficients:
    """ms_to_coeff static method edge cases."""

    def test_ms_to_coeff_positive(self):
        coeff = ADSREnvelope._ms_to_coeff(100.0, 48000)
        assert 0.0 < coeff < 1.0

    def test_ms_to_coeff_very_short(self):
        coeff = ADSREnvelope._ms_to_coeff(0.01, 48000)
        assert 0.0 < coeff < 1.0

    def test_ms_to_coeff_very_long(self):
        coeff = ADSREnvelope._ms_to_coeff(10000.0, 48000)
        assert 0.0 < coeff < 1.0

    def test_ms_to_coeff_zero(self):
        coeff = ADSREnvelope._ms_to_coeff(0.0, 48000)
        assert 0.0 < coeff < 1.0

    def test_shorter_gives_smaller_coeff(self):
        fast = ADSREnvelope._ms_to_coeff(10.0, 48000)
        slow = ADSREnvelope._ms_to_coeff(1000.0, 48000)
        assert fast < slow


class TestADSREdgeCases:
    """Edge cases and robustness."""

    def test_sustain_zero(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("decay_ms", 50.0)
        env.set_param("sustain", 0.0)
        env.gate_on(1.0)
        out = env.generate(sr)
        assert out[-1] < 0.01

    def test_sustain_one(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1.0)
        env.set_param("sustain", 1.0)
        env.gate_on(1.0)
        env.generate(sr // 10)
        out = env.generate(256)
        np.testing.assert_allclose(out, 1.0, atol=0.01)

    def test_rapid_retrigger(self, sr):
        env = ADSREnvelope(sr)
        for _ in range(100):
            env.gate_on(1.0)
            env.generate(10)
            env.gate_off()
            env.generate(10)
        assert isinstance(env.level, float)

    def test_multiple_gate_off_safe(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        env.gate_off()
        env.gate_off()
        assert env.state == ADSRState.RELEASE

    def test_different_sample_rates(self):
        for target_sr in [22050, 44100, 48000, 96000]:
            env = ADSREnvelope(target_sr)
            env.set_param("attack_ms", 100.0)
            env.gate_on(1.0)
            n = int(target_sr * 0.1)
            out = env.generate(n)
            # Exponential approach reaches ~63% after one time constant
            assert out[-1] > 0.5, f"sr={target_sr}: level={out[-1]}"
"""Tests for ADSR envelope generator."""
import numpy as np
import pytest
from vynth.dsp.adsr import ADSREnvelope, ADSRState

class TestADSR:
    def test_initial_state_is_idle(self, sr):
        env = ADSREnvelope(sr)
        assert not env.is_active
    
    def test_gate_on_transitions_to_attack(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        assert env.is_active
    
    def test_gate_off_triggers_release(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        env.generate(sr)  # let it reach sustain
        env.gate_off()
        out = env.generate(sr)
        # Should be decaying
        assert out[-1] < out[0]
    
    def test_full_cycle_returns_to_idle(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 10)
        env.set_param("decay_ms", 10)
        env.set_param("sustain", 0.5)
        env.set_param("release_ms", 10)
        env.gate_on(1.0)
        env.generate(sr // 10)  # attack+decay
        env.gate_off()
        env.generate(sr)  # long release
        assert not env.is_active
    
    def test_velocity_scales_peak(self, sr):
        env = ADSREnvelope(sr)
        env.set_param("attack_ms", 1)
        env.gate_on(0.5)
        out = env.generate(sr // 10)
        assert out.max() <= 0.55  # ~0.5 ± tolerance
    
    def test_process_multiplies_audio(self, sr, sine_440):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        n = 1024
        result = env.process(sine_440[:n])
        # At beginning of attack, should be near zero
        assert np.abs(result[0]) < 0.1
    
    def test_reset_clears_state(self, sr):
        env = ADSREnvelope(sr)
        env.gate_on(1.0)
        env.generate(100)
        env.reset()
        assert not env.is_active

    def test_bypassed_adsr_still_decays_on_release(self, sr):
        env = ADSREnvelope(sr)
        env.bypassed = True
        env.set_param("release_ms", 20)
        env.gate_on(1.0)

        data = np.ones(512, dtype=np.float32)
        sustained = env.process(data)
        np.testing.assert_allclose(sustained, data, atol=1e-6)

        env.gate_off()
        released = env.process(data)
        assert released[0] > released[-1]
        assert released[-1] < 0.99
