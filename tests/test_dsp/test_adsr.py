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
