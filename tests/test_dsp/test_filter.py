"""Tests for biquad filter."""
import numpy as np
from vynth.dsp.filter import BiquadFilter

class TestBiquadFilter:
    def test_lowpass_attenuates_high_freq(self, sr):
        # 100Hz sine should pass, 10kHz should be attenuated
        t = np.arange(sr, dtype=np.float32) / sr
        low = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        high = np.sin(2 * np.pi * 10000 * t).astype(np.float32)
        
        filt = BiquadFilter(sr)
        filt.set_param("mode", 0)  # LowPass
        filt.set_param("frequency", 500.0)
        filt.set_param("q", 0.707)
        
        out_low = filt.process(low)
        filt.reset()
        out_high = filt.process(high)
        
        # Low freq should pass mostly intact
        assert np.abs(out_low[sr//2:]).max() > 0.5
        # High freq should be heavily attenuated
        assert np.abs(out_high[sr//2:]).max() < 0.1
    
    def test_highpass_attenuates_low_freq(self, sr):
        t = np.arange(sr, dtype=np.float32) / sr
        low = np.sin(2 * np.pi * 50 * t).astype(np.float32)
        
        filt = BiquadFilter(sr)
        filt.set_param("mode", 1)  # HighPass
        filt.set_param("frequency", 5000.0)
        
        out = filt.process(low)
        assert np.abs(out[sr//2:]).max() < 0.05
    
    def test_reset_clears_state(self, sr):
        filt = BiquadFilter(sr)
        filt.process(np.random.randn(1024).astype(np.float32))
        filt.reset()
        out = filt.process(np.zeros(512, dtype=np.float32))
        assert np.abs(out).max() < 0.001
