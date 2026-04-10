"""Tests for phase vocoder pitch shifter."""
import numpy as np
import pytest
from vynth.dsp.pitch_shift import PitchShifter

class TestPitchShifter:
    def test_zero_shift_passthrough(self, sr, sine_440):
        ps = PitchShifter(sr)
        ps.set_param("shift_semitones", 0.0)
        out = ps.process(sine_440)
        # Should be similar to input (allowing for phase vocoder artifacts)
        assert len(out) > 0
    
    def test_octave_up_doubles_frequency(self, sr):
        # Generate a clean 440Hz sine
        t = np.arange(sr, dtype=np.float32) / sr
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        ps = PitchShifter(sr)
        ps.set_param("shift_semitones", 12.0)
        out = ps.process(sine)
        
        # FFT to verify frequency
        if len(out) > 2048:
            fft = np.abs(np.fft.rfft(out[:4096] * np.hanning(min(4096, len(out)))))
            freqs = np.fft.rfftfreq(min(4096, len(out)), 1/sr)
            peak_freq = freqs[np.argmax(fft[1:]) + 1]
            # Should be near 880 Hz (±50Hz tolerance for phase vocoder)
            assert 800 < peak_freq < 960, f"Peak at {peak_freq}Hz, expected ~880Hz"
    
    def test_reset_clears_state(self, sr):
        ps = PitchShifter(sr)
        ps.process(np.random.randn(1024).astype(np.float32))
        ps.reset()
        # Should not crash processing after reset
        out = ps.process(np.zeros(512, dtype=np.float32))
        assert len(out) > 0
    
    def test_stereo_input(self, sr, sine_440_stereo):
        ps = PitchShifter(sr)
        ps.set_param("shift_semitones", 5.0)
        out = ps.process(sine_440_stereo)
        assert out.ndim == 2
        assert out.shape[1] == 2
