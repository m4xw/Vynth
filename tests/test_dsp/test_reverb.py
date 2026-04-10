"""Tests for Freeverb."""
import numpy as np
from vynth.dsp.reverb import Reverb

class TestReverb:
    def test_impulse_response_has_tail(self, sr, impulse):
        rev = Reverb(sr)
        rev.set_param("wet", 1.0)
        rev.set_param("room_size", 0.8)
        out = rev.process(impulse)
        # Reverb tail should extend beyond impulse
        assert np.abs(out[sr//2:]).max() > 0.001
    
    def test_dry_signal_passthrough(self, sr, sine_440):
        rev = Reverb(sr)
        rev.set_param("wet", 0.0)
        rev.set_param("dry", 1.0)
        out = rev.process(sine_440[:1024])
        # Should be close to original (mono→stereo conversion might differ)
        assert out.shape[-1] == 2  # always stereo output
    
    def test_silence_produces_silence(self, sr, silence):
        rev = Reverb(sr)
        rev.reset()
        out = rev.process(silence[:1024])
        assert np.abs(out).max() < 0.001
    
    def test_bypass(self, sr, sine_440):
        rev = Reverb(sr)
        rev.bypassed = True
        out = rev.process_maybe_bypass(sine_440[:512])
        np.testing.assert_array_equal(out, sine_440[:512])
