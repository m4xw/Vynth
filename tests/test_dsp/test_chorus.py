"""Tests for chorus/unison effect."""
import numpy as np
from vynth.dsp.chorus import Chorus

class TestChorus:
    def test_output_is_stereo(self, sr, sine_440):
        ch = Chorus(sr)
        out = ch.process(sine_440[:1024])
        assert out.ndim == 2
        assert out.shape[1] == 2
    
    def test_stereo_spread(self, sr, sine_440):
        ch = Chorus(sr)
        ch.set_param("spread", 1.0)
        ch.set_param("mix", 1.0)
        out = ch.process(sine_440[:2048])
        # L and R should differ
        if out.shape[0] > 0:
            diff = np.abs(out[:, 0] - out[:, 1]).max()
            assert diff > 0.001
    
    def test_zero_mix_passthrough(self, sr, sine_440):
        ch = Chorus(sr)
        ch.set_param("mix", 0.0)
        out = ch.process(sine_440[:512])
        assert out.shape[0] == 512
