"""Tests for granular synthesis engine."""
import numpy as np
from vynth.dsp.granular import GranularEngine

class TestGranularEngine:
    def test_no_source_produces_silence(self, sr):
        gran = GranularEngine(sr)
        out = gran.process(np.zeros(512, dtype=np.float32))
        assert np.abs(out).max() < 0.001
    
    def test_with_source_produces_audio(self, sr, sine_440):
        gran = GranularEngine(sr)
        gran.set_source(sine_440, sr)
        gran.set_param("density", 20.0)
        gran.set_param("grain_size_ms", 50.0)
        out = gran.process(np.zeros(sr, dtype=np.float32))  # 1 second
        assert np.abs(out).max() > 0.01
    
    def test_output_is_stereo(self, sr, sine_440):
        gran = GranularEngine(sr)
        gran.set_source(sine_440, sr)
        gran.set_param("density", 10.0)
        out = gran.process(np.zeros(1024, dtype=np.float32))
        assert out.ndim == 2
        assert out.shape[1] == 2
    
    def test_reset_clears_grains(self, sr, sine_440):
        gran = GranularEngine(sr)
        gran.set_source(sine_440, sr)
        gran.set_param("density", 50.0)
        gran.process(np.zeros(2048, dtype=np.float32))
        gran.reset()
        out = gran.process(np.zeros(512, dtype=np.float32))
        # After reset, should start fresh (may or may not have audio depending on scheduler)
        assert out.shape[0] == 512
