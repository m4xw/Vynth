"""Tests for sample manager."""
import numpy as np
import tempfile
import os
import pytest
from vynth.sampler.sample import Sample
from vynth.sampler.sample_manager import SampleManager
from vynth.config import SAMPLE_RATE

class TestSampleManager:
    def test_add_and_get(self):
        mgr = SampleManager()
        data = np.random.randn(48000).astype(np.float32)
        sample = Sample.from_buffer(data, SAMPLE_RATE, "test_sample")
        mgr.add_sample(sample)
        assert mgr.get_sample("test_sample") is not None
    
    def test_remove(self):
        mgr = SampleManager()
        data = np.random.randn(48000).astype(np.float32)
        sample = Sample.from_buffer(data, SAMPLE_RATE, "test_sample")
        mgr.add_sample(sample)
        mgr.remove_sample("test_sample")
        assert mgr.get_sample("test_sample") is None
    
    def test_get_names(self):
        mgr = SampleManager()
        for name in ["a", "b", "c"]:
            data = np.zeros(1000, dtype=np.float32)
            mgr.add_sample(Sample.from_buffer(data, SAMPLE_RATE, name))
        names = mgr.get_names()
        assert len(names) == 3
    
    def test_load_wav_file(self):
        mgr = SampleManager()
        # Create temp WAV
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            data = np.random.randn(48000).astype(np.float32)
            sf.write(path, data, SAMPLE_RATE)
            sample = mgr.load_sample(path)
            assert sample.length == 48000
        finally:
            os.unlink(path)
    
    def test_select_sample(self):
        mgr = SampleManager()
        data = np.zeros(1000, dtype=np.float32)
        mgr.add_sample(Sample.from_buffer(data, SAMPLE_RATE, "sel_test"))
        mgr.select_sample("sel_test")
        assert mgr.get_selected() is not None
        assert mgr.get_selected().name == "sel_test"
