"""Tests for WAV export."""
import numpy as np
import tempfile
import os
import soundfile as sf
from vynth.engine.export import Exporter
from vynth.sampler.sample import Sample
from vynth.config import SAMPLE_RATE

class TestExport:
    def test_export_wav_16bit(self):
        exporter = Exporter()
        data = np.sin(np.linspace(0, 2*np.pi*440, SAMPLE_RATE, dtype=np.float32))
        sample = Sample.from_buffer(data, SAMPLE_RATE, "export_test")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            exporter.export_sample(sample, path, sample_rate=44100, bit_depth=16)
            assert os.path.exists(path)
            info = sf.info(path)
            assert info.samplerate == 44100
            assert info.subtype == "PCM_16"
        finally:
            os.unlink(path)
    
    def test_export_wav_24bit(self):
        exporter = Exporter()
        data = np.sin(np.linspace(0, 2*np.pi*440, SAMPLE_RATE, dtype=np.float32))
        sample = Sample.from_buffer(data, SAMPLE_RATE, "export_test")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            exporter.export_sample(sample, path, sample_rate=48000, bit_depth=24)
            info = sf.info(path)
            assert info.samplerate == 48000
            assert info.subtype == "PCM_24"
        finally:
            os.unlink(path)
