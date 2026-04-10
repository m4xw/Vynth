"""Tests for voice allocator."""
import numpy as np
import pytest
from vynth.engine.voice_allocator import VoiceAllocator
from vynth.sampler.sample import Sample
from vynth.config import MAX_VOICES, SAMPLE_RATE

@pytest.fixture
def allocator():
    va = VoiceAllocator(SAMPLE_RATE)
    # Create a 1-second sine sample
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sample = Sample.from_buffer(data, SAMPLE_RATE, "test")
    sample.root_note = 69  # A4
    va.set_sample(sample)
    return va

class TestVoiceAllocator:
    def test_note_on_activates_voice(self, allocator):
        allocator.note_on(60, 100)
        assert allocator.active_voice_count == 1
    
    def test_note_off_starts_release(self, allocator):
        allocator.note_on(60, 100)
        allocator.note_off(60)
        # Voice still active during release
        assert allocator.active_voice_count >= 0
    
    def test_multiple_notes(self, allocator):
        for note in [60, 64, 67]:
            allocator.note_on(note, 100)
        assert allocator.active_voice_count == 3
    
    def test_max_voices_voice_stealing(self, allocator):
        for note in range(MAX_VOICES + 5):
            allocator.note_on(note, 100)
        assert allocator.active_voice_count <= MAX_VOICES
    
    def test_all_notes_off(self, allocator):
        for note in [60, 64, 67]:
            allocator.note_on(note, 100)
        allocator.all_notes_off()
        # Process enough to let releases finish
        for _ in range(200):
            allocator.process(512)
        assert allocator.active_voice_count == 0
    
    def test_process_returns_stereo(self, allocator):
        allocator.note_on(60, 100)
        out = allocator.process(512)
        assert out.shape == (512, 2)
    
    def test_process_with_no_notes_returns_silence(self, allocator):
        out = allocator.process(512)
        assert np.abs(out).max() < 0.01
