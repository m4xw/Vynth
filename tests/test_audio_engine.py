"""Tests for audio engine (non-audio-device tests)."""
import numpy as np
import pytest
from vynth.config import SessionSettings, SAMPLE_RATE
from vynth.engine.audio_engine import AudioEngine
from vynth.utils.thread_safe_queue import Command, CommandType

class TestAudioEngine:
    def test_creation(self):
        settings = SessionSettings()
        engine = AudioEngine(settings)
        assert engine is not None
        assert not engine.is_running
    
    def test_command_queue(self):
        settings = SessionSettings()
        engine = AudioEngine(settings)
        cmd = Command(type=CommandType.NOTE_ON, note=60, velocity=100)
        engine.push_command(cmd)
        # Command should be in queue
        assert len(engine._command_queue) == 1
    
    def test_master_volume(self):
        settings = SessionSettings()
        engine = AudioEngine(settings)
        engine.set_master_volume(0.5)
        assert abs(engine._master_volume - 0.5) < 0.01
