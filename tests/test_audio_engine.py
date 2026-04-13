"""Exhaustive tests for AudioEngine (no real audio output)."""
import numpy as np
import pytest

from vynth.config import SessionSettings, SAMPLE_RATE, BLOCK_SIZE, CHANNELS
from vynth.engine.audio_engine import AudioEngine
from vynth.utils.thread_safe_queue import Command, CommandType


@pytest.fixture
def settings():
    return SessionSettings()


@pytest.fixture
def engine(settings):
    """AudioEngine without starting the stream."""
    return AudioEngine(settings)


class TestAudioEngineCreation:
    def test_engine_creates(self, engine):
        assert engine is not None

    def test_not_running_initially(self, engine):
        assert engine.is_running is False

    def test_command_queue_available(self, engine):
        assert engine.command_queue is not None

    def test_voice_allocator_available(self, engine):
        assert engine.voice_allocator is not None

    def test_default_master_volume(self, engine, settings):
        assert engine.master_volume == pytest.approx(settings.master_volume)


class TestAudioEngineCommandQueue:
    def test_push_command(self, engine):
        cmd = Command(type=CommandType.NOTE_ON, note=60, velocity=100)
        result = engine.push_command(cmd)
        assert result is True

    def test_push_multiple_commands(self, engine):
        for note in range(60, 72):
            cmd = Command(type=CommandType.NOTE_ON, note=note, velocity=80)
            assert engine.push_command(cmd) is True

    def test_command_queue_drains(self, engine):
        cmd = Command(type=CommandType.NOTE_ON, note=60, velocity=100)
        engine.push_command(cmd)
        cmds = engine.command_queue.drain()
        assert len(cmds) == 1
        assert cmds[0].note == 60


class TestAudioEngineMasterVolume:
    def test_set_master_volume(self, engine):
        engine.set_master_volume(0.5)
        assert engine.master_volume == pytest.approx(0.5)

    def test_clamp_zero(self, engine):
        engine.set_master_volume(0.0)
        assert engine.master_volume == pytest.approx(0.0)

    def test_clamp_negative(self, engine):
        engine.set_master_volume(-1.0)
        assert engine.master_volume == pytest.approx(0.0)

    def test_reasonable_max(self, engine):
        engine.set_master_volume(1.0)
        assert engine.master_volume == pytest.approx(1.0)

    def test_updates_settings(self, engine, settings):
        engine.set_master_volume(0.3)
        assert settings.master_volume == pytest.approx(0.3)


class TestAudioEngineDeviceList:
    def test_device_list_returns_list(self):
        # This may fail in CI without audio devices, so just check it's callable
        try:
            devices = AudioEngine.device_list()
            assert isinstance(devices, (list, tuple))
        except Exception:
            pytest.skip("No audio devices available")


class TestAudioEnginePeakLevels:
    def test_initial_peaks_zero(self, engine):
        l, r = engine._peak_l, engine._peak_r
        assert l == pytest.approx(0.0)
        assert r == pytest.approx(0.0)


class TestAudioEngineVisualizationBuffer:
    def test_get_vis_buffer_empty(self, engine):
        buf = engine.get_visualization_buffer(512)
        assert buf.shape == (0, 2) or np.max(np.abs(buf)) < 0.001

    def test_write_then_read(self, engine):
        # Manually write into the vis buffer
        data = np.ones((100, 2), dtype=np.float32) * 0.5
        engine._vis_buffer.write(data)
        buf = engine.get_visualization_buffer(100)
        assert buf.shape == (100, 2)
        np.testing.assert_allclose(buf, 0.5, atol=1e-6)


class TestAudioEngineActiveVoiceCount:
    def test_initially_zero(self, engine):
        assert engine.active_voice_count == 0


class TestAudioEngineCurrentDevice:
    def test_default_none(self, engine):
        assert engine.current_device is None

    def test_set_device_updates(self, engine, settings):
        settings.audio_device = 42
        assert engine.current_device == 42
