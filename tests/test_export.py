"""Exhaustive tests for the Exporter (offline WAV rendering)."""
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

import soundfile as sf

from vynth.config import SAMPLE_RATE
from vynth.engine.export import Exporter, _validate_path, _resample, _clone_allocator_params
from vynth.engine.voice_allocator import VoiceAllocator
from vynth.sampler.sample import Sample


def _make_sample(sr=48000, duration_s=0.1, channels=1):
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    data = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    if channels == 2:
        data = np.column_stack([data, data])
    return Sample.from_buffer(data, sr, "test")


class TestExportSample:
    def test_export_16bit(self, tmp_path):
        s = _make_sample()
        path = str(tmp_path / "out_16.wav")
        Exporter.export_sample(s, path, sample_rate=48000, bit_depth=16)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_export_24bit(self, tmp_path):
        s = _make_sample()
        path = str(tmp_path / "out_24.wav")
        Exporter.export_sample(s, path, sample_rate=48000, bit_depth=24)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_export_stereo(self, tmp_path):
        s = _make_sample(channels=2)
        path = str(tmp_path / "stereo.wav")
        Exporter.export_sample(s, path, sample_rate=48000, bit_depth=24)
        assert os.path.isfile(path)

    def test_export_resample_44100(self, tmp_path):
        s = _make_sample(sr=48000)
        path = str(tmp_path / "resampled.wav")
        Exporter.export_sample(s, path, sample_rate=44100, bit_depth=24)
        assert os.path.isfile(path)

    def test_export_empty_sample(self, tmp_path):
        s = Sample.from_buffer(np.array([], dtype=np.float32), 48000, "empty")
        path = str(tmp_path / "empty.wav")
        Exporter.export_sample(s, path, sample_rate=48000, bit_depth=16)
        assert os.path.isfile(path)


class TestExportSampleErrors:
    def test_unsupported_sample_rate(self, tmp_path):
        s = _make_sample()
        path = str(tmp_path / "bad_sr.wav")
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            Exporter.export_sample(s, path, sample_rate=22050, bit_depth=16)

    def test_unsupported_bit_depth(self, tmp_path):
        s = _make_sample()
        path = str(tmp_path / "bad_bits.wav")
        with pytest.raises(ValueError, match="Unsupported bit depth"):
            Exporter.export_sample(s, path, sample_rate=48000, bit_depth=32)


class TestValidatePath:
    def test_valid_path(self, tmp_path):
        path = str(tmp_path / "test.wav")
        _validate_path(path)  # Should not raise

    def test_path_traversal_rejected(self, tmp_path):
        path = str(tmp_path / ".." / ".." / "test.wav")
        with pytest.raises(ValueError, match="traversal"):
            _validate_path(path)

    def test_creates_parent_dir(self, tmp_path):
        path = str(tmp_path / "subdir" / "test.wav")
        _validate_path(path)
        assert (tmp_path / "subdir").is_dir()


class TestResampleHelper:
    def test_resample_mono(self):
        data = np.sin(np.linspace(0, 2 * np.pi * 440, 48000)).astype(np.float32)
        out = _resample(data, 48000, 44100)
        expected_len = int(48000 * 44100 / 48000)
        assert abs(len(out) - expected_len) < 10

    def test_resample_stereo(self):
        mono = np.sin(np.linspace(0, 2 * np.pi * 440, 48000)).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        out = _resample(stereo, 48000, 44100)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_resample_dtype(self):
        data = np.ones(1000, dtype=np.float32)
        out = _resample(data, 48000, 44100)
        assert out.dtype == np.float32


class TestRenderPerformance:
    def test_render_simple_performance(self, tmp_path):
        from vynth.engine.voice_allocator import VoiceAllocator
        va = VoiceAllocator()
        sample = _make_sample(sr=48000, duration_s=1.0)
        va.set_sample(sample)

        events = [
            {"time_s": 0.0, "type": "note_on", "note": 60, "velocity": 100},
            {"time_s": 0.5, "type": "note_off", "note": 60},
        ]

        path = str(tmp_path / "performance.wav")
        Exporter.render_performance(va, events, 1.0, 48000, path)
        assert os.path.isfile(path)

    def test_render_empty_events(self, tmp_path):
        from vynth.engine.voice_allocator import VoiceAllocator
        va = VoiceAllocator()

        path = str(tmp_path / "silent.wav")
        Exporter.render_performance(va, [], 0.5, 48000, path)
        assert os.path.isfile(path)

    def test_render_multiple_notes(self, tmp_path):
        from vynth.engine.voice_allocator import VoiceAllocator
        va = VoiceAllocator()
        sample = _make_sample(sr=48000, duration_s=2.0)
        va.set_sample(sample)

        events = [
            {"time_s": 0.0, "type": "note_on", "note": 60, "velocity": 100},
            {"time_s": 0.1, "type": "note_on", "note": 64, "velocity": 80},
            {"time_s": 0.2, "type": "note_on", "note": 67, "velocity": 90},
            {"time_s": 0.5, "type": "note_off", "note": 60},
            {"time_s": 0.6, "type": "note_off", "note": 64},
            {"time_s": 0.7, "type": "note_off", "note": 67},
        ]

        path = str(tmp_path / "chord.wav")
        Exporter.render_performance(va, events, 1.0, 48000, path)
        assert os.path.isfile(path)


# ── render_sample (export with effects) ──────────────────────────────────


def _make_long_sample(sr=48000, duration_s=0.5, freq=440):
    """Create a sample long enough for meaningful DSP processing."""
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    data = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    s = Sample.from_buffer(data, sr, "test_long")
    s.root_note = 60
    return s


class TestRenderSample:
    def test_creates_file(self, tmp_path):
        va = VoiceAllocator()
        sample = _make_long_sample()
        va.set_sample(sample)

        path = str(tmp_path / "rendered.wav")
        Exporter.render_sample(sample, va, path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_output_is_stereo(self, tmp_path):
        va = VoiceAllocator()
        sample = _make_long_sample()
        va.set_sample(sample)

        path = str(tmp_path / "rendered_stereo.wav")
        Exporter.render_sample(sample, va, path, sample_rate=48000, bit_depth=24)
        data, sr = sf.read(path)
        assert data.ndim == 2
        assert data.shape[1] == 2
        assert sr == 48000

    def test_output_contains_audio(self, tmp_path):
        va = VoiceAllocator()
        sample = _make_long_sample()
        va.set_sample(sample)
        # Use fast ADSR so audio appears quickly
        va.set_param("adsr_attack_ms", 1.0)
        va.set_param("adsr_sustain", 1.0)

        path = str(tmp_path / "rendered_audio.wav")
        Exporter.render_sample(sample, va, path)
        data, _ = sf.read(path)
        assert np.max(np.abs(data)) > 0.01

    def test_effects_are_applied(self, tmp_path):
        """Rendered export with heavy reverb should differ from dry export."""
        sample = _make_long_sample(duration_s=0.3)

        # Dry export (no effects)
        va_dry = VoiceAllocator()
        va_dry.set_param("adsr_attack_ms", 1.0)
        va_dry.set_param("adsr_sustain", 1.0)
        va_dry.set_param("reverb_bypass", 1.0)
        va_dry.set_param("chorus_bypass", 1.0)
        va_dry.set_param("delay_bypass", 1.0)

        path_dry = str(tmp_path / "dry.wav")
        Exporter.render_sample(sample, va_dry, path_dry, release_tail_s=0.1)
        dry_data, _ = sf.read(path_dry)

        # Wet export (heavy reverb)
        va_wet = VoiceAllocator()
        va_wet.set_param("adsr_attack_ms", 1.0)
        va_wet.set_param("adsr_sustain", 1.0)
        va_wet.set_param("reverb_room_size", 0.9)
        va_wet.set_param("reverb_wet", 0.8)
        va_wet.set_param("chorus_bypass", 1.0)
        va_wet.set_param("delay_bypass", 1.0)

        path_wet = str(tmp_path / "wet.wav")
        Exporter.render_sample(sample, va_wet, path_wet, release_tail_s=0.5)
        wet_data, _ = sf.read(path_wet)

        # Wet file should be longer (reverb tail) or differ in content
        assert len(wet_data) != len(dry_data) or not np.allclose(
            dry_data[:min(len(dry_data), len(wet_data))],
            wet_data[:min(len(dry_data), len(wet_data))],
            atol=0.01,
        )

    def test_resample_to_44100(self, tmp_path):
        va = VoiceAllocator()
        sample = _make_long_sample()
        path = str(tmp_path / "rendered_44100.wav")
        Exporter.render_sample(sample, va, path, sample_rate=44100)
        _, sr = sf.read(path)
        assert sr == 44100

    def test_16bit_export(self, tmp_path):
        va = VoiceAllocator()
        sample = _make_long_sample()
        path = str(tmp_path / "rendered_16.wav")
        Exporter.render_sample(sample, va, path, bit_depth=16)
        assert os.path.isfile(path)

    def test_empty_sample(self, tmp_path):
        va = VoiceAllocator()
        empty = Sample.from_buffer(np.array([], dtype=np.float32), 48000, "empty")
        path = str(tmp_path / "empty.wav")
        Exporter.render_sample(empty, va, path)
        assert os.path.isfile(path)

    def test_master_volume_applied(self, tmp_path):
        sample = _make_long_sample(duration_s=0.2)

        va = VoiceAllocator()
        va.set_param("adsr_attack_ms", 1.0)
        va.set_param("adsr_sustain", 1.0)

        path_full = str(tmp_path / "vol_full.wav")
        Exporter.render_sample(sample, va, path_full, master_volume=1.0, release_tail_s=0.05)
        full_data, _ = sf.read(path_full)

        path_half = str(tmp_path / "vol_half.wav")
        va2 = VoiceAllocator()
        va2.set_param("adsr_attack_ms", 1.0)
        va2.set_param("adsr_sustain", 1.0)
        Exporter.render_sample(sample, va2, path_half, master_volume=0.5, release_tail_s=0.05)
        half_data, _ = sf.read(path_half)

        # Half-volume peak should be roughly half of full-volume peak
        peak_full = np.max(np.abs(full_data))
        peak_half = np.max(np.abs(half_data))
        if peak_full > 0.01:
            assert peak_half < peak_full * 0.8

    def test_unsupported_sample_rate(self, tmp_path):
        va = VoiceAllocator()
        sample = _make_long_sample()
        path = str(tmp_path / "bad.wav")
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            Exporter.render_sample(sample, va, path, sample_rate=22050)

    def test_unsupported_bit_depth(self, tmp_path):
        va = VoiceAllocator()
        sample = _make_long_sample()
        path = str(tmp_path / "bad.wav")
        with pytest.raises(ValueError, match="Unsupported bit depth"):
            Exporter.render_sample(sample, va, path, bit_depth=32)

    def test_path_traversal_rejected(self, tmp_path):
        va = VoiceAllocator()
        sample = _make_long_sample()
        path = str(tmp_path / ".." / ".." / "evil.wav")
        with pytest.raises(ValueError, match="traversal"):
            Exporter.render_sample(sample, va, path)


class TestCloneAllocatorParams:
    def test_copies_adsr_params(self):
        source = VoiceAllocator()
        source.set_param("adsr_attack_ms", 50.0)
        source.set_param("adsr_sustain", 0.3)

        clone = _clone_allocator_params(source)
        assert clone.get_param("adsr_attack_ms") == pytest.approx(50.0)
        assert clone.get_param("adsr_sustain") == pytest.approx(0.3)

    def test_copies_filter_params(self):
        source = VoiceAllocator()
        source.set_param("filter_frequency", 2000.0)
        source.set_param("filter_q", 5.0)

        clone = _clone_allocator_params(source)
        assert clone.get_param("filter_frequency") == pytest.approx(2000.0)
        assert clone.get_param("filter_q") == pytest.approx(5.0)

    def test_copies_chorus_params(self):
        source = VoiceAllocator()
        source.set_param("chorus_rate_hz", 3.0)
        source.set_param("chorus_mix", 0.7)

        clone = _clone_allocator_params(source)
        assert clone.get_param("chorus_rate_hz") == pytest.approx(3.0)
        assert clone.get_param("chorus_mix") == pytest.approx(0.7)

    def test_copies_delay_params(self):
        source = VoiceAllocator()
        source.set_param("delay_time_ms", 500.0)
        source.set_param("delay_feedback", 0.6)

        clone = _clone_allocator_params(source)
        assert clone.get_param("delay_time_ms") == pytest.approx(500.0)
        assert clone.get_param("delay_feedback") == pytest.approx(0.6)

    def test_copies_reverb_params(self):
        source = VoiceAllocator()
        source.set_param("reverb_room_size", 0.9)
        source.set_param("reverb_wet", 0.8)

        clone = _clone_allocator_params(source)
        assert clone.get_param("reverb_room_size") == pytest.approx(0.9)
        assert clone.get_param("reverb_wet") == pytest.approx(0.8)

    def test_copies_limiter_params(self):
        source = VoiceAllocator()
        source.set_param("limiter_threshold_db", -6.0)

        clone = _clone_allocator_params(source)
        assert clone.get_param("limiter_threshold_db") == pytest.approx(-6.0)

    def test_copies_gain_param(self):
        source = VoiceAllocator()
        source.set_param("gain_gain_db", -12.0)

        clone = _clone_allocator_params(source)
        assert clone.get_param("gain_gain_db") == pytest.approx(-12.0)

    def test_copies_bypass_states(self):
        source = VoiceAllocator()
        source.set_param("chorus_bypass", 1.0)
        source.set_param("delay_bypass", 1.0)

        clone = _clone_allocator_params(source)
        assert clone._chorus.bypassed is True
        assert clone._delay.bypassed is True
        # Others should NOT be bypassed
        assert clone._reverb.bypassed is False
        assert clone._limiter.bypassed is False

    def test_clone_is_independent(self):
        source = VoiceAllocator()
        source.set_param("reverb_wet", 0.5)

        clone = _clone_allocator_params(source)
        clone.set_param("reverb_wet", 0.9)

        assert source.get_param("reverb_wet") == pytest.approx(0.5)
        assert clone.get_param("reverb_wet") == pytest.approx(0.9)
