"""Exhaustive tests for VoiceAllocator (64-voice polyphony + master DSP)."""
import numpy as np
import pytest

from vynth.config import MAX_VOICES, SAMPLE_RATE
from vynth.engine.voice_allocator import VoiceAllocator
from vynth.engine.voice import PlaybackMode
from vynth.sampler.sample import Sample


def _make_sample(sr=48000, duration_s=1.0):
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    data = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    return Sample.from_buffer(data, sr, "test")


@pytest.fixture
def va():
    return VoiceAllocator()


@pytest.fixture
def va_with_sample(va):
    va.set_sample(_make_sample())
    return va


class TestVoiceAllocatorInit:
    def test_voice_count(self, va):
        assert len(va.voices) == MAX_VOICES

    def test_no_active_voices(self, va):
        assert va.active_voice_count == 0

    def test_no_sample(self, va):
        assert va._current_sample is None


class TestVoiceAllocatorNoteOn:
    def test_note_on_activates_voice(self, va_with_sample):
        va_with_sample.note_on(60, 100)
        assert va_with_sample.active_voice_count >= 1

    def test_multiple_notes(self, va_with_sample):
        for note in range(60, 72):
            va_with_sample.note_on(note, 100)
        assert va_with_sample.active_voice_count == 12

    def test_note_on_without_sample_ignored(self, va):
        va.note_on(60, 100)
        assert va.active_voice_count == 0


class TestVoiceAllocatorNoteOff:
    def test_note_off_triggers_release(self, va_with_sample):
        va_with_sample.note_on(60, 100)
        va_with_sample.note_off(60)
        # Voice is still active during release phase
        # but note_off was called
        # Process some frames to move toward idle
        for _ in range(100):
            va_with_sample.process(512)
        # Eventually should become inactive
        assert va_with_sample.active_voice_count == 0

    def test_all_notes_off(self, va_with_sample):
        for note in range(60, 72):
            va_with_sample.note_on(note, 100)
        va_with_sample.all_notes_off()
        # Process to complete release
        for _ in range(100):
            va_with_sample.process(512)
        assert va_with_sample.active_voice_count == 0


class TestVoiceAllocatorStealing:
    def test_steal_oldest_when_all_busy(self, va_with_sample):
        # Fill all 64 voices
        for i in range(MAX_VOICES):
            va_with_sample.note_on(i % 128, 100)
        # One more should steal
        va_with_sample.note_on(127, 100)
        assert va_with_sample.active_voice_count <= MAX_VOICES


class TestVoiceAllocatorProcess:
    def test_silent_without_notes(self, va):
        out = va.process(512)
        assert out.shape == (512, 2)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_produces_audio_with_note(self, va_with_sample):
        va_with_sample.set_param("adsr_attack_ms", 1.0)
        va_with_sample.set_param("adsr_sustain", 1.0)
        va_with_sample.note_on(60, 127)
        # Process several blocks to let ADSR ramp up
        for _ in range(5):
            out = va_with_sample.process(512)
        assert out.shape == (512, 2)
        assert np.max(np.abs(out)) > 0.001

    def test_output_dtype(self, va_with_sample):
        va_with_sample.note_on(60, 100)
        out = va_with_sample.process(512)
        assert out.dtype == np.float32

    def test_no_nan(self, va_with_sample):
        va_with_sample.note_on(60, 100)
        for _ in range(10):
            out = va_with_sample.process(512)
            assert not np.any(np.isnan(out))

    def test_multiple_notes_mix(self, va_with_sample):
        va_with_sample.set_param("adsr_attack_ms", 1.0)
        va_with_sample.set_param("adsr_sustain", 1.0)
        va_with_sample.note_on(60, 127)
        va_with_sample.note_on(64, 100)
        va_with_sample.note_on(67, 100)
        for _ in range(5):
            out = va_with_sample.process(512)
        assert np.max(np.abs(out)) > 0.001


class TestVoiceAllocatorParams:
    def test_set_adsr_param(self, va_with_sample):
        va_with_sample.set_param("adsr_attack_ms", 50.0)
        assert va_with_sample.get_param("adsr_attack_ms") == pytest.approx(50.0)

    def test_set_filter_param(self, va_with_sample):
        va_with_sample.set_param("filter_frequency", 2000.0)
        assert va_with_sample.get_param("filter_frequency") == pytest.approx(2000.0)

    def test_set_chorus_param(self, va_with_sample):
        va_with_sample.set_param("chorus_rate", 3.0)
        assert va_with_sample.get_param("chorus_rate") == pytest.approx(3.0)

    def test_set_delay_param(self, va_with_sample):
        va_with_sample.set_param("delay_time_ms", 500.0)
        assert va_with_sample.get_param("delay_time_ms") == pytest.approx(500.0)

    def test_set_reverb_param(self, va_with_sample):
        va_with_sample.set_param("reverb_room_size", 0.8)
        assert va_with_sample.get_param("reverb_room_size") == pytest.approx(0.8)

    def test_set_limiter_param(self, va_with_sample):
        va_with_sample.set_param("limiter_threshold_db", -6.0)
        assert va_with_sample.get_param("limiter_threshold_db") == pytest.approx(-6.0)

    def test_set_gain_param(self, va_with_sample):
        va_with_sample.set_param("gain_gain_db", 6.0)
        assert va_with_sample.get_param("gain_gain_db") == pytest.approx(6.0)

    def test_unknown_prefix(self, va):
        # Should not crash
        va.set_param("unknown_param", 1.0)
        assert va.get_param("unknown_param") == 0.0


class TestVoiceAllocatorBypass:
    def test_bypass_chorus(self, va):
        va.set_param("chorus_bypass", 1.0)
        assert va._chorus.bypassed is True

    def test_unbypass_chorus(self, va):
        va.set_param("chorus_bypass", 1.0)
        va.set_param("chorus_bypass", 0.0)
        assert va._chorus.bypassed is False

    def test_bypass_delay(self, va):
        va.set_param("delay_bypass", 1.0)
        assert va._delay.bypassed is True

    def test_bypass_reverb(self, va):
        va.set_param("reverb_bypass", 1.0)
        assert va._reverb.bypassed is True

    def test_bypass_limiter(self, va):
        va.set_param("limiter_bypass", 1.0)
        assert va._limiter.bypassed is True

    def test_bypass_filter(self, va):
        va.set_param("filter_bypass", 1.0)
        assert all(v._filter.bypassed for v in va.voices)

    def test_bypass_adsr(self, va):
        va.set_param("adsr_bypass", 1.0)
        assert all(v._adsr.bypassed for v in va.voices)


class TestVoiceAllocatorReset:
    def test_reset_clears_voices(self, va_with_sample):
        va_with_sample.note_on(60, 100)
        va_with_sample.process(512)
        va_with_sample.reset()
        assert va_with_sample.active_voice_count == 0

    def test_reset_clears_dsp(self, va_with_sample):
        va_with_sample.note_on(60, 100)
        va_with_sample.process(512)
        va_with_sample.reset()
        out = va_with_sample.process(512)
        assert np.max(np.abs(out)) < 0.01


class TestVoiceAllocatorPlaybackMode:
    def test_set_sampler_mode(self, va):
        va.set_playback_mode(PlaybackMode.SAMPLER)
        assert all(v.mode == PlaybackMode.SAMPLER for v in va.voices)

    def test_set_granular_mode(self, va):
        va.set_playback_mode(PlaybackMode.GRANULAR)
        assert all(v.mode == PlaybackMode.GRANULAR for v in va.voices)

    def test_set_slice_mode(self, va):
        va.set_playback_mode(PlaybackMode.SLICE)
        assert all(v.mode == PlaybackMode.SLICE for v in va.voices)


class TestVoiceAllocatorSliceConfig:
    def test_set_slice_config_propagates(self, va):
        va.set_slice_config(8, 48)
        assert all(v._slice_num_slices == 8 for v in va.voices)
        assert all(v._slice_start_note == 48 for v in va.voices)

    def test_slice_config_via_param(self, va):
        va.set_param("slice_num_slices", 32.0)
        assert all(v._slice_num_slices == 32 for v in va.voices)

    def test_slice_start_note_via_param(self, va):
        va.set_param("slice_start_note", 60.0)
        assert all(v._slice_start_note == 60 for v in va.voices)

    def test_slice_region_start_via_param(self, va):
        va.set_param("slice_region_start", 1234.0)
        assert all(v._slice_region_start == 1234 for v in va.voices)

    def test_slice_region_end_via_param(self, va):
        va.set_param("slice_region_end", 5678.0)
        assert all(v._slice_region_end == 5678 for v in va.voices)

    def test_slice_mode_produces_audio(self, va_with_sample):
        va_with_sample.set_playback_mode(PlaybackMode.SLICE)
        va_with_sample.set_slice_config(16, 36)
        va_with_sample.set_param("adsr_attack_ms", 1.0)
        va_with_sample.set_param("adsr_sustain", 1.0)
        va_with_sample.note_on(36, 127)  # first slice
        for _ in range(5):
            out = va_with_sample.process(512)
        assert np.max(np.abs(out)) > 0.001

    def test_slice_out_of_range_silent(self, va_with_sample):
        va_with_sample.set_playback_mode(PlaybackMode.SLICE)
        va_with_sample.set_slice_config(16, 36)
        va_with_sample.note_on(10, 127)  # below start_note
        out = va_with_sample.process(512)
        assert np.max(np.abs(out)) < 0.01

    def test_different_slices_produce_different_audio(self, va_with_sample):
        va_with_sample.set_playback_mode(PlaybackMode.SLICE)
        va_with_sample.set_slice_config(4, 36)
        va_with_sample.set_param("adsr_attack_ms", 1.0)
        va_with_sample.set_param("adsr_sustain", 1.0)

        # Play first slice
        va_with_sample.note_on(36, 127)
        blocks_a = []
        for _ in range(3):
            blocks_a.append(va_with_sample.process(512).copy())
        va_with_sample.all_notes_off()
        for _ in range(100):
            va_with_sample.process(512)

        # Play last slice
        va_with_sample.note_on(39, 127)
        blocks_b = []
        for _ in range(3):
            blocks_b.append(va_with_sample.process(512).copy())

        a = np.concatenate(blocks_a)
        b = np.concatenate(blocks_b)
        # The two slices should not be identical
        assert not np.allclose(a, b, atol=1e-6)
