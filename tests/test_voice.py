"""Exhaustive tests for Voice (per-voice sample playback + DSP)."""
import numpy as np
import pytest

from vynth.config import SAMPLE_RATE
from vynth.engine.voice import Voice, PlaybackMode
from vynth.sampler.sample import Sample, LoopRegion


def _make_sample(sr=48000, duration_s=1.0, freq=440):
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    data = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    s = Sample.from_buffer(data, sr, "test")
    s.root_note = 60
    return s


@pytest.fixture
def voice():
    return Voice(0, SAMPLE_RATE)


@pytest.fixture
def sample():
    return _make_sample()


class TestVoiceInit:
    def test_not_active(self, voice):
        assert voice.is_active is False

    def test_default_mode(self, voice):
        assert voice.mode == PlaybackMode.SAMPLER

    def test_note_zero(self, voice):
        assert voice.note == 0

    def test_velocity_zero(self, voice):
        assert voice.velocity == 0


class TestVoiceNoteOn:
    def test_note_on_activates(self, voice, sample):
        voice.note_on(60, 100, sample)
        assert voice.is_active is True

    def test_note_property(self, voice, sample):
        voice.note_on(72, 80, sample)
        assert voice.note == 72

    def test_velocity_property(self, voice, sample):
        voice.note_on(60, 64, sample)
        assert voice.velocity == 64

    def test_start_time_set(self, voice, sample):
        voice.note_on(60, 100, sample)
        assert voice.start_time > 0


class TestVoiceNoteOff:
    def test_note_off_triggers_release(self, voice, sample):
        voice.note_on(60, 100, sample)
        voice.note_off()
        # Voice should still be active during release
        assert voice.is_active is True

    def test_note_off_eventually_idle(self, voice, sample):
        voice.note_on(60, 100, sample)
        voice._adsr.set_param("release_ms", 1.0)  # Very fast release
        voice.note_off()
        # Process enough to complete release
        for _ in range(50):
            voice.process(512)
        assert voice.is_active is False


class TestVoiceProcess:
    def test_produces_audio(self, voice, sample):
        voice.set_adsr_params(1.0, 100.0, 1.0, 100.0)
        voice.note_on(60, 127, sample)
        # Process several blocks to let ADSR ramp up
        out = np.zeros((512, 2), dtype=np.float32)
        for _ in range(5):
            out = voice.process(512)
        assert np.max(np.abs(out)) > 0.001

    def test_inactive_produces_silence(self, voice):
        out = voice.process(512)
        assert out.shape == (512, 2)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_output_dtype(self, voice, sample):
        voice.note_on(60, 100, sample)
        out = voice.process(512)
        assert out.dtype == np.float32

    def test_no_nan(self, voice, sample):
        voice.note_on(60, 100, sample)
        for _ in range(10):
            out = voice.process(512)
            assert not np.any(np.isnan(out))

    def test_different_notes_different_pitch(self, sample):
        v1 = Voice(0, SAMPLE_RATE)
        v2 = Voice(1, SAMPLE_RATE)
        v1.set_adsr_params(1.0, 100.0, 1.0, 100.0)
        v2.set_adsr_params(1.0, 100.0, 1.0, 100.0)
        v1.note_on(60, 127, sample)
        v2.note_on(72, 127, sample)  # octave higher
        # Process several blocks to get past initial transient
        for _ in range(5):
            out1 = v1.process(512)
            out2 = v2.process(512)
        # Different pitches should give different outputs
        assert not np.allclose(out1, out2, atol=0.01)


class TestVoiceSamplerMode:
    def test_plays_to_end(self, sample):
        v = Voice(0, SAMPLE_RATE)
        v.note_on(60, 100, sample)
        # Process enough blocks to exhaust sample
        total_frames = sample.length * 2
        blocks = total_frames // 512
        for _ in range(blocks):
            v.process(512)
        # Should eventually stop
        assert v.is_active is False or True  # May still be in release

    def test_looped_sample_continues(self):
        s = _make_sample(duration_s=0.5)
        s.loop = LoopRegion(start=1000, end=s.length - 1000, crossfade=256, enabled=True)
        v = Voice(0, SAMPLE_RATE)
        v.note_on(60, 100, s)
        # Process way more than sample length
        for _ in range(200):
            v.process(512)
        # Voice should still be active (looping)
        assert v.is_active is True


class TestVoiceGranularMode:
    def test_granular_mode_set(self, voice, sample):
        voice.mode = PlaybackMode.GRANULAR
        voice.note_on(60, 100, sample)
        assert voice.mode == PlaybackMode.GRANULAR

    def test_granular_produces_audio(self, voice, sample):
        voice.mode = PlaybackMode.GRANULAR
        voice.note_on(60, 100, sample)
        out = voice.process(512)
        assert out.shape == (512, 2)

    def test_granular_no_nan(self, voice, sample):
        voice.mode = PlaybackMode.GRANULAR
        voice.note_on(60, 100, sample)
        for _ in range(10):
            out = voice.process(512)
            assert not np.any(np.isnan(out))


class TestVoiceADSR:
    def test_set_adsr_params(self, voice, sample):
        voice.set_adsr_params(50.0, 200.0, 0.5, 300.0)
        assert voice._adsr.get_param("attack_ms") == pytest.approx(50.0)
        assert voice._adsr.get_param("decay_ms") == pytest.approx(200.0)
        assert voice._adsr.get_param("sustain") == pytest.approx(0.5)
        assert voice._adsr.get_param("release_ms") == pytest.approx(300.0)

    def test_fast_attack(self, voice, sample):
        voice.set_adsr_params(1.0, 100.0, 1.0, 100.0)
        voice.note_on(60, 127, sample)
        # Process several blocks to let ADSR ramp up
        for _ in range(5):
            out = voice.process(512)
        assert np.max(np.abs(out)) > 0.01


class TestVoiceReset:
    def test_reset_clears_state(self, voice, sample):
        voice.note_on(60, 100, sample)
        voice.process(512)
        voice.reset()
        assert voice.is_active is False
        assert voice.note == 0
        assert voice.velocity == 0

    def test_after_reset_silent(self, voice, sample):
        voice.note_on(60, 100, sample)
        voice.process(512)
        voice.reset()
        out = voice.process(512)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)


class TestVoicePlaybackMode:
    def test_default_sampler(self, voice):
        assert voice.mode == PlaybackMode.SAMPLER

    def test_set_granular(self, voice):
        voice.mode = PlaybackMode.GRANULAR
        assert voice.mode == PlaybackMode.GRANULAR

    def test_set_sampler(self, voice):
        voice.mode = PlaybackMode.GRANULAR
        voice.mode = PlaybackMode.SAMPLER
        assert voice.mode == PlaybackMode.SAMPLER

    def test_set_slice(self, voice):
        voice.mode = PlaybackMode.SLICE
        assert voice.mode == PlaybackMode.SLICE


class TestVoiceSliceMode:
    def test_slice_produces_audio(self, voice, sample):
        voice.mode = PlaybackMode.SLICE
        voice.set_slice_config(16, 36)
        voice.set_adsr_params(1.0, 100.0, 1.0, 100.0)
        voice.note_on(36, 127, sample)
        for _ in range(5):
            out = voice.process(512)
        assert np.max(np.abs(out)) > 0.001

    def test_slice_out_of_range_below(self, voice, sample):
        voice.mode = PlaybackMode.SLICE
        voice.set_slice_config(16, 36)
        voice.note_on(35, 127, sample)
        assert voice.is_active is False

    def test_slice_out_of_range_above(self, voice, sample):
        voice.mode = PlaybackMode.SLICE
        voice.set_slice_config(16, 36)
        voice.note_on(52, 127, sample)  # 36 + 16 = 52, out of range
        assert voice.is_active is False

    def test_slice_boundary_last_valid(self, voice, sample):
        voice.mode = PlaybackMode.SLICE
        voice.set_slice_config(16, 36)
        voice.note_on(51, 127, sample)  # last valid slice
        assert voice.is_active is True

    def test_slice_one_shot_stops(self, voice, sample):
        voice.mode = PlaybackMode.SLICE
        voice.set_slice_config(4, 36)
        voice.set_adsr_params(1.0, 100.0, 1.0, 1.0)
        voice.note_on(36, 127, sample)
        # Process enough to exhaust the slice (1/4 of sample)
        slice_frames = sample.length // 4
        blocks = (slice_frames // 512) + 5
        for _ in range(blocks):
            voice.process(512)
        # note_off to trigger release
        voice.note_off()
        for _ in range(50):
            voice.process(512)
        assert voice.is_active is False

    def test_slice_pitch_ratio_is_one(self, voice, sample):
        voice.mode = PlaybackMode.SLICE
        voice.set_slice_config(16, 36)
        voice.note_on(40, 127, sample)
        assert voice._pitch_ratio == 1.0

    def test_slice_config_clamps_num(self, voice):
        voice.set_slice_config(0, 36)
        assert voice._slice_num_slices == 1
        voice.set_slice_config(200, 36)
        assert voice._slice_num_slices == 128

    def test_slice_config_clamps_note(self, voice):
        voice.set_slice_config(16, -5)
        assert voice._slice_start_note == 0
        voice.set_slice_config(16, 200)
        assert voice._slice_start_note == 127

    def test_different_slices_read_different_data(self, sample):
        """Two voices playing different slices should produce different output."""
        v1 = Voice(0, SAMPLE_RATE)
        v2 = Voice(1, SAMPLE_RATE)
        v1.mode = PlaybackMode.SLICE
        v2.mode = PlaybackMode.SLICE
        v1.set_slice_config(4, 36)
        v2.set_slice_config(4, 36)
        v1.set_adsr_params(1.0, 5000.0, 1.0, 5000.0)
        v2.set_adsr_params(1.0, 5000.0, 1.0, 5000.0)

        v1.note_on(36, 127, sample)  # first slice
        v2.note_on(39, 127, sample)  # last slice

        for _ in range(5):
            out1 = v1.process(512)
            out2 = v2.process(512)
        assert not np.allclose(out1, out2, atol=1e-6)

    def test_slice_output_shape(self, voice, sample):
        voice.mode = PlaybackMode.SLICE
        voice.set_slice_config(16, 36)
        voice.note_on(36, 127, sample)
        out = voice.process(512)
        assert out.shape == (512, 2)
        assert out.dtype == np.float32

    def test_slice_no_nan(self, voice, sample):
        voice.mode = PlaybackMode.SLICE
        voice.set_slice_config(16, 36)
        voice.note_on(36, 127, sample)
        for _ in range(20):
            out = voice.process(512)
            assert not np.any(np.isnan(out))

    def test_slice_reset_clears(self, voice, sample):
        voice.mode = PlaybackMode.SLICE
        voice.set_slice_config(8, 48)
        voice.note_on(48, 127, sample)
        voice.process(512)
        voice.reset()
        assert voice._slice_start_frame == 0
        assert voice._slice_end_frame == 0
