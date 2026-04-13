"""Exhaustive tests for audio utility functions."""
import numpy as np
import pytest

from vynth.utils.audio_utils import (
    mono_to_stereo,
    stereo_to_mono,
    normalize,
    db_to_linear,
    linear_to_db,
    note_to_freq,
    freq_to_note,
    crossfade,
    resample_linear,
)


class TestMonoToStereo:
    def test_mono_1d_to_stereo(self):
        mono = np.ones(100, dtype=np.float32)
        out = mono_to_stereo(mono)
        assert out.shape == (100, 2)
        np.testing.assert_array_equal(out[:, 0], mono)
        np.testing.assert_array_equal(out[:, 1], mono)

    def test_already_stereo_unchanged(self):
        stereo = np.ones((100, 2), dtype=np.float32) * 0.5
        out = mono_to_stereo(stereo)
        np.testing.assert_array_equal(out, stereo)

    def test_single_channel_2d(self):
        data = np.ones((100, 1), dtype=np.float32)
        out = mono_to_stereo(data)
        assert out.shape == (100, 2)


class TestStereoToMono:
    def test_stereo_to_mono(self):
        stereo = np.column_stack([
            np.ones(100, dtype=np.float32),
            np.ones(100, dtype=np.float32) * 0.5,
        ])
        mono = stereo_to_mono(stereo)
        assert mono.ndim == 1
        np.testing.assert_allclose(mono, 0.75, atol=1e-6)

    def test_already_mono(self):
        mono = np.ones(100, dtype=np.float32)
        out = stereo_to_mono(mono)
        np.testing.assert_array_equal(out, mono)


class TestNormalize:
    def test_normalize_to_peak(self):
        data = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        out = normalize(data, peak=1.0)
        assert np.max(np.abs(out)) == pytest.approx(1.0)

    def test_normalize_custom_peak(self):
        data = np.array([0.5, -0.5], dtype=np.float32)
        out = normalize(data, peak=0.5)
        assert np.max(np.abs(out)) == pytest.approx(0.5)

    def test_silence_unchanged(self):
        data = np.zeros(100, dtype=np.float32)
        out = normalize(data)
        np.testing.assert_array_equal(out, 0.0)

    def test_very_quiet(self):
        data = np.ones(100, dtype=np.float32) * 1e-12
        out = normalize(data)
        # Should not amplify near-zero signal (under threshold)
        np.testing.assert_array_equal(out, data)


class TestDbToLinear:
    def test_zero_db(self):
        assert db_to_linear(0.0) == pytest.approx(1.0)

    def test_minus_6_db(self):
        assert db_to_linear(-6.0) == pytest.approx(0.5012, rel=0.01)

    def test_plus_6_db(self):
        assert db_to_linear(6.0) == pytest.approx(1.9953, rel=0.01)

    def test_minus_20_db(self):
        assert db_to_linear(-20.0) == pytest.approx(0.1, rel=0.01)


class TestLinearToDb:
    def test_unity(self):
        assert linear_to_db(1.0) == pytest.approx(0.0, abs=0.01)

    def test_half(self):
        assert linear_to_db(0.5) == pytest.approx(-6.02, abs=0.1)

    def test_zero(self):
        assert linear_to_db(0.0) == pytest.approx(-120.0)

    def test_very_small(self):
        assert linear_to_db(1e-12) == pytest.approx(-120.0)


class TestNoteToFreq:
    def test_a4(self):
        assert note_to_freq(69) == pytest.approx(440.0)

    def test_middle_c(self):
        assert note_to_freq(60) == pytest.approx(261.63, rel=0.01)

    def test_a3(self):
        assert note_to_freq(57) == pytest.approx(220.0, rel=0.01)

    def test_custom_a4(self):
        assert note_to_freq(69, a4=432.0) == pytest.approx(432.0)

    @pytest.mark.parametrize("note", [0, 21, 60, 69, 108, 127])
    def test_various_notes(self, note):
        freq = note_to_freq(note)
        assert freq > 0


class TestFreqToNote:
    def test_a4(self):
        assert freq_to_note(440.0) == pytest.approx(69.0)

    def test_middle_c(self):
        assert freq_to_note(261.63) == pytest.approx(60.0, abs=0.1)

    def test_zero_freq(self):
        assert freq_to_note(0.0) == 0.0

    def test_negative_freq(self):
        assert freq_to_note(-100.0) == 0.0

    def test_roundtrip(self):
        for note in range(21, 108):
            freq = note_to_freq(note)
            result = freq_to_note(freq)
            assert result == pytest.approx(note, abs=0.01)


class TestCrossfade:
    def test_basic_crossfade(self):
        a = np.ones(100, dtype=np.float32) * 0.5
        b = np.ones(100, dtype=np.float32) * 1.0
        out = crossfade(a, b, 20)
        assert len(out) == 180  # 100 + 100 - 20

    def test_zero_length(self):
        a = np.ones(50, dtype=np.float32)
        b = np.ones(50, dtype=np.float32)
        out = crossfade(a, b, 0)
        assert len(out) == 100

    def test_crossfade_values(self):
        a = np.ones(50, dtype=np.float32)
        b = np.zeros(50, dtype=np.float32)
        out = crossfade(a, b, 10)
        # In the crossfade region, should blend from 1.0 to 0.0
        xfade_region = out[40:50]
        assert xfade_region[0] > xfade_region[-1]

    def test_crossfade_longer_than_signal(self):
        a = np.ones(10, dtype=np.float32)
        b = np.ones(10, dtype=np.float32)
        out = crossfade(a, b, 100)  # length clamped to min(10, 10)
        assert len(out) == 10


class TestResampleLinear:
    def test_ratio_1_copy(self):
        data = np.ones(100, dtype=np.float32)
        out = resample_linear(data, 1.0)
        np.testing.assert_array_equal(out, data)

    def test_upsample(self):
        data = np.ones(100, dtype=np.float32)
        out = resample_linear(data, 0.5)  # ratio < 1 = more samples
        assert len(out) > 100

    def test_downsample(self):
        data = np.ones(100, dtype=np.float32)
        out = resample_linear(data, 2.0)  # ratio > 1 = fewer samples
        assert len(out) < 100

    def test_stereo(self):
        data = np.ones((100, 2), dtype=np.float32) * 0.5
        out = resample_linear(data, 2.0)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_very_high_ratio(self):
        data = np.ones(100, dtype=np.float32)
        out = resample_linear(data, 100.0)
        assert len(out) >= 1
