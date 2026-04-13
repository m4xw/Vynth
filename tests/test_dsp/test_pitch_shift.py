"""Exhaustive tests for phase vocoder pitch shifter."""
import numpy as np
import pytest

from vynth.dsp.pitch_shift import PitchShifter


@pytest.fixture
def ps(sr):
    return PitchShifter(sr)


def _make_sine(sr, freq, duration_s=1.0):
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def _peak_frequency(signal, sr, min_freq=20):
    n = len(signal)
    fft = np.abs(np.fft.rfft(signal * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = freqs >= min_freq
    return float(freqs[mask][np.argmax(fft[mask])])


class TestPitchShifterInit:
    def test_default_shift(self, sr):
        p = PitchShifter(sr)
        assert p.shift_semitones == pytest.approx(0.0)

    def test_custom_fft_hop(self):
        p = PitchShifter(48000, fft_size=4096, hop_size=1024)
        assert p._fft_size == 4096
        assert p._hop_size == 1024

    def test_custom_shift(self):
        p = PitchShifter(48000, shift_semitones=7.0)
        assert p.shift_semitones == pytest.approx(7.0)

    def test_pitch_ratio_property(self):
        p = PitchShifter(48000, shift_semitones=12.0)
        assert p._pitch_ratio == pytest.approx(2.0, rel=1e-4)

    def test_pitch_ratio_negative(self):
        p = PitchShifter(48000, shift_semitones=-12.0)
        assert p._pitch_ratio == pytest.approx(0.5, rel=1e-4)


class TestPitchShifterZeroShift:
    def test_preserves_length(self, sr):
        p = PitchShifter(sr)
        sig = _make_sine(sr, 440)
        out = p.process(sig)
        assert len(out) == len(sig)

    def test_preserves_frequency(self, sr):
        p = PitchShifter(sr)
        sig = _make_sine(sr, 440)
        out = p.process(sig)
        if len(out) > 4096:
            freq = _peak_frequency(out[:8192], sr)
            assert abs(freq - 440) < 50


class TestPitchShifterOctaveUp:
    def test_doubles_frequency(self, sr):
        sig = _make_sine(sr, 440)
        p = PitchShifter(sr)
        p.set_param("shift_semitones", 12.0)
        out = p.process(sig)
        if len(out) > 4096:
            freq = _peak_frequency(out[:8192], sr)
            assert 800 < freq < 960, f"Peak at {freq}Hz, expected ~880Hz"


class TestPitchShifterOctaveDown:
    def test_halves_frequency(self, sr):
        sig = _make_sine(sr, 880)
        p = PitchShifter(sr)
        p.set_param("shift_semitones", -12.0)
        out = p.process(sig)
        if len(out) > 4096:
            freq = _peak_frequency(out[:8192], sr)
            assert 380 < freq < 500, f"Peak at {freq}Hz, expected ~440Hz"


class TestPitchShifterVariousIntervals:
    @pytest.mark.parametrize("semitones", [-7, -5, -3, 3, 5, 7])
    def test_shift_no_crash(self, sr, semitones):
        p = PitchShifter(sr)
        p.set_param("shift_semitones", float(semitones))
        sig = _make_sine(sr, 440, 0.5)
        out = p.process(sig)
        assert len(out) == len(sig)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    @pytest.mark.parametrize("semitones", [-24, -12, 0, 12, 24])
    def test_extreme_shifts(self, sr, semitones):
        p = PitchShifter(sr)
        p.set_param("shift_semitones", float(semitones))
        sig = _make_sine(sr, 440, 0.2)
        out = p.process(sig)
        assert not np.any(np.isnan(out))


class TestPitchShifterMono:
    def test_mono_output_1d(self, sr):
        p = PitchShifter(sr)
        p.set_param("shift_semitones", 5.0)
        sig = _make_sine(sr, 440, 0.1)
        out = p.process(sig)
        assert out.ndim == 1

    def test_mono_dtype(self, sr):
        p = PitchShifter(sr)
        sig = _make_sine(sr, 440, 0.1)
        out = p.process(sig)
        assert out.dtype == np.float32


class TestPitchShifterStereo:
    def test_stereo_output_shape(self, sr, sine_440_stereo):
        p = PitchShifter(sr)
        p.set_param("shift_semitones", 5.0)
        out = p.process(sine_440_stereo)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_stereo_channels_processed_independently(self, sr, sine_440_stereo):
        p = PitchShifter(sr)
        p.set_param("shift_semitones", 3.0)
        out = p.process(sine_440_stereo[:4096])
        # Both channels are processed; output should be non-zero on both
        assert np.max(np.abs(out[:, 0])) > 0.01
        assert np.max(np.abs(out[:, 1])) > 0.01


class TestPitchShifterEdgeCases:
    def test_empty_input(self, sr):
        p = PitchShifter(sr)
        empty = np.array([], dtype=np.float32)
        out = p.process(empty)
        assert out.size == 0

    def test_very_short_input(self, sr):
        p = PitchShifter(sr)
        short = np.ones(10, dtype=np.float32) * 0.5
        out = p.process(short)
        assert len(out) == 10

    def test_silence_produces_near_silence(self, sr):
        p = PitchShifter(sr)
        p.set_param("shift_semitones", 7.0)
        silence = np.zeros(4096, dtype=np.float32)
        out = p.process(silence)
        assert np.max(np.abs(out)) < 0.01

    def test_dc_offset_input(self, sr):
        p = PitchShifter(sr)
        dc = np.ones(4096, dtype=np.float32) * 0.5
        out = p.process(dc)
        assert not np.any(np.isnan(out))


class TestPitchShifterReset:
    def test_reset_no_crash(self, sr):
        p = PitchShifter(sr)
        p.process(_make_sine(sr, 440, 0.1))
        p.reset()
        out = p.process(np.zeros(512, dtype=np.float32))
        assert len(out) > 0

    def test_reset_clears_state(self, sr):
        p = PitchShifter(sr)
        p.process(_make_sine(sr, 440))
        p.reset()
        assert np.all(p._input_buffer == 0)
        assert np.all(p._output_accum == 0)

    def test_reset_preserves_params(self, sr):
        p = PitchShifter(sr)
        p.set_param("shift_semitones", 5.0)
        p.reset()
        assert p.shift_semitones == pytest.approx(5.0)


class TestPitchShifterBypass:
    def test_bypass_passthrough(self, sr):
        p = PitchShifter(sr)
        p.bypassed = True
        sig = _make_sine(sr, 440, 0.1)
        out = p.process_maybe_bypass(sig.copy())
        np.testing.assert_array_equal(out, sig)


class TestPitchShifterProperty:
    def test_shift_semitones_setter(self, sr):
        p = PitchShifter(sr)
        p.shift_semitones = 7.0
        assert p.shift_semitones == pytest.approx(7.0)

    def test_shift_semitones_negative(self, sr):
        p = PitchShifter(sr)
        p.shift_semitones = -5.0
        assert p.shift_semitones == pytest.approx(-5.0)


class TestPitchShifterConsecutiveBlocks:
    def test_streaming_no_clicks(self, sr):
        p = PitchShifter(sr)
        p.set_param("shift_semitones", 3.0)
        sig = _make_sine(sr, 440)
        block_size = 512
        outputs = []
        for i in range(0, len(sig), block_size):
            block = sig[i: i + block_size]
            if len(block) == 0:
                break
            out = p.process(block)
            outputs.append(out)
        full = np.concatenate(outputs)
        # Check for big jumps at block boundaries
        for i in range(len(outputs) - 1):
            if len(outputs[i]) > 0 and len(outputs[i + 1]) > 0:
                jump = abs(float(outputs[i + 1][0]) - float(outputs[i][-1]))
                assert jump < 2.0, f"Click at block boundary {i}: {jump}"

    def test_state_save_restore(self, sr):
        p = PitchShifter(sr)
        p.process(_make_sine(sr, 440, 0.1))
        state = p._save_state()
        assert "input_buffer" in state
        assert "last_phase" in state
        p._restore_state(state)
        out = p.process(_make_sine(sr, 440, 0.1))
        assert not np.any(np.isnan(out))
