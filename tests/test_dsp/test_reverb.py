"""Exhaustive tests for Freeverb."""
import numpy as np
import pytest

from vynth.dsp.reverb import Reverb, _CombFilter, _AllpassFilter


@pytest.fixture
def rev(sr):
    return Reverb(sr)


def _make_impulse(sr, n=None):
    if n is None:
        n = sr
    buf = np.zeros(n, dtype=np.float32)
    buf[0] = 1.0
    return buf


def _make_sine(sr, freq, duration_s=0.5):
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


class TestReverbInit:
    def test_default_params(self, sr):
        r = Reverb(sr)
        assert r.get_param("room_size") == pytest.approx(0.5)
        assert r.get_param("damping") == pytest.approx(0.5)
        assert r.get_param("wet") == pytest.approx(0.3)
        assert r.get_param("dry") == pytest.approx(0.7)
        assert r.get_param("width") == pytest.approx(1.0)

    def test_comb_filter_count(self, sr):
        r = Reverb(sr)
        assert len(r._combs_l) == 8
        assert len(r._combs_r) == 8

    def test_allpass_filter_count(self, sr):
        r = Reverb(sr)
        assert len(r._allpasses_l) == 4
        assert len(r._allpasses_r) == 4

    def test_custom_sample_rate(self):
        r = Reverb(44100)
        assert r.sample_rate == 44100


class TestReverbProcess:
    def test_impulse_response_has_tail(self, sr, rev):
        imp = _make_impulse(sr)
        rev.set_param("wet", 1.0)
        rev.set_param("room_size", 0.8)
        out = rev.process(imp)
        assert np.abs(out[sr // 2:]).max() > 0.001

    def test_output_always_stereo(self, sr, rev):
        mono = _make_sine(sr, 440, 0.1)
        out = rev.process(mono)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_stereo_input(self, sr, rev):
        mono = _make_sine(sr, 440, 0.1)
        stereo = np.column_stack([mono, mono])
        out = rev.process(stereo)
        assert out.shape == stereo.shape

    def test_output_dtype(self, sr, rev):
        mono = _make_sine(sr, 440, 0.1)
        out = rev.process(mono)
        assert out.dtype == np.float32

    def test_no_nan(self, sr, rev):
        sig = _make_sine(sr, 440)
        out = rev.process(sig)
        assert not np.any(np.isnan(out))

    def test_no_inf(self, sr, rev):
        sig = _make_sine(sr, 440)
        out = rev.process(sig)
        assert not np.any(np.isinf(out))


class TestReverbDryWet:
    def test_dry_only(self, sr):
        r = Reverb(sr)
        r.set_param("wet", 0.0)
        r.set_param("dry", 1.0)
        sig = _make_sine(sr, 440, 0.1)
        out = r.process(sig)
        # L channel should be close to input
        np.testing.assert_allclose(out[:, 0], sig, atol=0.01)

    def test_wet_only(self, sr):
        r = Reverb(sr)
        r.set_param("wet", 1.0)
        r.set_param("dry", 0.0)
        sig = _make_sine(sr, 440, 0.1)
        out = r.process(sig)
        # Should differ from dry
        diff = np.max(np.abs(out[:, 0] - sig))
        assert diff > 0.01

    def test_zero_wet_zero_dry_silence(self, sr):
        r = Reverb(sr)
        r.set_param("wet", 0.0)
        r.set_param("dry", 0.0)
        r.reset()
        sig = _make_sine(sr, 440, 0.1)
        out = r.process(sig)
        assert np.max(np.abs(out)) < 0.01


class TestReverbRoomSize:
    def test_small_room_shorter_tail(self, sr):
        r_small = Reverb(sr)
        r_small.set_param("room_size", 0.1)
        r_small.set_param("wet", 1.0)

        r_large = Reverb(sr)
        r_large.set_param("room_size", 0.9)
        r_large.set_param("wet", 1.0)

        imp = _make_impulse(sr)
        out_small = r_small.process(imp.copy())
        out_large = r_large.process(imp.copy())

        # Larger room should have more energy late in the tail
        energy_small = np.sum(out_small[sr // 2:] ** 2)
        energy_large = np.sum(out_large[sr // 2:] ** 2)
        assert energy_large > energy_small


class TestReverbDamping:
    def test_high_damping_less_brightness(self, sr):
        r_low = Reverb(sr)
        r_low.set_param("damping", 0.1)
        r_low.set_param("wet", 1.0)
        r_low.set_param("room_size", 0.8)

        r_high = Reverb(sr)
        r_high.set_param("damping", 0.9)
        r_high.set_param("wet", 1.0)
        r_high.set_param("room_size", 0.8)

        imp = _make_impulse(sr)
        out_low = r_low.process(imp.copy())
        out_high = r_high.process(imp.copy())

        # Both should produce a tail
        assert np.max(np.abs(out_low)) > 0.001
        assert np.max(np.abs(out_high)) > 0.001


class TestReverbWidth:
    def test_zero_width_mono_output(self, sr):
        r = Reverb(sr)
        r.set_param("width", 0.0)
        r.set_param("wet", 1.0)
        r.set_param("dry", 0.0)
        sig = _make_sine(sr, 440, 0.1)
        out = r.process(sig)
        # With width=0, L and R should be identical
        np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-5)

    def test_full_width_stereo(self, sr):
        r = Reverb(sr)
        r.set_param("width", 1.0)
        r.set_param("wet", 1.0)
        r.set_param("dry", 0.0)
        sig = _make_sine(sr, 440, 0.1)
        out = r.process(sig)
        # L and R should differ
        diff = np.max(np.abs(out[:, 0] - out[:, 1]))
        assert diff > 0.001


class TestReverbSilence:
    def test_silence_produces_silence(self, sr):
        r = Reverb(sr)
        r.reset()
        silence = np.zeros(1024, dtype=np.float32)
        out = r.process(silence)
        assert np.abs(out).max() < 0.001


class TestReverbReset:
    def test_reset_clears_tail(self, sr):
        r = Reverb(sr)
        r.set_param("wet", 1.0)
        r.set_param("room_size", 0.9)
        imp = _make_impulse(sr)
        r.process(imp)
        r.reset()
        silence = np.zeros(1024, dtype=np.float32)
        out = r.process(silence)
        assert np.max(np.abs(out)) < 0.001

    def test_reset_preserves_params(self, sr):
        r = Reverb(sr)
        r.set_param("room_size", 0.9)
        r.set_param("wet", 0.8)
        r.reset()
        assert r.get_param("room_size") == pytest.approx(0.9)
        assert r.get_param("wet") == pytest.approx(0.8)


class TestReverbBypass:
    def test_bypass_passthrough_mono(self, sr, sine_440):
        r = Reverb(sr)
        r.bypassed = True
        out = r.process_maybe_bypass(sine_440[:512])
        np.testing.assert_array_equal(out, sine_440[:512])

    def test_bypass_passthrough_stereo(self, sr, sine_440_stereo):
        r = Reverb(sr)
        r.bypassed = True
        block = sine_440_stereo[:512]
        out = r.process_maybe_bypass(block.copy())
        np.testing.assert_array_equal(out, block)


class TestCombFilter:
    def test_comb_creates_echo(self):
        c = _CombFilter(100)
        # Feed impulse
        out_val = c.process(1.0, 0.5, 0.0, 1.0)
        # First output should be 0 (buffer was empty)
        assert out_val == pytest.approx(0.0)

    def test_comb_reset(self):
        c = _CombFilter(100)
        c.process(1.0, 0.5, 0.0, 1.0)
        c.reset()
        assert np.all(c._buffer == 0.0)
        assert c._idx == 0


class TestAllpassFilter:
    def test_allpass_is_not_passthrough(self):
        ap = _AllpassFilter(50)
        vals = []
        for i in range(100):
            v = ap.process(1.0 if i == 0 else 0.0)
            vals.append(v)
        assert np.array(vals).max() > 0.0

    def test_allpass_reset(self):
        ap = _AllpassFilter(50)
        ap.process(1.0)
        ap.reset()
        assert np.all(ap._buffer == 0.0)
        assert ap._idx == 0


class TestReverbConsecutiveBlocks:
    def test_no_clicks_at_boundaries(self, sr):
        r = Reverb(sr)
        r.set_param("wet", 0.5)
        sig = _make_sine(sr, 440)
        bs = 512
        prev_end = None
        for i in range(0, len(sig), bs):
            block = sig[i: i + bs]
            if len(block) == 0:
                break
            out = r.process(block)
            if prev_end is not None:
                jump = np.max(np.abs(out[0] - prev_end))
                assert jump < 1.0, f"Click at block {i // bs}"
            prev_end = out[-1].copy()
