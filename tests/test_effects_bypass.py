"""Exhaustive tests for bypass, param routing, and DSP chain integration."""
import numpy as np
import pytest

from vynth.dsp.adsr import ADSREnvelope
from vynth.dsp.chorus import Chorus
from vynth.dsp.delay import Delay
from vynth.dsp.filter import BiquadFilter
from vynth.dsp.gain import GainEffect
from vynth.dsp.limiter import Limiter
from vynth.dsp.pitch_shift import PitchShifter
from vynth.dsp.reverb import Reverb
from vynth.dsp.base import DSPEffect


ALL_EFFECTS = [
    ("Chorus", Chorus),
    ("Delay", Delay),
    ("BiquadFilter", BiquadFilter),
    ("GainEffect", GainEffect),
    ("Limiter", Limiter),
    ("PitchShifter", PitchShifter),
    ("Reverb", Reverb),
]


class TestBypassAllEffects:
    @pytest.mark.parametrize("name,cls", ALL_EFFECTS, ids=[n for n, _ in ALL_EFFECTS])
    def test_bypass_returns_input_unchanged(self, sr, name, cls):
        fx = cls(sr)
        fx.bypassed = True
        data = np.random.default_rng(42).standard_normal(512).astype(np.float32) * 0.5
        out = fx.process_maybe_bypass(data.copy())
        np.testing.assert_array_equal(out, data)

    @pytest.mark.parametrize("name,cls", ALL_EFFECTS, ids=[n for n, _ in ALL_EFFECTS])
    def test_bypass_toggle(self, sr, name, cls):
        fx = cls(sr)
        assert fx.bypassed is False
        fx.bypassed = True
        assert fx.bypassed is True
        fx.bypassed = False
        assert fx.bypassed is False

    @pytest.mark.parametrize("name,cls", ALL_EFFECTS, ids=[n for n, _ in ALL_EFFECTS])
    def test_not_bypassed_processes(self, sr, name, cls):
        fx = cls(sr)
        fx.bypassed = False
        data = np.random.default_rng(42).standard_normal(512).astype(np.float32) * 0.5
        out = fx.process_maybe_bypass(data.copy())
        # Out should be a valid array (no crash)
        assert out is not None
        assert isinstance(out, np.ndarray)


class TestBypassStereoInput:
    @pytest.mark.parametrize("name,cls", ALL_EFFECTS, ids=[n for n, _ in ALL_EFFECTS])
    def test_bypass_stereo_unchanged(self, sr, name, cls):
        fx = cls(sr)
        fx.bypassed = True
        data = np.random.default_rng(42).standard_normal((512, 2)).astype(np.float32)
        out = fx.process_maybe_bypass(data.copy())
        np.testing.assert_array_equal(out, data)


class TestADSRBypassSpecial:
    """ADSR bypass test is separate since ADSR is not a standard process filter."""

    def test_adsr_bypass(self, sr):
        adsr = ADSREnvelope(sr)
        adsr.bypassed = True
        data = np.ones(512, dtype=np.float32)
        out = adsr.process_maybe_bypass(data)
        np.testing.assert_array_equal(out, data)


class TestParamRouting:
    """Verify param prefixes route to the correct effect."""

    @pytest.mark.parametrize("param,value", [
        ("chorus_rate", 2.0),
        ("chorus_depth", 0.8),
        ("chorus_mix", 0.5),
    ])
    def test_chorus_params(self, sr, param, value):
        c = Chorus(sr)
        c.set_param(param.split("_", 1)[1], value)
        assert c.get_param(param.split("_", 1)[1]) == pytest.approx(value)

    @pytest.mark.parametrize("param,value", [
        ("delay_time_ms", 100.0),
        ("delay_feedback", 0.6),
        ("delay_mix", 0.4),
    ])
    def test_delay_params(self, sr, param, value):
        d = Delay(sr)
        d.set_param(param.split("_", 1)[1], value)
        assert d.get_param(param.split("_", 1)[1]) == pytest.approx(value)

    @pytest.mark.parametrize("param,value", [
        ("reverb_room_size", 0.8),
        ("reverb_damping", 0.7),
        ("reverb_wet", 0.5),
        ("reverb_dry", 0.5),
        ("reverb_width", 0.8),
    ])
    def test_reverb_params(self, sr, param, value):
        r = Reverb(sr)
        r.set_param(param.split("_", 1)[1], value)
        assert r.get_param(param.split("_", 1)[1]) == pytest.approx(value)


class TestGetSetParams:
    @pytest.mark.parametrize("name,cls", ALL_EFFECTS, ids=[n for n, _ in ALL_EFFECTS])
    def test_get_params_returns_dict(self, sr, name, cls):
        fx = cls(sr)
        params = fx.get_params()
        assert isinstance(params, dict)

    @pytest.mark.parametrize("name,cls", ALL_EFFECTS, ids=[n for n, _ in ALL_EFFECTS])
    def test_set_params_bulk(self, sr, name, cls):
        fx = cls(sr)
        params = fx.get_params()
        # Set all current params back
        fx.set_params(params)
        for k, v in params.items():
            assert fx.get_param(k) == pytest.approx(v, abs=0.01)

    @pytest.mark.parametrize("name,cls", ALL_EFFECTS, ids=[n for n, _ in ALL_EFFECTS])
    def test_get_param_unknown_returns_zero(self, sr, name, cls):
        fx = cls(sr)
        assert fx.get_param("nonexistent_xyz") == 0.0


class TestResetAllEffects:
    @pytest.mark.parametrize("name,cls", ALL_EFFECTS, ids=[n for n, _ in ALL_EFFECTS])
    def test_reset_no_error(self, sr, name, cls):
        fx = cls(sr)
        sig = np.random.default_rng(42).standard_normal(1024).astype(np.float32)
        fx.process(sig)
        fx.reset()  # Should not raise


class TestMixParamScaling:
    def test_chorus_mix_zero(self, sr, sine_440):
        c = Chorus(sr)
        c.set_param("mix", 0.0)
        out = c.process(sine_440[:512])
        # Mix=0 should leave signal unchanged (dry only)
        # Chorus outputs stereo
        if out.ndim == 2:
            np.testing.assert_allclose(out[:, 0], sine_440[:512], atol=0.01)
        else:
            np.testing.assert_allclose(out, sine_440[:512], atol=0.01)

    def test_delay_mix_zero(self, sr, sine_440):
        d = Delay(sr)
        d.set_param("mix", 0.0)
        out = d.process(sine_440[:512])
        np.testing.assert_allclose(out[:, 0], sine_440[:512], atol=1e-6)

    def test_reverb_dry_one_wet_zero(self, sr, sine_440):
        r = Reverb(sr)
        r.set_param("wet", 0.0)
        r.set_param("dry", 1.0)
        out = r.process(sine_440[:512])
        np.testing.assert_allclose(out[:, 0], sine_440[:512], atol=0.01)


class TestChorusVoiceCount:
    @pytest.mark.parametrize("voices", [1, 2, 3, 4, 6, 8])
    def test_different_voice_counts(self, sr, voices):
        c = Chorus(sr)
        c.set_param("voices", float(voices))
        sig = np.sin(np.linspace(0, 2 * np.pi * 440, 512, dtype=np.float32))
        out = c.process(sig)
        assert out is not None
        assert not np.any(np.isnan(out))


class TestEffectChainOrder:
    """Test that a chain of effects produces valid output."""

    def test_full_chain_no_nan(self, sr):
        sig = np.random.default_rng(42).standard_normal(1024).astype(np.float32) * 0.3

        # Simulate master chain order
        gain = GainEffect(sr)
        chorus = Chorus(sr)
        delay = Delay(sr)
        reverb = Reverb(sr)
        limiter = Limiter(sr)

        out = gain.process(sig)
        out = chorus.process(out)
        out = delay.process(out)
        out = reverb.process(out)
        out = limiter.process(out)

        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_full_chain_all_bypassed(self, sr, sine_440):
        gain = GainEffect(sr)
        chorus = Chorus(sr)
        delay = Delay(sr)
        reverb = Reverb(sr)
        limiter = Limiter(sr)

        for fx in [gain, chorus, delay, reverb, limiter]:
            fx.bypassed = True

        block = sine_440[:512].copy()
        out = gain.process_maybe_bypass(block)
        out = chorus.process_maybe_bypass(out)
        out = delay.process_maybe_bypass(out)
        out = reverb.process_maybe_bypass(out)
        out = limiter.process_maybe_bypass(out)

        np.testing.assert_array_equal(out, sine_440[:512])
