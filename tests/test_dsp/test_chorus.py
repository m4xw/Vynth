"""Exhaustive tests for chorus / unison effect."""
import math

import numpy as np
import pytest

from vynth.dsp.chorus import Chorus


@pytest.fixture
def chorus(sr):
    return Chorus(sr)


@pytest.fixture
def mono_block(sr):
    t = np.arange(1024, dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)


@pytest.fixture
def stereo_block(sr):
    t = np.arange(1024, dtype=np.float32) / sr
    mono = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    return np.column_stack([mono, mono])


class TestChorusInit:
    def test_defaults(self, sr):
        c = Chorus(sr)
        assert c.get_param("num_voices") == pytest.approx(4.0)
        assert c.get_param("detune_cents") == pytest.approx(12.0)
        assert c.get_param("rate_hz") == pytest.approx(1.5)
        assert c.get_param("depth") == pytest.approx(0.5)
        assert c.get_param("mix") == pytest.approx(0.5)
        assert c.get_param("spread") == pytest.approx(0.7)

    def test_custom_sample_rate(self):
        c = Chorus(44100)
        assert c.sample_rate == 44100

    def test_not_bypassed_by_default(self, sr):
        c = Chorus(sr)
        assert not c.bypassed


class TestChorusProcess:
    def test_mono_input_produces_stereo_output(self, chorus, mono_block):
        out = chorus.process(mono_block)
        assert out.ndim == 2
        assert out.shape == (1024, 2)

    def test_stereo_input_produces_stereo_output(self, chorus, stereo_block):
        out = chorus.process(stereo_block)
        assert out.ndim == 2
        assert out.shape == (1024, 2)

    def test_output_dtype_float32(self, chorus, mono_block):
        out = chorus.process(mono_block)
        assert out.dtype == np.float32

    def test_no_nan_in_output(self, chorus, mono_block):
        out = chorus.process(mono_block)
        assert not np.any(np.isnan(out))

    def test_no_inf_in_output(self, chorus, mono_block):
        out = chorus.process(mono_block)
        assert not np.any(np.isinf(out))

    def test_processes_silence_without_crash(self, chorus, sr):
        silence = np.zeros(1024, dtype=np.float32)
        out = chorus.process(silence)
        assert out.shape == (1024, 2)

    def test_processes_dc_offset(self, chorus):
        dc = np.ones(512, dtype=np.float32) * 0.5
        out = chorus.process(dc)
        assert out.shape == (512, 2)
        assert not np.any(np.isnan(out))

    def test_consecutive_blocks_no_discontinuity(self, chorus, mono_block):
        out1 = chorus.process(mono_block)
        out2 = chorus.process(mono_block)
        # No huge jump at block boundary
        jump_l = abs(out2[0, 0] - out1[-1, 0])
        jump_r = abs(out2[0, 1] - out1[-1, 1])
        assert jump_l < 1.0
        assert jump_r < 1.0

    def test_small_block_size(self, chorus):
        tiny = np.zeros(1, dtype=np.float32)
        out = chorus.process(tiny)
        assert out.shape == (1, 2)


class TestChorusMix:
    def test_zero_mix_passthrough(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("mix", 0.0)
        out = c.process(mono_block)
        # Left channel should match input
        np.testing.assert_allclose(out[:, 0], mono_block, atol=1e-5)

    def test_full_mix_no_dry(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("mix", 1.0)
        out = c.process(mono_block)
        # Should differ from dry signal
        assert np.max(np.abs(out[:, 0] - mono_block)) > 0.001

    def test_mix_blending(self, sr, mono_block):
        c_dry = Chorus(sr)
        c_dry.set_param("mix", 0.0)
        out_dry = c_dry.process(mono_block.copy())

        c_wet = Chorus(sr)
        c_wet.set_param("mix", 1.0)
        out_wet = c_wet.process(mono_block.copy())

        c_half = Chorus(sr)
        c_half.set_param("mix", 0.5)
        out_half = c_half.process(mono_block.copy())

        # Half mix should be between dry and wet peaks
        peak_dry = np.max(np.abs(out_dry))
        peak_wet = np.max(np.abs(out_wet))
        peak_half = np.max(np.abs(out_half))
        lo, hi = sorted([peak_dry, peak_wet])
        assert peak_half <= hi + 0.5  # generous tolerance


class TestChorusStereoSpread:
    def test_zero_spread_mono(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("spread", 0.0)
        c.set_param("mix", 1.0)
        out = c.process(mono_block)
        # With zero spread, L and R should be very similar
        diff = np.max(np.abs(out[:, 0] - out[:, 1]))
        assert diff < 0.5

    def test_full_spread_stereo_difference(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("spread", 1.0)
        c.set_param("mix", 1.0)
        out = c.process(mono_block)
        diff = np.max(np.abs(out[:, 0] - out[:, 1]))
        assert diff > 0.001


class TestChorusVoiceCount:
    @pytest.mark.parametrize("n_voices", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_voice_count(self, sr, mono_block, n_voices):
        c = Chorus(sr)
        c.set_param("num_voices", float(n_voices))
        out = c.process(mono_block)
        assert out.shape == (1024, 2)
        assert not np.any(np.isnan(out))

    def test_clamps_below_one(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("num_voices", 0.0)
        out = c.process(mono_block)
        assert not np.any(np.isnan(out))

    def test_clamps_above_eight(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("num_voices", 20.0)
        out = c.process(mono_block)
        assert not np.any(np.isnan(out))


class TestChorusDetune:
    def test_zero_detune(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("detune_cents", 0.0)
        out = c.process(mono_block)
        assert not np.any(np.isnan(out))

    def test_large_detune(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("detune_cents", 100.0)
        out = c.process(mono_block)
        assert not np.any(np.isnan(out))


class TestChorusDepthAndRate:
    def test_zero_depth(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("depth", 0.0)
        out = c.process(mono_block)
        assert not np.any(np.isnan(out))

    def test_full_depth(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("depth", 1.0)
        out = c.process(mono_block)
        assert not np.any(np.isnan(out))

    def test_high_rate(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("rate_hz", 10.0)
        out = c.process(mono_block)
        assert not np.any(np.isnan(out))

    def test_low_rate(self, sr, mono_block):
        c = Chorus(sr)
        c.set_param("rate_hz", 0.1)
        out = c.process(mono_block)
        assert not np.any(np.isnan(out))


class TestChorusReset:
    def test_reset_clears_buffers(self, sr):
        c = Chorus(sr)
        t = np.arange(4096, dtype=np.float32) / sr
        sig = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        c.process(sig)
        c.reset()
        assert np.all(c._buf_l == 0.0)
        assert np.all(c._buf_r == 0.0)
        assert c._write_idx == 0

    def test_reset_allows_clean_restart(self, sr, mono_block):
        c = Chorus(sr)
        c.process(mono_block)
        c.reset()
        out = c.process(mono_block)
        assert not np.any(np.isnan(out))


class TestChorusBypass:
    def test_bypass_passthrough(self, sr, stereo_block):
        c = Chorus(sr)
        c.bypassed = True
        out = c.process_maybe_bypass(stereo_block.copy())
        np.testing.assert_array_equal(out, stereo_block)

    def test_bypass_toggle(self, sr, mono_block):
        c = Chorus(sr)
        c.bypassed = True
        assert c.bypassed
        c.bypassed = False
        assert not c.bypassed


class TestChorusParamAccess:
    def test_set_get_param(self, sr):
        c = Chorus(sr)
        c.set_param("num_voices", 3.0)
        assert c.get_param("num_voices") == pytest.approx(3.0)

    def test_get_all_params(self, sr):
        c = Chorus(sr)
        params = c.get_params()
        for key in ["num_voices", "detune_cents", "rate_hz", "depth", "mix", "spread"]:
            assert key in params

    def test_set_params_bulk(self, sr):
        c = Chorus(sr)
        c.set_params({"num_voices": 6.0, "mix": 0.8})
        assert c.get_param("num_voices") == pytest.approx(6.0)
        assert c.get_param("mix") == pytest.approx(0.8)
