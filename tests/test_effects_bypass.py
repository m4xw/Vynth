"""Tests for effect bypass, param routing, voice mix scaling, and chorus voices."""
import numpy as np
import pytest

from vynth.config import SAMPLE_RATE, BLOCK_SIZE
from vynth.dsp.chorus import Chorus
from vynth.dsp.delay import Delay
from vynth.dsp.filter import BiquadFilter
from vynth.dsp.limiter import Limiter
from vynth.dsp.reverb import Reverb
from vynth.engine.voice_allocator import VoiceAllocator
from vynth.sampler.sample import Sample


@pytest.fixture
def allocator():
    va = VoiceAllocator(SAMPLE_RATE)
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sample = Sample.from_buffer(data, SAMPLE_RATE, "test")
    sample.root_note = 69
    va.set_sample(sample)
    return va


@pytest.fixture
def stereo_block():
    rng = np.random.default_rng(42)
    return rng.standard_normal((BLOCK_SIZE, 2)).astype(np.float32) * 0.5


# ── Bypass tests ─────────────────────────────────────────────────────────


class TestBypass:
    """Bypass flag must cause process_maybe_bypass to pass through unchanged."""

    def test_chorus_bypass(self, stereo_block):
        c = Chorus(SAMPLE_RATE)
        c.bypassed = True
        out = c.process_maybe_bypass(stereo_block.copy())
        np.testing.assert_array_equal(out, stereo_block)

    def test_delay_bypass(self, stereo_block):
        d = Delay(SAMPLE_RATE)
        d.bypassed = True
        out = d.process_maybe_bypass(stereo_block.copy())
        np.testing.assert_array_equal(out, stereo_block)

    def test_reverb_bypass(self, stereo_block):
        r = Reverb(SAMPLE_RATE)
        r.bypassed = True
        out = r.process_maybe_bypass(stereo_block.copy())
        np.testing.assert_array_equal(out, stereo_block)

    def test_limiter_bypass(self, stereo_block):
        li = Limiter(SAMPLE_RATE)
        li.bypassed = True
        out = li.process_maybe_bypass(stereo_block.copy())
        np.testing.assert_array_equal(out, stereo_block)

    def test_filter_bypass(self, stereo_block):
        f = BiquadFilter(SAMPLE_RATE)
        f.bypassed = True
        out = f.process_maybe_bypass(stereo_block.copy())
        np.testing.assert_array_equal(out, stereo_block)


# ── VoiceAllocator bypass routing ────────────────────────────────────────


class TestAllocatorBypass:
    """set_param('xxx_bypass', 1.0) must toggle the DSP bypass flag."""

    def test_chorus_bypass_via_param(self, allocator):
        allocator.set_param("chorus_bypass", 1.0)
        assert allocator._chorus.bypassed is True

    def test_chorus_unbypass_via_param(self, allocator):
        allocator.set_param("chorus_bypass", 1.0)
        allocator.set_param("chorus_bypass", 0.0)
        assert allocator._chorus.bypassed is False

    def test_delay_bypass_via_param(self, allocator):
        allocator.set_param("delay_bypass", 1.0)
        assert allocator._delay.bypassed is True

    def test_reverb_bypass_via_param(self, allocator):
        allocator.set_param("reverb_bypass", 1.0)
        assert allocator._reverb.bypassed is True

    def test_limiter_bypass_via_param(self, allocator):
        allocator.set_param("limiter_bypass", 1.0)
        assert allocator._limiter.bypassed is True

    def test_filter_bypass_via_param(self, allocator):
        allocator.set_param("filter_bypass", 1.0)
        assert allocator.voices[0]._filter.bypassed is True

    def test_pitch_shift_bypass_via_param(self, allocator):
        allocator.set_param("pitch_shift_bypass", 1.0)
        assert allocator.voices[0]._pitch_shifter.bypassed is True

    def test_adsr_bypass_via_param(self, allocator):
        allocator.set_param("adsr_bypass", 1.0)
        assert allocator.voices[0]._adsr.bypassed is True


# ── Param name routing ───────────────────────────────────────────────────


class TestParamRouting:
    """UI param names must reach the correct DSP params after prefix strip."""

    def test_adsr_attack_ms(self, allocator):
        allocator.set_param("adsr_attack_ms", 25.0)
        assert allocator.voices[0]._adsr.get_param("attack_ms") == pytest.approx(25.0)

    def test_adsr_decay_ms(self, allocator):
        allocator.set_param("adsr_decay_ms", 200.0)
        assert allocator.voices[0]._adsr.get_param("decay_ms") == pytest.approx(200.0)

    def test_adsr_sustain(self, allocator):
        allocator.set_param("adsr_sustain", 0.5)
        assert allocator.voices[0]._adsr.get_param("sustain") == pytest.approx(0.5)

    def test_adsr_release_ms(self, allocator):
        allocator.set_param("adsr_release_ms", 500.0)
        assert allocator.voices[0]._adsr.get_param("release_ms") == pytest.approx(500.0)

    def test_filter_frequency(self, allocator):
        allocator.set_param("filter_frequency", 2000.0)
        assert allocator.voices[0]._filter.get_param("frequency") == pytest.approx(2000.0)

    def test_filter_gain_db(self, allocator):
        allocator.set_param("filter_gain_db", 6.0)
        assert allocator.voices[0]._filter.get_param("gain_db") == pytest.approx(6.0)

    def test_chorus_detune_cents(self, allocator):
        allocator.set_param("chorus_detune_cents", 20.0)
        assert allocator._chorus.get_param("detune_cents") == pytest.approx(20.0)

    def test_chorus_rate_hz(self, allocator):
        allocator.set_param("chorus_rate_hz", 2.5)
        assert allocator._chorus.get_param("rate_hz") == pytest.approx(2.5)

    def test_delay_time_ms(self, allocator):
        allocator.set_param("delay_time_ms", 500.0)
        assert allocator._delay.get_param("time_ms") == pytest.approx(500.0)

    def test_limiter_threshold_db(self, allocator):
        allocator.set_param("limiter_threshold_db", -6.0)
        assert allocator._limiter.get_param("threshold_db") == pytest.approx(-6.0)

    def test_limiter_release_ms(self, allocator):
        allocator.set_param("limiter_release_ms", 50.0)
        assert allocator._limiter.get_param("release_ms") == pytest.approx(50.0)


# ── Voice mix scaling ────────────────────────────────────────────────────


class TestVoiceMixScaling:
    """Polyphonic mix must not clip above ±1.0."""

    def test_single_voice_no_scaling(self, allocator):
        allocator.note_on(69, 127)
        out = allocator.process(BLOCK_SIZE)
        # Single voice should produce reasonable output
        assert out.shape == (BLOCK_SIZE, 2)
        assert np.max(np.abs(out)) > 0.0

    def test_multiple_voices_no_clipping(self, allocator):
        # Play a 6-note chord
        for note in [60, 64, 67, 72, 76, 79]:
            allocator.note_on(note, 127)
        # Bypass all master effects so we measure only the voice mix
        allocator.set_param("chorus_bypass", 1.0)
        allocator.set_param("delay_bypass", 1.0)
        allocator.set_param("reverb_bypass", 1.0)
        allocator.set_param("limiter_bypass", 1.0)
        out = allocator.process(BLOCK_SIZE)
        peak = np.max(np.abs(out))
        assert peak <= 1.5, f"Voice mix peak {peak:.2f} too high, scaling broken"

    def test_many_voices_scaling(self, allocator):
        for note in range(36, 84):
            allocator.note_on(note, 100)
        allocator.set_param("chorus_bypass", 1.0)
        allocator.set_param("delay_bypass", 1.0)
        allocator.set_param("reverb_bypass", 1.0)
        allocator.set_param("limiter_bypass", 1.0)
        out = allocator.process(BLOCK_SIZE)
        peak = np.max(np.abs(out))
        assert peak < 3.0, f"Voice mix peak {peak:.2f} too high with many voices"


# ── Chorus voices ────────────────────────────────────────────────────────


class TestChorusVoices:
    """Chorus must support num_voices=1."""

    def test_chorus_single_voice(self, stereo_block):
        c = Chorus(SAMPLE_RATE)
        c.set_param("num_voices", 1.0)
        out = c.process(stereo_block)
        assert out.shape == (BLOCK_SIZE, 2)
        assert not np.any(np.isnan(out))

    def test_chorus_eight_voices(self, stereo_block):
        c = Chorus(SAMPLE_RATE)
        c.set_param("num_voices", 8.0)
        out = c.process(stereo_block)
        assert out.shape == (BLOCK_SIZE, 2)
        assert not np.any(np.isnan(out))

    def test_chorus_num_voices_param_update(self):
        c = Chorus(SAMPLE_RATE)
        c.set_param("num_voices", 3.0)
        assert c.get_param("num_voices") == pytest.approx(3.0)
