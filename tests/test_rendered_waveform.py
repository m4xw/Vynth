"""Exhaustive tests for RenderedWaveformProcessor (offline effect chain rendering)."""
import numpy as np
import pytest

from vynth.ui.widgets.rendered_waveform_view import RenderedWaveformProcessor, RenderContext


@pytest.fixture
def processor():
    return RenderedWaveformProcessor()


@pytest.fixture
def default_ctx():
    return RenderContext(params={}, bypass={})


def _sine(sr=48000, freq=440, duration_s=0.1):
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


class TestRenderedProcessorInit:
    def test_creates(self):
        p = RenderedWaveformProcessor()
        assert p is not None


class TestRenderedProcessorOutput:
    def test_mono_input_stereo_output(self, processor, default_ctx):
        sig = _sine()
        out = processor.render(sig, 48000, default_ctx)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_stereo_input(self, processor, default_ctx):
        sig = _sine()
        stereo = np.column_stack([sig, sig])
        out = processor.render(stereo, 48000, default_ctx)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_output_dtype(self, processor, default_ctx):
        out = processor.render(_sine(), 48000, default_ctx)
        assert out.dtype == np.float32

    def test_output_clipped(self, processor, default_ctx):
        out = processor.render(_sine(), 48000, default_ctx)
        assert np.all(out >= -1.0)
        assert np.all(out <= 1.0)

    def test_no_nan(self, processor, default_ctx):
        out = processor.render(_sine(), 48000, default_ctx)
        assert not np.any(np.isnan(out))

    def test_empty_input(self, processor, default_ctx):
        empty = np.array([], dtype=np.float32)
        out = processor.render(empty, 48000, default_ctx)
        assert out.shape == (0, 2)


class TestRenderContextParams:
    def test_with_filter_params(self, processor):
        ctx = RenderContext(
            params={"filter_freq": 500.0, "filter_q": 2.0, "filter_mode": 0},
            bypass={},
        )
        out = processor.render(_sine(), 48000, ctx)
        assert out.shape[0] == len(_sine())

    def test_with_reverb_params(self, processor):
        ctx = RenderContext(
            params={"reverb_room_size": 0.9, "reverb_wet": 0.5},
            bypass={},
        )
        out = processor.render(_sine(), 48000, ctx)
        assert not np.any(np.isnan(out))

    def test_with_all_effects(self, processor):
        ctx = RenderContext(
            params={
                "filter_freq": 1000.0,
                "gain_gain_db": 3.0,
                "chorus_rate": 2.0,
                "delay_time_ms": 100.0,
                "reverb_room_size": 0.5,
            },
            bypass={},
        )
        out = processor.render(_sine(), 48000, ctx)
        assert not np.any(np.isnan(out))


class TestRenderContextBypass:
    def test_all_bypassed(self, processor):
        ctx = RenderContext(
            params={},
            bypass={"filter": True, "gain": True, "chorus": True,
                    "delay": True, "reverb": True, "limiter": True},
        )
        sig = _sine()
        out = processor.render(sig, 48000, ctx)
        # With all bypassed, output ≈ stereo version of input
        expected = np.column_stack([sig, sig])
        np.testing.assert_allclose(out, np.clip(expected, -1, 1), atol=0.01)

    def test_filter_bypassed(self, processor):
        ctx = RenderContext(
            params={"filter_freq": 100.0, "filter_q": 10.0},
            bypass={"filter": True},
        )
        out = processor.render(_sine(), 48000, ctx)
        assert not np.any(np.isnan(out))


class TestRenderContext:
    def test_dataclass_slots(self):
        ctx = RenderContext(params={"a": 1.0}, bypass={"b": True})
        assert ctx.params == {"a": 1.0}
        assert ctx.bypass == {"b": True}

    def test_empty_context(self):
        ctx = RenderContext(params={}, bypass={})
        assert len(ctx.params) == 0
        assert len(ctx.bypass) == 0
