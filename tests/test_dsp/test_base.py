"""Exhaustive tests for DSPEffect abstract base class."""
import numpy as np
import pytest

from vynth.dsp.base import DSPEffect


class ConcreteDSP(DSPEffect):
    """Minimal concrete implementation for testing."""

    def __init__(self, sample_rate=48000):
        super().__init__(sample_rate)
        self._params["gain"] = 1.0
        self._last_changed = None

    def process(self, data: np.ndarray) -> np.ndarray:
        return data * self._params["gain"]

    def _on_param_changed(self, name, value):
        self._last_changed = (name, value)


class TestDSPEffectInit:
    def test_default_sample_rate(self):
        d = ConcreteDSP()
        assert d.sample_rate == 48000

    def test_custom_sample_rate(self):
        d = ConcreteDSP(44100)
        assert d.sample_rate == 44100

    def test_not_bypassed(self):
        d = ConcreteDSP()
        assert d.bypassed is False

    def test_empty_params_initially(self):
        # Our concrete adds "gain", but a bare subclass might not
        d = ConcreteDSP()
        assert "gain" in d._params


class TestDSPEffectBypass:
    def test_set_bypass(self):
        d = ConcreteDSP()
        d.bypassed = True
        assert d.bypassed is True

    def test_unset_bypass(self):
        d = ConcreteDSP()
        d.bypassed = True
        d.bypassed = False
        assert d.bypassed is False


class TestDSPEffectParams:
    def test_get_param(self):
        d = ConcreteDSP()
        assert d.get_param("gain") == pytest.approx(1.0)

    def test_get_param_missing(self):
        d = ConcreteDSP()
        assert d.get_param("nonexistent") == 0.0

    def test_set_param(self):
        d = ConcreteDSP()
        d.set_param("gain", 0.5)
        assert d.get_param("gain") == pytest.approx(0.5)

    def test_set_param_triggers_callback(self):
        d = ConcreteDSP()
        d.set_param("gain", 2.0)
        assert d._last_changed == ("gain", 2.0)

    def test_set_param_new_key(self):
        d = ConcreteDSP()
        d.set_param("new_param", 42.0)
        assert d.get_param("new_param") == pytest.approx(42.0)

    def test_get_params(self):
        d = ConcreteDSP()
        d.set_param("another", 5.0)
        params = d.get_params()
        assert "gain" in params
        assert "another" in params
        assert params["another"] == pytest.approx(5.0)

    def test_get_params_returns_copy(self):
        d = ConcreteDSP()
        params = d.get_params()
        params["gain"] = 999.0
        assert d.get_param("gain") == pytest.approx(1.0)

    def test_set_params_bulk(self):
        d = ConcreteDSP()
        d.set_params({"gain": 0.1, "extra": 3.0})
        assert d.get_param("gain") == pytest.approx(0.1)
        assert d.get_param("extra") == pytest.approx(3.0)


class TestDSPEffectProcess:
    def test_process_mono(self):
        d = ConcreteDSP()
        d.set_param("gain", 0.5)
        data = np.ones(100, dtype=np.float32)
        out = d.process(data)
        np.testing.assert_allclose(out, 0.5, atol=1e-7)

    def test_process_stereo(self):
        d = ConcreteDSP()
        d.set_param("gain", 2.0)
        data = np.ones((100, 2), dtype=np.float32) * 0.3
        out = d.process(data)
        np.testing.assert_allclose(out, 0.6, atol=1e-6)


class TestDSPEffectProcessMaybeBypass:
    def test_not_bypassed_processes(self):
        d = ConcreteDSP()
        d.set_param("gain", 0.5)
        data = np.ones(100, dtype=np.float32)
        out = d.process_maybe_bypass(data)
        np.testing.assert_allclose(out, 0.5, atol=1e-7)

    def test_bypassed_returns_input(self):
        d = ConcreteDSP()
        d.set_param("gain", 0.5)
        d.bypassed = True
        data = np.ones(100, dtype=np.float32)
        out = d.process_maybe_bypass(data)
        np.testing.assert_allclose(out, 1.0, atol=1e-7)

    def test_bypass_returns_same_object(self):
        d = ConcreteDSP()
        d.bypassed = True
        data = np.ones(100, dtype=np.float32)
        out = d.process_maybe_bypass(data)
        assert out is data


class TestDSPEffectReset:
    def test_reset_does_not_raise(self):
        d = ConcreteDSP()
        d.reset()  # base implementation is a no-op


class TestDSPEffectAbstract:
    def test_cannot_instantiate_bare_base(self):
        with pytest.raises(TypeError):
            DSPEffect()
