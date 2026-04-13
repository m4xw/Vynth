"""Exhaustive tests for FormantPreserver (LPC-based spectral envelope correction)."""
import numpy as np
import pytest

from vynth.dsp.formant import FormantPreserver, _lpc_coefficients, _spectral_envelope


@pytest.fixture
def fp(sr):
    return FormantPreserver(sample_rate=sr)


def _sine(sr, freq=440, duration_s=0.1, amp=1.0):
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


class TestFormantInit:
    def test_default_amount(self, sr):
        fp = FormantPreserver(sample_rate=sr)
        assert fp.preservation_amount == pytest.approx(1.0)

    def test_custom_amount(self, sr):
        fp = FormantPreserver(preservation_amount=0.5, sample_rate=sr)
        assert fp.preservation_amount == pytest.approx(0.5)

    def test_amount_clamped_high(self, sr):
        fp = FormantPreserver(preservation_amount=5.0, sample_rate=sr)
        assert fp.preservation_amount == pytest.approx(1.0)

    def test_amount_clamped_low(self, sr):
        fp = FormantPreserver(preservation_amount=-1.0, sample_rate=sr)
        assert fp.preservation_amount == pytest.approx(0.0)

    def test_no_reference(self, sr):
        fp = FormantPreserver(sample_rate=sr)
        assert fp._reference is None


class TestFormantProcess:
    def test_no_reference_returns_copy(self, sr, fp, sine_440):
        out = fp.process(sine_440[:512])
        np.testing.assert_array_equal(out, sine_440[:512])

    def test_with_reference_modifies_signal(self, sr, fp, sine_440):
        ref = _sine(sr, 220, 0.1)
        shifted = _sine(sr, 880, 0.1)
        fp.set_reference(ref)
        out = fp.process(shifted)
        # Output should differ from input
        assert out.shape == shifted.shape
        assert not np.allclose(out, shifted, atol=0.01)

    def test_mono_output_shape(self, sr, fp):
        ref = _sine(sr, 440, 0.1)
        sig = _sine(sr, 880, 0.1)
        fp.set_reference(ref)
        out = fp.process(sig)
        assert out.ndim == 1
        assert len(out) == len(sig)

    def test_stereo_output_shape(self, sr, fp):
        ref = _sine(sr, 440, 0.1)
        sig = _sine(sr, 880, 0.1)
        stereo = np.column_stack([sig, sig])
        fp.set_reference(ref)
        out = fp.process(stereo)
        assert out.ndim == 2
        assert out.shape == stereo.shape

    def test_dtype_float32(self, sr, fp):
        ref = _sine(sr, 440, 0.1)
        sig = _sine(sr, 880, 0.1)
        fp.set_reference(ref)
        out = fp.process(sig)
        assert out.dtype == np.float32

    def test_empty_input_returns_empty(self, sr, fp):
        empty = np.array([], dtype=np.float32)
        out = fp.process(empty)
        assert out.size == 0

    def test_no_nan(self, sr, fp):
        ref = _sine(sr, 440, 0.2)
        sig = _sine(sr, 880, 0.2)
        fp.set_reference(ref)
        out = fp.process(sig)
        assert not np.any(np.isnan(out))

    def test_no_inf(self, sr, fp):
        ref = _sine(sr, 440, 0.2)
        sig = _sine(sr, 880, 0.2)
        fp.set_reference(ref)
        out = fp.process(sig)
        assert not np.any(np.isinf(out))


class TestFormantPreservationAmount:
    def test_zero_amount_passthrough(self, sr):
        fp = FormantPreserver(preservation_amount=0.0, sample_rate=sr)
        ref = _sine(sr, 440, 0.1)
        sig = _sine(sr, 880, 0.1)
        fp.set_reference(ref)
        out = fp.process(sig)
        np.testing.assert_allclose(out, sig, atol=0.01)

    def test_full_amount_differs(self, sr):
        fp = FormantPreserver(preservation_amount=1.0, sample_rate=sr)
        ref = _sine(sr, 440, 0.1)
        sig = _sine(sr, 880, 0.1)
        fp.set_reference(ref)
        out = fp.process(sig)
        diff = np.max(np.abs(out - sig))
        assert diff > 0.001

    def test_half_amount_intermediate(self, sr):
        fp_full = FormantPreserver(preservation_amount=1.0, sample_rate=sr)
        fp_half = FormantPreserver(preservation_amount=0.5, sample_rate=sr)
        ref = _sine(sr, 440, 0.1)
        sig = _sine(sr, 880, 0.1)
        fp_full.set_reference(ref)
        fp_half.set_reference(ref.copy())
        out_full = fp_full.process(sig.copy())
        out_half = fp_half.process(sig.copy())
        # Half should be between original and full correction
        diff_full = np.sum(np.abs(out_full - sig))
        diff_half = np.sum(np.abs(out_half - sig))
        assert diff_half < diff_full


class TestFormantSetReference:
    def test_set_reference_stores(self, sr, fp):
        ref = _sine(sr, 440, 0.1)
        fp.set_reference(ref)
        assert fp._reference is not None
        assert len(fp._reference) == len(ref)

    def test_set_reference_copies(self, sr, fp):
        ref = _sine(sr, 440, 0.1)
        fp.set_reference(ref)
        ref[:] = 0.0
        assert np.max(np.abs(fp._reference)) > 0.0

    def test_stereo_reference(self, sr, fp):
        ref = _sine(sr, 440, 0.1)
        stereo_ref = np.column_stack([ref, ref])
        fp.set_reference(stereo_ref)
        assert fp._reference.shape == stereo_ref.shape


class TestFormantReset:
    def test_reset_clears_reference(self, sr, fp):
        fp.set_reference(_sine(sr, 440, 0.1))
        fp.reset()
        assert fp._reference is None


class TestFormantBypass:
    def test_bypass_passthrough(self, sr, sine_440):
        fp = FormantPreserver(sample_rate=sr)
        fp.set_reference(_sine(sr, 220, 0.1))
        fp.bypassed = True
        out = fp.process_maybe_bypass(sine_440[:512])
        np.testing.assert_array_equal(out, sine_440[:512])


class TestLPCCoefficients:
    def test_lpc_length(self):
        sig = np.random.randn(1024).astype(np.float64)
        a = _lpc_coefficients(sig, 10)
        assert len(a) == 11
        assert a[0] == pytest.approx(1.0)

    def test_lpc_zero_signal(self):
        sig = np.zeros(1024, dtype=np.float64)
        a = _lpc_coefficients(sig, 10)
        assert a[0] == 1.0

    def test_lpc_short_signal(self):
        sig = np.array([1.0, 2.0], dtype=np.float64)
        a = _lpc_coefficients(sig, 10)
        assert len(a) == 11
        assert a[0] == 1.0

    @pytest.mark.parametrize("order", [2, 10, 28, 50])
    def test_lpc_various_orders(self, order):
        sig = np.random.randn(2048).astype(np.float64)
        a = _lpc_coefficients(sig, order)
        assert len(a) == order + 1


class TestSpectralEnvelope:
    def test_envelope_shape(self):
        sig = np.random.randn(1024).astype(np.float64)
        env = _spectral_envelope(sig, 2048, 28)
        assert len(env) == 2048 // 2 + 1

    def test_envelope_positive(self):
        sig = np.random.randn(1024).astype(np.float64)
        env = _spectral_envelope(sig, 2048, 28)
        assert np.all(env > 0)

    def test_envelope_no_nan(self):
        sig = np.random.randn(1024).astype(np.float64)
        env = _spectral_envelope(sig, 2048, 28)
        assert not np.any(np.isnan(env))


class TestFormantMatchLength:
    def test_shorter_gets_padded(self, sr):
        arr = np.ones(50, dtype=np.float32)
        out = FormantPreserver._match_length(arr, 100)
        assert len(out) == 100
        np.testing.assert_array_equal(out[:50], 1.0)
        np.testing.assert_array_equal(out[50:], 0.0)

    def test_longer_gets_trimmed(self, sr):
        arr = np.ones(100, dtype=np.float32)
        out = FormantPreserver._match_length(arr, 50)
        assert len(out) == 50

    def test_exact_length_unchanged(self, sr):
        arr = np.ones(100, dtype=np.float32)
        out = FormantPreserver._match_length(arr, 100)
        assert len(out) == 100
