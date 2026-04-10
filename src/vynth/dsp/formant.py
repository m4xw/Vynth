"""Formant preservation — compensate spectral envelope after pitch shifting."""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter
from scipy.fft import rfft, irfft

from vynth.dsp.base import DSPEffect


def _lpc_coefficients(signal: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC coefficients via the autocorrelation (Levinson-Durbin) method.

    Returns
    -------
    np.ndarray
        LPC coefficients of length ``order + 1`` with ``a[0] == 1.0``.
    """
    signal = signal.astype(np.float64)
    n = len(signal)
    if n <= order:
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        return a

    # Autocorrelation
    r = np.correlate(signal, signal, mode="full")[n - 1 : n + order]
    if r[0] == 0.0:
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        return a

    # Levinson-Durbin recursion
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    err = r[0]

    for i in range(1, order + 1):
        acc = np.dot(a[1:i], r[1:i][::-1]) if i > 1 else 0.0
        k = -(r[i] + acc) / err
        a[1 : i + 1] = a[1 : i + 1] + k * a[1 : i + 1][::-1].copy()
        a[i] = k
        err *= 1.0 - k * k
        if err <= 0.0:
            break

    return a


def _spectral_envelope(signal: np.ndarray, fft_size: int, lpc_order: int) -> np.ndarray:
    """Return the magnitude spectral envelope (rfft bins) estimated via LPC.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain mono audio block.
    fft_size : int
        FFT size (determines number of output bins).
    lpc_order : int
        LPC model order.

    Returns
    -------
    np.ndarray
        Spectral envelope magnitudes, shape ``(fft_size // 2 + 1,)``.
    """
    a = _lpc_coefficients(signal, lpc_order)
    # Envelope = 1 / |A(z)| evaluated on the unit circle
    a_spectrum = rfft(a, n=fft_size)
    magnitude = np.abs(a_spectrum)
    # Avoid division by zero — clamp to a very small positive value
    magnitude = np.maximum(magnitude, 1e-10)
    return (1.0 / magnitude).astype(np.float64)


class FormantPreserver(DSPEffect):
    """Preserve formants after pitch shifting.

    Prevents the *chipmunk effect* by re-imposing the original signal's
    spectral envelope onto the pitch-shifted signal.

    Parameters
    ----------
    preservation_amount : float
        Blend between unmodified shifted signal (0.0) and fully formant-
        corrected signal (1.0).
    sample_rate : int
        Audio sample rate in Hz.
    fft_size : int
        Analysis FFT size.
    lpc_order : int
        LPC model order for envelope estimation.
    """

    def __init__(
        self,
        preservation_amount: float = 1.0,
        sample_rate: int = 48_000,
        fft_size: int = 2048,
        lpc_order: int = 28,
    ):
        super().__init__(sample_rate)
        self._fft_size = fft_size
        self._lpc_order = lpc_order
        self._params["preservation_amount"] = np.clip(preservation_amount, 0.0, 1.0)

        self._reference: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    @property
    def preservation_amount(self) -> float:
        return self._params["preservation_amount"]

    @preservation_amount.setter
    def preservation_amount(self, value: float) -> None:
        self.set_param("preservation_amount", float(np.clip(value, 0.0, 1.0)))

    def _on_param_changed(self, name: str, value: float) -> None:
        if name == "preservation_amount":
            self._params[name] = float(np.clip(value, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Reference signal
    # ------------------------------------------------------------------

    def set_reference(self, original_data: np.ndarray) -> None:
        """Store the *original* (pre-pitch-shift) signal for envelope comparison.

        Parameters
        ----------
        original_data : np.ndarray
            Mono ``(N,)`` or stereo ``(N, 2)`` float32 audio.
        """
        self._reference = original_data.astype(np.float32, copy=True)

    # ------------------------------------------------------------------
    # Core formant correction (mono)
    # ------------------------------------------------------------------

    def _process_mono(
        self, shifted: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Apply formant correction to a mono block.

        Parameters
        ----------
        shifted : np.ndarray
            Pitch-shifted mono signal.
        reference : np.ndarray
            Original (pre-shift) mono signal of the same length.

        Returns
        -------
        np.ndarray
            Formant-corrected mono signal (float32).
        """
        n = len(shifted)

        # --- Edge cases ---
        if n < self._lpc_order + 1:
            return shifted.copy()

        rms = np.sqrt(np.mean(shifted ** 2))
        if rms < 1e-8:
            return shifted.copy()

        ref_rms = np.sqrt(np.mean(reference ** 2))
        if ref_rms < 1e-8:
            return shifted.copy()

        # Remove DC offset for cleaner LPC
        shifted_dc = shifted - np.mean(shifted)
        reference_dc = reference - np.mean(reference)

        # Spectral envelopes via LPC
        env_original = _spectral_envelope(reference_dc, self._fft_size, self._lpc_order)
        env_shifted = _spectral_envelope(shifted_dc, self._fft_size, self._lpc_order)

        # Correction filter: original_envelope / shifted_envelope
        correction = env_original / np.maximum(env_shifted, 1e-10)

        # Apply correction in the frequency domain
        shifted_spectrum = rfft(shifted_dc.astype(np.float64), n=self._fft_size)
        corrected_spectrum = shifted_spectrum * correction
        corrected = irfft(corrected_spectrum, n=self._fft_size).astype(np.float32)

        # Trim / pad to original length
        corrected = corrected[:n]
        if len(corrected) < n:
            corrected = np.pad(corrected, (0, n - len(corrected)))

        # Restore DC of the shifted signal
        corrected += np.mean(shifted)

        # Blend according to preservation_amount
        amount = self._params["preservation_amount"]
        if amount < 1.0:
            corrected = amount * corrected + (1.0 - amount) * shifted

        return corrected.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply formant preservation to pitch-shifted audio.

        Call :meth:`set_reference` with the original signal **before** calling
        this method.  If no reference is set the data is returned unchanged.

        Parameters
        ----------
        data : np.ndarray
            Pitch-shifted mono ``(N,)`` or stereo ``(N, 2)`` float32 audio.

        Returns
        -------
        np.ndarray
            Formant-corrected audio with the same shape and dtype.
        """
        if data.size == 0:
            return data.copy()

        data = data.astype(np.float32, copy=False)

        if self._reference is None:
            return data.copy()

        ref = self._reference

        if data.ndim == 1:
            # Mono — ensure reference length matches
            ref_mono = ref if ref.ndim == 1 else ref[:, 0]
            ref_mono = self._match_length(ref_mono, len(data))
            return self._process_mono(data, ref_mono)

        # Stereo / multi-channel
        channels = data.shape[1]
        out = np.empty_like(data)
        for ch in range(channels):
            if ref.ndim == 1:
                ref_ch = ref
            elif ch < ref.shape[1]:
                ref_ch = ref[:, ch]
            else:
                ref_ch = ref[:, 0]
            ref_ch = self._match_length(ref_ch, data.shape[0])
            out[:, ch] = self._process_mono(data[:, ch], ref_ch)
        return out

    def reset(self) -> None:
        """Clear stored reference data."""
        self._reference = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_length(arr: np.ndarray, target: int) -> np.ndarray:
        """Trim or zero-pad *arr* to *target* length."""
        if len(arr) >= target:
            return arr[:target]
        return np.pad(arr, (0, target - len(arr))).astype(np.float32)
