"""Phase vocoder pitch shifter — STFT-based real-time pitch shifting."""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, irfft

from vynth.dsp.base import DSPEffect


class PitchShifter(DSPEffect):
    """Real-time pitch shifter using a phase vocoder.

    Parameters
    ----------
    shift_semitones : float
        Pitch shift in semitones (positive = up, negative = down).
    sample_rate : int
        Audio sample rate in Hz.
    fft_size : int
        FFT window size.
    hop_size : int
        Hop (step) size between successive STFT frames.
    """

    FFT_SIZE = 2048
    HOP_SIZE = 512

    def __init__(
        self,
        sample_rate: int = 48_000,
        fft_size: int = FFT_SIZE,
        hop_size: int = HOP_SIZE,
        shift_semitones: float = 0.0,
    ):
        super().__init__(sample_rate)
        self._fft_size = fft_size
        self._hop_size = hop_size
        self._params["shift_semitones"] = shift_semitones

        self._window = np.hanning(self._fft_size).astype(np.float32)
        self._bin_count = self._fft_size // 2 + 1
        self._freq_per_bin = sample_rate / self._fft_size
        self._expected_phase_advance = (
            2.0 * np.pi * self._hop_size / self._fft_size * np.arange(self._bin_count)
        ).astype(np.float64)

        self._reset_state()

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Allocate / zero all internal streaming buffers."""
        self._input_buffer = np.zeros(self._fft_size, dtype=np.float32)
        self._output_buffer = np.zeros(self._fft_size, dtype=np.float32)
        self._last_phase = np.zeros(self._bin_count, dtype=np.float64)
        self._sum_phase = np.zeros(self._bin_count, dtype=np.float64)
        self._output_accum = np.zeros(self._fft_size * 2, dtype=np.float32)
        self._output_pos = 0

    def reset(self) -> None:  # noqa: D401
        """Clear all internal buffers so the next block starts fresh."""
        self._reset_state()

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    @property
    def shift_semitones(self) -> float:
        return self._params["shift_semitones"]

    @shift_semitones.setter
    def shift_semitones(self, value: float) -> None:
        self.set_param("shift_semitones", value)

    @property
    def _pitch_ratio(self) -> float:
        return 2.0 ** (self._params["shift_semitones"] / 12.0)

    # ------------------------------------------------------------------
    # Core phase-vocoder processing (mono)
    # ------------------------------------------------------------------

    def _process_mono(self, data: np.ndarray) -> np.ndarray:
        """Process a mono block of float32 samples and return pitch-shifted output."""
        n_samples = len(data)
        output = np.zeros(n_samples, dtype=np.float32)
        ratio = self._pitch_ratio

        pos = 0
        while pos < n_samples:
            # How many samples we can consume in this iteration
            chunk = min(self._hop_size, n_samples - pos)

            # Shift input buffer left and append new samples
            self._input_buffer[: -chunk] = self._input_buffer[chunk:]
            self._input_buffer[-chunk:] = data[pos : pos + chunk]

            # ---------- Analysis ----------
            windowed = self._input_buffer * self._window
            spectrum = rfft(windowed.astype(np.float64))
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Phase difference & unwrap
            phase_diff = phase - self._last_phase
            self._last_phase[:] = phase
            phase_diff -= self._expected_phase_advance
            phase_diff = phase_diff - 2.0 * np.pi * np.round(phase_diff / (2.0 * np.pi))
            true_freq = self._expected_phase_advance + phase_diff  # per-hop radians

            # ---------- Synthesis (pitch shift by re-mapping bins) ----------
            synth_magnitude = np.zeros(self._bin_count, dtype=np.float64)
            synth_freq = np.zeros(self._bin_count, dtype=np.float64)

            for k in range(self._bin_count):
                new_bin = int(k * ratio)
                if 0 <= new_bin < self._bin_count:
                    synth_magnitude[new_bin] += magnitude[k]
                    synth_freq[new_bin] = true_freq[k] * ratio

            self._sum_phase += synth_freq
            synth_spectrum = synth_magnitude * np.exp(1j * self._sum_phase)

            # ---------- IFFT & overlap-add ----------
            frame = irfft(synth_spectrum, n=self._fft_size).astype(np.float32)
            frame *= self._window

            self._output_accum[: self._fft_size] += frame
            output[pos : pos + chunk] = self._output_accum[:chunk]

            # Shift accumulator
            self._output_accum[: -chunk] = self._output_accum[chunk:]
            self._output_accum[-chunk:] = 0.0

            pos += chunk

        return output

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process audio data.

        Parameters
        ----------
        data : np.ndarray
            Mono ``(N,)`` or stereo ``(N, 2)`` float32 audio.

        Returns
        -------
        np.ndarray
            Pitch-shifted audio with the same shape and dtype.
        """
        if data.size == 0:
            return data.copy()

        data = data.astype(np.float32, copy=False)

        if data.ndim == 1:
            return self._process_mono(data)

        # Stereo / multi-channel: process each channel independently
        channels = data.shape[1]
        # We need per-channel state — use the first channel with current state,
        # and create temporary copies for additional channels.
        out = np.empty_like(data)
        saved_state = self._save_state()
        for ch in range(channels):
            if ch > 0:
                self._restore_state(saved_state)
            out[:, ch] = self._process_mono(data[:, ch])
            if ch == 0:
                saved_state = self._save_state()
        self._restore_state(saved_state)
        return out

    # ------------------------------------------------------------------
    # State snapshot helpers (for multi-channel processing)
    # ------------------------------------------------------------------

    def _save_state(self) -> dict:
        return {
            "input_buffer": self._input_buffer.copy(),
            "output_buffer": self._output_buffer.copy(),
            "last_phase": self._last_phase.copy(),
            "sum_phase": self._sum_phase.copy(),
            "output_accum": self._output_accum.copy(),
        }

    def _restore_state(self, state: dict) -> None:
        self._input_buffer[:] = state["input_buffer"]
        self._output_buffer[:] = state["output_buffer"]
        self._last_phase[:] = state["last_phase"]
        self._sum_phase[:] = state["sum_phase"]
        self._output_accum[:] = state["output_accum"]
