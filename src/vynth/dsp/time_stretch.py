"""WSOLA (Waveform Similarity Overlap-Add) time stretcher."""
from __future__ import annotations

import numpy as np

from vynth.dsp.base import DSPEffect


class TimeStretch(DSPEffect):
    """Offline WSOLA time-stretcher.

    Processes entire buffers — **not** a real-time streaming effect.

    Parameters
    ----------
    stretch_ratio : float
        Time-stretch factor (0.25 … 4.0, default 1.0).
        Values > 1.0 make the signal longer (slower);
        values < 1.0 make it shorter (faster).
    """

    WINDOW_SIZE: int = 2048

    PARAM_DEFAULTS: dict[str, tuple[float, float, float]] = {
        "stretch_ratio": (0.25, 4.0, 1.0),
    }

    def __init__(self, sample_rate: int = 48_000):
        super().__init__(sample_rate)
        for name, (lo, hi, default) in self.PARAM_DEFAULTS.items():
            self._params[name] = default

        self._window = np.hanning(self.WINDOW_SIZE).astype(np.float64)

    # ------------------------------------------------------------------
    def _on_param_changed(self, name: str, value: float) -> None:
        lo, hi, _ = self.PARAM_DEFAULTS[name]
        self._params[name] = float(np.clip(value, lo, hi))

    # ------------------------------------------------------------------
    @staticmethod
    def _best_overlap_offset(
        src: np.ndarray, target: np.ndarray, tolerance: int
    ) -> int:
        """Find the offset within *src* that best matches *target* via cross-correlation.

        Parameters
        ----------
        src : np.ndarray
            Source region to search (length ≥ len(target) + 2*tolerance).
        target : np.ndarray
            Reference waveform to match against.
        tolerance : int
            Half-width of the search range in samples.

        Returns
        -------
        int
            Best offset into *src*.
        """
        best_offset = 0
        best_corr = -np.inf
        t_len = len(target)

        for offset in range(2 * tolerance + 1):
            segment = src[offset : offset + t_len]
            if len(segment) < t_len:
                break
            corr = np.dot(segment, target)
            if corr > best_corr:
                best_corr = corr
                best_offset = offset

        return best_offset

    # ------------------------------------------------------------------
    def _stretch_mono(self, signal: np.ndarray) -> np.ndarray:
        """Time-stretch a single-channel signal via WSOLA."""
        ratio = self._params["stretch_ratio"]
        if abs(ratio - 1.0) < 1e-6:
            return signal.copy()

        win_size = self.WINDOW_SIZE
        analysis_hop = win_size // 4
        synthesis_hop = max(1, int(round(analysis_hop * ratio)))
        tolerance = win_size // 4
        n_in = len(signal)

        # Estimated output length (generous)
        est_out_len = int(n_in * ratio) + win_size * 2
        output = np.zeros(est_out_len, dtype=np.float64)
        norm = np.zeros(est_out_len, dtype=np.float64)

        analysis_pos = 0
        synthesis_pos = 0

        while analysis_pos + win_size <= n_in:
            # Determine search region for best overlap
            search_start = max(0, analysis_pos - tolerance)
            search_end = min(n_in, analysis_pos + win_size + tolerance)
            search_region = signal[search_start:search_end]

            # Target = Hann-windowed frame at nominal analysis position
            nominal_frame = signal[analysis_pos : analysis_pos + win_size]
            if len(nominal_frame) < win_size:
                break

            # Find best overlap offset
            if search_end - search_start >= win_size:
                offset = self._best_overlap_offset(
                    search_region, nominal_frame, tolerance
                )
            else:
                offset = 0

            actual_start = search_start + offset
            frame = signal[actual_start : actual_start + win_size]
            if len(frame) < win_size:
                break

            windowed = frame * self._window

            # Overlap-add into output
            out_end = synthesis_pos + win_size
            if out_end > est_out_len:
                # Extend buffers
                extra = out_end - est_out_len + win_size
                output = np.concatenate([output, np.zeros(extra, dtype=np.float64)])
                norm = np.concatenate([norm, np.zeros(extra, dtype=np.float64)])
                est_out_len = len(output)

            output[synthesis_pos:out_end] += windowed
            norm[synthesis_pos:out_end] += self._window

            analysis_pos += analysis_hop
            synthesis_pos += synthesis_hop

        # Normalise where we have overlapping windows
        mask = norm > 1e-8
        output[mask] /= norm[mask]

        # Trim to actual content
        final_len = synthesis_pos
        if final_len > len(output):
            final_len = len(output)
        return output[:final_len]

    # ------------------------------------------------------------------
    def process(self, data: np.ndarray) -> np.ndarray:
        """Time-stretch an entire audio buffer.

        Parameters
        ----------
        data : np.ndarray
            Mono ``(frames,)`` or stereo ``(frames, 2)`` float32 audio.

        Returns
        -------
        np.ndarray
            Stretched audio in float32, same channel layout as input.
        """
        if data.size == 0:
            return data

        work = data.astype(np.float64)

        if work.ndim == 1:
            result = self._stretch_mono(work)
        else:
            channels = [
                self._stretch_mono(work[:, ch]) for ch in range(work.shape[1])
            ]
            # Ensure equal length across channels
            min_len = min(len(c) for c in channels)
            result = np.column_stack([c[:min_len] for c in channels])

        return result.astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        # No streaming state to clear; included for interface consistency.
        pass
