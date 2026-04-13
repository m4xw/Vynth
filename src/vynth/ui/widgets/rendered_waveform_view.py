"""Rendered waveform view and offline processing helper."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from vynth.dsp.chorus import Chorus
from vynth.dsp.delay import Delay
from vynth.dsp.filter import BiquadFilter
from vynth.dsp.gain import GainEffect
from vynth.dsp.limiter import Limiter
from vynth.dsp.reverb import Reverb
from vynth.ui.theme import Colors
from vynth.ui.widgets.waveform_view import WaveformView


@dataclass(slots=True)
class RenderContext:
    """Data container for offline waveform rendering."""

    params: dict[str, float]
    bypass: dict[str, bool]
    selection: tuple[int, int] | None = None


class RenderedWaveformProcessor:
    """Offline renderer that applies the active effect chain to a sample."""

    def render(self, data: np.ndarray, sample_rate: int, ctx: RenderContext) -> np.ndarray:
        if data.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        stereo = self._to_stereo(data)
        start, end = self._resolve_selection(stereo.shape[0], ctx.selection)

        # Approximate the audible chain for visualization with deterministic state.
        if start == 0 and end == stereo.shape[0]:
            out = self._apply_filter(stereo, sample_rate, ctx)
            out = self._apply_master_effects(out, sample_rate, ctx)
        else:
            out = stereo.copy()
            selected = out[start:end]
            selected = self._apply_filter(selected, sample_rate, ctx)
            selected = self._apply_master_effects(selected, sample_rate, ctx)
            out[start:end] = selected

        return np.clip(out, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _resolve_selection(
        length: int,
        selection: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if selection is None:
            return (0, length)

        start = max(0, min(int(selection[0]), length))
        end = max(0, min(int(selection[1]), length))
        if end <= start:
            return (0, length)
        return (start, end)

    @staticmethod
    def _to_stereo(data: np.ndarray) -> np.ndarray:
        if data.ndim == 2 and data.shape[1] == 2:
            return data.astype(np.float32, copy=True)
        mono = data.reshape(-1).astype(np.float32, copy=False)
        return np.column_stack((mono, mono)).astype(np.float32, copy=False)

    @staticmethod
    def _apply_params(effect, prefix: str, params: dict[str, float]) -> None:
        prefix_token = f"{prefix}_"
        for key, value in params.items():
            if not key.startswith(prefix_token):
                continue
            param_name = key[len(prefix_token) :]
            try:
                effect.set_param(param_name, float(value))
            except Exception:
                # Unknown or unsupported params should not break visualization rendering.
                continue

    def _apply_filter(self, data: np.ndarray, sample_rate: int, ctx: RenderContext) -> np.ndarray:
        fx = BiquadFilter(sample_rate)
        self._apply_params(fx, "filter", ctx.params)
        fx.bypassed = bool(ctx.bypass.get("filter", False))
        return fx.process_maybe_bypass(data)

    def _apply_master_effects(self, data: np.ndarray, sample_rate: int, ctx: RenderContext) -> np.ndarray:
        chain = [
            ("gain", GainEffect(sample_rate)),
            ("chorus", Chorus(sample_rate)),
            ("delay", Delay(sample_rate)),
            ("reverb", Reverb(sample_rate)),
            ("limiter", Limiter(sample_rate)),
        ]

        out = data
        for prefix, effect in chain:
            self._apply_params(effect, prefix, ctx.params)
            effect.bypassed = bool(ctx.bypass.get(prefix, False))
            out = effect.process_maybe_bypass(out)
        return out


class RenderedWaveformView(QWidget):
    """UI wrapper around WaveformView for offline effect-rendered waveforms."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._hint = QLabel("Rendered Waveform (post effects)")
        self._hint.setStyleSheet(
            f"padding: 4px 8px; color: {Colors.TEXT_SECONDARY}; "
            f"background: {Colors.BG_MEDIUM}; border-bottom: 1px solid {Colors.BORDER};"
        )
        layout.addWidget(self._hint)

        self._waveform = WaveformView(self)
        layout.addWidget(self._waveform, stretch=1)

    def set_rendered_data(self, data: np.ndarray, sample_rate: int) -> None:
        self._waveform.set_data(data, sample_rate)

    def clear(self) -> None:
        self._waveform.clear()

    def set_selection(self, start: int, end: int) -> None:
        """Show selection region on the rendered waveform."""
        self._waveform.set_selection_region(start, end)

    def clear_selection(self) -> None:
        self._waveform.clear_selection_region()

    def set_loop_points(self, start: int, end: int) -> None:
        """Show loop markers on the rendered waveform."""
        self._waveform.set_loop_points(start, end)

    def clear_loop_points(self) -> None:
        self._waveform.clear_loop_points()

    def set_filter_overlay(
        self,
        frequency_hz: float,
        q: float,
        mode: int,
        gain_db: float,
        scope_frames: tuple[int, int] | None = None,
    ) -> None:
        self._waveform.set_filter_overlay(
            frequency_hz,
            q,
            mode,
            gain_db,
            scope_frames=scope_frames,
        )

    def clear_filter_overlay(self) -> None:
        self._waveform.clear_filter_overlay()
