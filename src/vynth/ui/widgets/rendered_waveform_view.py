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


class RenderedWaveformProcessor:
    """Offline renderer that applies the active effect chain to a sample."""

    def render(self, data: np.ndarray, sample_rate: int, ctx: RenderContext) -> np.ndarray:
        if data.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        stereo = self._to_stereo(data)

        # Approximate the audible chain for visualization with deterministic state.
        stereo = self._apply_filter(stereo, sample_rate, ctx)
        stereo = self._apply_master_effects(stereo, sample_rate, ctx)

        return np.clip(stereo, -1.0, 1.0).astype(np.float32)

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

    def set_filter_overlay(self, frequency_hz: float, q: float, mode: int, gain_db: float) -> None:
        self._waveform.set_filter_overlay(frequency_hz, q, mode, gain_db)

    def clear_filter_overlay(self) -> None:
        self._waveform.clear_filter_overlay()
