"""Live scrolling waveform display for real-time audio output."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from vynth.config import SAMPLE_RATE
from vynth.ui.theme import Colors


class LiveWaveformView(QWidget):
    """Rolling waveform display showing the live audio output."""

    # How many seconds of audio history to display
    DISPLAY_SECONDS = 2.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sample_rate = SAMPLE_RATE
        self._display_frames = int(self.DISPLAY_SECONDS * self._sample_rate)

        # Pre-allocated circular buffer for display
        self._buffer = np.zeros(self._display_frames, dtype=np.float32)
        self._write_pos = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._hint = QLabel("Live Waveform")
        self._hint.setStyleSheet(
            f"padding: 4px 8px; color: {Colors.TEXT_SECONDARY}; "
            f"background: {Colors.BG_MEDIUM}; border-bottom: 1px solid {Colors.BORDER};"
        )
        layout.addWidget(self._hint)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(Colors.WAVEFORM_BG)
        layout.addWidget(self._plot_widget, stretch=1)

        plot = self._plot_widget.getPlotItem()
        plot.setLabel("bottom", "Time", units="s")
        plot.setLabel("left", "Amplitude")
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.getAxis("bottom").setPen(pg.mkPen(Colors.TEXT_SECONDARY, width=1))
        plot.getAxis("left").setPen(pg.mkPen(Colors.TEXT_SECONDARY, width=1))
        plot.getAxis("bottom").setTextPen(Colors.TEXT_SECONDARY)
        plot.getAxis("left").setTextPen(Colors.TEXT_SECONDARY)

        plot.setYRange(-1.0, 1.0)
        plot.setXRange(0.0, self.DISPLAY_SECONDS, padding=0)
        plot.getViewBox().setMouseEnabled(x=False, y=False)

        # Waveform curve — filled
        fill_color = QColor(Colors.ACCENT_SECONDARY)
        fill_color.setAlpha(60)

        self._curve_max = plot.plot(pen=pg.mkPen(Colors.ACCENT_SECONDARY, width=1))
        self._curve_min = plot.plot(pen=pg.mkPen(Colors.ACCENT_SECONDARY, width=1))
        self._fill = pg.FillBetweenItem(self._curve_min, self._curve_max, brush=fill_color)
        plot.addItem(self._fill)

        # Downsample factor for display (show ~800 points)
        self._display_points = 800

    def push_audio(self, data: np.ndarray) -> None:
        """Push new audio frames into the rolling buffer.

        *data* shape: (frames,) or (frames, 2).  Stereo is mixed to mono.
        """
        if data.size == 0:
            return
        mono = data.mean(axis=1) if data.ndim == 2 else data.ravel()

        n = len(mono)
        if n >= self._display_frames:
            # More data than buffer — take the tail
            self._buffer[:] = mono[-self._display_frames:]
            self._write_pos = 0
        else:
            space = self._display_frames - self._write_pos
            if n <= space:
                self._buffer[self._write_pos : self._write_pos + n] = mono
                self._write_pos += n
            else:
                self._buffer[self._write_pos :] = mono[:space]
                remainder = n - space
                self._buffer[:remainder] = mono[space:]
                self._write_pos = remainder

    def update_display(self) -> None:
        """Redraw the waveform from the current buffer state."""
        # Unroll circular buffer into linear order
        if self._write_pos == 0:
            linear = self._buffer
        else:
            linear = np.concatenate(
                (self._buffer[self._write_pos:], self._buffer[:self._write_pos])
            )

        # Downsample via min/max envelope
        n = len(linear)
        chunk_size = max(1, n // self._display_points)
        n_chunks = n // chunk_size
        if n_chunks < 2:
            self._curve_max.setData([], [])
            self._curve_min.setData([], [])
            return

        seg = linear[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
        env_max = seg.max(axis=1)
        env_min = seg.min(axis=1)
        t = np.linspace(0.0, self.DISPLAY_SECONDS, n_chunks)

        self._curve_max.setData(t, env_max)
        self._curve_min.setData(t, env_min)
        self._fill.setCurves(self._curve_min, self._curve_max)

    def clear(self) -> None:
        """Reset the display buffer."""
        self._buffer[:] = 0.0
        self._write_pos = 0
        self._curve_max.setData([], [])
        self._curve_min.setData([], [])
