"""Real-time FFT spectrum analyzer display."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QLinearGradient
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from vynth.config import SAMPLE_RATE, SPECTRUM_FFT_SIZE, UI_FPS
from vynth.ui.theme import Colors

pg.setConfigOptions(antialias=True)

# Precompute Hann window
_HANN_WINDOW = np.hanning(SPECTRUM_FFT_SIZE).astype(np.float32)

# Frequency markers for grid
_FREQ_MARKERS = [20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000]


class SpectrumView(QWidget):
    """Real-time FFT spectrum analyzer display."""

    MIN_DB = -90.0
    MAX_DB = 0.0
    FREQ_MIN = 20.0
    FREQ_MAX = 20_000.0
    SMOOTHING_ALPHA = 0.3
    PEAK_DECAY_RATE = 0.3  # dB per frame

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sample_rate: int = SAMPLE_RATE
        self._fft_size: int = SPECTRUM_FFT_SIZE
        self._spectrum_db: np.ndarray | None = None
        self._peak_db: np.ndarray | None = None
        self._pending_block: np.ndarray | None = None
        self._visible = True
        self._linear_mode = False

        self._setup_ui()
        self._setup_plot()
        self._setup_timer()

    # ── UI setup ──────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toggle button bar
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 0)
        bar.addStretch()
        self._btn_mag_toggle = QPushButton("dB")
        self._btn_mag_toggle.setCheckable(True)
        self._btn_mag_toggle.setChecked(False)
        self._btn_mag_toggle.setFixedSize(40, 18)
        self._btn_mag_toggle.setToolTip("Toggle between dB and linear magnitude")
        self._btn_mag_toggle.setStyleSheet(
            f"""
            QPushButton {{
                background: {Colors.BG_MEDIUM}; border: 1px solid {Colors.BORDER};
                border-radius: 3px; font-size: 10px; color: {Colors.TEXT_SECONDARY};
            }}
            QPushButton:checked {{
                background: {Colors.ACCENT_PRIMARY}; color: #fff;
            }}
            """
        )
        self._btn_mag_toggle.toggled.connect(self._on_mag_mode_toggled)
        bar.addWidget(self._btn_mag_toggle)
        layout.addLayout(bar)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(Colors.WAVEFORM_BG)
        layout.addWidget(self._plot_widget)

    def _setup_plot(self) -> None:
        plot = self._plot_widget.getPlotItem()
        plot.setLogMode(x=True, y=False)
        plot.setLabel("bottom", "Frequency", units="Hz")
        plot.setLabel("left", "Magnitude", units="dB")

        plot.setXRange(
            np.log10(self.FREQ_MIN),
            np.log10(self.FREQ_MAX),
            padding=0,
        )
        plot.setYRange(self.MIN_DB, self.MAX_DB, padding=0)
        plot.setLimits(
            xMin=np.log10(self.FREQ_MIN),
            xMax=np.log10(self.FREQ_MAX),
            yMin=self.MIN_DB - 5,
            yMax=self.MAX_DB + 5,
        )

        # Axis styling
        for axis_name in ("bottom", "left"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(Colors.TEXT_SECONDARY, width=1))
            axis.setTextPen(Colors.TEXT_SECONDARY)

        # Grid
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.getAxis("bottom").setGrid(100)
        plot.getAxis("left").setGrid(100)

        # Custom tick labels for frequency axis
        freq_ticks = [(np.log10(f), self._freq_label(f)) for f in _FREQ_MARKERS]
        plot.getAxis("bottom").setTicks([freq_ticks])

        # Spectrum fill curve
        fill_color = QColor(Colors.ACCENT_PRIMARY)
        fill_color.setAlpha(80)
        self._spectrum_curve = plot.plot(
            pen=pg.mkPen(Colors.ACCENT_PRIMARY, width=1.5),
        )
        self._fill_curve = pg.PlotCurveItem(pen=pg.mkPen(width=0))
        plot.addItem(self._fill_curve)

        # Gradient fill below spectrum
        self._fill_item = pg.FillBetweenItem(
            self._fill_curve, self._spectrum_curve, brush=fill_color,
        )
        plot.addItem(self._fill_item)

        # Peak hold line
        self._peak_curve = plot.plot(
            pen=pg.mkPen(Colors.ACCENT_WARM, width=1, style=pg.QtCore.Qt.PenStyle.DotLine),
        )

        # Disable interactive zoom/pan on Y
        vb = plot.getViewBox()
        vb.setMouseEnabled(x=False, y=False)

    def _setup_timer(self) -> None:
        interval_ms = max(1, 1000 // UI_FPS)
        self._timer = QTimer(self)
        self._timer.setTimerType(pg.QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._update_display)
        self._timer.start(interval_ms)

    # ── Public API ────────────────────────────────────────────────────

    def push_audio_block(self, data: np.ndarray) -> None:
        """Feed an audio block from the engine for FFT analysis.

        *data* can be mono (1-D) or stereo (2-D). If stereo, channels
        are averaged to mono before analysis.
        """
        if data.ndim == 2:
            data = data.mean(axis=1)
        self._pending_block = data

    def set_sample_rate(self, sr: int) -> None:
        """Update the sample rate used for the frequency axis."""
        self._sample_rate = sr

    def clear(self) -> None:
        """Reset the spectrum display."""
        self._spectrum_db = None
        self._peak_db = None
        self._pending_block = None
        self._spectrum_curve.setData([], [])
        self._fill_curve.setData([], [])
        self._peak_curve.setData([], [])

    # ── Mode toggle ───────────────────────────────────────────────────

    def _on_mag_mode_toggled(self, linear: bool) -> None:
        self._linear_mode = linear
        self._btn_mag_toggle.setText("Lin" if linear else "dB")
        plot = self._plot_widget.getPlotItem()
        if linear:
            plot.setLabel("left", "Magnitude", units="")
            plot.setYRange(0.0, 1.0, padding=0)
            plot.setLimits(yMin=-0.05, yMax=1.05)
        else:
            plot.setLabel("left", "Magnitude", units="dB")
            plot.setYRange(self.MIN_DB, self.MAX_DB, padding=0)
            plot.setLimits(yMin=self.MIN_DB - 5, yMax=self.MAX_DB + 5)
        # Reset smoothed state so there's no jump artefact on switch
        self._spectrum_db = None
        self._peak_db = None

    # ── Internal ──────────────────────────────────────────────────────

    def _update_display(self) -> None:
        if self._pending_block is None:
            # Decay peaks even when no new audio arrives
            if self._peak_db is not None and self._spectrum_db is not None:
                self._peak_db -= self.PEAK_DECAY_RATE
                np.maximum(self._peak_db, self._spectrum_db, out=self._peak_db)
                np.clip(self._peak_db, self.MIN_DB, self.MAX_DB, out=self._peak_db)
            return

        if not self._visible or not self.isVisible():
            return

        block = self._pending_block
        self._pending_block = None

        spectrum_db = self._compute_fft(block)
        if spectrum_db is None:
            return

        # Exponential moving average smoothing
        if self._spectrum_db is not None and len(self._spectrum_db) == len(spectrum_db):
            self._spectrum_db += self.SMOOTHING_ALPHA * (spectrum_db - self._spectrum_db)
        else:
            self._spectrum_db = spectrum_db.copy()

        # Peak hold
        if self._peak_db is None or len(self._peak_db) != len(self._spectrum_db):
            self._peak_db = self._spectrum_db.copy()
        else:
            self._peak_db -= self.PEAK_DECAY_RATE
            np.maximum(self._peak_db, self._spectrum_db, out=self._peak_db)

        # Frequency axis
        freqs = np.fft.rfftfreq(self._fft_size, d=1.0 / self._sample_rate)
        mask = (freqs >= self.FREQ_MIN) & (freqs <= self.FREQ_MAX)
        freqs_masked = freqs[mask]
        db_masked = self._spectrum_db[mask]
        peak_masked = self._peak_db[mask]

        if len(freqs_masked) == 0:
            return

        if self._linear_mode:
            # Convert smoothed dB back to linear (0–1) for display
            lin = np.power(10.0, self._spectrum_db[mask] / 20.0)
            peak_lin = np.power(10.0, self._peak_db[mask] / 20.0)
            self._spectrum_curve.setData(freqs_masked, lin)
            baseline = np.zeros_like(lin)
            self._fill_curve.setData(freqs_masked, baseline)
            self._peak_curve.setData(freqs_masked, peak_lin)
        else:
            # Update curves (pyqtgraph handles log10 conversion via setLogMode)
            self._spectrum_curve.setData(freqs_masked, db_masked)

            # Fill baseline
            baseline = np.full_like(db_masked, self.MIN_DB)
            self._fill_curve.setData(freqs_masked, baseline)

            self._peak_curve.setData(freqs_masked, peak_masked)

    def _compute_fft(self, block: np.ndarray) -> np.ndarray | None:
        """Compute windowed FFT and return magnitude in dB."""
        n = len(block)
        if n == 0:
            return None

        # Pad or truncate to FFT size
        if n < self._fft_size:
            padded = np.zeros(self._fft_size, dtype=np.float32)
            padded[:n] = block[:n]
            block = padded
        else:
            block = block[-self._fft_size :]

        windowed = block * _HANN_WINDOW
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result) / self._fft_size

        # Avoid log of zero
        magnitude = np.clip(magnitude, 1e-10, None)
        db = 20.0 * np.log10(magnitude)
        np.clip(db, self.MIN_DB, self.MAX_DB, out=db)
        return db

    @staticmethod
    def _freq_label(freq: float) -> str:
        """Format a frequency value for axis display."""
        if freq >= 1000:
            return f"{freq / 1000:.0f}k"
        return f"{freq:.0f}"

    def showEvent(self, event) -> None:  # noqa: N802
        self._visible = True
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._visible = False
        super().hideEvent(event)
