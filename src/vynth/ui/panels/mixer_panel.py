"""Master mixer with volume and meters."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSizePolicy, QComboBox,
)

from vynth.ui.theme import Colors
from vynth.ui.widgets.fader import Fader
from vynth.ui.widgets.knob import RotaryKnob
from vynth.ui.widgets.led_meter import LEDMeter


class MixerPanel(QWidget):
    """Master mixer with volume and meters."""

    volumeChanged = pyqtSignal(float)
    visualizerModeChanged = pyqtSignal(str)
    MIN_DB = -60.0
    MAX_DB = 50.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        group = QGroupBox("Master")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(group)

        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        # --- Fader + meter row -----------------------------------------------
        fader_row = QHBoxLayout()
        fader_row.setSpacing(4)

        self._fader = Fader(
            orientation=Qt.Orientation.Vertical,
            minimum=self.MIN_DB,
            maximum=self.MAX_DB,
            value=self._linear_to_db(0.8),
            default_value=self._linear_to_db(0.8),
            show_db_scale=True,
        )
        self._fader.valueChanged.connect(self._on_fader)
        fader_row.addWidget(self._fader, alignment=Qt.AlignmentFlag.AlignCenter)

        self._meter = LEDMeter()
        fader_row.addWidget(self._meter, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(fader_row, 1)

        # --- dB readout ------------------------------------------------------
        self._db_label = QLabel("−∞ dB")
        self._db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._db_label.setStyleSheet(
            f"font-family: monospace; font-size: 13px; color: {Colors.TEXT_PRIMARY};"
        )
        layout.addWidget(self._db_label)

        # --- Pan knob --------------------------------------------------------
        self._pan = RotaryKnob(
            name="Pan", minimum=-1.0, maximum=1.0, value=0.0,
            default_value=0.0, suffix="", decimals=2,
        )
        layout.addWidget(self._pan, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- Visualizer mode -----------------------------------------------
        vis_row = QHBoxLayout()
        vis_row.setSpacing(6)
        vis_label = QLabel("Visualizer")
        vis_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        vis_row.addWidget(vis_label)

        self._visualizer_mode = QComboBox()
        self._visualizer_mode.addItem("Spectrum", "spectrum")
        self._visualizer_mode.addItem("Rendered", "rendered")
        self._visualizer_mode.addItem("Live", "live")
        self._visualizer_mode.currentIndexChanged.connect(self._on_visualizer_mode_changed)
        vis_row.addWidget(self._visualizer_mode)
        layout.addLayout(vis_row)

        # --- Info labels -----------------------------------------------------
        info_row = QVBoxLayout()
        info_row.setSpacing(2)

        self._voice_label = QLabel("0/64 voices")
        self._voice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._voice_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        info_row.addWidget(self._voice_label)

        self._cpu_label = QLabel("CPU: 0 %")
        self._cpu_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cpu_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        info_row.addWidget(self._cpu_label)

        layout.addLayout(info_row)

    # -- public API ----------------------------------------------------------

    def set_levels(self, left: float, right: float) -> None:
        self._meter.set_levels(left, right)

    def set_voice_count(self, active: int, total: int = 64) -> None:
        self._voice_label.setText(f"{active}/{total} voices")

    def set_volume(self, value: float) -> None:
        """Set the fader position programmatically."""
        self._fader.blockSignals(True)
        self._fader.value = self._linear_to_db(value)
        self._fader.blockSignals(False)
        self._on_fader(self._fader.value)

    def set_cpu_load(self, percent: float) -> None:
        self._cpu_label.setText(f"CPU: {percent:.0f} %")

    def set_visualizer_mode(self, mode: str) -> None:
        idx = self._visualizer_mode.findData(mode)
        if idx < 0:
            return
        self._visualizer_mode.blockSignals(True)
        self._visualizer_mode.setCurrentIndex(idx)
        self._visualizer_mode.blockSignals(False)

    # -- internals -----------------------------------------------------------

    @classmethod
    def _db_to_linear(cls, db: float) -> float:
        if db <= cls.MIN_DB:
            return 0.0
        return float(math.pow(10.0, db / 20.0))

    @classmethod
    def _linear_to_db(cls, value: float) -> float:
        if value <= 0.0:
            return cls.MIN_DB
        return max(cls.MIN_DB, min(cls.MAX_DB, 20.0 * math.log10(value)))

    def _on_fader(self, value: float) -> None:
        if value <= self.MIN_DB:
            self._db_label.setText("−∞ dB")
        else:
            self._db_label.setText(f"{value:+.1f} dB")
        self.volumeChanged.emit(self._db_to_linear(value))

    def _on_visualizer_mode_changed(self) -> None:
        mode = self._visualizer_mode.currentData()
        if isinstance(mode, str):
            self.visualizerModeChanged.emit(mode)
