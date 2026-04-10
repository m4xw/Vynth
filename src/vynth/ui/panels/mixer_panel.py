"""Master mixer with volume and meters."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSizePolicy,
)

from vynth.ui.theme import Colors
from vynth.ui.widgets.fader import Fader
from vynth.ui.widgets.knob import RotaryKnob
from vynth.ui.widgets.led_meter import LEDMeter


class MixerPanel(QWidget):
    """Master mixer with volume and meters."""

    volumeChanged = pyqtSignal(float)

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
            minimum=0.0, maximum=1.0, value=0.8,
            default_value=0.8, show_db_scale=True,
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

    def set_cpu_load(self, percent: float) -> None:
        self._cpu_label.setText(f"CPU: {percent:.0f} %")

    # -- internals -----------------------------------------------------------

    def _on_fader(self, value: float) -> None:
        if value <= 0.0:
            self._db_label.setText("−∞ dB")
        else:
            db = 20.0 * math.log10(value)
            self._db_label.setText(f"{db:+.1f} dB")
        self.volumeChanged.emit(value)
