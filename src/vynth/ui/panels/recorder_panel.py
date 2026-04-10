"""Audio recording controls with level meter and preview."""

from __future__ import annotations

import collections

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QComboBox, QLabel, QCheckBox, QSizePolicy,
)

from vynth.ui.theme import Colors
from vynth.ui.widgets.led_meter import LEDMeter


class _WaveformPreview(QWidget):
    """Minimal waveform visualisation of the last ~3 seconds of audio."""

    _MAX_SAMPLES = 300  # visual samples kept

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._samples: collections.deque[float] = collections.deque(maxlen=self._MAX_SAMPLES)

    def push_sample(self, value: float) -> None:
        self._samples.append(max(-1.0, min(1.0, value)))
        self.update()

    def clear(self) -> None:
        self._samples.clear()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg = QColor(Colors.WAVEFORM_BG)
        p.fillRect(self.rect(), bg)

        if not self._samples:
            p.end()
            return

        pen = QPen(QColor(Colors.WAVEFORM), 1)
        p.setPen(pen)

        w = self.width()
        h = self.height()
        mid = h / 2.0
        n = len(self._samples)

        for i, s in enumerate(self._samples):
            x = int(i * w / n)
            y_off = s * mid
            p.drawLine(x, int(mid - y_off), x, int(mid + y_off))

        p.end()


class RecorderPanel(QWidget):
    """Audio recording controls with level meter and preview."""

    recordToggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        group = QGroupBox("Recorder")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(group)

        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        # --- Record button ---------------------------------------------------
        self._rec_btn = QPushButton("● REC")
        self._rec_btn.setCheckable(True)
        self._rec_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {Colors.BG_MEDIUM};
                color: {Colors.ACCENT_WARM};
                border: 2px solid {Colors.ACCENT_WARM};
                border-radius: 6px;
                font-weight: bold;
                font-size: 15px;
                min-height: 32px;
            }}
            QPushButton:checked {{
                background-color: {Colors.ACCENT_WARM};
                color: #ffffff;
            }}
            """
        )
        self._rec_btn.toggled.connect(self._on_record_toggled)
        layout.addWidget(self._rec_btn)

        # --- Blink timer for recording animation ---
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._blink)
        self._blink_visible = True

        # --- Input device selector -------------------------------------------
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Input:"))
        self._device_combo = QComboBox()
        self._device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        dev_row.addWidget(self._device_combo)
        layout.addLayout(dev_row)

        # --- Level meter + duration ------------------------------------------
        meter_row = QHBoxLayout()
        self._meter = LEDMeter()
        meter_row.addWidget(self._meter)

        self._duration_label = QLabel("00:00.0")
        self._duration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._duration_label.setStyleSheet(
            f"font-family: monospace; font-size: 18px; color: {Colors.TEXT_PRIMARY};"
        )
        meter_row.addWidget(self._duration_label, 1)
        layout.addLayout(meter_row)

        # --- Waveform preview ------------------------------------------------
        self._waveform = _WaveformPreview()
        layout.addWidget(self._waveform)

        # --- Monitor toggle --------------------------------------------------
        self._monitor_cb = QCheckBox("Monitor input")
        layout.addWidget(self._monitor_cb)

        self._recording = False

    # -- public API ----------------------------------------------------------

    def set_recording_state(self, recording: bool) -> None:
        self._recording = recording
        self._rec_btn.blockSignals(True)
        self._rec_btn.setChecked(recording)
        self._rec_btn.blockSignals(False)
        if recording:
            self._blink_timer.start()
        else:
            self._blink_timer.stop()
            self._rec_btn.setText("● REC")
            self._blink_visible = True

    def update_level(self, level: float) -> None:
        self._meter.set_levels(level, level)
        if self._recording:
            self._waveform.push_sample(level * 2.0 - 1.0)

    def set_devices(self, devices: list[str]) -> None:
        self._device_combo.clear()
        self._device_combo.addItems(devices)

    def update_duration(self, seconds: float) -> None:
        mins = int(seconds) // 60
        secs = seconds - mins * 60
        self._duration_label.setText(f"{mins:02d}:{secs:04.1f}")

    # -- internals -----------------------------------------------------------

    def _on_record_toggled(self, checked: bool) -> None:
        self.set_recording_state(checked)
        if not checked:
            self._waveform.clear()
        self.recordToggled.emit(checked)

    def _blink(self) -> None:
        self._blink_visible = not self._blink_visible
        self._rec_btn.setText("● REC" if self._blink_visible else "  REC")
