"""Audio and MIDI device settings dialog."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QComboBox, QPushButton, QTabWidget, QWidget,
    QDialogButtonBox, QSpinBox,
)

from vynth.ui.theme import Colors


class SettingsDialog(QDialog):
    """Audio and MIDI device settings."""

    applyRequested = pyqtSignal(dict)
    refreshRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # ── Audio tab ─────────────────────────────────────────
        audio_tab = QWidget()
        audio_form = QFormLayout(audio_tab)
        audio_form.setSpacing(8)

        self._output_device = QComboBox()
        audio_form.addRow("Output Device:", self._output_device)

        self._sample_rate = QComboBox()
        self._sample_rate.addItems(["44100", "48000"])
        self._sample_rate.currentTextChanged.connect(self._update_latency)
        audio_form.addRow("Sample Rate:", self._sample_rate)

        self._buffer_size = QComboBox()
        self._buffer_size.addItems(["128", "256", "512", "1024", "2048"])
        self._buffer_size.setCurrentText("512")
        self._buffer_size.currentTextChanged.connect(self._update_latency)
        audio_form.addRow("Buffer Size:", self._buffer_size)

        self._latency_label = QLabel()
        self._latency_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        audio_form.addRow("Latency:", self._latency_label)
        self._update_latency()

        tabs.addTab(audio_tab, "Audio")

        # ── MIDI tab ──────────────────────────────────────────
        midi_tab = QWidget()
        midi_form = QFormLayout(midi_tab)
        midi_form.setSpacing(8)

        self._midi_device = QComboBox()
        midi_form.addRow("Input Device:", self._midi_device)

        self._midi_channel = QComboBox()
        self._midi_channel.addItem("All")
        for ch in range(1, 17):
            self._midi_channel.addItem(str(ch))
        midi_form.addRow("Channel Filter:", self._midi_channel)

        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self._refresh_devices)
        midi_form.addRow("", refresh_btn)

        tabs.addTab(midi_tab, "MIDI")

        layout.addWidget(tabs)

        # ── Buttons ───────────────────────────────────────────
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        self._apply_btn = btn_box.button(QDialogButtonBox.StandardButton.Apply)
        if self._apply_btn:
            self._apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(btn_box)

    # -- public API ----------------------------------------------------------

    def get_settings(self) -> dict:
        return {
            "output_device": self._output_device.currentText(),
            "output_device_index": self._output_device.currentIndex(),
            "sample_rate": int(self._sample_rate.currentText()),
            "buffer_size": int(self._buffer_size.currentText()),
            "midi_device": self._midi_device.currentText(),
            "midi_device_index": self._midi_device.currentIndex(),
            "midi_channel": self._midi_channel.currentText(),
            "midi_channel_index": self._midi_channel.currentIndex(),
        }

    def set_audio_devices(self, devices: list[str]) -> None:
        self._output_device.clear()
        self._output_device.addItem("(Default audio device)")
        self._output_device.addItems(devices)

    def set_midi_devices(self, devices: list[str]) -> None:
        self._midi_device.clear()
        self._midi_device.addItem("(No MIDI device)")
        self._midi_device.addItems(devices)

    def set_midi_settings(self, *, midi_device: int | None, midi_channel: int | None) -> None:
        device_index = 0 if midi_device is None else midi_device + 1
        if 0 <= device_index < self._midi_device.count():
            self._midi_device.setCurrentIndex(device_index)

        channel_index = 0 if midi_channel is None else midi_channel + 1
        if 0 <= channel_index < self._midi_channel.count():
            self._midi_channel.setCurrentIndex(channel_index)

    def set_audio_settings(
        self,
        *,
        sample_rate: int,
        buffer_size: int,
        output_device: int | None,
    ) -> None:
        self._sample_rate.setCurrentText(str(sample_rate))
        self._buffer_size.setCurrentText(str(buffer_size))

        device_index = 0 if output_device is None else output_device + 1
        if 0 <= device_index < self._output_device.count():
            self._output_device.setCurrentIndex(device_index)

        self._update_latency()

    # -- internals -----------------------------------------------------------

    def _update_latency(self) -> None:
        try:
            sr = int(self._sample_rate.currentText())
            buf = int(self._buffer_size.currentText())
            ms = buf / sr * 1000.0
            self._latency_label.setText(f"{ms:.1f} ms")
        except (ValueError, ZeroDivisionError):
            self._latency_label.setText("—")

    def _refresh_devices(self) -> None:
        self.refreshRequested.emit()

    def _on_apply(self) -> None:
        self.applyRequested.emit(self.get_settings())
