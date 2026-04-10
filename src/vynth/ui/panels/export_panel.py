"""Export dialog for rendering audio to WAV."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QProgressBar, QFileDialog, QGroupBox,
    QDialogButtonBox, QWidget,
)

from vynth.ui.theme import Colors


class ExportPanel(QDialog):
    """Export dialog for rendering audio to WAV."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Audio")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- File path -------------------------------------------------------
        path_group = QGroupBox("Output")
        path_layout = QHBoxLayout(path_group)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select output file…")
        path_layout.addWidget(self._path_edit, 1)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(browse_btn)
        layout.addWidget(path_group)

        # --- Format settings -------------------------------------------------
        fmt_group = QGroupBox("Format")
        fmt_layout = QVBoxLayout(fmt_group)

        # Sample rate
        sr_row = QHBoxLayout()
        sr_row.addWidget(QLabel("Sample Rate:"))
        self._sr_combo = QComboBox()
        self._sr_combo.addItems(["44100", "48000"])
        self._sr_combo.currentTextChanged.connect(self._update_estimate)
        sr_row.addWidget(self._sr_combo)
        fmt_layout.addLayout(sr_row)

        # Bit depth
        bits_row = QHBoxLayout()
        bits_row.addWidget(QLabel("Bit Depth:"))
        self._bits_combo = QComboBox()
        self._bits_combo.addItems(["16", "24"])
        self._bits_combo.currentTextChanged.connect(self._update_estimate)
        bits_row.addWidget(self._bits_combo)
        fmt_layout.addLayout(bits_row)

        # Format (read-only for V1)
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItem("WAV")
        self._format_combo.setEnabled(False)
        format_row.addWidget(self._format_combo)
        fmt_layout.addLayout(format_row)

        layout.addWidget(fmt_group)

        # --- File size estimate ----------------------------------------------
        self._estimate_label = QLabel("Estimated size: —")
        self._estimate_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(self._estimate_label)

        # --- Progress --------------------------------------------------------
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # --- Buttons ---------------------------------------------------------
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._duration_s: float = 0.0
        self._channels: int = 2
        self._update_estimate()

    # -- public API ----------------------------------------------------------

    def get_settings(self) -> dict:
        return {
            "path": self._path_edit.text(),
            "sr": int(self._sr_combo.currentText()),
            "bits": int(self._bits_combo.currentText()),
        }

    def set_duration(self, seconds: float, channels: int = 2) -> None:
        self._duration_s = seconds
        self._channels = channels
        self._update_estimate()

    def set_progress(self, percent: int) -> None:
        self._progress.setVisible(True)
        self._progress.setValue(percent)

    # -- internals -----------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Audio", "", "WAV Files (*.wav)"
        )
        if path:
            self._path_edit.setText(path)

    def _update_estimate(self) -> None:
        if self._duration_s <= 0:
            self._estimate_label.setText("Estimated size: —")
            return
        sr = int(self._sr_combo.currentText())
        bits = int(self._bits_combo.currentText())
        bytes_per_sec = sr * self._channels * (bits // 8)
        total = bytes_per_sec * self._duration_s
        if total < 1_048_576:
            self._estimate_label.setText(f"Estimated size: {total / 1024:.1f} KB")
        else:
            self._estimate_label.setText(f"Estimated size: {total / 1_048_576:.1f} MB")
