"""Simple about dialog for Vynth."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QWidget,
)

from vynth.ui.theme import Colors


class AboutDialog(QDialog):
    """Simple about dialog."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About Vynth")
        self.setFixedSize(340, 260)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Title ---
        title = QLabel("Vynth")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 32px; font-weight: bold; color: {Colors.ACCENT_PRIMARY};"
        )
        layout.addWidget(title)

        # --- Version ---
        version = QLabel("Version 1.0.0")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 13px;")
        layout.addWidget(version)

        # --- Description ---
        desc = QLabel("Professional Voice Sampler Synthesizer")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 13px;")
        layout.addWidget(desc)

        # --- Credits ---
        credits = QLabel(
            "Built with PyQt6 &amp; NumPy<br>"
            "DSP Engine by Vynth Team<br>"
            "© 2026 Vynth Project"
        )
        credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credits.setWordWrap(True)
        credits.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 11px;")
        layout.addWidget(credits)

        layout.addStretch()

        # --- Close button ---
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)
