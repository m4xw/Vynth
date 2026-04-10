"""Waveform view with editing tools — selection, loop points, right-click menu."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from vynth.sampler.sample import Sample
from vynth.ui.theme import Colors
from vynth.ui.widgets.waveform_view import WaveformView


class WaveformEditor(QWidget):
    """Waveform view with editing tools — selection, loop points, right-click menu."""

    selectionChanged = pyqtSignal(int, int)
    loopPointsChanged = pyqtSignal(int, int)
    editRequested = pyqtSignal(str)  # edit action name

    class _Mode:
        SELECTION = "selection"
        LOOP = "loop"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sample: Sample | None = None
        self._mode = self._Mode.SELECTION
        self._loop_start: int = 0
        self._loop_end: int = 0

        self._setup_ui()
        self._connect_signals()

    # ── UI setup ──────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)
        self._toolbar.setStyleSheet(
            f"QToolBar {{ background: {Colors.BG_MEDIUM}; border-bottom: 1px solid {Colors.BORDER}; spacing: 4px; padding: 2px; }}"
        )

        self._btn_zoom_in = QPushButton("Zoom +")
        self._btn_zoom_out = QPushButton("Zoom −")
        self._btn_zoom_fit = QPushButton("Fit")
        self._btn_zoom_sel = QPushButton("Sel")

        self._btn_mode_select = QPushButton("Select")
        self._btn_mode_select.setCheckable(True)
        self._btn_mode_select.setChecked(True)
        self._btn_mode_loop = QPushButton("Loop")
        self._btn_mode_loop.setCheckable(True)

        for btn in (self._btn_zoom_in, self._btn_zoom_out, self._btn_zoom_fit, self._btn_zoom_sel):
            btn.setFixedHeight(24)
            btn.setMinimumWidth(40)
            self._toolbar.addWidget(btn)

        self._toolbar.addSeparator()

        for btn in (self._btn_mode_select, self._btn_mode_loop):
            btn.setFixedHeight(24)
            btn.setMinimumWidth(50)
            self._toolbar.addWidget(btn)

        layout.addWidget(self._toolbar)

        # Waveform view
        self._waveform = WaveformView(self)
        layout.addWidget(self._waveform, stretch=1)

        # Info bar
        self._info_bar = QWidget()
        self._info_bar.setStyleSheet(
            f"background: {Colors.BG_DARK}; border-top: 1px solid {Colors.BORDER}; padding: 2px 6px;"
        )
        info_layout = QHBoxLayout(self._info_bar)
        info_layout.setContentsMargins(6, 2, 6, 2)
        info_layout.setSpacing(16)

        self._lbl_total = QLabel("Duration: —")
        self._lbl_selection = QLabel("Selection: —")
        self._lbl_sr = QLabel("SR: —")

        for lbl in (self._lbl_total, self._lbl_selection, self._lbl_sr):
            lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; background: transparent;")
            info_layout.addWidget(lbl)

        info_layout.addStretch()
        layout.addWidget(self._info_bar)

        # Context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _connect_signals(self) -> None:
        self._btn_zoom_in.clicked.connect(self._zoom_in)
        self._btn_zoom_out.clicked.connect(self._zoom_out)
        self._btn_zoom_fit.clicked.connect(self._waveform.zoom_to_fit)
        self._btn_zoom_sel.clicked.connect(self._waveform.zoom_to_selection)

        self._btn_mode_select.clicked.connect(lambda: self._set_mode(self._Mode.SELECTION))
        self._btn_mode_loop.clicked.connect(lambda: self._set_mode(self._Mode.LOOP))

        self._waveform.selectionChanged.connect(self._on_selection_changed)
        self._waveform.positionClicked.connect(self._on_position_clicked)

    # ── Public API ────────────────────────────────────────────────────

    def set_sample(self, sample: Sample) -> None:
        """Load sample data into the editor."""
        self._sample = sample
        self._waveform.set_data(sample.data, sample.sample_rate)

        if sample.loop.enabled:
            self._loop_start = sample.loop.start
            self._loop_end = sample.loop.end
            self._waveform.set_loop_points(self._loop_start, self._loop_end)

        self._update_info()

    def get_selection(self) -> tuple[int, int]:
        """Get current selection in frames."""
        sel = self._waveform.get_selection()
        if sel is None:
            return (0, 0)
        return sel

    def get_loop_points(self) -> tuple[int, int]:
        """Return current loop start and end frames."""
        return (self._loop_start, self._loop_end)

    def set_loop_points(self, start: int, end: int) -> None:
        """Update loop markers."""
        self._loop_start = start
        self._loop_end = end
        self._waveform.set_loop_points(start, end)
        self.loopPointsChanged.emit(start, end)

    # ── Mode switching ────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._btn_mode_select.setChecked(mode == self._Mode.SELECTION)
        self._btn_mode_loop.setChecked(mode == self._Mode.LOOP)

    # ── Zoom helpers ──────────────────────────────────────────────────

    def _zoom_in(self) -> None:
        vr = self._waveform._plot_widget.viewRange()[0]
        center = (vr[0] + vr[1]) / 2.0
        half = (vr[1] - vr[0]) / 2.0 * 0.5
        self._waveform._plot_widget.setXRange(center - half, center + half, padding=0)

    def _zoom_out(self) -> None:
        vr = self._waveform._plot_widget.viewRange()[0]
        center = (vr[0] + vr[1]) / 2.0
        half = (vr[1] - vr[0]) / 2.0 * 2.0
        duration = 0.0
        if self._sample is not None:
            duration = self._sample.duration_s
        new_min = max(0, center - half)
        new_max = min(duration, center + half) if duration > 0 else center + half
        self._waveform._plot_widget.setXRange(new_min, new_max, padding=0)

    # ── Selection / position callbacks ────────────────────────────────

    def _on_selection_changed(self, start: int, end: int) -> None:
        if self._mode == self._Mode.LOOP:
            self.set_loop_points(start, end)
        else:
            self.selectionChanged.emit(start, end)
        self._update_info()

    def _on_position_clicked(self, frame: int) -> None:
        if self._mode == self._Mode.LOOP:
            pass  # Position click doesn't affect loop mode
        self._update_info()

    # ── Info bar ──────────────────────────────────────────────────────

    def _update_info(self) -> None:
        if self._sample is None:
            self._lbl_total.setText("Duration: —")
            self._lbl_selection.setText("Selection: —")
            self._lbl_sr.setText("SR: —")
            return

        self._lbl_total.setText(f"Duration: {self._sample.duration_s:.3f}s")
        self._lbl_sr.setText(f"SR: {self._sample.sample_rate} Hz")

        sel = self._waveform.get_selection()
        if sel is not None:
            s, e = sel
            dur = (e - s) / self._sample.sample_rate
            self._lbl_selection.setText(f"Selection: {dur:.3f}s ({e - s} frames)")
        else:
            self._lbl_selection.setText("Selection: —")

    # ── Context menu ──────────────────────────────────────────────────

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: {Colors.BG_MEDIUM}; border: 1px solid {Colors.BORDER}; padding: 4px; }}"
            f"QMenu::item {{ padding: 4px 20px; }}"
            f"QMenu::item:selected {{ background: {Colors.ACCENT_PRIMARY}; }}"
            f"QMenu::separator {{ height: 1px; background: {Colors.BORDER}; margin: 4px 0; }}"
        )

        edit_actions = [
            ("Trim to Selection", "trim"),
            ("Normalize", "normalize"),
            ("Reverse Selection", "reverse"),
            ("Fade In", "fade_in"),
            ("Fade Out", "fade_out"),
        ]
        for label, action_name in edit_actions:
            action = menu.addAction(label)
            action.triggered.connect(lambda checked, n=action_name: self.editRequested.emit(n))

        menu.addSeparator()

        loop_actions = [
            ("Set Loop Start", "set_loop_start"),
            ("Set Loop End", "set_loop_end"),
            ("Clear Loop", "clear_loop"),
        ]
        for label, action_name in loop_actions:
            action = menu.addAction(label)
            action.triggered.connect(lambda checked, n=action_name: self._handle_loop_action(n))

        menu.addSeparator()

        zoom_sel = menu.addAction("Zoom to Selection")
        zoom_sel.triggered.connect(self._waveform.zoom_to_selection)
        zoom_fit = menu.addAction("Zoom to Fit")
        zoom_fit.triggered.connect(self._waveform.zoom_to_fit)

        menu.exec(self.mapToGlobal(pos))

    def _handle_loop_action(self, action: str) -> None:
        if action == "set_loop_start":
            sel = self._waveform.get_selection()
            if sel is not None:
                self.set_loop_points(sel[0], self._loop_end)
        elif action == "set_loop_end":
            sel = self._waveform.get_selection()
            if sel is not None:
                self.set_loop_points(self._loop_start, sel[1])
        elif action == "clear_loop":
            self._loop_start = 0
            self._loop_end = 0
            self._waveform._loop_start_line.hide()
            self._waveform._loop_end_line.hide()
            self._waveform._loop_region.hide()
            self.loopPointsChanged.emit(0, 0)
