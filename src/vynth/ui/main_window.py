"""Main application window for Vynth — Voice Sampler Synthesizer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from vynth.engine.audio_engine import AudioEngine
    from vynth.engine.midi_engine import MidiEngine
    from vynth.engine.recorder import Recorder
    from vynth.sampler.sample_manager import SampleManager

from vynth.ui.theme import Colors

logger = logging.getLogger(__name__)


def _placeholder(name: str, accent: str = Colors.ACCENT_PRIMARY) -> QLabel:
    """Create a styled placeholder label for a panel that will be replaced later."""
    label = QLabel(name)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet(
        f"QLabel {{ color: {accent}; font-size: 18px; font-weight: bold; "
        f"background-color: {Colors.BG_DARK}; border: 1px dashed {Colors.BORDER_LIGHT}; "
        f"border-radius: 6px; }}"
    )
    label.setMinimumHeight(120)
    return label


class MainWindow(QMainWindow):
    """Main Vynth window with dockable panels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._audio_engine: AudioEngine | None = None
        self._midi_engine: MidiEngine | None = None
        self._recorder: Recorder | None = None
        self._sample_manager: SampleManager | None = None

        self.setWindowTitle("Vynth \u2014 Voice Sampler Synthesizer")
        self.setMinimumSize(800, 500)

        self._create_menus()
        self._create_toolbar()
        self._create_central_widget()
        self._create_dock_widgets()
        self._create_status_bar()

        # Fit window to available screen space and center
        self._fit_to_screen()

        # Periodic status-bar refresh (CPU / voice count)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(500)

    # ── Engine wiring ────────────────────────────────────────

    def set_engine(
        self,
        audio_engine: AudioEngine,
        midi_engine: MidiEngine,
        recorder: Recorder,
        sample_manager: SampleManager,
    ) -> None:
        """Wire engine references into the window and its child panels."""
        self._audio_engine = audio_engine
        self._midi_engine = midi_engine
        self._recorder = recorder
        self._sample_manager = sample_manager

    # ── Menu bar ─────────────────────────────────────────────

    def _create_menus(self) -> None:
        menu_bar = self.menuBar()

        # File
        file_menu = menu_bar.addMenu("&File")
        self._act_new = file_menu.addAction("New Session")
        self._act_new.setShortcut(QKeySequence("Ctrl+N"))

        self._act_load_session = file_menu.addAction("Load Session\u2026")
        self._act_load_session.setShortcut(QKeySequence("Ctrl+Shift+O"))

        self._act_load = file_menu.addAction("Load Sample\u2026")
        self._act_load.setShortcut(QKeySequence("Ctrl+O"))

        self._act_save = file_menu.addAction("Save Session")
        self._act_save.setShortcut(QKeySequence("Ctrl+S"))

        self._act_export = file_menu.addAction("Export WAV\u2026")
        self._act_export.setShortcut(QKeySequence("Ctrl+E"))

        file_menu.addSeparator()
        self._act_exit = file_menu.addAction("Exit")
        self._act_exit.setShortcut(QKeySequence("Alt+F4"))
        self._act_exit.triggered.connect(self.close)

        # Edit
        edit_menu = menu_bar.addMenu("&Edit")
        self._act_undo = edit_menu.addAction("Undo")
        self._act_undo.setShortcut(QKeySequence("Ctrl+Z"))

        self._act_redo = edit_menu.addAction("Redo")
        self._act_redo.setShortcut(QKeySequence("Ctrl+Shift+Z"))

        edit_menu.addSeparator()
        self._act_prefs = edit_menu.addAction("Preferences\u2026")

        # View — populated after dock widgets exist
        self._view_menu = menu_bar.addMenu("&View")

        # Help
        help_menu = menu_bar.addMenu("&Help")
        self._act_about = help_menu.addAction("About Vynth")

    # ── Toolbar ──────────────────────────────────────────────

    def _create_toolbar(self) -> None:
        tb = QToolBar("Transport")
        tb.setObjectName("toolbar_transport")
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        self._act_record = QAction("\u23fa Record", self)
        self._act_record.setShortcut(QKeySequence("Space"))
        self._act_record.setToolTip("Toggle recording (Space)")
        tb.addAction(self._act_record)

        self._act_stop = QAction("\u23f9 Stop", self)
        self._act_stop.setToolTip("Stop playback / recording")
        tb.addAction(self._act_stop)

        self._act_play = QAction("\u25b6 Play", self)
        self._act_play.setToolTip("Play selected sample")
        tb.addAction(self._act_play)

        tb.addSeparator()

        # MIDI device selector
        midi_label = QLabel(" MIDI: ")
        midi_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        tb.addWidget(midi_label)
        self._combo_midi = QComboBox()
        self._combo_midi.setMinimumWidth(180)
        self._combo_midi.setToolTip("Select MIDI input device")
        self._combo_midi.addItem("(No MIDI device)")
        tb.addWidget(self._combo_midi)

        tb.addSeparator()

        # Audio device selector
        audio_label = QLabel(" Audio: ")
        audio_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        tb.addWidget(audio_label)
        self._combo_audio = QComboBox()
        self._combo_audio.setMinimumWidth(180)
        self._combo_audio.setToolTip("Select audio output device")
        self._combo_audio.addItem("(Default audio device)")
        tb.addWidget(self._combo_audio)

    # ── Central widget ───────────────────────────────────────

    def _create_central_widget(self) -> None:
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(3)

        self.waveform_editor = _placeholder("Waveform Editor", Colors.ACCENT_PRIMARY)
        self.waveform_editor.setMinimumHeight(200)
        splitter.addWidget(self.waveform_editor)

        self.spectrum_analyzer = _placeholder("Spectrum Analyzer", Colors.ACCENT_SECONDARY)
        self.spectrum_analyzer.setMinimumHeight(120)
        splitter.addWidget(self.spectrum_analyzer)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    # ── Dock widgets ─────────────────────────────────────────

    def _create_dock_widgets(self) -> None:
        # -- Left: Sample Browser --
        self.dock_sample_browser = self._make_dock(
            "Sample Browser",
            _placeholder("Sample Browser", Colors.ACCENT_PRIMARY),
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.dock_sample_browser.setMinimumWidth(220)

        # -- Right: Effects Rack --
        self.dock_effects = self._make_dock(
            "Effects Rack",
            _placeholder("Effects Rack", Colors.ACCENT_WARM),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.dock_effects.setMinimumWidth(240)

        # -- Bottom: Recorder + MIDI Keyboard (tabbed) --
        self.dock_recorder = self._make_dock(
            "Recorder",
            _placeholder("Recorder Panel", Colors.METER_GREEN),
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.dock_midi_keyboard = self._make_dock(
            "MIDI Keyboard",
            _placeholder("MIDI Keyboard Display", Colors.ACCENT_GOLD),
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.tabifyDockWidget(self.dock_recorder, self.dock_midi_keyboard)
        self.dock_recorder.raise_()

        # -- Bottom-right: Mixer --
        self.dock_mixer = self._make_dock(
            "Mixer",
            _placeholder("Mixer", Colors.ACCENT_SECONDARY),
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.dock_mixer.setMinimumWidth(260)

        # Populate View menu with toggle actions
        for dock in (
            self.dock_sample_browser,
            self.dock_effects,
            self.dock_recorder,
            self.dock_midi_keyboard,
            self.dock_mixer,
        ):
            self._view_menu.addAction(dock.toggleViewAction())

    def _make_dock(
        self, title: str, widget: QWidget, area: Qt.DockWidgetArea
    ) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower().replace(' ', '_')}")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    # ── Status bar ───────────────────────────────────────────

    def _create_status_bar(self) -> None:
        sb = self.statusBar()

        self._lbl_sample_info = QLabel("No sample loaded")
        sb.addWidget(self._lbl_sample_info, stretch=1)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet(f"color: {Colors.BORDER};")
        sb.addWidget(sep1)

        self._lbl_midi_activity = QLabel("\u25cf MIDI")
        self._lbl_midi_activity.setStyleSheet(f"color: {Colors.TEXT_DIM};")
        self._lbl_midi_activity.setToolTip("MIDI activity indicator")
        sb.addWidget(self._lbl_midi_activity)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f"color: {Colors.BORDER};")
        sb.addWidget(sep2)

        self._lbl_cpu = QLabel("CPU: 0.0 %")
        self._lbl_cpu.setFixedWidth(100)
        sb.addWidget(self._lbl_cpu)

        self._lbl_voices = QLabel("Voices: 0")
        self._lbl_voices.setFixedWidth(80)
        sb.addWidget(self._lbl_voices)

    def _refresh_status(self) -> None:
        """Periodically update status bar fields from the engine."""
        if self._audio_engine is not None:
            cpu = getattr(self._audio_engine, "cpu_load", 0.0)
            voices = getattr(self._audio_engine, "active_voices", 0)
            self._lbl_cpu.setText(f"CPU: {cpu:.1f} %")
            self._lbl_voices.setText(f"Voices: {voices}")

    # ── Public helpers ───────────────────────────────────────

    def _fit_to_screen(self) -> None:
        """Resize and center the window within the available screen area."""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 800)
            return
        avail = screen.availableGeometry()
        # Use 90% of available space, but no larger than 1280×800
        w = min(1280, int(avail.width() * 0.9))
        h = min(800, int(avail.height() * 0.9))
        self.resize(w, h)
        # Center on screen
        x = avail.x() + (avail.width() - w) // 2
        y = avail.y() + (avail.height() - h) // 2
        self.move(x, y)

    def set_sample_info(self, text: str) -> None:
        """Update the sample info text in the status bar."""
        self._lbl_sample_info.setText(text)

    def flash_midi_activity(self) -> None:
        """Briefly highlight the MIDI indicator."""
        self._lbl_midi_activity.setStyleSheet(f"color: {Colors.METER_GREEN};")
        QTimer.singleShot(
            120,
            lambda: self._lbl_midi_activity.setStyleSheet(f"color: {Colors.TEXT_DIM};"),
        )

    # ── Close ────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802
        """Stop engines gracefully on window close."""
        self._status_timer.stop()
        try:
            if self._recorder is not None and self._recorder.is_recording:
                self._recorder.stop_recording()
            if self._midi_engine is not None:
                self._midi_engine.stop()
            if self._audio_engine is not None:
                self._audio_engine.stop()
        except Exception:
            logger.exception("Error while shutting down engines")
        super().closeEvent(event)
