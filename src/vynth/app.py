"""Application controller — wires UI, engine, and MIDI together."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from vynth.config import (
    BLOCK_SIZE,
    SAMPLE_RATE,
    SPECTRUM_FFT_SIZE,
    UI_FPS,
    SessionSettings,
)
from vynth.engine.audio_engine import AudioEngine
from vynth.engine.export import Exporter
from vynth.engine.midi_engine import MIDIEngine
from vynth.engine.recorder import Recorder
from vynth.sampler.sample import Sample
from vynth.sampler.sample_editor import SampleEditor
from vynth.sampler.sample_manager import SampleManager
from vynth.ui.dialogs.about_dialog import AboutDialog
from vynth.ui.dialogs.settings_dialog import SettingsDialog
from vynth.ui.main_window import MainWindow
from vynth.ui.panels.effects_rack import EffectsRack
from vynth.ui.panels.export_panel import ExportPanel
from vynth.ui.panels.mixer_panel import MixerPanel
from vynth.ui.panels.recorder_panel import RecorderPanel
from vynth.ui.panels.sample_browser import SampleBrowser
from vynth.ui.theme import apply_theme
from vynth.ui.widgets.midi_keyboard import MIDIKeyboardWidget
from vynth.ui.widgets.spectrum_view import SpectrumView
from vynth.ui.widgets.waveform_editor import WaveformEditor
from vynth.utils.thread_safe_queue import Command, CommandType

log = logging.getLogger(__name__)

_SESSION_FILE = "vynth_session.json"


class VynthApp:
    """Application controller — wires UI, engine, and MIDI together."""

    def __init__(self) -> None:
        self._qapp = QApplication(sys.argv)
        self._qapp.setApplicationName("Vynth")
        apply_theme(self._qapp)

        # ── settings ─────────────────────────────────────────────────────
        self._settings = SessionSettings()

        # ── engines ──────────────────────────────────────────────────────
        self._audio_engine = AudioEngine(self._settings)
        self._midi_engine = MIDIEngine(self._audio_engine.command_queue)
        self._recorder = Recorder(sample_rate=self._settings.sample_rate)
        self._sample_manager = SampleManager()
        self._sample_editor = SampleEditor()
        self._exporter = Exporter()

        # ── main window ──────────────────────────────────────────────────
        self._window = MainWindow()
        self._window.set_engine(
            self._audio_engine,
            self._midi_engine,
            self._recorder,
            self._sample_manager,
        )

        # ── real panels ──────────────────────────────────────────────────
        self._waveform_editor = WaveformEditor()
        self._spectrum_view = SpectrumView()
        self._recorder_panel = RecorderPanel()
        self._sample_browser = SampleBrowser()
        self._effects_rack = EffectsRack()
        self._mixer_panel = MixerPanel()
        self._midi_keyboard = MIDIKeyboardWidget()

        self._replace_placeholders()

        # ── signal wiring ────────────────────────────────────────────────
        self._wire_midi_signals()
        self._wire_recorder_signals()
        self._wire_sample_browser_signals()
        self._wire_effects_signals()
        self._wire_mixer_signals()
        self._wire_waveform_editor_signals()
        self._wire_menu_actions()

        # ── visualization timer ──────────────────────────────────────────
        interval_ms = max(1, int(1000 / UI_FPS))
        self._vis_timer = QTimer()
        self._vis_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._vis_timer.timeout.connect(self._on_vis_tick)
        self._vis_timer.start(interval_ms)

        # ── populate device combos ───────────────────────────────────────
        self._populate_audio_devices()
        self._populate_midi_devices()

        # ── start engines ────────────────────────────────────────────────
        self._start_audio_engine()
        self._start_midi_engine()

    # ── public ───────────────────────────────────────────────────────────

    def run(self) -> int:
        """Show the window and enter the Qt event loop."""
        self._window.show()
        return self._qapp.exec()

    # ── placeholder replacement ──────────────────────────────────────────

    def _replace_placeholders(self) -> None:
        """Swap the placeholder labels in MainWindow with real panels."""
        # Central area — waveform editor + spectrum
        splitter = self._window.centralWidget()
        splitter.replaceWidget(0, self._waveform_editor)
        splitter.replaceWidget(1, self._spectrum_view)

        # Docks
        self._window.dock_sample_browser.setWidget(self._sample_browser)
        self._window.dock_effects.setWidget(self._effects_rack)
        self._window.dock_recorder.setWidget(self._recorder_panel)
        self._window.dock_midi_keyboard.setWidget(self._midi_keyboard)
        self._window.dock_mixer.setWidget(self._mixer_panel)

    # ── MIDI → Engine → UI ───────────────────────────────────────────────

    def _wire_midi_signals(self) -> None:
        self._midi_engine.note_on_received.connect(self._on_midi_note_on)
        self._midi_engine.note_off_received.connect(self._on_midi_note_off)
        self._midi_engine.device_changed.connect(
            lambda name: self._window.statusBar().showMessage(
                f"MIDI: {name}", 3000
            )
        )
        self._midi_engine.devices_updated.connect(self._on_midi_devices_updated)

        # MIDI keyboard mouse clicks → audio engine
        self._midi_keyboard.notePressed.connect(self._on_midi_note_on)
        self._midi_keyboard.noteReleased.connect(self._on_midi_note_off)

        # Toolbar MIDI device combo
        self._window._combo_midi.currentIndexChanged.connect(
            self._on_midi_device_selected
        )

    def _on_midi_note_on(self, note: int, velocity: int) -> None:
        self._audio_engine.push_command(
            Command(type=CommandType.NOTE_ON, note=note, velocity=velocity)
        )
        self._midi_keyboard.highlight_note(note, velocity)
        self._window.flash_midi_activity()

    def _on_midi_note_off(self, note: int) -> None:
        self._audio_engine.push_command(
            Command(type=CommandType.NOTE_OFF, note=note)
        )
        self._midi_keyboard.release_note(note)

    def _on_midi_device_selected(self, index: int) -> None:
        if index <= 0:
            self._midi_engine.close_device()
            return
        # index 0 is the "(No MIDI device)" placeholder
        self._midi_engine.open_device(index - 1)

    def _on_midi_devices_updated(self, devices: list[str]) -> None:
        combo = self._window._combo_midi
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(No MIDI device)")
        combo.addItems(devices)
        # Restore previous selection if still available
        idx = combo.findText(current)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)

    # ── Recorder → UI ────────────────────────────────────────────────────

    def _wire_recorder_signals(self) -> None:
        self._recorder_panel.recordToggled.connect(self._on_record_toggled)
        self._recorder.level_updated.connect(self._recorder_panel.update_level)

    def _on_record_toggled(self, recording: bool) -> None:
        if recording:
            try:
                self._recorder.start_recording()
            except Exception:
                log.exception("Failed to start recording")
                QMessageBox.warning(
                    self._window,
                    "Recording Error",
                    "Could not open the audio input device.\n"
                    "Check your recording device in Settings.",
                )
        else:
            sample = self._recorder.stop_recording()
            if sample.length > 0:
                self._sample_manager.add_sample(sample)
                self._waveform_editor.set_sample(sample)
                self._window.set_sample_info(
                    f"{sample.name}  \u2014  {sample.duration_s:.2f}s  "
                    f"{sample.sample_rate} Hz  {sample.channels}ch"
                )

    # ── Sample Browser → Engine ──────────────────────────────────────────

    def _wire_sample_browser_signals(self) -> None:
        self._sample_browser.sampleSelected.connect(self._on_sample_selected)
        self._sample_manager.sample_added.connect(self._on_sample_added_to_browser)
        self._sample_manager.sample_removed.connect(
            self._sample_browser.remove_sample
        )

    def _on_sample_added_to_browser(self, name: str) -> None:
        sample = self._sample_manager.get_sample(name)
        if sample is None:
            return
        self._sample_browser.add_sample(
            name, sample.duration_s, sample.sample_rate, sample.channels,
        )

    def _on_sample_selected(self, name: str) -> None:
        self._sample_manager.select_sample(name)
        sample = self._sample_manager.get_sample(name)
        if sample is None:
            return
        # Push to audio engine
        self._audio_engine.push_command(
            Command(type=CommandType.SET_SAMPLE, data=sample)
        )
        # Update UI
        self._waveform_editor.set_sample(sample)
        self._window.set_sample_info(
            f"{sample.name}  \u2014  {sample.duration_s:.2f}s  "
            f"{sample.sample_rate} Hz  {sample.channels}ch"
        )

    # ── Effects Rack → Engine ────────────────────────────────────────────

    def _wire_effects_signals(self) -> None:
        self._effects_rack.paramChanged.connect(self._on_effect_param_changed)

    def _on_effect_param_changed(self, name: str, value: float) -> None:
        self._audio_engine.push_command(
            Command(
                type=CommandType.PARAM_CHANGE,
                param_name=name,
                param_value=value,
            )
        )

    # ── Mixer → Engine ───────────────────────────────────────────────────

    def _wire_mixer_signals(self) -> None:
        self._mixer_panel.volumeChanged.connect(
            self._audio_engine.set_master_volume
        )

    # ── Waveform Editor → Sample Editor ──────────────────────────────────

    def _wire_waveform_editor_signals(self) -> None:
        self._waveform_editor.editRequested.connect(self._on_edit_requested)

    def _on_edit_requested(self, action: str) -> None:
        sample = self._sample_manager.get_selected()
        if sample is None:
            return

        try:
            match action:
                case "trim":
                    start, end = self._waveform_editor.get_selection()
                    if start >= end:
                        return
                    new_sample = self._sample_editor.trim(sample, start, end)
                case "normalize":
                    new_sample = self._sample_editor.normalize(sample)
                case "reverse":
                    new_sample = self._sample_editor.reverse(sample)
                case "fade_in":
                    new_sample = self._sample_editor.fade_in(sample, 50.0)
                case "fade_out":
                    new_sample = self._sample_editor.fade_out(sample, 50.0)
                case _:
                    log.warning("Unknown edit action: %s", action)
                    return
        except Exception:
            log.exception("Edit action '%s' failed", action)
            QMessageBox.warning(
                self._window, "Edit Error",
                f"Could not apply '{action}' to the sample.",
            )
            return

        # Replace sample in manager and refresh UI
        self._sample_manager.remove_sample(sample.name)
        self._sample_manager.add_sample(new_sample)
        self._sample_manager.select_sample(new_sample.name)
        self._waveform_editor.set_sample(new_sample)
        self._audio_engine.push_command(
            Command(type=CommandType.SET_SAMPLE, data=new_sample)
        )

    # ── Visualization timer ──────────────────────────────────────────────

    def _on_vis_tick(self) -> None:
        if not self._audio_engine.is_running:
            return

        # Peak levels → mixer meter
        left, right = self._audio_engine.read_peak_levels()
        self._mixer_panel.set_levels(left, right)

        # Voice count → mixer
        voices = self._audio_engine.active_voice_count
        self._mixer_panel.set_voice_count(voices)

        # Visualization buffer → spectrum analyzer
        buf = self._audio_engine.get_visualization_buffer(SPECTRUM_FFT_SIZE)
        if buf is not None and buf.size > 0:
            mono = buf.mean(axis=1) if buf.ndim == 2 else buf
            self._spectrum_view.push_audio_block(mono)

    # ── Menu actions ─────────────────────────────────────────────────────

    def _wire_menu_actions(self) -> None:
        w = self._window
        w._act_load.triggered.connect(self._on_load_sample)
        w._act_export.triggered.connect(self._on_export)
        w._act_save.triggered.connect(self._on_save_session)
        w._act_new.triggered.connect(self._on_new_session)
        w._act_prefs.triggered.connect(self._on_preferences)
        w._act_about.triggered.connect(self._on_about)

        # Toolbar transport — toggle the record button which emits recordToggled
        w._act_record.triggered.connect(
            lambda: self._recorder_panel._rec_btn.toggle()
        )
        w._act_play.triggered.connect(self._on_play_sample)
        w._act_stop.triggered.connect(self._on_stop)

        # Audio device combo
        w._combo_audio.currentIndexChanged.connect(self._on_audio_device_selected)

    def _on_load_sample(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self._window,
            "Load Samples",
            "",
            "Audio Files (*.wav *.flac *.ogg *.aiff *.aif);;All Files (*)",
        )
        for path in paths:
            try:
                sample = self._sample_manager.load_sample(path)
                if self._settings.recent_samples.count(path) == 0:
                    self._settings.recent_samples.append(path)
            except Exception:
                log.exception("Failed to load %s", path)
                QMessageBox.warning(
                    self._window,
                    "Load Error",
                    f"Could not load:\n{path}",
                )

    def _on_export(self) -> None:
        sample = self._sample_manager.get_selected()
        if sample is None:
            QMessageBox.information(
                self._window, "Export", "No sample selected to export."
            )
            return

        dlg = ExportPanel(self._window)
        dlg.set_duration(sample.duration_s, sample.channels)
        if dlg.exec() != ExportPanel.DialogCode.Accepted:
            return

        export_cfg = dlg.get_settings()
        path = export_cfg["path"]
        if not path:
            return

        try:
            self._exporter.export_sample(
                sample,
                path,
                sample_rate=export_cfg["sr"],
                bit_depth=export_cfg["bits"],
            )
            QMessageBox.information(
                self._window, "Export Complete",
                f"Exported to:\n{path}",
            )
        except Exception:
            log.exception("Export failed")
            QMessageBox.critical(
                self._window, "Export Error",
                f"Failed to export to:\n{path}",
            )

    def _on_save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self._window, "Save Session", _SESSION_FILE,
            "Vynth Session (*.json)",
        )
        if not path:
            return
        data = {
            "master_volume": self._settings.master_volume,
            "sample_rate": self._settings.sample_rate,
            "block_size": self._settings.block_size,
            "samples": [
                s.file_path
                for s in (
                    self._sample_manager.get_sample(n)
                    for n in self._sample_manager.get_names()
                )
                if s is not None and s.file_path
            ],
        }
        try:
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._window.statusBar().showMessage(f"Session saved: {path}", 3000)
        except Exception:
            log.exception("Failed to save session")
            QMessageBox.warning(
                self._window, "Save Error", f"Could not save session to:\n{path}"
            )

    def _on_new_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._window, "Load Session", "",
            "Vynth Session (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            log.exception("Failed to read session file")
            QMessageBox.warning(
                self._window, "Load Error", f"Could not read:\n{path}"
            )
            return

        # Apply settings
        self._settings.master_volume = data.get(
            "master_volume", self._settings.master_volume
        )
        self._audio_engine.set_master_volume(self._settings.master_volume)

        # Load samples
        for sample_path in data.get("samples", []):
            if Path(sample_path).exists():
                try:
                    self._sample_manager.load_sample(sample_path)
                except Exception:
                    log.warning("Could not reload sample: %s", sample_path)

        self._window.statusBar().showMessage(f"Session loaded: {path}", 3000)

    def _on_preferences(self) -> None:
        dlg = SettingsDialog(self._window)
        # Populate device lists
        audio_devs = [d["name"] for d in AudioEngine.device_list()]
        dlg.set_audio_devices(audio_devs)
        midi_devs = self._midi_engine.get_devices()
        dlg.set_midi_devices(midi_devs)

        if dlg.exec() != SettingsDialog.DialogCode.Accepted:
            return

        cfg = dlg.get_settings()
        # Apply audio settings
        new_sr = cfg.get("sample_rate", self._settings.sample_rate)
        new_bs = cfg.get("buffer_size", self._settings.block_size)
        if new_sr != self._settings.sample_rate or new_bs != self._settings.block_size:
            self._settings.sample_rate = new_sr
            self._settings.block_size = new_bs
            # Restart audio engine with new settings
            self._audio_engine.stop()
            self._start_audio_engine()

    def _on_about(self) -> None:
        AboutDialog(self._window).exec()

    def _on_play_sample(self) -> None:
        sample = self._sample_manager.get_selected()
        if sample is None:
            return
        # Trigger middle-C note-on so the voice allocator plays the sample
        self._audio_engine.push_command(
            Command(type=CommandType.NOTE_ON, note=60, velocity=100)
        )

    def _on_stop(self) -> None:
        self._audio_engine.push_command(
            Command(type=CommandType.ALL_NOTES_OFF)
        )
        if self._recorder.is_recording:
            self._recorder_panel._rec_btn.setChecked(False)

    # ── Device management ────────────────────────────────────────────────

    def _populate_audio_devices(self) -> None:
        combo = self._window._combo_audio
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(Default audio device)")
        try:
            for dev in AudioEngine.device_list():
                combo.addItem(dev["name"])
        except Exception:
            log.warning("Could not enumerate audio devices")
        combo.blockSignals(False)

    def _populate_midi_devices(self) -> None:
        combo = self._window._combo_midi
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(No MIDI device)")
        try:
            for name in self._midi_engine.get_devices():
                combo.addItem(name)
        except Exception:
            log.warning("Could not enumerate MIDI devices")
        combo.blockSignals(False)

    def _on_audio_device_selected(self, index: int) -> None:
        device_id = None if index <= 0 else index - 1
        try:
            self._audio_engine.set_device(device_id)
        except Exception:
            log.exception("Failed to switch audio device")
            QMessageBox.warning(
                self._window,
                "Audio Device Error",
                "Could not open the selected audio device.\n"
                "Falling back to the default device.",
            )
            self._window._combo_audio.blockSignals(True)
            self._window._combo_audio.setCurrentIndex(0)
            self._window._combo_audio.blockSignals(False)
            self._audio_engine.set_device(None)

    # ── Engine startup ───────────────────────────────────────────────────

    def _start_audio_engine(self) -> None:
        try:
            self._audio_engine.start()
        except Exception:
            log.exception("Audio engine failed to start")
            QMessageBox.critical(
                self._window,
                "Audio Error",
                "Could not open the audio output device.\n"
                "Please check your audio settings under Edit > Preferences.",
            )

    def _start_midi_engine(self) -> None:
        if not self._midi_engine.available:
            log.info("MIDI disabled (python-rtmidi not installed)")
            return
        try:
            self._midi_engine.start()
        except Exception:
            log.exception("MIDI engine failed to start")
            # Non-fatal — the app works without MIDI


def main() -> None:
    """Entry point for ``python -m vynth``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    app = VynthApp()
    sys.exit(app.run())
