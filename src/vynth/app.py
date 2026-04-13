"""Application controller — wires UI, engine, and MIDI together."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QStackedWidget

from vynth.config import (
    AppConfig,
    BLOCK_SIZE,
    SAMPLE_RATE,
    SPECTRUM_FFT_SIZE,
    UI_FPS,
    SessionSettings,
    default_midi_controller_profile,
)
from vynth.engine.audio_engine import AudioEngine
from vynth.engine.export import Exporter
from vynth.engine.midi_engine import MIDIEngine
from vynth.engine.recorder import Recorder
from vynth.engine.voice import PlaybackMode
from vynth.sampler.sample import LoopRegion, Sample
from vynth.sampler.sample_editor import SampleEditor
from vynth.sampler.sample_manager import SampleManager
from vynth.ui.dialogs.about_dialog import AboutDialog
from vynth.ui.dialogs.midi_controller_dialog import MIDIControllerDialog
from vynth.ui.dialogs.settings_dialog import SettingsDialog
from vynth.ui.main_window import MainWindow
from vynth.ui.panels.effects_rack import EffectsRack
from vynth.ui.panels.export_panel import ExportPanel
from vynth.ui.panels.mixer_panel import MixerPanel
from vynth.ui.panels.recorder_panel import RecorderPanel
from vynth.ui.panels.sample_browser import SampleBrowser
from vynth.ui.theme import apply_theme
from vynth.ui.widgets.midi_keyboard import MIDIKeyboardWidget
from vynth.ui.widgets.rendered_waveform_view import (
    RenderContext,
    RenderedWaveformProcessor,
    RenderedWaveformView,
)
from vynth.ui.widgets.live_waveform_view import LiveWaveformView
from vynth.ui.widgets.spectrum_view import SpectrumView
from vynth.ui.widgets.waveform_editor import WaveformEditor
from vynth.utils.thread_safe_queue import Command, CommandType

log = logging.getLogger(__name__)

_SESSION_FILE = "vynth_session.json"
_CONTROLLER_RUNTIME_TARGETS = {
    "division",
    "swing",
    "mode",
    "octave",
    "latch",
    "sync",
    "gate",
    "gpm",
}


class VynthApp:
    """Application controller — wires UI, engine, and MIDI together."""

    def __init__(self) -> None:
        self._qapp = QApplication(sys.argv)
        self._qapp.setApplicationName("Vynth")
        apply_theme(self._qapp)

        # ── settings ─────────────────────────────────────────────────────
        self._settings = SessionSettings()
        self._app_config = AppConfig()

        # ── engines ──────────────────────────────────────────────────────
        self._audio_engine = AudioEngine(self._settings)
        self._midi_engine = MIDIEngine(self._audio_engine.command_queue)
        self._midi_controller_profile = self._app_config.midi_controller_profile
        if not self._midi_controller_profile:
            self._midi_controller_profile = default_midi_controller_profile()
        self._midi_engine.set_controller_profile(self._midi_controller_profile)
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
        self._rendered_waveform_view = RenderedWaveformView()
        self._live_waveform_view = LiveWaveformView()
        self._recorder_panel = RecorderPanel()
        self._sample_browser = SampleBrowser()
        self._effects_rack = EffectsRack()
        self._mixer_panel = MixerPanel()
        self._midi_keyboard = MIDIKeyboardWidget()

        self._visualizer_mode = "spectrum"
        self._current_sample: Sample | None = None
        self._render_processor = RenderedWaveformProcessor()
        self._render_invalidated = True
        self._render_invalidate_ts = 0.0
        self._render_debounce_s = 0.06
        self._controller_runtime: dict[str, float] = {
            "division": 0.25,
            "swing": 0.0,
            "mode": 0.0,
            "octave": 0.0,
            "latch": 0.0,
            "sync": 0.0,
            "gate": 0.5,
            "gpm": 120.0,
        }
        self._octave_shift = 0

        fx_state = self._effects_rack.get_all_state()
        self._effect_params: dict[str, float] = {
            str(k): float(v) for k, v in fx_state.get("params", {}).items()
        }
        self._effect_bypass: dict[str, bool] = {
            str(k): bool(v) for k, v in fx_state.get("bypass", {}).items()
        }

        self._replace_placeholders()

        # ── signal wiring ────────────────────────────────────────────────
        self._wire_midi_signals()
        self._wire_recorder_signals()
        self._wire_sample_browser_signals()
        self._wire_effects_signals()
        self._wire_mixer_signals()
        self._wire_waveform_editor_signals()
        self._wire_menu_actions()

        # Push initial bypass defaults to engine (signals fired during EffectsRack
        # construction before wiring was connected, so they were silently dropped)
        self._effects_rack.force_emit_all_bypass()
        self._sync_playback_slice_state()

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

        # ── auto-load last session ───────────────────────────────────────
        last = self._app_config.last_session_path
        if last and Path(last).is_file():
            log.info("Auto-loading last session: %s", last)
            self._load_session_from_path(Path(last))

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
        self._visualizer_stack = QStackedWidget()
        self._visualizer_stack.addWidget(self._spectrum_view)
        self._visualizer_stack.addWidget(self._rendered_waveform_view)
        self._visualizer_stack.addWidget(self._live_waveform_view)
        splitter.replaceWidget(1, self._visualizer_stack)
        self._set_visualizer_mode("spectrum")

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
        self._midi_engine.mapped_action_triggered.connect(self._on_mapped_action)
        self._midi_engine.mapped_param_changed.connect(self._on_mapped_param_changed)

        # MIDI keyboard mouse clicks → audio engine
        self._midi_keyboard.notePressed.connect(self._on_midi_note_on)
        self._midi_keyboard.noteReleased.connect(self._on_midi_note_off)

        # Toolbar MIDI device combo
        self._window._combo_midi.currentIndexChanged.connect(
            self._on_midi_device_selected
        )

    def _on_midi_note_on(self, note: int, velocity: int) -> None:
        shifted_note = max(0, min(127, note + (self._octave_shift * 12)))
        self._sync_playback_slice_state()
        start_frame, end_frame = self._resolve_note_region()
        self._audio_engine.push_command(
            Command(
                type=CommandType.NOTE_ON,
                note=shifted_note,
                velocity=velocity,
                data=(start_frame, end_frame),
            )
        )
        self._midi_keyboard.highlight_note(shifted_note, velocity)
        self._window.flash_midi_activity()

    def _on_midi_note_off(self, note: int) -> None:
        shifted_note = max(0, min(127, note + (self._octave_shift * 12)))
        self._audio_engine.push_command(
            Command(type=CommandType.NOTE_OFF, note=shifted_note)
        )
        self._midi_keyboard.release_note(shifted_note)

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
                self._current_sample = sample
                self._waveform_editor.set_sample(sample)
                self._invalidate_rendered_waveform()
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
        self._current_sample = sample
        # Push to audio engine
        self._audio_engine.push_command(
            Command(type=CommandType.SET_SAMPLE, data=sample)
        )
        # Update UI
        self._waveform_editor.set_sample(sample)
        sel_start, sel_end = sample.selection_range
        if sel_end > sel_start:
            self._waveform_editor.set_selection(sel_start, sel_end)
        else:
            self._waveform_editor.clear_selection()
        self._push_slice_region_params()
        self._invalidate_rendered_waveform()
        self._window.set_sample_info(
            f"{sample.name}  \u2014  {sample.duration_s:.2f}s  "
            f"{sample.sample_rate} Hz  {sample.channels}ch"
        )

    # ── Effects Rack → Engine ────────────────────────────────────────────

    def _wire_effects_signals(self) -> None:
        self._effects_rack.paramChanged.connect(self._on_effect_param_changed)
        self._effects_rack.bypassChanged.connect(self._on_effect_bypass_changed)
        self._effects_rack.playbackModeChanged.connect(self._on_playback_mode_changed)
        self._effects_rack.sliceConfigChanged.connect(self._on_slice_config_changed)

    def _on_effect_param_changed(self, name: str, value: float) -> None:
        self._effect_params[name] = value
        self._invalidate_rendered_waveform()
        self._audio_engine.push_command(
            Command(
                type=CommandType.PARAM_CHANGE,
                param_name=name,
                param_value=value,
            )
        )

    def _on_effect_bypass_changed(self, prefix: str, bypassed: bool) -> None:
        self._effect_bypass[prefix] = bypassed
        self._invalidate_rendered_waveform()
        self._audio_engine.push_command(
            Command(
                type=CommandType.PARAM_CHANGE,
                param_name=f"{prefix}_bypass",
                param_value=1.0 if bypassed else 0.0,
            )
        )

    def _on_playback_mode_changed(self, index: int) -> None:
        mode = [PlaybackMode.SAMPLER, PlaybackMode.GRANULAR, PlaybackMode.SLICE][index]
        self._audio_engine.push_command(
            Command(type=CommandType.SET_PLAYBACK_MODE, data=mode)
        )

    def _on_slice_config_changed(self, num_slices: int, start_note: int) -> None:
        self._audio_engine.push_command(
            Command(
                type=CommandType.PARAM_CHANGE,
                param_name="slice_num_slices",
                param_value=float(num_slices),
            )
        )
        self._audio_engine.push_command(
            Command(
                type=CommandType.PARAM_CHANGE,
                param_name="slice_start_note",
                param_value=float(start_note),
            )
        )
        self._push_slice_region_params()

    def _sync_playback_slice_state(self) -> None:
        """Push current playback/slice UI state to engine immediately."""
        self._on_playback_mode_changed(self._effects_rack.playback_mode_index)
        self._on_slice_config_changed(
            self._effects_rack.slice_num_slices,
            self._effects_rack.slice_start_note,
        )

    # ── Mixer → Engine ───────────────────────────────────────────────────

    def _wire_mixer_signals(self) -> None:
        self._mixer_panel.volumeChanged.connect(self._on_master_volume_changed)
        self._mixer_panel.visualizerModeChanged.connect(self._set_visualizer_mode)

    def _on_master_volume_changed(self, value: float) -> None:
        self._settings.master_volume = float(value)
        self._audio_engine.set_master_volume(float(value))

    # ── Waveform Editor → Sample Editor ──────────────────────────────────

    def _wire_waveform_editor_signals(self) -> None:
        self._waveform_editor.editRequested.connect(self._on_edit_requested)
        self._waveform_editor.loopPointsChanged.connect(self._on_loop_points_changed)
        self._waveform_editor.selectionChanged.connect(self._on_selection_changed)

    def _on_selection_changed(self, start: int, end: int) -> None:
        sample = self._sample_manager.get_selected()
        if sample is not None:
            sample.selection_range = (int(start), int(end))
        self._push_slice_region_params()
        self._invalidate_rendered_waveform()

    def _on_loop_points_changed(self, start: int, end: int) -> None:
        sample = self._sample_manager.get_selected()
        if sample is None:
            return
        if end > start > 0:
            sample.loop.start = start
            sample.loop.end = end
            sample.loop.enabled = True
        else:
            sample.loop.enabled = False
        # Re-push sample so the voice sees updated loop region
        self._audio_engine.push_command(
            Command(type=CommandType.SET_SAMPLE, data=sample)
        )
        self._push_slice_region_params()
        self._invalidate_rendered_waveform()

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
        self._current_sample = new_sample
        self._waveform_editor.set_sample(new_sample)
        self._audio_engine.push_command(
            Command(type=CommandType.SET_SAMPLE, data=new_sample)
        )
        self._push_slice_region_params()
        self._invalidate_rendered_waveform()

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

        if self._visualizer_mode == "spectrum":
            # Visualization buffer → spectrum analyzer
            buf = self._audio_engine.get_visualization_buffer(SPECTRUM_FFT_SIZE)
            if buf is not None and buf.size > 0:
                mono = buf.mean(axis=1) if buf.ndim == 2 else buf
                self._spectrum_view.push_audio_block(mono)
            return

        if self._visualizer_mode == "live":
            buf = self._audio_engine.read_visualization_buffer()
            if buf is not None and buf.size > 0:
                self._live_waveform_view.push_audio(buf)
                self._live_waveform_view.update_display()
            return

        self._update_rendered_waveform_if_needed()

    def _set_visualizer_mode(self, mode: str) -> None:
        target = mode if mode in ("spectrum", "rendered", "live") else "spectrum"
        self._visualizer_mode = target
        self._mixer_panel.set_visualizer_mode(target)
        if target == "rendered":
            self._visualizer_stack.setCurrentWidget(self._rendered_waveform_view)
            self._invalidate_rendered_waveform()
        elif target == "live":
            self._visualizer_stack.setCurrentWidget(self._live_waveform_view)
        else:
            self._visualizer_stack.setCurrentWidget(self._spectrum_view)

    def _invalidate_rendered_waveform(self) -> None:
        self._render_invalidated = True
        self._render_invalidate_ts = time.monotonic()

    def _update_rendered_waveform_if_needed(self) -> None:
        if not self._render_invalidated:
            return
        if (time.monotonic() - self._render_invalidate_ts) < self._render_debounce_s:
            return

        sample = self._current_sample or self._sample_manager.get_selected()
        if sample is None:
            self._rendered_waveform_view.clear()
            self._render_invalidated = False
            return

        rendered = self._render_processor.render(
            sample.data,
            sample.sample_rate,
            RenderContext(params=self._effect_params, bypass=self._effect_bypass),
        )
        self._rendered_waveform_view.set_rendered_data(rendered, sample.sample_rate)

        # Carry over selection and loop from the editor
        sel = self._waveform_editor.get_selection()
        if sel[1] > sel[0]:
            self._rendered_waveform_view.set_selection(sel[0], sel[1])
        else:
            self._rendered_waveform_view.clear_selection()

        loop_start, loop_end = self._waveform_editor.get_loop_points()
        if loop_end > loop_start > 0:
            self._rendered_waveform_view.set_loop_points(loop_start, loop_end)
        else:
            self._rendered_waveform_view.clear_loop_points()

        if self._effect_bypass.get("filter", False):
            self._rendered_waveform_view.clear_filter_overlay()
        else:
            self._rendered_waveform_view.set_filter_overlay(
                frequency_hz=self._effect_params.get(
                    "filter_frequency",
                    self._audio_engine.voice_allocator.get_param("filter_frequency"),
                ),
                q=self._effect_params.get(
                    "filter_q",
                    self._audio_engine.voice_allocator.get_param("filter_q"),
                ),
                mode=int(
                    self._effect_params.get(
                        "filter_mode",
                        self._audio_engine.voice_allocator.get_param("filter_mode"),
                    )
                ),
                gain_db=self._effect_params.get(
                    "filter_gain_db",
                    self._audio_engine.voice_allocator.get_param("filter_gain_db"),
                ),
            )

        self._render_invalidated = False

    # ── Menu actions ─────────────────────────────────────────────────────

    def _wire_menu_actions(self) -> None:
        w = self._window
        w._act_load.triggered.connect(self._on_load_sample)
        w._act_export.triggered.connect(self._on_export)
        w._act_save.triggered.connect(self._on_save_session)
        w._act_new.triggered.connect(self._on_new_session)
        w._act_load_session.triggered.connect(self._on_load_session)
        w._act_prefs.triggered.connect(self._on_preferences)
        w._act_midi_controller.triggered.connect(self._on_midi_controller_editor)
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
            if export_cfg.get("apply_effects", False):
                self._exporter.render_sample(
                    sample,
                    self._audio_engine.voice_allocator,
                    path,
                    sample_rate=export_cfg["sr"],
                    bit_depth=export_cfg["bits"],
                    master_volume=self._audio_engine.master_volume,
                )
            else:
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

        # Save embedded samples (recordings, edits) next to the session file.
        # Also persist per-sample playback metadata needed for faithful restore.
        session_dir = Path(path).parent
        samples_dir = session_dir / (Path(path).stem + "_samples")
        sample_entries: list[dict] = []
        sample_names = self._sample_manager.get_names()
        selected_name = self._sample_manager.get_selected().name if self._sample_manager.get_selected() else None

        sel_start, sel_end = self._waveform_editor.get_selection()
        selection_data = None
        if sel_end > sel_start:
            selection_data = {"start": int(sel_start), "end": int(sel_end)}

        for idx, name in enumerate(sample_names):
            s = self._sample_manager.get_sample(name)
            if s is None:
                continue
            if s.file_path and Path(s.file_path).exists():
                sample_path = s.file_path
            else:
                # Save embedded sample to disk
                samples_dir.mkdir(parents=True, exist_ok=True)
                safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
                wav_path = samples_dir / f"{safe}.wav"
                s.save(wav_path)
                sample_path = str(wav_path)

            sample_entries.append(
                {
                    "path": sample_path,
                    "name": s.name,
                    "root_note": int(s.root_note),
                    "note_range": [int(s.note_range[0]), int(s.note_range[1])],
                    "velocity_range": [int(s.velocity_range[0]), int(s.velocity_range[1])],
                    "loop": {
                        "start": int(s.loop.start),
                        "end": int(s.loop.end),
                        "crossfade": int(s.loop.crossfade),
                        "enabled": bool(s.loop.enabled),
                    },
                    "selected": bool(name == selected_name),
                    "selection": selection_data if name == selected_name else None,
                    "index": idx,
                }
            )
            sample_sel_start, sample_sel_end = s.selection_range
            if sample_sel_end > sample_sel_start:
                sample_entries[-1]["selection"] = {
                    "start": int(sample_sel_start),
                    "end": int(sample_sel_end),
                }

        selected_index = next(
            (i for i, n in enumerate(sample_names) if n == selected_name),
            -1,
        )

        data = {
            "master_volume": float(self._audio_engine.master_volume),
            "sample_rate": self._settings.sample_rate,
            "block_size": self._settings.block_size,
            "samples": sample_entries,
            "effects": self._effects_rack.get_all_state(),
            "midi_controller_profile": self._midi_controller_profile,
            "selected_sample_index": selected_index,
            "selected_sample_name": selected_name,
            "waveform_selection": selection_data,
            "runtime": {
                "octave_shift": int(self._octave_shift),
                "controller": {k: float(v) for k, v in self._controller_runtime.items()},
            },
        }
        try:
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._app_config.last_session_path = path
            self._window.statusBar().showMessage(f"Session saved: {path}", 3000)
        except Exception:
            log.exception("Failed to save session")
            QMessageBox.warning(
                self._window, "Save Error", f"Could not save session to:\n{path}"
            )

    def _on_load_session(self) -> None:
        """Open a file dialog to choose and load a session file."""
        path, _ = QFileDialog.getOpenFileName(
            self._window, "Load Session", "",
            "Vynth Session (*.json);;All Files (*)",
        )
        if not path:
            return
        self._load_session_from_path(Path(path))

    def _on_new_session(self) -> None:
        """Reset to a blank session (clear all samples and effects)."""
        # Stop all notes
        self._audio_engine.push_command(
            Command(type=CommandType.ALL_NOTES_OFF)
        )
        # Clear engine sample
        self._audio_engine.push_command(
            Command(type=CommandType.SET_SAMPLE, data=None)
        )
        # Clear sample manager
        for name in list(self._sample_manager.get_names()):
            self._sample_manager.remove_sample(name)
        # Reset effects rack to defaults
        self._effects_rack.reset_all()
        self._sync_playback_slice_state()
        # Reset master volume
        self._settings.master_volume = 0.8
        self._audio_engine.set_master_volume(0.8)
        self._mixer_panel.set_volume(0.8)
        # Clear waveform
        self._waveform_editor.clear()
        self._rendered_waveform_view.clear()
        self._live_waveform_view.clear()
        self._current_sample = None
        self._push_slice_region_params()
        self._window.set_sample_info("")
        self._invalidate_rendered_waveform()
        self._window.statusBar().showMessage("New session", 3000)

    def _load_session_from_path(self, path: Path) -> None:
        """Load a session from *path* (no dialog)."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Failed to read session file")
            QMessageBox.warning(
                self._window, "Load Error", f"Could not read:\n{path}"
            )
            return

        # Reset current in-memory state so load is deterministic.
        self._on_new_session()

        # Apply settings
        self._settings.master_volume = data.get(
            "master_volume", self._settings.master_volume
        )
        self._audio_engine.set_master_volume(self._settings.master_volume)
        self._mixer_panel.set_volume(self._settings.master_volume)

        # Load samples
        loaded_names: list[str] = []
        selected_name_saved = data.get("selected_sample_name")
        selected_index_saved = int(data.get("selected_sample_index", -1))
        selection_saved = data.get("waveform_selection")

        for entry in data.get("samples", []):
            # Support both old format (plain string) and new format (dict)
            if isinstance(entry, str):
                sample_path = entry
                root_note = 60
                note_range = (0, 127)
                velocity_range = (0, 127)
                loop_data = None
                sample_selection = None
            elif isinstance(entry, dict):
                sample_path = entry.get("path", "")
                root_note = entry.get("root_note", 60)
                note_range_raw = entry.get("note_range", [0, 127])
                velocity_range_raw = entry.get("velocity_range", [0, 127])
                if isinstance(note_range_raw, (list, tuple)) and len(note_range_raw) == 2:
                    note_range = (int(note_range_raw[0]), int(note_range_raw[1]))
                else:
                    note_range = (0, 127)
                if isinstance(velocity_range_raw, (list, tuple)) and len(velocity_range_raw) == 2:
                    velocity_range = (int(velocity_range_raw[0]), int(velocity_range_raw[1]))
                else:
                    velocity_range = (0, 127)
                loop_data = entry.get("loop")
                sample_selection = entry.get("selection")
            else:
                continue
            if not sample_path or not Path(sample_path).exists():
                log.warning("Sample file not found: %s", sample_path)
                continue
            try:
                sample = self._sample_manager.load_sample(sample_path)
                sample.root_note = root_note
                sample.note_range = note_range
                sample.velocity_range = velocity_range

                if isinstance(loop_data, dict):
                    ls = int(loop_data.get("start", 0))
                    le = int(loop_data.get("end", 0))
                    lc = int(loop_data.get("crossfade", 256))
                    enabled = bool(loop_data.get("enabled", False))
                    sample.loop = LoopRegion(start=ls, end=le, crossfade=lc, enabled=enabled)

                if isinstance(sample_selection, dict):
                    ss = int(sample_selection.get("start", 0))
                    se = int(sample_selection.get("end", 0))
                    sample.selection_range = (ss, se)

                loaded_names.append(sample.name)
            except Exception:
                log.warning("Could not reload sample: %s", sample_path)

        # Select the saved sample if possible, otherwise keep backward-compatible behavior.
        selected_name = None
        if selected_name_saved in loaded_names:
            selected_name = selected_name_saved
        elif 0 <= selected_index_saved < len(loaded_names):
            selected_name = loaded_names[selected_index_saved]
        elif loaded_names:
            selected_name = loaded_names[-1]

        if selected_name:
            self._on_sample_selected(selected_name)

        # Restore waveform selection if present.
        if isinstance(selection_saved, dict):
            s = int(selection_saved.get("start", 0))
            e = int(selection_saved.get("end", 0))
            if e > s and self._current_sample is not None:
                s = max(0, min(s, self._current_sample.length))
                e = max(0, min(e, self._current_sample.length))
                self._waveform_editor.set_selection(s, e)
                self._current_sample.selection_range = (s, e)
            else:
                self._waveform_editor.clear_selection()
        else:
            if self._current_sample is not None:
                ss, se = self._current_sample.selection_range
                if se > ss:
                    ss = max(0, min(ss, self._current_sample.length))
                    se = max(0, min(se, self._current_sample.length))
                    self._waveform_editor.set_selection(ss, se)
                else:
                    self._waveform_editor.clear_selection()
            else:
                self._waveform_editor.clear_selection()

        # Restore effects state
        effects_state = data.get("effects")
        if effects_state:
            self._effects_rack.set_all_state(effects_state)
            self._effect_params = {
                str(k): float(v) for k, v in effects_state.get("params", {}).items()
            }
            self._effect_bypass = {
                str(k): bool(v) for k, v in effects_state.get("bypass", {}).items()
            }
            self._invalidate_rendered_waveform()
        # Force-push all bypass states to engine (Qt skips toggled() when the
        # checked state hasn't changed vs. what set_all_state just wrote)
        self._effects_rack.force_emit_all_bypass()
        self._sync_playback_slice_state()

        # Restore MIDI controller profile if embedded in session
        profile = data.get("midi_controller_profile")
        if isinstance(profile, dict):
            self._midi_controller_profile = profile
            self._midi_engine.set_controller_profile(profile)
            self._app_config.midi_controller_profile = profile

        # Restore runtime controller state.
        runtime = data.get("runtime")
        if isinstance(runtime, dict):
            self._octave_shift = int(runtime.get("octave_shift", self._octave_shift))
            controller = runtime.get("controller")
            if isinstance(controller, dict):
                for key in _CONTROLLER_RUNTIME_TARGETS:
                    if key in controller:
                        self._controller_runtime[key] = float(controller[key])

        self._push_slice_region_params()
        self._invalidate_rendered_waveform()

        self._app_config.last_session_path = str(path)
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

    def _on_midi_controller_editor(self) -> None:
        dlg = MIDIControllerDialog(self._midi_controller_profile, self._window)
        self._midi_engine.controller_cc_received.connect(dlg.on_midi_cc)
        self._midi_engine.controller_note_received.connect(dlg.on_midi_note)
        try:
            if dlg.exec() != MIDIControllerDialog.DialogCode.Accepted:
                return
            profile = dlg.get_profile()
            self._midi_controller_profile = profile
            self._midi_engine.set_controller_profile(profile)
            self._app_config.midi_controller_profile = profile
            self._window.statusBar().showMessage("MIDI controller profile updated", 3000)
        finally:
            self._midi_engine.controller_cc_received.disconnect(dlg.on_midi_cc)
            self._midi_engine.controller_note_received.disconnect(dlg.on_midi_note)

    def _on_mapped_action(self, action: str) -> None:
        if action == "play":
            self._on_play_sample()
        elif action == "stop":
            self._on_stop()
        elif action == "record_toggle":
            self._recorder_panel._rec_btn.toggle()
        elif action == "save_session":
            self._on_save_session()
        elif action == "export":
            self._on_export()
        elif action == "load_sample":
            self._on_load_sample()
        elif action == "new_session":
            self._on_new_session()

    def _on_mapped_param_changed(self, name: str, value: float) -> None:
        """Mirror mapped MIDI parameter changes in the visible UI controls."""
        if name in _CONTROLLER_RUNTIME_TARGETS:
            self._controller_runtime[name] = value
            if name == "mode":
                idx = max(0, min(2, int(round(value))))
                self._effects_rack.playback_mode_index = idx
            elif name == "octave":
                self._octave_shift = max(-4, min(4, int(round(value))))
            return

        if name == "master_volume":
            self._mixer_panel.set_volume(value)
            self._settings.master_volume = value
            return

        if name.endswith("_bypass"):
            prefix = name[:-7]
            self._effect_bypass[prefix] = value >= 0.5
            self._effects_rack.set_bypass(prefix, value >= 0.5)
            self._invalidate_rendered_waveform()
            return

        self._effects_rack.set_param(name, value)
        self._effect_params[name] = value
        self._invalidate_rendered_waveform()

    def _on_about(self) -> None:
        AboutDialog(self._window).exec()

    def _on_play_sample(self) -> None:
        sample = self._sample_manager.get_selected()
        if sample is None:
            return
        self._sync_playback_slice_state()

        preview_note = int(sample.root_note)
        start_frame, end_frame = self._resolve_note_region()
        self._audio_engine.push_command(
            Command(
                type=CommandType.NOTE_ON, note=preview_note, velocity=100,
                data=(start_frame, end_frame),
            )
        )

    def _resolve_note_start_frame(self) -> int:
        """Prefer selection start, then loop start, else zero."""
        sample = self._sample_manager.get_selected()
        if sample is None:
            return 0

        sel_start, sel_end = self._waveform_editor.get_selection()
        if sel_end > sel_start:
            return max(0, sel_start)

        if sample.loop.enabled and sample.loop.end > sample.loop.start:
            return max(0, int(sample.loop.start))

        return 0

    def _resolve_note_region(self) -> tuple[int, int]:
        """Resolve note playback region as (start, end).

        Selection limits playback to [start, end).
        Loop/no selection starts at loop start or zero and plays normally.
        """
        sample = self._sample_manager.get_selected()
        if sample is None:
            return (0, 0)

        sel_start, sel_end = self._waveform_editor.get_selection()
        if sel_end > sel_start:
            return (max(0, int(sel_start)), min(sample.length, int(sel_end)))

        if sample.loop.enabled and sample.loop.end > sample.loop.start:
            return (max(0, int(sample.loop.start)), 0)

        return (0, 0)

    def _resolve_slice_region(self) -> tuple[int, int]:
        """Resolve slice source region: selection, then loop, else full sample."""
        sample = self._sample_manager.get_selected()
        if sample is None:
            return (0, 0)

        sel_start, sel_end = self._waveform_editor.get_selection()
        if sel_end > sel_start:
            return (max(0, int(sel_start)), min(sample.length, int(sel_end)))

        if sample.loop.enabled and sample.loop.end > sample.loop.start:
            return (max(0, int(sample.loop.start)), min(sample.length, int(sample.loop.end)))

        return (0, int(sample.length))

    def _push_slice_region_params(self) -> None:
        """Send active slice source region to the audio engine."""
        start, end = self._resolve_slice_region()
        self._audio_engine.push_command(
            Command(
                type=CommandType.PARAM_CHANGE,
                param_name="slice_region_start",
                param_value=float(start),
            )
        )
        self._audio_engine.push_command(
            Command(
                type=CommandType.PARAM_CHANGE,
                param_name="slice_region_end",
                param_value=float(end),
            )
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
