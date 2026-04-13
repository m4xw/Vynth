"""Collapsible effects chain with knobs for each effect."""

from __future__ import annotations

from functools import partial

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QScrollArea,
    QPushButton, QComboBox, QCheckBox, QSpinBox, QSizePolicy,
)

from vynth.ui.theme import Colors
from vynth.ui.widgets.knob import RotaryKnob
from vynth.ui.widgets.adsr_display import ADSRDisplay


class _EffectModule(QGroupBox):
    """Collapsible module with a bypass button and content area."""

    bypassToggled = pyqtSignal(bool)
    enableToggled = pyqtSignal(bool)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggled)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(4, 4, 4, 4)
        self._content_layout.setSpacing(4)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 8, 4, 4)
        outer.setSpacing(0)

        # Bypass button in header
        self._bypass_btn = QPushButton("Bypass")
        self._bypass_btn.setCheckable(True)
        self._bypass_btn.setFixedHeight(20)
        self._bypass_btn.setStyleSheet(
            f"""
            QPushButton {{ background: {Colors.BG_MEDIUM}; border: 1px solid {Colors.BORDER};
                           border-radius: 3px; font-size: 11px; padding: 0 8px; }}
            QPushButton:checked {{ background: {Colors.ACCENT_WARM}; color: #fff; }}
            """
        )
        self._bypass_btn.toggled.connect(self.bypassToggled)

        header = QHBoxLayout()
        header.addStretch()
        header.addWidget(self._bypass_btn)
        outer.addLayout(header)
        outer.addWidget(self._content)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def _on_toggled(self, expanded: bool) -> None:
        self._content.setVisible(expanded)
        self.enableToggled.emit(expanded)


def _make_knob(name: str, minimum: float, maximum: float, value: float,
               suffix: str = "", decimals: int = 2) -> RotaryKnob:
    return RotaryKnob(
        name=name, minimum=minimum, maximum=maximum,
        value=value, default_value=value, suffix=suffix, decimals=decimals,
    )


def _knob_row(*knobs: RotaryKnob) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    for k in knobs:
        row.addWidget(k, alignment=Qt.AlignmentFlag.AlignCenter)
    return row


class EffectsRack(QWidget):
    """Collapsible effects chain with knobs for each effect."""

    paramChanged = pyqtSignal(str, float)
    bypassChanged = pyqtSignal(str, bool)  # (effect_prefix, bypassed)
    playbackModeChanged = pyqtSignal(int)  # PlaybackMode enum value
    sliceConfigChanged = pyqtSignal(int, int)  # (num_slices, start_note)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._knobs: dict[str, RotaryKnob] = {}
        self._modules: dict[str, _EffectModule] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setSpacing(6)
        self._layout.setContentsMargins(4, 4, 4, 4)

        self._build_playback_mode()
        self._build_adsr()
        self._build_pitch_shift()
        self._build_filter()
        self._build_gain()
        self._build_chorus()
        self._build_delay()
        self._build_reverb()
        self._build_granular()
        self._build_limiter()

        # Capture initial bypass states as defaults for reset_all()
        self._default_bypass: dict[str, bool] = {
            prefix: mod._bypass_btn.isChecked()
            for prefix, mod in self._modules.items()
        }

        self._layout.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll)

    # -- public API ----------------------------------------------------------

    def set_param(self, name: str, value: float) -> None:
        knob = self._knobs.get(name)
        if knob is not None:
            knob.blockSignals(True)
            knob.value = value
            knob.blockSignals(False)

    def get_all_state(self) -> dict:
        """Return all knob values and bypass/enable states for session save."""
        state: dict = {"params": {}, "bypass": {}, "enabled": {}}
        for key, knob in self._knobs.items():
            state["params"][key] = knob.value
        for prefix, mod in self._modules.items():
            state["bypass"][prefix] = mod._bypass_btn.isChecked()
            state["enabled"][prefix] = mod.isChecked()
        state["playback_mode"] = self._mode_combo.currentIndex()
        state["slice_num_slices"] = self._slice_num_spin.value()
        state["slice_start_note"] = self._slice_note_spin.value()
        return state

    def set_all_state(self, state: dict) -> None:
        """Restore knob values and bypass/enable states from session load."""
        for key, value in state.get("params", {}).items():
            knob = self._knobs.get(key)
            if knob is not None:
                knob.value = value
        for prefix, bypassed in state.get("bypass", {}).items():
            mod = self._modules.get(prefix)
            if mod is not None:
                mod._bypass_btn.setChecked(bypassed)
        for prefix, enabled in state.get("enabled", {}).items():
            mod = self._modules.get(prefix)
            if mod is not None:
                mod.setChecked(enabled)
        if "playback_mode" in state:
            self._mode_combo.setCurrentIndex(state["playback_mode"])
        if "slice_num_slices" in state:
            self._slice_num_spin.setValue(state["slice_num_slices"])
        if "slice_start_note" in state:
            self._slice_note_spin.setValue(state["slice_start_note"])

    def reset_all(self) -> None:
        """Reset all knobs to their default values and restore default bypass states."""
        for knob in self._knobs.values():
            knob.value = knob.default_value
        for prefix, mod in self._modules.items():
            mod._bypass_btn.setChecked(self._default_bypass.get(prefix, False))
            mod.setChecked(True)
        self._mode_combo.setCurrentIndex(0)
        self._slice_num_spin.setValue(16)
        self._slice_note_spin.setValue(36)
        self.force_emit_all_bypass()

    def force_emit_all_bypass(self) -> None:
        """Force-emit bypassChanged for every module regardless of prior state.

        Needed on startup/load because Qt skips toggled() when the checked
        state hasn't changed, so the audio engine would never get the command.
        """
        for prefix, mod in self._modules.items():
            self.bypassChanged.emit(prefix, mod._bypass_btn.isChecked())

    # -- internal helpers ----------------------------------------------------

    def _wire_module(self, prefix: str, mod: _EffectModule) -> None:
        """Wire bypass and enable signals for a module."""
        self._modules[prefix] = mod
        mod.bypassToggled.connect(
            lambda bypassed, p=prefix: self.bypassChanged.emit(p, bypassed)
        )
        mod.enableToggled.connect(
            lambda enabled, p=prefix: self.bypassChanged.emit(p, not enabled)
        )

    # -- builders ------------------------------------------------------------

    def _reg(self, prefix: str, knob: RotaryKnob) -> RotaryKnob:
        key = f"{prefix}_{knob.name}".lower().replace(" ", "_")
        self._knobs[key] = knob
        knob.valueChanged.connect(partial(self._emit, key))
        return knob

    def _emit(self, key: str, value: float) -> None:
        self.paramChanged.emit(key, value)

    # --- Playback Mode ------------------------------------------------------

    def _build_playback_mode(self) -> None:
        grp = QGroupBox("Playback Mode")
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(4, 8, 4, 4)
        lay.setSpacing(4)

        mode_row = QHBoxLayout()
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Sampler", "Granular", "Slice"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        lay.addLayout(mode_row)

        # Slice config (visible only in slice mode)
        self._slice_config_widget = QWidget()
        slice_lay = QHBoxLayout(self._slice_config_widget)
        slice_lay.setContentsMargins(0, 0, 0, 0)
        slice_lay.setSpacing(8)

        self._slice_num_spin = QSpinBox()
        self._slice_num_spin.setRange(2, 128)
        self._slice_num_spin.setValue(16)
        self._slice_num_spin.setPrefix("Slices: ")
        self._slice_num_spin.setToolTip("Number of equal slices")
        self._slice_num_spin.valueChanged.connect(self._on_slice_config_changed)
        slice_lay.addWidget(self._slice_num_spin)

        self._slice_note_spin = QSpinBox()
        self._slice_note_spin.setRange(0, 127)
        self._slice_note_spin.setValue(36)
        self._slice_note_spin.setPrefix("Start: ")
        self._slice_note_spin.setToolTip("MIDI note for the first slice (36 = C2)")
        self._slice_note_spin.valueChanged.connect(self._on_slice_config_changed)
        slice_lay.addWidget(self._slice_note_spin)

        self._slice_config_widget.setVisible(False)
        lay.addWidget(self._slice_config_widget)

        self._layout.addWidget(grp)

    def _on_mode_changed(self, index: int) -> None:
        self._slice_config_widget.setVisible(index == 2)
        self.playbackModeChanged.emit(index)

    def _on_slice_config_changed(self) -> None:
        self.sliceConfigChanged.emit(
            self._slice_num_spin.value(),
            self._slice_note_spin.value(),
        )

    @property
    def playback_mode_index(self) -> int:
        return self._mode_combo.currentIndex()

    @playback_mode_index.setter
    def playback_mode_index(self, index: int) -> None:
        self._mode_combo.setCurrentIndex(index)

    @property
    def slice_num_slices(self) -> int:
        return self._slice_num_spin.value()

    @property
    def slice_start_note(self) -> int:
        return self._slice_note_spin.value()

    # --- ADSR ---------------------------------------------------------------

    def _build_adsr(self) -> None:
        mod = _EffectModule("ADSR Envelope")
        self._wire_module("adsr", mod)
        self._adsr = ADSRDisplay()
        self._adsr.paramChanged.connect(lambda n, v: self.paramChanged.emit(f"adsr_{n}", v))
        mod.content_layout.addWidget(self._adsr)

        a = self._reg("adsr", _make_knob("attack_ms", 1, 5000, 50, " ms", 0))
        d = self._reg("adsr", _make_knob("decay_ms", 1, 5000, 150, " ms", 0))
        s = self._reg("adsr", _make_knob("sustain", 0, 1, 0.7, "", 2))
        r = self._reg("adsr", _make_knob("release_ms", 1, 5000, 300, " ms", 0))
        mod.content_layout.addLayout(_knob_row(a, d, s, r))
        self._layout.addWidget(mod)

    # --- Pitch Shift --------------------------------------------------------

    def _build_pitch_shift(self) -> None:
        mod = _EffectModule("Pitch Shift")
        self._wire_module("pitch_shift", mod)
        mod._bypass_btn.setChecked(True)
        semi = self._reg("pitch_shift", _make_knob("semitones", -24, 24, 0, " st", 1))
        mod.content_layout.addLayout(_knob_row(semi))

        self._formant_cb = QCheckBox("Preserve Formant")
        self._formant_cb.setChecked(True)
        mod.content_layout.addWidget(self._formant_cb)
        self._layout.addWidget(mod)

    # --- Filter -------------------------------------------------------------

    def _build_filter(self) -> None:
        mod = _EffectModule("Filter")
        self._wire_module("filter", mod)
        mod._bypass_btn.setChecked(True)

        mode_row = QHBoxLayout()
        self._filter_mode = QComboBox()
        self._filter_mode.addItems(["Low Pass", "High Pass", "Band Pass", "Notch"])
        mode_row.addWidget(self._filter_mode)
        mod.content_layout.addLayout(mode_row)

        freq = self._reg("filter", _make_knob("frequency", 20, 20000, 1000, " Hz", 0))
        q = self._reg("filter", _make_knob("q", 0.1, 20, 1.0, "", 2))
        gain = self._reg("filter", _make_knob("gain_db", -24, 24, 0, " dB", 1))
        mod.content_layout.addLayout(_knob_row(freq, q, gain))
        self._layout.addWidget(mod)

    # --- Gain / Boost -------------------------------------------------------

    def _build_gain(self) -> None:
        mod = _EffectModule("Gain / Boost")
        self._wire_module("gain", mod)
        gain = self._reg("gain", _make_knob("gain_db", -24, 24, 0, " dB", 1))
        mod.content_layout.addLayout(_knob_row(gain))
        self._layout.addWidget(mod)

    # --- Chorus -------------------------------------------------------------

    def _build_chorus(self) -> None:
        mod = _EffectModule("Chorus")
        self._wire_module("chorus", mod)
        mod._bypass_btn.setChecked(True)

        voices_row = QHBoxLayout()
        self._chorus_voices = QSpinBox()
        self._chorus_voices.setRange(1, 8)
        self._chorus_voices.setValue(4)
        self._chorus_voices.setPrefix("Voices: ")
        self._chorus_voices.valueChanged.connect(
            lambda v: self.paramChanged.emit("chorus_num_voices", float(v))
        )
        voices_row.addWidget(self._chorus_voices)
        mod.content_layout.addLayout(voices_row)

        detune = self._reg("chorus", _make_knob("detune_cents", 0, 50, 10, " ct", 1))
        rate = self._reg("chorus", _make_knob("rate_hz", 0.05, 5, 0.8, " Hz", 2))
        depth = self._reg("chorus", _make_knob("depth", 0, 1, 0.5))
        mix = self._reg("chorus", _make_knob("mix", 0, 1, 0.5))
        spread = self._reg("chorus", _make_knob("spread", 0, 1, 0.7))
        mod.content_layout.addLayout(_knob_row(detune, rate, depth))
        mod.content_layout.addLayout(_knob_row(mix, spread))
        self._layout.addWidget(mod)

    # --- Delay --------------------------------------------------------------

    def _build_delay(self) -> None:
        mod = _EffectModule("Delay")
        self._wire_module("delay", mod)
        mod._bypass_btn.setChecked(True)

        time = self._reg("delay", _make_knob("time_ms", 1, 2000, 250, " ms", 0))
        fb = self._reg("delay", _make_knob("feedback", 0, 0.95, 0.4))
        mix = self._reg("delay", _make_knob("mix", 0, 1, 0.3))
        mod.content_layout.addLayout(_knob_row(time, fb, mix))

        self._ping_pong_cb = QCheckBox("Ping-Pong")
        mod.content_layout.addWidget(self._ping_pong_cb)
        self._layout.addWidget(mod)

    # --- Reverb -------------------------------------------------------------

    def _build_reverb(self) -> None:
        mod = _EffectModule("Reverb")
        self._wire_module("reverb", mod)
        mod._bypass_btn.setChecked(True)

        room = self._reg("reverb", _make_knob("room_size", 0, 1, 0.5))
        damp = self._reg("reverb", _make_knob("damping", 0, 1, 0.5))
        wet = self._reg("reverb", _make_knob("wet", 0, 1, 0.3))
        mod.content_layout.addLayout(_knob_row(room, damp, wet))
        self._layout.addWidget(mod)

    # --- Granular -----------------------------------------------------------

    def _build_granular(self) -> None:
        mod = _EffectModule("Granular")
        self._wire_module("granular", mod)
        mod._bypass_btn.setChecked(True)

        grain = self._reg("granular", _make_knob("grain_size", 5, 500, 50, " ms", 0))
        overlap = self._reg("granular", _make_knob("overlap", 0, 1, 0.5))
        scatter = self._reg("granular", _make_knob("scatter", 0, 1, 0.2))
        density = self._reg("granular", _make_knob("density", 1, 64, 8, "", 0))
        pos = self._reg("granular", _make_knob("position", 0, 1, 0.0))
        pitch = self._reg("granular", _make_knob("pitch", -24, 24, 0, " st", 1))
        mod.content_layout.addLayout(_knob_row(grain, overlap, scatter))
        mod.content_layout.addLayout(_knob_row(density, pos, pitch))
        self._layout.addWidget(mod)

    # --- Limiter ------------------------------------------------------------

    def _build_limiter(self) -> None:
        mod = _EffectModule("Limiter")
        self._wire_module("limiter", mod)

        thresh = self._reg("limiter", _make_knob("threshold_db", -60, 0, -1, " dB", 1))
        rel = self._reg("limiter", _make_knob("release_ms", 1, 1000, 100, " ms", 0))
        mod.content_layout.addLayout(_knob_row(thresh, rel))
        self._layout.addWidget(mod)
