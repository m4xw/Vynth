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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._knobs: dict[str, RotaryKnob] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setSpacing(6)
        self._layout.setContentsMargins(4, 4, 4, 4)

        self._build_adsr()
        self._build_pitch_shift()
        self._build_filter()
        self._build_chorus()
        self._build_delay()
        self._build_reverb()
        self._build_granular()
        self._build_limiter()

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

    # -- builders ------------------------------------------------------------

    def _reg(self, prefix: str, knob: RotaryKnob) -> RotaryKnob:
        key = f"{prefix}_{knob.name}".lower().replace(" ", "_")
        self._knobs[key] = knob
        knob.valueChanged.connect(partial(self._emit, key))
        return knob

    def _emit(self, key: str, value: float) -> None:
        self.paramChanged.emit(key, value)

    # --- ADSR ---------------------------------------------------------------

    def _build_adsr(self) -> None:
        mod = _EffectModule("ADSR Envelope")
        self._adsr = ADSRDisplay()
        self._adsr.paramChanged.connect(lambda n, v: self.paramChanged.emit(f"adsr_{n}", v))
        mod.content_layout.addWidget(self._adsr)

        a = self._reg("adsr", _make_knob("attack", 1, 5000, 50, " ms", 0))
        d = self._reg("adsr", _make_knob("decay", 1, 5000, 150, " ms", 0))
        s = self._reg("adsr", _make_knob("sustain", 0, 1, 0.7, "", 2))
        r = self._reg("adsr", _make_knob("release", 1, 5000, 300, " ms", 0))
        mod.content_layout.addLayout(_knob_row(a, d, s, r))
        self._layout.addWidget(mod)

    # --- Pitch Shift --------------------------------------------------------

    def _build_pitch_shift(self) -> None:
        mod = _EffectModule("Pitch Shift")
        semi = self._reg("pitch_shift", _make_knob("semitones", -24, 24, 0, " st", 1))
        mod.content_layout.addLayout(_knob_row(semi))

        self._formant_cb = QCheckBox("Preserve Formant")
        self._formant_cb.setChecked(True)
        mod.content_layout.addWidget(self._formant_cb)
        self._layout.addWidget(mod)

    # --- Filter -------------------------------------------------------------

    def _build_filter(self) -> None:
        mod = _EffectModule("Filter")

        mode_row = QHBoxLayout()
        self._filter_mode = QComboBox()
        self._filter_mode.addItems(["Low Pass", "High Pass", "Band Pass", "Notch"])
        mode_row.addWidget(self._filter_mode)
        mod.content_layout.addLayout(mode_row)

        freq = self._reg("filter", _make_knob("frequency", 20, 20000, 1000, " Hz", 0))
        q = self._reg("filter", _make_knob("q", 0.1, 20, 1.0, "", 2))
        gain = self._reg("filter", _make_knob("gain", -24, 24, 0, " dB", 1))
        mod.content_layout.addLayout(_knob_row(freq, q, gain))
        self._layout.addWidget(mod)

    # --- Chorus -------------------------------------------------------------

    def _build_chorus(self) -> None:
        mod = _EffectModule("Chorus")

        voices_row = QHBoxLayout()
        self._chorus_voices = QSpinBox()
        self._chorus_voices.setRange(2, 8)
        self._chorus_voices.setValue(4)
        self._chorus_voices.setPrefix("Voices: ")
        voices_row.addWidget(self._chorus_voices)
        mod.content_layout.addLayout(voices_row)

        detune = self._reg("chorus", _make_knob("detune", 0, 50, 10, " ct", 1))
        rate = self._reg("chorus", _make_knob("rate", 0.05, 5, 0.8, " Hz", 2))
        depth = self._reg("chorus", _make_knob("depth", 0, 1, 0.5))
        mix = self._reg("chorus", _make_knob("mix", 0, 1, 0.5))
        spread = self._reg("chorus", _make_knob("spread", 0, 1, 0.7))
        mod.content_layout.addLayout(_knob_row(detune, rate, depth))
        mod.content_layout.addLayout(_knob_row(mix, spread))
        self._layout.addWidget(mod)

    # --- Delay --------------------------------------------------------------

    def _build_delay(self) -> None:
        mod = _EffectModule("Delay")

        time = self._reg("delay", _make_knob("time", 1, 2000, 250, " ms", 0))
        fb = self._reg("delay", _make_knob("feedback", 0, 0.95, 0.4))
        mix = self._reg("delay", _make_knob("mix", 0, 1, 0.3))
        mod.content_layout.addLayout(_knob_row(time, fb, mix))

        self._ping_pong_cb = QCheckBox("Ping-Pong")
        mod.content_layout.addWidget(self._ping_pong_cb)
        self._layout.addWidget(mod)

    # --- Reverb -------------------------------------------------------------

    def _build_reverb(self) -> None:
        mod = _EffectModule("Reverb")

        room = self._reg("reverb", _make_knob("room_size", 0, 1, 0.5))
        damp = self._reg("reverb", _make_knob("damping", 0, 1, 0.5))
        wet = self._reg("reverb", _make_knob("wet", 0, 1, 0.3))
        mod.content_layout.addLayout(_knob_row(room, damp, wet))
        self._layout.addWidget(mod)

    # --- Granular -----------------------------------------------------------

    def _build_granular(self) -> None:
        mod = _EffectModule("Granular")

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

        thresh = self._reg("limiter", _make_knob("threshold", -60, 0, -1, " dB", 1))
        rel = self._reg("limiter", _make_knob("release", 1, 1000, 100, " ms", 0))
        mod.content_layout.addLayout(_knob_row(thresh, rel))
        self._layout.addWidget(mod)
