"""MIDI controller mapping editor dialog."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vynth.config import default_midi_controller_profile
from vynth.ui.theme import Colors

_HEADERS = [
    "On",
    "Input",
    "Number",
    "Channel",
    "Trigger",
    "Mode",
    "Target Type",
    "Target",
    "Min",
    "Max",
]

_EFFECT_PARAM_TARGETS = [
    "adsr_attack_ms",
    "adsr_decay_ms",
    "adsr_sustain",
    "adsr_release_ms",
    "pitch_shift_semitones",
    "filter_frequency",
    "filter_q",
    "filter_gain_db",
    "gain_gain_db",
    "chorus_num_voices",
    "chorus_detune_cents",
    "chorus_rate_hz",
    "chorus_depth",
    "chorus_mix",
    "chorus_spread",
    "delay_time_ms",
    "delay_feedback",
    "delay_mix",
    "reverb_room_size",
    "reverb_damping",
    "reverb_wet",
    "granular_grain_size",
    "granular_overlap",
    "granular_scatter",
    "granular_density",
    "granular_position",
    "granular_pitch",
    "limiter_threshold_db",
    "limiter_release_ms",
    "adsr_bypass",
    "pitch_shift_bypass",
    "filter_bypass",
    "gain_bypass",
    "chorus_bypass",
    "delay_bypass",
    "reverb_bypass",
    "granular_bypass",
    "limiter_bypass",
    "master_volume",
]

_CUSTOM_PARAM_TARGETS = [
    "division",
    "swing",
    "mode",
    "octave",
    "latch",
    "sync",
    "gate",
    "gpm",
]

_ACTION_TARGETS = [
    "play",
    "stop",
    "record_toggle",
    "save_session",
    "export",
    "load_sample",
    "new_session",
]


class MIDIControllerDialog(QDialog):
    """Edit controller mappings for CC knobs and note buttons/pads."""

    def __init__(self, profile: dict | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MIDI Controller Editor")
        self.setMinimumSize(980, 460)

        self._learn_row: int | None = None

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Profile:"))
        self._name = QLineEdit()
        top.addWidget(self._name, 1)
        self._status = QLabel("Ready")
        self._status.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        top.addWidget(self._status)
        root.addLayout(top)

        self._table = QTableWidget(0, len(_HEADERS), self)
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(True)
        root.addWidget(self._table, 1)

        row_btns = QHBoxLayout()
        self._btn_add_cc = QPushButton("Add CC")
        self._btn_add_note = QPushButton("Add Note")
        self._btn_remove = QPushButton("Remove Selected")
        self._btn_learn = QPushButton("Learn Selected")
        self._btn_mpk = QPushButton("Load MPK Mini Plus Preset")
        row_btns.addWidget(self._btn_add_cc)
        row_btns.addWidget(self._btn_add_note)
        row_btns.addWidget(self._btn_remove)
        row_btns.addWidget(self._btn_learn)
        row_btns.addStretch(1)
        row_btns.addWidget(self._btn_mpk)
        root.addLayout(row_btns)

        info = QLabel(
            "Target dropdown includes all effect parameters plus: division, swing, mode, octave, latch, sync, gate, gpm"
        )
        info.setStyleSheet(f"color: {Colors.TEXT_DIM};")
        root.addWidget(info)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

        self._btn_add_cc.clicked.connect(lambda: self._add_row(default_input="cc"))
        self._btn_add_note.clicked.connect(lambda: self._add_row(default_input="note"))
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_learn.clicked.connect(self._arm_learn)
        self._btn_mpk.clicked.connect(self._load_mpk_defaults)

        self.set_profile(profile or default_midi_controller_profile())

    def set_profile(self, profile: dict) -> None:
        self._table.setRowCount(0)
        self._name.setText(str(profile.get("name", "Controller Profile")))
        mappings = profile.get("mappings", [])
        if not isinstance(mappings, list):
            return
        for mapping in mappings:
            if isinstance(mapping, dict):
                self._add_row(mapping=mapping)

    def get_profile(self) -> dict:
        mappings: list[dict] = []
        for row in range(self._table.rowCount()):
            target_combo = self._table.cellWidget(row, 7)
            target_value = ""
            if isinstance(target_combo, QComboBox):
                target_value = target_combo.currentText().strip()
            mappings.append(
                {
                    "enabled": self._get_combo_text(row, 0) == "on",
                    "input": self._get_combo_text(row, 1),
                    "number": self._to_int(self._table.item(row, 2), 0),
                    "channel": self._parse_channel(self._get_combo_text(row, 3)),
                    "trigger": self._get_combo_text(row, 4),
                    "mode": self._get_combo_text(row, 5),
                    "target_type": self._get_combo_text(row, 6),
                    "target": target_value,
                    "min": self._to_float(self._table.item(row, 8), 0.0),
                    "max": self._to_float(self._table.item(row, 9), 1.0),
                }
            )
        return {
            "name": self._name.text().strip() or "Controller Profile",
            "mappings": mappings,
        }

    def on_midi_cc(self, channel: int, cc_number: int, _value: int) -> None:
        if self._learn_row is None:
            return
        row = self._learn_row
        self._set_combo_text(row, 1, "cc")
        self._set_item_text(row, 2, str(cc_number))
        self._set_combo_text(row, 3, str(channel + 1))
        self._set_combo_text(row, 4, "change")
        self._status.setText(f"Learned CC {cc_number} on CH {channel + 1}")
        self._learn_row = None

    def on_midi_note(self, channel: int, note: int, _velocity: int, pressed: bool) -> None:
        if self._learn_row is None or not pressed:
            return
        row = self._learn_row
        self._set_combo_text(row, 1, "note")
        self._set_item_text(row, 2, str(note))
        self._set_combo_text(row, 3, str(channel + 1))
        self._set_combo_text(row, 4, "press")
        self._status.setText(f"Learned NOTE {note} on CH {channel + 1}")
        self._learn_row = None

    def _arm_learn(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            self._status.setText("Select a row first")
            return
        self._learn_row = row
        self._status.setText("Learn armed: move a knob or press a pad")

    def _load_mpk_defaults(self) -> None:
        self.set_profile(default_midi_controller_profile())
        self._status.setText("Loaded MPK Mini Plus preset")

    def _add_row(self, mapping: dict | None = None, default_input: str = "cc") -> None:
        data = {
            "enabled": True,
            "input": default_input,
            "number": 0,
            "channel": "all",
            "trigger": "change" if default_input == "cc" else "press",
            "mode": "absolute" if default_input == "cc" else "momentary",
            "target_type": "param",
            "target": "filter_frequency",
            "min": 0.0,
            "max": 1.0,
        }
        if isinstance(mapping, dict):
            data.update(mapping)

        row = self._table.rowCount()
        self._table.insertRow(row)

        self._table.setCellWidget(row, 0, self._combo(["on", "off"], "on" if data["enabled"] else "off"))
        self._table.setCellWidget(row, 1, self._combo(["cc", "note"], str(data["input"])))
        self._table.setItem(row, 2, QTableWidgetItem(str(data["number"])))
        self._table.setCellWidget(
            row,
            3,
            self._combo(["all"] + [str(i) for i in range(1, 17)], str(data["channel"])),
        )
        self._table.setCellWidget(row, 4, self._combo(["change", "press", "release"], str(data["trigger"])))
        self._table.setCellWidget(row, 5, self._combo(["absolute", "momentary", "toggle"], str(data["mode"])))
        target_type_combo = self._combo(["param", "action"], str(data["target_type"]))
        self._table.setCellWidget(row, 6, target_type_combo)
        self._table.setCellWidget(row, 7, self._target_combo(str(data["target_type"]), str(data["target"])))
        self._table.setItem(row, 8, QTableWidgetItem(str(data["min"])))
        self._table.setItem(row, 9, QTableWidgetItem(str(data["max"])))
        target_type_combo.currentTextChanged.connect(
            lambda _text, r=row: self._refresh_target_combo(r)
        )

    def _remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for row in rows:
            self._table.removeRow(row)
        self._status.setText("Removed selected rows")

    @staticmethod
    def _combo(options: list[str], current: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(options)
        idx = combo.findText(current)
        combo.setCurrentIndex(max(0, idx))
        return combo

    def _get_combo_text(self, row: int, col: int) -> str:
        combo = self._table.cellWidget(row, col)
        if isinstance(combo, QComboBox):
            return combo.currentText().strip().lower()
        return ""

    def _set_combo_text(self, row: int, col: int, value: str) -> None:
        combo = self._table.cellWidget(row, col)
        if isinstance(combo, QComboBox):
            idx = combo.findText(value)
            combo.setCurrentIndex(max(0, idx))

    def _get_item_text(self, row: int, col: int) -> str:
        item = self._table.item(row, col)
        if item is None:
            return ""
        return item.text().strip()

    def _set_item_text(self, row: int, col: int, value: str) -> None:
        item = self._table.item(row, col)
        if item is None:
            self._table.setItem(row, col, QTableWidgetItem(value))
            return
        item.setText(value)

    def _target_combo(self, target_type: str, current: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        if target_type == "action":
            combo.addItems(_ACTION_TARGETS)
        else:
            combo.addItems(_EFFECT_PARAM_TARGETS + _CUSTOM_PARAM_TARGETS)
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentText(current)
        return combo

    def _refresh_target_combo(self, row: int) -> None:
        target_type = self._get_combo_text(row, 6)
        current_target = ""
        current_widget = self._table.cellWidget(row, 7)
        if isinstance(current_widget, QComboBox):
            current_target = current_widget.currentText().strip()
        self._table.setCellWidget(row, 7, self._target_combo(target_type, current_target))

    @staticmethod
    def _to_int(item: QTableWidgetItem | None, default: int) -> int:
        if item is None:
            return default
        try:
            return int(item.text().strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_float(item: QTableWidgetItem | None, default: float) -> float:
        if item is None:
            return default
        try:
            return float(item.text().strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_channel(value: str) -> str | int:
        if value == "all":
            return "all"
        try:
            return int(value)
        except (TypeError, ValueError):
            return "all"
