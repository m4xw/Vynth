"""Sample library browser with list view."""

from __future__ import annotations

import os
from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLineEdit, QPushButton, QMenu, QSizePolicy, QAbstractItemView,
)

from vynth.ui.theme import Colors


@dataclass
class _SampleInfo:
    name: str
    duration_s: float
    sr: int
    channels: int


class SampleBrowser(QWidget):
    """Sample library browser with list view."""

    sampleSelected = pyqtSignal(str)
    sampleDoubleClicked = pyqtSignal(str)

    _ROLE_META = Qt.ItemDataRole.UserRole

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- search / filter ---
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search samples…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)

        # --- list ---
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setDragEnabled(True)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.currentItemChanged.connect(self._on_selection)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {Colors.BG_DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 3px;
            }}
            QListWidget::item {{
                padding: 4px 6px;
            }}
            QListWidget::item:selected {{
                background-color: {Colors.BG_HIGHLIGHT};
            }}
            """
        )
        layout.addWidget(self._list, 1)

        # --- toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        self._btn_load = QPushButton("Load")
        self._btn_remove = QPushButton("Remove")
        self._btn_sort = QPushButton("Sort A-Z")

        for btn in (self._btn_load, self._btn_remove, self._btn_sort):
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._btn_load.clicked.connect(lambda: self.sampleDoubleClicked.emit("__load__"))
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_sort.clicked.connect(lambda: self._list.sortItems(Qt.SortOrder.AscendingOrder))

        toolbar.addWidget(self._btn_load)
        toolbar.addWidget(self._btn_remove)
        toolbar.addWidget(self._btn_sort)
        layout.addLayout(toolbar)

    # -- public API ----------------------------------------------------------

    def add_sample(self, name: str, duration_s: float, sr: int, channels: int) -> None:
        ch_icon = "🔊" if channels >= 2 else "🔈"
        label = f"{name}  ({duration_s:.2f}s · {sr} Hz · {ch_icon})"
        item = QListWidgetItem(label)
        item.setData(self._ROLE_META, _SampleInfo(name, duration_s, sr, channels))
        self._list.addItem(item)

    def remove_sample(self, name: str) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            meta: _SampleInfo | None = item.data(self._ROLE_META) if item else None
            if meta and meta.name == name:
                self._list.takeItem(i)
                return

    def get_selected(self) -> str | None:
        item = self._list.currentItem()
        if item is None:
            return None
        meta: _SampleInfo | None = item.data(self._ROLE_META)
        return meta.name if meta else None

    def clear(self) -> None:
        self._list.clear()

    # -- internals -----------------------------------------------------------

    def _apply_filter(self, text: str) -> None:
        lower = text.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None:
                item.setHidden(lower not in (item.text().lower()))

    def _on_selection(self, current: QListWidgetItem | None, _prev) -> None:
        if current is None:
            return
        meta: _SampleInfo | None = current.data(self._ROLE_META)
        if meta:
            self.sampleSelected.emit(meta.name)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        meta: _SampleInfo | None = item.data(self._ROLE_META)
        if meta:
            self.sampleDoubleClicked.emit(meta.name)

    def _remove_selected(self) -> None:
        name = self.get_selected()
        if name:
            self.remove_sample(name)

    def _show_context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        meta: _SampleInfo | None = item.data(self._ROLE_META)
        if meta is None:
            return

        menu = QMenu(self)
        menu.addAction("Load", lambda: self.sampleDoubleClicked.emit(meta.name))
        menu.addAction("Remove", lambda: self.remove_sample(meta.name))
        menu.addAction("Rename", lambda: self._list.editItem(item))
        menu.addAction("Set as Active", lambda: self.sampleSelected.emit(meta.name))
        menu.addAction("Show in Explorer", lambda: self._open_explorer(meta.name))
        menu.exec(self._list.mapToGlobal(pos))

    @staticmethod
    def _open_explorer(name: str) -> None:
        path = os.path.abspath(name)
        folder = os.path.dirname(path) if os.path.isfile(path) else path
        if os.path.isdir(folder):
            os.startfile(folder)  # type: ignore[attr-defined]
