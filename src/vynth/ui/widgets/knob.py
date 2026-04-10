from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics, QMouseEvent
from PyQt6.QtWidgets import QWidget, QLineEdit, QVBoxLayout

from vynth.ui.theme import Colors


class RotaryKnob(QWidget):
    """Custom rotary knob with value display."""

    valueChanged = pyqtSignal(float)

    # Arc geometry
    _ARC_START = -225  # degrees (Qt uses 0=3-o'clock, CCW positive)
    _ARC_SPAN = 270

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        name: str = "",
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: float = 0.0,
        default_value: float | None = None,
        suffix: str = "",
        decimals: int = 2,
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._minimum = minimum
        self._maximum = maximum
        self._value = self._clamp(value)
        self._default_value = default_value if default_value is not None else value
        self._suffix = suffix
        self._decimals = decimals

        self._drag_start_y: int | None = None
        self._drag_start_value: float = 0.0
        self._sensitivity = 0.005  # value-range fraction per pixel

        self._edit: QLineEdit | None = None

        self.setFixedSize(60, 80)
        self.setMouseTracking(True)

    # -- public properties ---------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, v: str) -> None:
        self._name = v
        self.update()

    @property
    def minimum(self) -> float:
        return self._minimum

    @minimum.setter
    def minimum(self, v: float) -> None:
        self._minimum = v
        self._value = self._clamp(self._value)
        self.update()

    @property
    def maximum(self) -> float:
        return self._maximum

    @maximum.setter
    def maximum(self, v: float) -> None:
        self._maximum = v
        self._value = self._clamp(self._value)
        self.update()

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, v: float) -> None:
        v = self._clamp(v)
        if v != self._value:
            self._value = v
            self.valueChanged.emit(self._value)
            self.update()

    @property
    def default_value(self) -> float:
        return self._default_value

    @default_value.setter
    def default_value(self, v: float) -> None:
        self._default_value = v

    @property
    def suffix(self) -> str:
        return self._suffix

    @suffix.setter
    def suffix(self, v: str) -> None:
        self._suffix = v
        self.update()

    @property
    def decimals(self) -> int:
        return self._decimals

    @decimals.setter
    def decimals(self, v: int) -> None:
        self._decimals = v
        self.update()

    # -- helpers -------------------------------------------------------------

    def _clamp(self, v: float) -> float:
        return max(self._minimum, min(self._maximum, v))

    def _normalized(self) -> float:
        rng = self._maximum - self._minimum
        if rng == 0:
            return 0.0
        return (self._value - self._minimum) / rng

    def _format_value(self) -> str:
        v = self._value
        if self._suffix == "dB":
            if v <= self._minimum:
                return "-inf"
            return f"{v:.{self._decimals}f}"
        if self._suffix in ("Hz", "hz"):
            if abs(v) >= 1000:
                return f"{v / 1000:.1f}k"
            return f"{v:.0f}"
        return f"{v:.{self._decimals}f}"

    # -- painting ------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        knob_size = 50
        knob_x = (self.width() - knob_size) / 2
        knob_y = 0.0
        knob_rect = QRectF(knob_x, knob_y, knob_size, knob_size)

        arc_margin = 5.0
        arc_rect = knob_rect.adjusted(arc_margin, arc_margin, -arc_margin, -arc_margin)

        # Background circle
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(Colors.BG_LIGHT))
        p.drawEllipse(knob_rect)

        # Background arc (full sweep)
        pen_bg = QPen(QColor(Colors.BG_MEDIUM), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen_bg)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(arc_rect, int(self._ARC_START * 16), int(self._ARC_SPAN * 16))

        # Value arc
        norm = self._normalized()
        if norm > 0:
            value_span = self._ARC_SPAN * norm
            pen_val = QPen(QColor(Colors.ACCENT_PRIMARY), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(pen_val)
            p.drawArc(arc_rect, int(self._ARC_START * 16), int(value_span * 16))

        # Dot indicator on arc
        angle_deg = -225 + self._ARC_SPAN * norm
        angle_rad = math.radians(angle_deg)
        cx = knob_rect.center().x()
        cy = knob_rect.center().y()
        dot_r = arc_rect.width() / 2
        dot_x = cx + dot_r * math.cos(angle_rad)
        dot_y = cy - dot_r * math.sin(angle_rad)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(Colors.ACCENT_PRIMARY))
        p.drawEllipse(QPointF(dot_x, dot_y), 3.0, 3.0)

        # Center value text
        value_text = self._format_value()
        if self._suffix and self._suffix not in ("Hz", "hz"):
            value_text += self._suffix
        font_val = QFont("Segoe UI", 8)
        font_val.setBold(True)
        p.setFont(font_val)
        p.setPen(QColor(Colors.TEXT_PRIMARY))
        fm = QFontMetrics(font_val)
        tr = fm.boundingRect(value_text)
        tx = cx - tr.width() / 2
        ty = cy + tr.height() / 4
        p.drawText(QPointF(tx, ty), value_text)

        # Label below knob
        if self._name:
            font_lbl = QFont("Segoe UI", 7)
            p.setFont(font_lbl)
            p.setPen(QColor(Colors.TEXT_SECONDARY))
            fm2 = QFontMetrics(font_lbl)
            lw = fm2.horizontalAdvance(self._name)
            lx = (self.width() - lw) / 2
            ly = knob_size + 16
            p.drawText(QPointF(lx, ly), self._name)

        p.end()

    # -- mouse interaction ---------------------------------------------------

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_start_y = int(ev.position().y())
            self._drag_start_value = self._value
            self.setCursor(Qt.CursorShape.BlankCursor)
        elif ev.button() == Qt.MouseButton.RightButton:
            self.value = self._default_value

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start_y is not None:
            dy = self._drag_start_y - int(ev.position().y())
            rng = self._maximum - self._minimum
            self.value = self._drag_start_value + dy * self._sensitivity * rng

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_start_y = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self._show_editor()

    def wheelEvent(self, ev) -> None:  # noqa: N802
        delta = ev.angleDelta().y()
        rng = self._maximum - self._minimum
        step = rng * 0.01
        self.value = self._value + (step if delta > 0 else -step)

    # -- inline editor -------------------------------------------------------

    def _show_editor(self) -> None:
        if self._edit is not None:
            return
        self._edit = QLineEdit(self)
        self._edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._edit.setStyleSheet(
            f"background:{Colors.BG_MEDIUM};color:{Colors.TEXT_PRIMARY};"
            f"border:1px solid {Colors.ACCENT_PRIMARY};border-radius:3px;"
            "font-size:9px;padding:1px;"
        )
        self._edit.setText(f"{self._value:.{self._decimals}f}")
        self._edit.selectAll()
        self._edit.setGeometry(5, 15, 50, 20)
        self._edit.setFocus()
        self._edit.returnPressed.connect(self._commit_editor)
        self._edit.editingFinished.connect(self._close_editor)
        self._edit.show()

    def _commit_editor(self) -> None:
        if self._edit is None:
            return
        try:
            self.value = float(self._edit.text())
        except ValueError:
            pass
        self._close_editor()

    def _close_editor(self) -> None:
        if self._edit is not None:
            self._edit.deleteLater()
            self._edit = None
