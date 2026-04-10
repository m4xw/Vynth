from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics, QMouseEvent, QLinearGradient
from PyQt6.QtWidgets import QWidget

from vynth.ui.theme import Colors


class Fader(QWidget):
    """Custom fader/slider with professional styling."""

    valueChanged = pyqtSignal(float)

    _THUMB_W = 12
    _THUMB_H = 20
    _TRACK_WIDTH = 2

    # dB markings for vertical mode (value, label)
    _DB_MARKS: list[tuple[float, str]] = [
        (1.0, "0"),
        (0.75, "-6"),
        (0.5, "-12"),
        (0.25, "-24"),
        (0.1, "-48"),
        (0.0, "-∞"),
    ]

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        orientation: Qt.Orientation = Qt.Orientation.Vertical,
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: float = 0.0,
        default_value: float | None = None,
        show_db_scale: bool = True,
    ) -> None:
        super().__init__(parent)
        self._orientation = orientation
        self._minimum = minimum
        self._maximum = maximum
        self._value = self._clamp(value)
        self._default_value = default_value if default_value is not None else value
        self._show_db_scale = show_db_scale
        self._dragging = False

        if orientation == Qt.Orientation.Vertical:
            self.setFixedSize(30, 120)
        else:
            self.setFixedSize(120, 30)

    # -- properties ----------------------------------------------------------

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
    def orientation(self) -> Qt.Orientation:
        return self._orientation

    # -- helpers -------------------------------------------------------------

    def _clamp(self, v: float) -> float:
        return max(self._minimum, min(self._maximum, v))

    def _normalized(self) -> float:
        rng = self._maximum - self._minimum
        if rng == 0:
            return 0.0
        return (self._value - self._minimum) / rng

    def _track_rect(self) -> QRectF:
        """Usable track area (thumb center travels here)."""
        half_thumb = self._THUMB_H / 2
        if self._orientation == Qt.Orientation.Vertical:
            cx = self.width() / 2
            return QRectF(cx, half_thumb, 0, self.height() - self._THUMB_H)
        else:
            cy = self.height() / 2
            return QRectF(half_thumb, cy, self.width() - self._THUMB_H, 0)

    def _pos_to_value(self, x: float, y: float) -> float:
        tr = self._track_rect()
        if self._orientation == Qt.Orientation.Vertical:
            norm = 1.0 - (y - tr.top()) / tr.height() if tr.height() else 0.0
        else:
            norm = (x - tr.left()) / tr.width() if tr.width() else 0.0
        norm = max(0.0, min(1.0, norm))
        return self._minimum + norm * (self._maximum - self._minimum)

    def _thumb_center(self) -> tuple[float, float]:
        tr = self._track_rect()
        norm = self._normalized()
        if self._orientation == Qt.Orientation.Vertical:
            y = tr.bottom() - norm * tr.height()
            return (self.width() / 2, y)
        else:
            x = tr.left() + norm * tr.width()
            return (x, self.height() / 2)

    # -- painting ------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        vertical = self._orientation == Qt.Orientation.Vertical
        tr = self._track_rect()
        tcx, tcy = self._thumb_center()

        # Track background
        track_pen = QPen(QColor(Colors.BG_LIGHT), self._TRACK_WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(track_pen)
        if vertical:
            p.drawLine(int(tr.x()), int(tr.top()), int(tr.x()), int(tr.bottom()))
        else:
            p.drawLine(int(tr.left()), int(tr.y()), int(tr.right()), int(tr.y()))

        # Filled portion
        fill_pen = QPen(QColor(Colors.ACCENT_PRIMARY), self._TRACK_WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(fill_pen)
        if vertical:
            p.drawLine(int(tr.x()), int(tcy), int(tr.x()), int(tr.bottom()))
        else:
            p.drawLine(int(tr.left()), int(tr.y()), int(tcx), int(tr.y()))

        # dB scale markings (vertical only)
        if vertical and self._show_db_scale:
            mark_font = QFont("Segoe UI", 6)
            p.setFont(mark_font)
            p.setPen(QColor(Colors.TEXT_SECONDARY))
            for norm_val, label in self._DB_MARKS:
                my = tr.bottom() - norm_val * tr.height()
                # small tick
                p.drawLine(int(tr.x() - 4), int(my), int(tr.x() - 2), int(my))
                fm = QFontMetrics(mark_font)
                lw = fm.horizontalAdvance(label)
                # Only draw if space allows (left side)
                if tr.x() - 5 - lw >= 0:
                    p.drawText(int(tr.x() - 5 - lw), int(my + fm.ascent() / 2 - 1), label)

        # Thumb
        tw, th = self._THUMB_W, self._THUMB_H
        if not vertical:
            tw, th = th, tw  # swap for horizontal
        thumb_rect = QRectF(tcx - tw / 2, tcy - th / 2, tw, th)

        p.setPen(QPen(QColor(Colors.BORDER), 1.0))
        grad = QLinearGradient(thumb_rect.topLeft(), thumb_rect.bottomLeft())
        grad.setColorAt(0.0, QColor("#3a3a5e"))
        grad.setColorAt(1.0, QColor("#252540"))
        p.setBrush(grad)
        p.drawRoundedRect(thumb_rect, 3.0, 3.0)

        # Center line on thumb
        p.setPen(QPen(QColor(Colors.ACCENT_PRIMARY), 1.0))
        if vertical:
            p.drawLine(int(thumb_rect.left() + 3), int(tcy), int(thumb_rect.right() - 3), int(tcy))
        else:
            p.drawLine(int(tcx), int(thumb_rect.top() + 3), int(tcx), int(thumb_rect.bottom() - 3))

        p.end()

    # -- mouse interaction ---------------------------------------------------

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.value = self._pos_to_value(ev.position().x(), ev.position().y())

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self.value = self._pos_to_value(ev.position().x(), ev.position().y())

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self.value = self._default_value

    def wheelEvent(self, ev) -> None:  # noqa: N802
        delta = ev.angleDelta().y()
        rng = self._maximum - self._minimum
        step = rng * 0.01
        self.value = self._value + (step if delta > 0 else -step)
