"""Interactive ADSR envelope display with draggable handles."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QPainterPath,
    QLinearGradient, QMouseEvent,
)
from PyQt6.QtWidgets import QWidget, QSizePolicy

from vynth.ui.theme import Colors

# Constraints
_MIN_TIME_MS = 1.0
_MAX_TIME_MS = 5000.0
_SUSTAIN_DISPLAY_RATIO = 0.2  # fixed fraction of width for sustain segment


class ADSRDisplay(QWidget):
    """Visual ADSR curve with interactive drag handles."""

    paramChanged = pyqtSignal(str, float)  # param_name, value

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # ADSR parameters
        self._attack_ms: float = 50.0
        self._decay_ms: float = 150.0
        self._sustain: float = 0.7     # 0.0–1.0
        self._release_ms: float = 300.0

        # Drag state
        self._dragging: int | None = None  # handle index 0-3
        self._handle_radius: float = 4.0

        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_params(self, attack_ms: float, decay_ms: float, sustain: float, release_ms: float) -> None:
        self._attack_ms = max(_MIN_TIME_MS, min(_MAX_TIME_MS, attack_ms))
        self._decay_ms = max(_MIN_TIME_MS, min(_MAX_TIME_MS, decay_ms))
        self._sustain = max(0.0, min(1.0, sustain))
        self._release_ms = max(_MIN_TIME_MS, min(_MAX_TIME_MS, release_ms))
        self.update()

    def get_params(self) -> tuple[float, float, float, float]:
        return (self._attack_ms, self._decay_ms, self._sustain, self._release_ms)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _margins(self) -> tuple[float, float, float, float]:
        """Return (left, top, right, bottom) padding."""
        return 30.0, 12.0, 12.0, 20.0

    def _draw_area(self) -> QRectF:
        ml, mt, mr, mb = self._margins()
        return QRectF(ml, mt, self.width() - ml - mr, self.height() - mt - mb)

    def _segment_widths(self, area: QRectF) -> tuple[float, float, float, float]:
        """Distribute available width among A, D, S, R segments proportionally."""
        total_time = self._attack_ms + self._decay_ms + self._release_ms
        sustain_w = area.width() * _SUSTAIN_DISPLAY_RATIO
        time_w = area.width() - sustain_w

        if total_time <= 0:
            a_w = d_w = r_w = time_w / 3
        else:
            a_w = time_w * (self._attack_ms / total_time)
            d_w = time_w * (self._decay_ms / total_time)
            r_w = time_w * (self._release_ms / total_time)

        return a_w, d_w, sustain_w, r_w

    def _handle_points(self) -> list[QPointF]:
        """Return the four drag-handle positions in widget coordinates."""
        area = self._draw_area()
        a_w, d_w, s_w, r_w = self._segment_widths(area)

        x0 = area.left()
        y_bottom = area.bottom()
        y_top = area.top()
        sustain_y = y_bottom - self._sustain * area.height()

        p_attack = QPointF(x0 + a_w, y_top)                          # handle 0
        p_decay = QPointF(x0 + a_w + d_w, sustain_y)                 # handle 1
        p_sustain = QPointF(x0 + a_w + d_w + s_w, sustain_y)         # handle 2
        p_release = QPointF(x0 + a_w + d_w + s_w + r_w, y_bottom)    # handle 3

        return [p_attack, p_decay, p_sustain, p_release]

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        area = self._draw_area()
        a_w, d_w, s_w, r_w = self._segment_widths(area)
        accent = QColor(Colors.ACCENT_PRIMARY)

        x0 = area.left()
        y_bottom = area.bottom()
        y_top = area.top()
        sustain_y = y_bottom - self._sustain * area.height()

        # --- Background ---
        painter.fillRect(self.rect(), QColor(Colors.BG_DARK))

        # --- Grid lines ---
        grid_pen = QPen(QColor(Colors.BORDER), 1, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        # Horizontal lines at 25 / 50 / 75 %
        for frac in (0.25, 0.5, 0.75):
            gy = y_bottom - frac * area.height()
            painter.drawLine(QPointF(area.left(), gy), QPointF(area.right(), gy))
        # Vertical segment boundaries
        seg_x = x0
        for sw in (a_w, d_w, s_w):
            seg_x += sw
            painter.drawLine(QPointF(seg_x, y_top), QPointF(seg_x, y_bottom))

        # --- Build ADSR path ---
        origin = QPointF(x0, y_bottom)

        path = QPainterPath()
        path.moveTo(origin)

        # Attack: exponential-ish curve (quadratic Bézier)
        p_attack = QPointF(x0 + a_w, y_top)
        ctrl_a = QPointF(x0 + a_w * 0.4, y_top)
        path.quadTo(ctrl_a, p_attack)

        # Decay: exponential curve to sustain
        p_decay = QPointF(x0 + a_w + d_w, sustain_y)
        ctrl_d = QPointF(x0 + a_w + d_w * 0.3, sustain_y)
        path.quadTo(ctrl_d, p_decay)

        # Sustain: horizontal line
        p_sustain_end = QPointF(x0 + a_w + d_w + s_w, sustain_y)
        path.lineTo(p_sustain_end)

        # Release: exponential curve to zero
        p_release = QPointF(x0 + a_w + d_w + s_w + r_w, y_bottom)
        ctrl_r = QPointF(x0 + a_w + d_w + s_w + r_w * 0.3, y_bottom)
        path.quadTo(ctrl_r, p_release)

        # --- Fill under curve ---
        fill_path = QPainterPath(path)
        fill_path.lineTo(QPointF(x0 + a_w + d_w + s_w + r_w, y_bottom))
        fill_path.lineTo(origin)
        fill_path.closeSubpath()

        fill_grad = QLinearGradient(QPointF(0, y_top), QPointF(0, y_bottom))
        fill_color = QColor(accent)
        fill_color.setAlpha(60)
        fill_grad.setColorAt(0.0, fill_color)
        fill_color_bot = QColor(accent)
        fill_color_bot.setAlpha(10)
        fill_grad.setColorAt(1.0, fill_color_bot)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill_grad))
        painter.drawPath(fill_path)

        # --- Curve stroke ---
        painter.setPen(QPen(accent, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # --- Drag handles ---
        handles = self._handle_points()
        for i, pt in enumerate(handles):
            if i == self._dragging:
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.setBrush(QBrush(accent.lighter(140)))
            else:
                painter.setPen(QPen(accent, 1.5))
                painter.setBrush(QBrush(QColor(Colors.BG_DARK)))
            painter.drawEllipse(pt, self._handle_radius, self._handle_radius)

        # --- Axis labels ---
        font = QFont("Segoe UI", 7)
        painter.setFont(font)
        painter.setPen(QColor(Colors.TEXT_SECONDARY))

        # Segment labels
        seg_labels = ["A", "D", "S", "R"]
        seg_starts = [x0, x0 + a_w, x0 + a_w + d_w, x0 + a_w + d_w + s_w]
        seg_ws = [a_w, d_w, s_w, r_w]
        for label, sx, sw in zip(seg_labels, seg_starts, seg_ws):
            painter.drawText(QRectF(sx, y_bottom + 2, sw, 16),
                             Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, label)

        # Value labels near handles
        value_font = QFont("Segoe UI", 6)
        painter.setFont(value_font)
        painter.setPen(QColor(Colors.TEXT_SECONDARY))
        value_texts = [
            f"{self._attack_ms:.0f}ms",
            f"{self._decay_ms:.0f}ms",
            f"{self._sustain:.0%}",
            f"{self._release_ms:.0f}ms",
        ]
        for pt, txt in zip(handles, value_texts):
            painter.drawText(QRectF(pt.x() - 24, pt.y() - 16, 48, 12),
                             Qt.AlignmentFlag.AlignCenter, txt)

        # Y-axis amplitude markers
        painter.setPen(QColor(Colors.TEXT_SECONDARY))
        for frac, label in ((1.0, "1.0"), (0.5, "0.5"), (0.0, "0.0")):
            gy = y_bottom - frac * area.height()
            painter.drawText(QRectF(0, gy - 6, area.left() - 4, 12),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _hit_handle(self, pos: QPointF) -> int | None:
        handles = self._handle_points()
        hit_radius = self._handle_radius + 4
        for i, pt in enumerate(handles):
            dx = pos.x() - pt.x()
            dy = pos.y() - pt.y()
            if dx * dx + dy * dy <= hit_radius * hit_radius:
                return i
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        hit = self._hit_handle(event.position())
        if hit is not None:
            self._dragging = hit
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()

        if self._dragging is None:
            # Hover cursor
            hit = self._hit_handle(pos)
            self.setCursor(Qt.CursorShape.PointingHandCursor if hit is not None else Qt.CursorShape.ArrowCursor)
            return

        area = self._draw_area()
        a_w, d_w, s_w, r_w = self._segment_widths(area)
        total_time_w = area.width() * (1.0 - _SUSTAIN_DISPLAY_RATIO)
        total_time = self._attack_ms + self._decay_ms + self._release_ms

        def _x_to_time(x_offset: float) -> float:
            """Convert pixel offset to milliseconds (clamped)."""
            if total_time_w <= 0:
                return _MIN_TIME_MS
            ratio = max(0.0, min(1.0, x_offset / total_time_w))
            return max(_MIN_TIME_MS, min(_MAX_TIME_MS, ratio * total_time))

        def _y_to_sustain(y: float) -> float:
            return max(0.0, min(1.0, (area.bottom() - y) / area.height()))

        x0 = area.left()

        if self._dragging == 0:
            # Attack handle: horizontal only
            new_a = _x_to_time(pos.x() - x0)
            if new_a != self._attack_ms:
                self._attack_ms = new_a
                self.paramChanged.emit("attack", self._attack_ms)

        elif self._dragging == 1:
            # Decay handle: horizontal for decay, vertical for sustain
            attack_px = a_w
            new_d = _x_to_time(pos.x() - x0 - attack_px)
            new_s = _y_to_sustain(pos.y())
            changed = False
            if new_d != self._decay_ms:
                self._decay_ms = new_d
                self.paramChanged.emit("decay", self._decay_ms)
                changed = True
            if new_s != self._sustain:
                self._sustain = new_s
                self.paramChanged.emit("sustain", self._sustain)
                changed = True

        elif self._dragging == 2:
            # Sustain handle: vertical only
            new_s = _y_to_sustain(pos.y())
            if new_s != self._sustain:
                self._sustain = new_s
                self.paramChanged.emit("sustain", self._sustain)

        elif self._dragging == 3:
            # Release handle: horizontal only
            release_start_x = x0 + a_w + d_w + s_w
            new_r = _x_to_time(pos.x() - release_start_x)
            if new_r != self._release_ms:
                self._release_ms = new_r
                self.paramChanged.emit("release", self._release_ms)

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._dragging is not None:
            self._dragging = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
