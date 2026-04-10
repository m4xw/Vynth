"""On-screen MIDI piano keyboard widget with note highlighting."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QLinearGradient, QMouseEvent
from PyQt6.QtWidgets import QWidget, QSizePolicy

from vynth.ui.theme import Colors

# Piano layout constants
_FIRST_NOTE = 36   # C1
_LAST_NOTE = 96    # C6
_OCTAVES = 5
_WHITE_NOTES_PER_OCTAVE = 7
_TOTAL_WHITE_KEYS = _OCTAVES * _WHITE_NOTES_PER_OCTAVE + 1  # +1 for final C6

# Which notes in an octave (0-11) are white keys
_WHITE_OFFSETS = {0, 2, 4, 5, 7, 9, 11}
# Black key positions relative to their left white key (0-indexed semitone -> white key index)
_BLACK_KEY_MAP = {1: 0, 3: 1, 6: 3, 8: 4, 10: 5}  # semitone -> preceding white key index


def _is_white(note: int) -> bool:
    return (note % 12) in _WHITE_OFFSETS


def _white_key_index(note: int) -> int:
    """Return the cumulative white-key index for a given MIDI note."""
    octave = (note - _FIRST_NOTE) // 12
    semitone = (note - _FIRST_NOTE) % 12
    white_count = sum(1 for s in range(_WHITE_OFFSETS.__len__()) if s < semitone and s in {0, 2, 4, 5, 7, 9, 11})
    # Count whites below this semitone in the octave
    whites_below = 0
    for s in range(semitone):
        if s in _WHITE_OFFSETS:
            whites_below += 1
    return octave * _WHITE_NOTES_PER_OCTAVE + whites_below


class MIDIKeyboardWidget(QWidget):
    """Visual MIDI piano keyboard with note highlighting."""

    notePressed = pyqtSignal(int, int)   # note, velocity
    noteReleased = pyqtSignal(int)       # note

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # {note: velocity} for currently active notes
        self._active_notes: dict[int, int] = {}
        self._pressed_note: int | None = None

        # Pre-build key geometry lists (rebuilt on resize)
        self._white_rects: list[tuple[int, QRectF]] = []  # (note, rect)
        self._black_rects: list[tuple[int, QRectF]] = []

        self._build_key_rects()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def highlight_note(self, note: int, velocity: int) -> None:
        """Highlight a note from an external MIDI source."""
        if _FIRST_NOTE <= note <= _LAST_NOTE:
            self._active_notes[note] = max(1, min(velocity, 127))
            self.update()

    def release_note(self, note: int) -> None:
        """Remove highlight from a note."""
        if note in self._active_notes:
            del self._active_notes[note]
            self.update()

    def clear_all(self) -> None:
        """Remove all active highlights."""
        self._active_notes.clear()
        self.update()

    # ------------------------------------------------------------------
    # Internal geometry
    # ------------------------------------------------------------------

    def _build_key_rects(self) -> None:
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        white_w = w / _TOTAL_WHITE_KEYS
        black_w = white_w * 0.6
        black_h = h * 0.6

        self._white_rects.clear()
        self._black_rects.clear()

        white_idx = 0
        for note in range(_FIRST_NOTE, _LAST_NOTE + 1):
            if _is_white(note):
                x = white_idx * white_w
                self._white_rects.append((note, QRectF(x, 0, white_w, h)))
                white_idx += 1
            else:
                # Black key sits between its two neighboring white keys
                x = white_idx * white_w - black_w / 2
                self._black_rects.append((note, QRectF(x, 0, black_w, black_h)))

        # Because black keys are enumerated when we hit them in the loop,
        # white_idx only advances on white keys, so positions are correct.

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._build_key_rects()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        accent = QColor(Colors.ACCENT_PRIMARY)
        border_color = QColor(Colors.BORDER)

        # --- White keys ---
        for note, rect in self._white_rects:
            velocity = self._active_notes.get(note, 0)
            if velocity:
                brightness = 0.4 + 0.6 * (velocity / 127.0)
                base = QColor(Colors.ACCENT_PRIMARY)
                fill = QColor(
                    int(base.red() * brightness + 255 * (1 - brightness)),
                    int(base.green() * brightness + 255 * (1 - brightness)),
                    int(base.blue() * brightness + 255 * (1 - brightness)),
                )
            else:
                fill = QColor("#e0e0e0")

            painter.setPen(QPen(border_color, 1))
            painter.setBrush(QBrush(fill))
            painter.drawRoundedRect(rect, 2, 2)

        # --- Black keys (drawn on top) ---
        for note, rect in self._black_rects:
            velocity = self._active_notes.get(note, 0)
            if velocity:
                brightness = 0.5 + 0.5 * (velocity / 127.0)
                base = QColor(Colors.ACCENT_PRIMARY)
                fill = QColor(
                    int(base.red() * brightness),
                    int(base.green() * brightness),
                    int(base.blue() * brightness),
                )
            else:
                fill = QColor(Colors.BG_MEDIUM)

            # Subtle top-to-bottom gradient for depth
            grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            grad.setColorAt(0.0, fill.lighter(120))
            grad.setColorAt(1.0, fill.darker(130))

            painter.setPen(QPen(QColor("#000000"), 1))
            painter.setBrush(QBrush(grad))
            painter.drawRoundedRect(rect, 2, 2)

        # --- C-key labels ---
        font = QFont("Segoe UI", 7)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(Colors.TEXT_SECONDARY))
        for note, rect in self._white_rects:
            if note % 12 == 0:  # C notes
                octave = (note // 12) - 1
                label = f"C{octave}"
                painter.drawText(rect.adjusted(2, 0, 0, -4), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, label)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _note_at(self, x: float, y: float) -> int | None:
        """Return the MIDI note at the given pixel position (black keys first)."""
        for note, rect in self._black_rects:
            if rect.contains(x, y):
                return note
        for note, rect in self._white_rects:
            if rect.contains(x, y):
                return note
        return None

    def _velocity_from_y(self, y: float, rect: QRectF) -> int:
        """Map vertical click position to velocity (top=soft, bottom=loud)."""
        ratio = max(0.0, min(1.0, (y - rect.top()) / rect.height()))
        return max(1, min(127, int(30 + ratio * 97)))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        note = self._note_at(pos.x(), pos.y())
        if note is not None:
            # Find the rect for velocity calculation
            vel = 100
            for n, rect in self._black_rects + self._white_rects:
                if n == note:
                    vel = self._velocity_from_y(pos.y(), rect)
                    break
            self._pressed_note = note
            self._active_notes[note] = vel
            self.notePressed.emit(note, vel)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._pressed_note is not None:
            note = self._pressed_note
            self._active_notes.pop(note, None)
            self.noteReleased.emit(note)
            self._pressed_note = None
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._pressed_note is None:
            return
        pos = event.position()
        note = self._note_at(pos.x(), pos.y())
        if note != self._pressed_note:
            # Release old note
            old = self._pressed_note
            self._active_notes.pop(old, None)
            self.noteReleased.emit(old)
            # Press new note
            if note is not None:
                vel = 100
                for n, rect in self._black_rects + self._white_rects:
                    if n == note:
                        vel = self._velocity_from_y(pos.y(), rect)
                        break
                self._pressed_note = note
                self._active_notes[note] = vel
                self.notePressed.emit(note, vel)
            else:
                self._pressed_note = None
            self.update()
