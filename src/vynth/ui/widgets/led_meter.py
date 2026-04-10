from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QRectF, QTimer
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtWidgets import QWidget

from vynth.ui.theme import Colors


class LEDMeter(QWidget):
    """Stereo LED peak meter."""

    _NUM_SEGMENTS = 24
    _SEG_GAP = 1
    _BAR_GAP = 2  # gap between L and R bars
    _PEAK_HOLD_SEC = 1.5
    _DECAY_DB_PER_SEC = 20.0
    _FPS = 60

    # Segment color zones (inclusive ranges)
    _GREEN_END = 14   # 0..14
    _YELLOW_END = 19  # 15..19
    # 20..23 = red

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(20, 120)

        # Current display levels (0..1 linear)
        self._level_l: float = 0.0
        self._level_r: float = 0.0

        # Smoothed display levels (for decay)
        self._display_l: float = 0.0
        self._display_r: float = 0.0

        # Peak hold
        self._peak_l: float = 0.0
        self._peak_r: float = 0.0
        self._peak_l_time: float = 0.0
        self._peak_r_time: float = 0.0

        self._last_tick: float = time.monotonic()

        # Refresh timer
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // self._FPS)

        # Pre-compute segment colors
        self._seg_colors: list[QColor] = []
        for i in range(self._NUM_SEGMENTS):
            if i <= self._GREEN_END:
                self._seg_colors.append(QColor(Colors.METER_GREEN))
            elif i <= self._YELLOW_END:
                self._seg_colors.append(QColor(Colors.METER_YELLOW))
            else:
                self._seg_colors.append(QColor(Colors.METER_RED))

        self._seg_off_color = QColor(Colors.BG_LIGHT)

    # -- public API ----------------------------------------------------------

    def set_levels(self, left: float, right: float) -> None:
        """Set current input levels (0.0 – 1.0 linear)."""
        self._level_l = max(0.0, min(1.0, left))
        self._level_r = max(0.0, min(1.0, right))

    # -- internal ------------------------------------------------------------

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        # Decay factor: convert dB/s to linear multiplier per frame
        # -20 dB/s  -> multiply by 10^(-20*dt/20) = 10^(-dt)
        decay = 10.0 ** (-dt * self._DECAY_DB_PER_SEC / 20.0)

        for ch in ("l", "r"):
            level = getattr(self, f"_level_{ch}")
            display = getattr(self, f"_display_{ch}")
            peak = getattr(self, f"_peak_{ch}")
            peak_time = getattr(self, f"_peak_{ch}_time")

            # Rise instantly, decay smoothly
            if level >= display:
                display = level
            else:
                display *= decay
                if display < 0.001:
                    display = 0.0

            # Peak hold
            if level >= peak:
                peak = level
                peak_time = now
            elif now - peak_time > self._PEAK_HOLD_SEC:
                peak *= decay
                if peak < 0.001:
                    peak = 0.0

            setattr(self, f"_display_{ch}", display)
            setattr(self, f"_peak_{ch}", peak)
            setattr(self, f"_peak_{ch}_time", peak_time)

        self.update()

    # -- painting ------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        bar_w = (w - self._BAR_GAP) / 2.0
        seg_h = (h - (self._NUM_SEGMENTS - 1) * self._SEG_GAP) / self._NUM_SEGMENTS

        for ch_idx, (display, peak) in enumerate(
            [(self._display_l, self._peak_l), (self._display_r, self._peak_r)]
        ):
            x0 = ch_idx * (bar_w + self._BAR_GAP)
            lit_count = int(display * self._NUM_SEGMENTS)
            peak_seg = int(peak * (self._NUM_SEGMENTS - 1)) if peak > 0 else -1

            for seg in range(self._NUM_SEGMENTS):
                # Segment 0 = bottom, so y goes bottom-up
                y = h - (seg + 1) * (seg_h + self._SEG_GAP) + self._SEG_GAP
                rect = QRectF(x0, y, bar_w, seg_h)

                if seg < lit_count:
                    p.setBrush(self._seg_colors[seg])
                elif seg == peak_seg:
                    p.setBrush(self._seg_colors[seg])
                else:
                    p.setBrush(self._seg_off_color)

                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(rect, 1.5, 1.5)

        p.end()
