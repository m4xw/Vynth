"""High-performance waveform display with zoom, scroll, and selection."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QScrollBar, QVBoxLayout, QWidget

from vynth.config import SAMPLE_RATE, WAVEFORM_DOWNSAMPLE_THRESHOLD
from vynth.ui.theme import Colors

pg.setConfigOptions(antialias=True)


class WaveformView(QWidget):
    """High-performance waveform display with zoom and scroll."""

    selectionChanged = pyqtSignal(int, int)  # start_frame, end_frame
    positionClicked = pyqtSignal(int)  # frame

    def __init__(self, parent: QWidget | None = None, stereo_overlay: bool = True) -> None:
        super().__init__(parent)
        self._data: np.ndarray | None = None
        self._sample_rate: int = SAMPLE_RATE
        self._stereo_overlay = stereo_overlay
        self._selection_start: int | None = None
        self._selection_end: int | None = None
        self._drag_active = False
        self._total_frames = 0

        self._setup_ui()
        self._setup_plot()

    # ── UI setup ──────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(Colors.WAVEFORM_BG)
        layout.addWidget(self._plot_widget)

        scroll_layout = QHBoxLayout()
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self._scrollbar.valueChanged.connect(self._on_scroll)
        scroll_layout.addWidget(self._scrollbar)
        layout.addLayout(scroll_layout)

    def _setup_plot(self) -> None:
        plot = self._plot_widget.getPlotItem()
        plot.setLabel("bottom", "Time", units="s")
        plot.setLabel("left", "Amplitude")
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.getAxis("bottom").setPen(pg.mkPen(Colors.TEXT_SECONDARY, width=1))
        plot.getAxis("left").setPen(pg.mkPen(Colors.TEXT_SECONDARY, width=1))
        plot.getAxis("bottom").setTextPen(Colors.TEXT_SECONDARY)
        plot.getAxis("left").setTextPen(Colors.TEXT_SECONDARY)

        plot.getAxis("bottom").setGrid(100)
        plot.getAxis("left").setGrid(100)

        # Default Y range for audio waveforms
        plot.setYRange(-1.0, 1.0)
        plot.setXRange(0.0, 1.0)

        # Waveform curves (L / R or mono)
        self._curve_l = plot.plot(pen=pg.mkPen(Colors.ACCENT_PRIMARY, width=1))
        self._curve_r = plot.plot(pen=pg.mkPen(Colors.ACCENT_PRIMARY, width=1))

        # Envelope fill items (for downsampled peak display)
        fill_color = QColor(Colors.ACCENT_PRIMARY)
        fill_color.setAlpha(60)
        self._fill_l = pg.FillBetweenItem(self._curve_l, self._curve_l, brush=fill_color)
        plot.addItem(self._fill_l)
        self._fill_l.hide()

        fill_color_r = QColor(Colors.ACCENT_PRIMARY)
        fill_color_r.setAlpha(40)
        self._fill_r = pg.FillBetweenItem(self._curve_r, self._curve_r, brush=fill_color_r)
        plot.addItem(self._fill_r)
        self._fill_r.hide()

        # Envelope boundary curves (hidden, used for fill)
        self._env_max_l = plot.plot(pen=pg.mkPen(Colors.ACCENT_PRIMARY, width=1))
        self._env_min_l = plot.plot(pen=pg.mkPen(Colors.ACCENT_PRIMARY, width=1))
        self._env_max_r = plot.plot(pen=pg.mkPen(Colors.ACCENT_PRIMARY, width=1))
        self._env_min_r = plot.plot(pen=pg.mkPen(Colors.ACCENT_PRIMARY, width=1))

        # Playback position line
        self._playback_line = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen(Colors.ACCENT_SECONDARY, width=2),
        )
        self._playback_line.hide()
        plot.addItem(self._playback_line)

        # Selection region
        sel_color = QColor(Colors.ACCENT_PRIMARY)
        sel_color.setAlpha(40)
        self._selection_region = pg.LinearRegionItem(
            values=[0, 0], brush=sel_color, movable=False,
            pen=pg.mkPen(Colors.ACCENT_PRIMARY, width=1),
        )
        self._selection_region.hide()
        plot.addItem(self._selection_region)

        # Loop markers
        loop_pen = pg.mkPen(Colors.ACCENT_SECONDARY, width=2, style=Qt.PenStyle.DashLine)
        self._loop_start_line = pg.InfiniteLine(pos=0, angle=90, pen=loop_pen)
        self._loop_end_line = pg.InfiniteLine(pos=0, angle=90, pen=loop_pen)
        self._loop_start_line.hide()
        self._loop_end_line.hide()
        plot.addItem(self._loop_start_line)
        plot.addItem(self._loop_end_line)

        loop_fill = QColor(Colors.ACCENT_SECONDARY)
        loop_fill.setAlpha(20)
        self._loop_region = pg.LinearRegionItem(
            values=[0, 0], brush=loop_fill, movable=False,
            pen=pg.mkPen(width=0),
        )
        self._loop_region.hide()
        plot.addItem(self._loop_region)

        # View linking
        vb = plot.getViewBox()
        vb.setMouseEnabled(x=True, y=False)
        vb.sigXRangeChanged.connect(self._on_range_changed)

        # Mouse interaction
        self._plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # ── Public API ────────────────────────────────────────────────────

    def set_data(self, data: np.ndarray, sample_rate: int) -> None:
        """Load sample data into the waveform view."""
        self._data = data
        self._sample_rate = sample_rate
        self._total_frames = data.shape[0] if data.ndim >= 1 else 0
        self._selection_start = None
        self._selection_end = None
        self._selection_region.hide()
        self._playback_line.hide()
        self._draw_waveform()
        self.zoom_to_fit()

    def set_playback_position(self, frame: int) -> None:
        """Update the playback position marker."""
        if self._data is None:
            return
        t = frame / self._sample_rate
        self._playback_line.setValue(t)
        self._playback_line.show()

    def set_loop_points(self, start: int, end: int) -> None:
        """Draw vertical markers for loop region."""
        if self._data is None:
            return
        t_start = start / self._sample_rate
        t_end = end / self._sample_rate
        self._loop_start_line.setValue(t_start)
        self._loop_end_line.setValue(t_end)
        self._loop_region.setRegion([t_start, t_end])
        self._loop_start_line.show()
        self._loop_end_line.show()
        self._loop_region.show()

    def clear(self) -> None:
        """Remove waveform data and reset display."""
        self._data = None
        self._total_frames = 0
        self._selection_start = None
        self._selection_end = None
        self._curve_l.setData([], [])
        self._curve_r.setData([], [])
        self._env_max_l.setData([], [])
        self._env_min_l.setData([], [])
        self._env_max_r.setData([], [])
        self._env_min_r.setData([], [])
        self._fill_l.hide()
        self._fill_r.hide()
        self._playback_line.hide()
        self._selection_region.hide()
        self._loop_start_line.hide()
        self._loop_end_line.hide()
        self._loop_region.hide()

    def zoom_to_selection(self) -> None:
        """Zoom view to the current selection."""
        if self._selection_start is None or self._selection_end is None:
            return
        s = min(self._selection_start, self._selection_end) / self._sample_rate
        e = max(self._selection_start, self._selection_end) / self._sample_rate
        margin = (e - s) * 0.05
        self._plot_widget.setXRange(s - margin, e + margin, padding=0)

    def zoom_to_fit(self) -> None:
        """Show the entire waveform."""
        if self._data is None:
            return
        duration = self._total_frames / self._sample_rate
        self._plot_widget.setXRange(0, duration, padding=0.02)
        self._plot_widget.setYRange(-1.05, 1.05, padding=0)

    def get_selection(self) -> tuple[int, int] | None:
        """Return the current selection as (start_frame, end_frame) or None."""
        if self._selection_start is None or self._selection_end is None:
            return None
        s = min(self._selection_start, self._selection_end)
        e = max(self._selection_start, self._selection_end)
        return (s, e)

    def set_stereo_overlay(self, overlay: bool) -> None:
        """Toggle between overlay and split stereo display."""
        self._stereo_overlay = overlay
        if self._data is not None:
            self._draw_waveform()

    # ── Drawing ───────────────────────────────────────────────────────

    def _draw_waveform(self) -> None:
        if self._data is None:
            return

        is_stereo = self._data.ndim == 2 and self._data.shape[1] == 2

        if is_stereo:
            left = self._data[:, 0]
            right = self._data[:, 1]
        else:
            left = self._data.ravel()
            right = None

        view_range = self._plot_widget.viewRange()
        x_min_t, x_max_t = view_range[0]
        frame_start = max(0, int(x_min_t * self._sample_rate))
        frame_end = min(self._total_frames, int(x_max_t * self._sample_rate))

        view_width = max(self._plot_widget.width(), 800)
        need_downsample = (frame_end - frame_start) > WAVEFORM_DOWNSAMPLE_THRESHOLD

        if need_downsample:
            self._draw_envelope(left, right, frame_start, frame_end, view_width)
        else:
            self._draw_direct(left, right, frame_start, frame_end)

    def _draw_direct(
        self,
        left: np.ndarray,
        right: np.ndarray | None,
        frame_start: int,
        frame_end: int,
    ) -> None:
        """Draw waveform directly without downsampling."""
        self._fill_l.hide()
        self._fill_r.hide()
        self._env_max_l.setData([], [])
        self._env_min_l.setData([], [])
        self._env_max_r.setData([], [])
        self._env_min_r.setData([], [])

        seg = left[frame_start:frame_end]
        t = np.linspace(frame_start / self._sample_rate, frame_end / self._sample_rate, len(seg))

        if right is not None and not self._stereo_overlay:
            self._curve_l.setData(t, seg * 0.5 + 0.5)
            seg_r = right[frame_start:frame_end]
            self._curve_r.setData(t, seg_r * 0.5 - 0.5)
            self._curve_r.show()
        else:
            self._curve_l.setData(t, seg)
            if right is not None:
                seg_r = right[frame_start:frame_end]
                self._curve_r.setData(t, seg_r)
                self._curve_r.show()
            else:
                self._curve_r.setData([], [])
                self._curve_r.hide()

    def _draw_envelope(
        self,
        left: np.ndarray,
        right: np.ndarray | None,
        frame_start: int,
        frame_end: int,
        view_width: int,
    ) -> None:
        """Draw min/max envelope for large datasets."""
        chunk_size = max(1, (frame_end - frame_start) // view_width)
        n_chunks = (frame_end - frame_start) // chunk_size

        seg = left[frame_start : frame_start + n_chunks * chunk_size]
        chunks = seg.reshape(n_chunks, chunk_size)
        env_min = chunks.min(axis=1)
        env_max = chunks.max(axis=1)
        t = np.linspace(frame_start / self._sample_rate, frame_end / self._sample_rate, n_chunks)

        if right is not None and not self._stereo_overlay:
            # Split view: L top half, R bottom half
            self._env_max_l.setData(t, env_max * 0.5 + 0.5)
            self._env_min_l.setData(t, env_min * 0.5 + 0.5)
            self._curve_l.setData([], [])

            seg_r = right[frame_start : frame_start + n_chunks * chunk_size]
            chunks_r = seg_r.reshape(n_chunks, chunk_size)
            self._env_max_r.setData(t, chunks_r.max(axis=1) * 0.5 - 0.5)
            self._env_min_r.setData(t, chunks_r.min(axis=1) * 0.5 - 0.5)
            self._curve_r.setData([], [])

            self._fill_l.setCurves(self._env_min_l, self._env_max_l)
            self._fill_r.setCurves(self._env_min_r, self._env_max_r)
            self._fill_l.show()
            self._fill_r.show()
        else:
            self._env_max_l.setData(t, env_max)
            self._env_min_l.setData(t, env_min)
            self._curve_l.setData([], [])

            self._fill_l.setCurves(self._env_min_l, self._env_max_l)
            self._fill_l.show()

            if right is not None:
                seg_r = right[frame_start : frame_start + n_chunks * chunk_size]
                chunks_r = seg_r.reshape(n_chunks, chunk_size)
                self._env_max_r.setData(t, chunks_r.max(axis=1))
                self._env_min_r.setData(t, chunks_r.min(axis=1))
                self._fill_r.setCurves(self._env_min_r, self._env_max_r)
                self._fill_r.show()
                self._curve_r.setData([], [])
            else:
                self._fill_r.hide()
                self._env_max_r.setData([], [])
                self._env_min_r.setData([], [])
                self._curve_r.setData([], [])

    # ── Scrollbar / zoom ──────────────────────────────────────────────

    def _on_range_changed(self) -> None:
        """Keep scrollbar in sync with the view."""
        if self._data is None:
            return
        x_range = self._plot_widget.viewRange()[0]
        duration = self._total_frames / self._sample_rate
        if duration <= 0:
            return
        page_size = x_range[1] - x_range[0]
        self._scrollbar.blockSignals(True)
        self._scrollbar.setMinimum(0)
        self._scrollbar.setMaximum(max(0, int((duration - page_size) * 1000)))
        self._scrollbar.setPageStep(int(page_size * 1000))
        self._scrollbar.setValue(int(x_range[0] * 1000))
        self._scrollbar.blockSignals(False)
        self._draw_waveform()

    def _on_scroll(self, value: int) -> None:
        t_start = value / 1000.0
        x_range = self._plot_widget.viewRange()[0]
        page = x_range[1] - x_range[0]
        self._plot_widget.setXRange(t_start, t_start + page, padding=0)

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom X axis with mouse wheel."""
        if self._data is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        factor = 0.85 if delta > 0 else 1.0 / 0.85
        x_range = self._plot_widget.viewRange()[0]
        center = (x_range[0] + x_range[1]) / 2.0
        half = (x_range[1] - x_range[0]) / 2.0 * factor
        duration = self._total_frames / self._sample_rate
        new_min = max(0, center - half)
        new_max = min(duration, center + half)
        self._plot_widget.setXRange(new_min, new_max, padding=0)
        event.accept()

    # ── Mouse interaction ─────────────────────────────────────────────

    def _on_mouse_clicked(self, event) -> None:
        if self._data is None:
            return
        pos = event.scenePos()
        vb = self._plot_widget.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        frame = int(mouse_point.x() * self._sample_rate)
        frame = max(0, min(frame, self._total_frames - 1))

        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if not self._drag_active:
                self._selection_start = frame
                self._selection_end = frame
                self._drag_active = True
                self.positionClicked.emit(frame)
            else:
                self._drag_active = False
                if self._selection_start is not None:
                    self._selection_end = frame
                    s = min(self._selection_start, self._selection_end)
                    e = max(self._selection_start, self._selection_end)
                    if e - s > 10:
                        self._selection_region.setRegion([
                            s / self._sample_rate,
                            e / self._sample_rate,
                        ])
                        self._selection_region.show()
                        self.selectionChanged.emit(s, e)
                    else:
                        self._selection_region.hide()
                        self._selection_start = None
                        self._selection_end = None

    def _on_mouse_moved(self, pos) -> None:
        if not self._drag_active or self._data is None:
            return
        vb = self._plot_widget.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        frame = int(mouse_point.x() * self._sample_rate)
        frame = max(0, min(frame, self._total_frames - 1))
        self._selection_end = frame
        s = min(self._selection_start, self._selection_end)
        e = max(self._selection_start, self._selection_end)
        self._selection_region.setRegion([s / self._sample_rate, e / self._sample_rate])
        self._selection_region.show()
