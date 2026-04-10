"""Voice recording engine using sounddevice InputStream."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, pyqtSignal

from vynth.config import SAMPLE_RATE
from vynth.sampler.sample import Sample

logger = logging.getLogger(__name__)

_MAX_DURATION_S = 300  # 5 minutes


class Recorder(QObject):
    """Records audio input to a numpy buffer."""

    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    level_updated = pyqtSignal(float)
    buffer_updated = pyqtSignal(np.ndarray)

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = 1,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._channels = channels
        self._recording = False
        self._buffer_list: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._max_frames = 0
        self._recorded_frames = 0
        self._device: int | None = None

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ── Device management ────────────────────────────────────────────────

    @staticmethod
    def get_input_devices() -> list[dict]:
        """Return list of available input devices with id, name, channels."""
        devices: list[dict] = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append({
                    "id": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": dev["default_samplerate"],
                })
        return devices

    def set_input_device(self, device_id: int | None) -> None:
        """Set the input device for recording. *None* = system default."""
        if self._recording:
            logger.warning("Cannot change device while recording.")
            return
        self._device = device_id

    # ── Recording ────────────────────────────────────────────────────────

    def start_recording(self, device: int | None = None,
                        max_duration_s: float = _MAX_DURATION_S) -> None:
        """Open an InputStream and begin recording."""
        if self._recording:
            logger.warning("Already recording.")
            return

        if device is not None:
            self._device = device

        self._buffer_list.clear()
        self._recorded_frames = 0
        self._max_frames = int(max_duration_s * self._sample_rate)

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                device=self._device,
                callback=self._input_callback,
            )
            self._stream.start()
            self._recording = True
            self.recording_started.emit()
            logger.info("Recording started (device=%s, sr=%d, ch=%d).",
                        self._device, self._sample_rate, self._channels)
        except sd.PortAudioError:
            logger.exception("Failed to open input stream.")
            self._recording = False

    def stop_recording(self) -> Sample:
        """Stop the stream and return the recorded audio as a *Sample*."""
        if not self._recording:
            logger.warning("Not currently recording.")
            return Sample.from_buffer(
                np.zeros(0, dtype=np.float32), self._sample_rate, "Empty"
            )

        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._buffer_list:
            data = np.concatenate(self._buffer_list, axis=0)
        else:
            data = np.zeros(0, dtype=np.float32)

        self._buffer_list.clear()
        self.recording_stopped.emit()
        logger.info("Recording stopped — %d frames captured.", len(data))
        return Sample.from_buffer(data, self._sample_rate, "Recording")

    # ── Stream callback (called from audio thread) ───────────────────────

    def _input_callback(self, indata: np.ndarray, frames: int,
                        time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("Input stream status: %s", status)

        if not self._recording:
            return

        # Auto-stop safety — honour maximum duration
        remaining = self._max_frames - self._recorded_frames
        if remaining <= 0:
            self._recording = False
            return

        chunk = indata[:min(frames, remaining)].copy()
        self._buffer_list.append(chunk)
        self._recorded_frames += len(chunk)

        # Peak level (RMS would be heavier; peak is fine for a meter)
        peak = float(np.abs(chunk).max())
        self.level_updated.emit(peak)
        self.buffer_updated.emit(chunk)
