"""Central audio engine — manages sounddevice output and master mixing."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, pyqtSignal

from vynth.config import (
    BLOCK_SIZE,
    CHANNELS,
    DTYPE,
    SAMPLE_RATE,
    SessionSettings,
)
from vynth.engine.voice_allocator import VoiceAllocator
from vynth.utils.ring_buffer import RingBuffer
from vynth.utils.thread_safe_queue import Command, CommandQueue, CommandType

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)


class AudioEngine(QObject):
    """Manages the sounddevice output stream and master mixing."""

    # Signals (emitted on the *main / timer* thread, never from the callback)
    level_updated = pyqtSignal(float, float)   # L / R peak
    underrun_detected = pyqtSignal()

    # ── construction ─────────────────────────────────────────────────────
    def __init__(self, settings: SessionSettings) -> None:
        super().__init__()
        self._settings = settings
        self._voice_allocator = VoiceAllocator()
        self._command_queue = CommandQueue()
        self._vis_buffer = RingBuffer(
            capacity=SAMPLE_RATE * 2,  # 2 seconds of audio
            channels=CHANNELS,
        )

        self._master_volume: float = settings.master_volume
        self._stream: sd.OutputStream | None = None

        # Peak levels written by callback, read by UI timer.
        # Plain floats are atomic on CPython (GIL).
        self._peak_l: float = 0.0
        self._peak_r: float = 0.0

    # ── public API ───────────────────────────────────────────────────────
    @property
    def command_queue(self) -> CommandQueue:
        return self._command_queue

    @property
    def voice_allocator(self) -> VoiceAllocator:
        return self._voice_allocator

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    # ── device management ────────────────────────────────────────────────
    @staticmethod
    def device_list() -> Sequence[dict]:
        """Return list of available output devices."""
        devices = sd.query_devices()
        if isinstance(devices, dict):
            devices = [devices]
        return [d for d in devices if d["max_output_channels"] > 0]

    @property
    def current_device(self) -> int | None:
        return self._settings.audio_device

    def set_device(self, device_id: int | None) -> None:
        """Switch output device.  Restarts the stream if running."""
        was_running = self.is_running
        if was_running:
            self.stop()
        self._settings.audio_device = device_id
        if was_running:
            self.start()

    # ── stream lifecycle ─────────────────────────────────────────────────
    def start(self) -> None:
        """Open and start the sounddevice output stream."""
        if self.is_running:
            return

        self._stream = sd.OutputStream(
            samplerate=self._settings.sample_rate,
            blocksize=self._settings.block_size,
            device=self._settings.audio_device,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
            finished_callback=self._stream_finished,
        )
        self._stream.start()
        log.info(
            "Audio stream started  sr=%d  bs=%d  dev=%s",
            self._settings.sample_rate,
            self._settings.block_size,
            self._settings.audio_device,
        )

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Audio stream stopped")

    # ── audio callback (real-time, NEVER block) ──────────────────────────
    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status.output_underflow:
            self._peak_l = 0.0
            self._peak_r = 0.0
            # Mark for UI (will be picked up by timer)
            self._underrun_flag = True

        # 1. Drain command queue
        commands = self._command_queue.drain()
        for cmd in commands:
            self._execute_command(cmd)

        # 2. Generate audio from voice allocator
        buf = self._voice_allocator.process(frames)

        # 3. Apply master volume
        buf *= self._master_volume

        # 4. Soft-clip (tanh) as safety limiter
        np.tanh(buf, out=buf)

        # 5. Write to output
        outdata[:frames] = buf[:frames]

        # 6. Feed visualization ring buffer
        self._vis_buffer.write(buf[:frames])

        # 7. Calculate peak levels (store, don't emit signal from callback)
        if CHANNELS >= 2:
            self._peak_l = float(np.max(np.abs(buf[:frames, 0])))
            self._peak_r = float(np.max(np.abs(buf[:frames, 1])))
        else:
            peak = float(np.max(np.abs(buf[:frames])))
            self._peak_l = peak
            self._peak_r = peak

    def _execute_command(self, cmd: Command) -> None:
        """Dispatch a command inside the audio callback (real-time safe)."""
        match cmd.type:
            case CommandType.NOTE_ON:
                start_frame = cmd.data if isinstance(cmd.data, int) else 0
                self._voice_allocator.note_on(cmd.note, cmd.velocity, start_frame)
            case CommandType.NOTE_OFF:
                self._voice_allocator.note_off(cmd.note)
            case CommandType.ALL_NOTES_OFF:
                self._voice_allocator.all_notes_off()
            case CommandType.PARAM_CHANGE:
                self._voice_allocator.set_param(cmd.param_name, cmd.param_value)
            case CommandType.SET_SAMPLE:
                self._voice_allocator.set_sample(cmd.data)
            case CommandType.PITCH_BEND:
                self._voice_allocator.set_param("pitch_bend", cmd.param_value)
            case CommandType.MOD_WHEEL:
                self._voice_allocator.set_param("mod_wheel", cmd.param_value)
            case CommandType.SUSTAIN_PEDAL:
                self._voice_allocator.set_param(
                    "sustain", 1.0 if cmd.param_value >= 64 else 0.0
                )
            case CommandType.SET_PLAYBACK_MODE:
                self._voice_allocator.set_param("playback_mode", cmd.param_value)

    def _stream_finished(self) -> None:
        log.debug("Stream finished callback fired")

    # ── master volume ────────────────────────────────────────────────────
    def set_master_volume(self, vol: float) -> None:
        self._master_volume = max(0.0, min(vol, float(10.0 ** (50.0 / 20.0))))
        self._settings.master_volume = self._master_volume

    @property
    def master_volume(self) -> float:
        return self._master_volume

    # ── UI helpers (called from main thread) ─────────────────────────────
    def push_command(self, cmd: Command) -> bool:
        """Thread-safe: push a command for the audio callback to process."""
        return self._command_queue.push(cmd)

    def read_peak_levels(self) -> tuple[float, float]:
        """Read latest peak levels and emit signal.  Call from a QTimer."""
        l, r = self._peak_l, self._peak_r
        self.level_updated.emit(l, r)
        if getattr(self, "_underrun_flag", False):
            self._underrun_flag = False
            self.underrun_detected.emit()
        return l, r

    def get_visualization_buffer(self, n_frames: int) -> np.ndarray:
        """Read the most recent *n_frames* from the ring buffer (UI thread)."""
        return self._vis_buffer.peek(n_frames)

    @property
    def active_voice_count(self) -> int:
        return self._voice_allocator.active_voice_count
