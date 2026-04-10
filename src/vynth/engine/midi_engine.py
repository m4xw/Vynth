"""MIDI input engine — polls rtmidi and routes messages to the audio engine."""
from __future__ import annotations

import logging
import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal

from vynth.config import MIDI_HOTPLUG_POLL_S, MIDI_POLL_INTERVAL_MS
from vynth.utils.thread_safe_queue import Command, CommandQueue, CommandType

log = logging.getLogger(__name__)

# ── optional rtmidi import ───────────────────────────────────────────────
try:
    import rtmidi  # type: ignore[import-untyped]

    _RTMIDI_AVAILABLE = True
except ImportError:
    _RTMIDI_AVAILABLE = False
    log.warning("python-rtmidi not installed — MIDI input disabled")

# ── MIDI status bytes ────────────────────────────────────────────────────
_NOTE_OFF = 0x80
_NOTE_ON = 0x90
_CONTROL_CHANGE = 0xB0
_PITCH_BEND = 0xE0

# Common CC numbers
_CC_MOD_WHEEL = 1
_CC_SUSTAIN = 64


class MIDIEngine(QObject):
    """Polls MIDI input devices and routes messages to the audio engine."""

    # Signals (for UI feedback only — audio goes through CommandQueue)
    note_on_received = pyqtSignal(int, int)   # note, velocity
    note_off_received = pyqtSignal(int)        # note
    device_changed = pyqtSignal(str)           # port name
    devices_updated = pyqtSignal(list)         # list[str]

    def __init__(self, command_queue: CommandQueue) -> None:
        super().__init__()
        self._command_queue = command_queue
        self._running = False
        self._current_port: int | None = None
        self._current_port_name: str = ""
        self._thread: threading.Thread | None = None
        self._known_devices: list[str] = []

        self._midiin: rtmidi.MidiIn | None = None
        if _RTMIDI_AVAILABLE:
            self._midiin = rtmidi.MidiIn()

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        """Start the MIDI polling thread."""
        if not _RTMIDI_AVAILABLE:
            log.warning("Cannot start MIDIEngine — rtmidi not available")
            return
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, name="midi-poll", daemon=True
        )
        self._thread.start()
        log.info("MIDI polling started")

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.close_device()
        log.info("MIDI polling stopped")

    # ── device management ────────────────────────────────────────────────
    def get_devices(self) -> list[str]:
        """Enumerate available MIDI input ports."""
        if self._midiin is None:
            return []
        return [self._midiin.get_port_name(i) for i in range(self._midiin.get_port_count())]

    def open_device(self, port_num: int) -> None:
        """Open a specific MIDI input port by index."""
        if self._midiin is None:
            return
        self.close_device()

        port_count = self._midiin.get_port_count()
        if port_num < 0 or port_num >= port_count:
            log.error("Invalid MIDI port %d (available: %d)", port_num, port_count)
            return

        self._midiin.open_port(port_num)
        self._current_port = port_num
        self._current_port_name = self._midiin.get_port_name(port_num)
        log.info("Opened MIDI port %d: %s", port_num, self._current_port_name)
        self.device_changed.emit(self._current_port_name)

    def close_device(self) -> None:
        """Close the currently open MIDI port."""
        if self._midiin is not None and self._current_port is not None:
            self._midiin.close_port()
            log.info("Closed MIDI port: %s", self._current_port_name)
        self._current_port = None
        self._current_port_name = ""

    @property
    def current_port(self) -> int | None:
        return self._current_port

    @property
    def current_port_name(self) -> str:
        return self._current_port_name

    @property
    def available(self) -> bool:
        return _RTMIDI_AVAILABLE

    # ── polling loop (runs in daemon thread) ─────────────────────────────
    def _poll_loop(self) -> None:
        poll_s = MIDI_POLL_INTERVAL_MS / 1000.0
        hotplug_deadline = time.monotonic() + MIDI_HOTPLUG_POLL_S

        while self._running:
            # ── read MIDI messages ───────────────────────────────────────
            if self._midiin is not None and self._current_port is not None:
                while True:
                    msg = self._midiin.get_message()
                    if msg is None:
                        break
                    midi_bytes, _delta = msg
                    self._dispatch(midi_bytes)

            # ── periodic device hotplug check ────────────────────────────
            now = time.monotonic()
            if now >= hotplug_deadline:
                hotplug_deadline = now + MIDI_HOTPLUG_POLL_S
                self._check_device_changes()

            time.sleep(poll_s)

    # ── message dispatch ─────────────────────────────────────────────────
    def _dispatch(self, data: list[int]) -> None:
        if len(data) < 1:
            return

        status = data[0] & 0xF0
        channel = data[0] & 0x0F

        if status == _NOTE_ON and len(data) >= 3:
            note, velocity = data[1], data[2]
            if velocity == 0:
                # Note-on with velocity 0 == note-off (common convention)
                self._command_queue.push(
                    Command(type=CommandType.NOTE_OFF, channel=channel, note=note)
                )
                self.note_off_received.emit(note)
            else:
                self._command_queue.push(
                    Command(
                        type=CommandType.NOTE_ON,
                        channel=channel,
                        note=note,
                        velocity=velocity,
                    )
                )
                self.note_on_received.emit(note, velocity)

        elif status == _NOTE_OFF and len(data) >= 3:
            note = data[1]
            self._command_queue.push(
                Command(type=CommandType.NOTE_OFF, channel=channel, note=note)
            )
            self.note_off_received.emit(note)

        elif status == _CONTROL_CHANGE and len(data) >= 3:
            cc_num, cc_val = data[1], data[2]
            if cc_num == _CC_SUSTAIN:
                self._command_queue.push(
                    Command(
                        type=CommandType.SUSTAIN_PEDAL,
                        channel=channel,
                        param_value=float(cc_val),
                    )
                )
            elif cc_num == _CC_MOD_WHEEL:
                self._command_queue.push(
                    Command(
                        type=CommandType.MOD_WHEEL,
                        channel=channel,
                        param_value=cc_val / 127.0,
                    )
                )

        elif status == _PITCH_BEND and len(data) >= 3:
            # 14-bit value: LSB = data[1], MSB = data[2]
            raw = (data[2] << 7) | data[1]
            # Normalize to -1.0 … +1.0  (center = 8192)
            normalized = (raw - 8192) / 8192.0
            self._command_queue.push(
                Command(
                    type=CommandType.PITCH_BEND,
                    channel=channel,
                    param_value=normalized,
                )
            )

    # ── hotplug detection ────────────────────────────────────────────────
    def _check_device_changes(self) -> None:
        current = self.get_devices()
        if current != self._known_devices:
            self._known_devices = current
            log.info("MIDI devices changed: %s", current)
            self.devices_updated.emit(current)
