"""Lock-free command queue for audio-thread communication.

The UI/MIDI thread pushes commands; the audio callback pops them.
Uses a pre-allocated list as a circular buffer to avoid GC pressure.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class CommandType(Enum):
    NOTE_ON = auto()
    NOTE_OFF = auto()
    ALL_NOTES_OFF = auto()
    PARAM_CHANGE = auto()
    PITCH_BEND = auto()
    MOD_WHEEL = auto()
    SUSTAIN_PEDAL = auto()
    SET_SAMPLE = auto()
    SET_PLAYBACK_MODE = auto()


@dataclass(slots=True)
class Command:
    type: CommandType
    channel: int = 0
    note: int = 0
    velocity: int = 0
    param_name: str = ""
    param_value: float = 0.0
    data: Any = None


class CommandQueue:
    """Thread-safe command queue optimized for audio use.

    Uses collections.deque which is thread-safe for append/popleft in CPython.
    """

    def __init__(self, maxlen: int = 4096):
        self._queue: deque[Command] = deque(maxlen=maxlen)
        self._dropped = 0

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def push(self, cmd: Command) -> bool:
        """Push a command. Returns False if queue is full (oldest dropped)."""
        if len(self._queue) >= self._queue.maxlen:
            self._dropped += 1
            return False
        self._queue.append(cmd)
        return True

    def pop(self) -> Command | None:
        """Pop the oldest command, or None if empty."""
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    def drain(self, max_count: int = 256) -> list[Command]:
        """Pop up to max_count commands at once."""
        result = []
        for _ in range(max_count):
            cmd = self.pop()
            if cmd is None:
                break
            result.append(cmd)
        return result

    def clear(self):
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)
