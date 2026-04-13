"""Global configuration constants for Vynth."""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# ── Audio ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000
BLOCK_SIZE = 512
CHANNELS = 2
DTYPE = "float32"

# ── Polyphony ────────────────────────────────────────────────────────────
MAX_VOICES = 64
VOICE_STOP_LAST_ON_RETRIGGER_DEFAULT = True

# ── MIDI ─────────────────────────────────────────────────────────────────
MIDI_POLL_INTERVAL_MS = 1
MIDI_HOTPLUG_POLL_S = 2.0
MIDI_CHANNELS = 16
A4_FREQUENCY = 440.0
A4_NOTE = 69  # MIDI note number for A4

# ── DSP ──────────────────────────────────────────────────────────────────
ADSR_MIN_MS = 1.0
ADSR_MAX_MS = 10_000.0
ADSR_DEFAULT_ATTACK_MS = 10.0
ADSR_DEFAULT_DECAY_MS = 100.0
ADSR_DEFAULT_SUSTAIN = 0.7  # 0..1
ADSR_DEFAULT_RELEASE_MS = 200.0

REVERB_DEFAULT_ROOM_SIZE = 0.5
REVERB_DEFAULT_DAMPING = 0.5
REVERB_DEFAULT_WET = 0.3

CHORUS_DEFAULT_VOICES = 4
CHORUS_DEFAULT_DETUNE_CENTS = 12.0
CHORUS_DEFAULT_RATE_HZ = 1.5

DELAY_DEFAULT_TIME_MS = 250.0
DELAY_DEFAULT_FEEDBACK = 0.4
DELAY_DEFAULT_MIX = 0.3

FILTER_DEFAULT_FREQ = 1000.0
FILTER_DEFAULT_Q = 0.707
FILTER_DEFAULT_GAIN_DB = 0.0

GRANULAR_DEFAULT_GRAIN_SIZE_MS = 50.0
GRANULAR_DEFAULT_OVERLAP = 0.5
GRANULAR_DEFAULT_SCATTER = 0.1
GRANULAR_DEFAULT_DENSITY = 10.0

# ── UI ───────────────────────────────────────────────────────────────────
UI_FPS = 60
WAVEFORM_DOWNSAMPLE_THRESHOLD = 100_000
SPECTRUM_FFT_SIZE = 4096

# ── Export ────────────────────────────────────────────────────────────────
EXPORT_SAMPLE_RATES = [44_100, 48_000]
EXPORT_BIT_DEPTHS = [16, 24]
EXPORT_DEFAULT_SR = 48_000
EXPORT_DEFAULT_BITS = 24


def default_midi_controller_profile() -> dict:
    """Return the built-in MPK Mini Plus starter mapping profile."""
    return {
        "name": "MPK Mini Plus Default",
        "mappings": [
            {
                "enabled": True,
                "input": "cc",
                "number": 1,
                "channel": "all",
                "trigger": "change",
                "mode": "absolute",
                "target_type": "param",
                "target": "mod_wheel",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "enabled": True,
                "input": "cc",
                "number": 74,
                "channel": "all",
                "trigger": "change",
                "mode": "absolute",
                "target_type": "param",
                "target": "filter_frequency",
                "min": 200.0,
                "max": 12000.0,
            },
            {
                "enabled": True,
                "input": "cc",
                "number": 71,
                "channel": "all",
                "trigger": "change",
                "mode": "absolute",
                "target_type": "param",
                "target": "filter_q",
                "min": 0.2,
                "max": 8.0,
            },
            {
                "enabled": True,
                "input": "cc",
                "number": 91,
                "channel": "all",
                "trigger": "change",
                "mode": "absolute",
                "target_type": "param",
                "target": "reverb_wet",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "enabled": True,
                "input": "cc",
                "number": 7,
                "channel": "all",
                "trigger": "change",
                "mode": "absolute",
                "target_type": "param",
                "target": "master_volume",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "enabled": True,
                "input": "note",
                "number": 36,
                "channel": "all",
                "trigger": "press",
                "mode": "momentary",
                "target_type": "action",
                "target": "play",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "enabled": True,
                "input": "note",
                "number": 37,
                "channel": "all",
                "trigger": "press",
                "mode": "momentary",
                "target_type": "action",
                "target": "stop",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "enabled": True,
                "input": "note",
                "number": 38,
                "channel": "all",
                "trigger": "press",
                "mode": "toggle",
                "target_type": "action",
                "target": "record_toggle",
                "min": 0.0,
                "max": 1.0,
            },
        ],
    }


@dataclass
class SessionSettings:
    """Mutable runtime settings."""
    audio_device: int | None = None
    midi_device: int | None = None
    midi_channel: int | None = None
    sample_rate: int = SAMPLE_RATE
    block_size: int = BLOCK_SIZE
    buffer_count: int = 3
    master_volume: float = 0.8
    # Last used export settings
    export_sr: int = EXPORT_DEFAULT_SR
    export_bits: int = EXPORT_DEFAULT_BITS
    export_dir: str = ""
    # Recent files
    recent_samples: list[str] = field(default_factory=list)


class AppConfig:
    """Persistent application config stored in user data directory.

    Survives across sessions. Uses QStandardPaths when available,
    falls back to ``~/.vynth/``.
    """

    _FILENAME = "vynth_app_config.json"

    def __init__(self) -> None:
        self._data: dict = {}
        self._path = self._resolve_path()
        self._load()

    # ── public API ───────────────────────────────────────────

    @property
    def last_session_path(self) -> str:
        return self._data.get("last_session_path", "")

    @last_session_path.setter
    def last_session_path(self, value: str) -> None:
        self._data["last_session_path"] = value
        self._save()

    @property
    def midi_controller_profile(self) -> dict:
        profile = self._data.get("midi_controller_profile")
        if isinstance(profile, dict):
            return copy.deepcopy(profile)
        return default_midi_controller_profile()

    @midi_controller_profile.setter
    def midi_controller_profile(self, value: dict) -> None:
        if not isinstance(value, dict):
            return
        self._data["midi_controller_profile"] = copy.deepcopy(value)
        self._save()

    @property
    def audio_settings(self) -> dict:
        settings = self._data.get("audio_settings")
        if not isinstance(settings, dict):
            return {}

        result: dict[str, int | None] = {}
        sample_rate = settings.get("sample_rate")
        if isinstance(sample_rate, int) and sample_rate > 0:
            result["sample_rate"] = sample_rate

        block_size = settings.get("block_size")
        if isinstance(block_size, int) and block_size > 0:
            result["block_size"] = block_size

        audio_device = settings.get("audio_device")
        if audio_device is None or isinstance(audio_device, int):
            result["audio_device"] = audio_device

        return result

    @audio_settings.setter
    def audio_settings(self, value: dict) -> None:
        if not isinstance(value, dict):
            return

        settings: dict[str, int | None] = {}
        sample_rate = value.get("sample_rate")
        if isinstance(sample_rate, int) and sample_rate > 0:
            settings["sample_rate"] = sample_rate

        block_size = value.get("block_size")
        if isinstance(block_size, int) and block_size > 0:
            settings["block_size"] = block_size

        audio_device = value.get("audio_device")
        if audio_device is None or isinstance(audio_device, int):
            settings["audio_device"] = audio_device

        self._data["audio_settings"] = settings
        self._save()

    @property
    def midi_settings(self) -> dict:
        settings = self._data.get("midi_settings")
        if not isinstance(settings, dict):
            return {}

        result: dict[str, int | None] = {}
        midi_device = settings.get("midi_device")
        if midi_device is None or isinstance(midi_device, int):
            result["midi_device"] = midi_device

        midi_channel = settings.get("midi_channel")
        if midi_channel is None or isinstance(midi_channel, int):
            result["midi_channel"] = midi_channel

        return result

    @midi_settings.setter
    def midi_settings(self, value: dict) -> None:
        if not isinstance(value, dict):
            return

        settings: dict[str, int | None] = {}
        midi_device = value.get("midi_device")
        if midi_device is None or isinstance(midi_device, int):
            settings["midi_device"] = midi_device

        midi_channel = value.get("midi_channel")
        if midi_channel is None or isinstance(midi_channel, int):
            settings["midi_channel"] = midi_channel

        self._data["midi_settings"] = settings
        self._save()

    @property
    def ui_settings(self) -> dict:
        settings = self._data.get("ui_settings")
        if not isinstance(settings, dict):
            return {}

        result: dict[str, str] = {}
        visualizer_mode = settings.get("visualizer_mode")
        if visualizer_mode in {"spectrum", "rendered", "live"}:
            result["visualizer_mode"] = visualizer_mode

        return result

    @ui_settings.setter
    def ui_settings(self, value: dict) -> None:
        if not isinstance(value, dict):
            return

        settings: dict[str, str] = {}
        visualizer_mode = value.get("visualizer_mode")
        if visualizer_mode in {"spectrum", "rendered", "live"}:
            settings["visualizer_mode"] = visualizer_mode

        self._data["ui_settings"] = settings
        self._save()

    # ── internals ────────────────────────────────────────────

    @staticmethod
    def _resolve_path() -> Path:
        try:
            from PyQt6.QtCore import QStandardPaths

            base = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppDataLocation
            )
            if base:
                return Path(base) / AppConfig._FILENAME
        except Exception:
            pass
        return Path.home() / ".vynth" / AppConfig._FILENAME

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("Corrupt app config at %s — starting fresh", self._path)
                self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2), encoding="utf-8"
            )
        except Exception:
            log.warning("Could not write app config to %s", self._path)
