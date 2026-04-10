"""Global configuration constants for Vynth."""
from dataclasses import dataclass, field

# ── Audio ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000
BLOCK_SIZE = 512
CHANNELS = 2
DTYPE = "float32"

# ── Polyphony ────────────────────────────────────────────────────────────
MAX_VOICES = 64

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


@dataclass
class SessionSettings:
    """Mutable runtime settings."""
    audio_device: int | None = None
    midi_device: int | None = None
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
