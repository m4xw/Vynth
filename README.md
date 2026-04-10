# 🎤 Vynth — Professional Voice Sampler Synthesizer

**Record your voice, shape it with studio-grade effects, and play it polyphonically via MIDI keyboard.**

Vynth turns your voice into a fully playable instrument. Record a vocal sample, edit it in the built-in waveform editor, then use your MIDI keyboard to create choirs, background vocals, pads, textures, and more — all in real time with 64-voice polyphony.

---

## Features

### 🎙️ Voice Recording
- Record directly from any audio input device
- Real-time level monitoring with VU meter
- Waveform preview during recording
- Up to 5 minutes per recording

### 🎹 MIDI Playback
- 64-voice polyphony — play full chords and clusters
- Automatic pitch mapping from root note
- Hot-plug MIDI device detection
- Note On/Off, Sustain pedal, Pitch Bend, Mod Wheel support
- On-screen piano keyboard with live note visualization

### 🎛️ DSP Effects Chain

| Effect | Description |
|--------|-------------|
| **ADSR Envelope** | Sample-accurate attack/decay/sustain/release with exponential curves |
| **Pitch Shifter** | Phase vocoder STFT pitch shifting (±24 semitones) |
| **Formant Preservation** | LPC-based spectral envelope correction — no chipmunk effect |
| **Time Stretch** | WSOLA waveform-similarity time stretching (0.25× – 4×) |
| **Reverb** | Freeverb algorithm (8 comb + 4 allpass filters) with room size, damping, width |
| **Delay** | Stereo delay with feedback saturation and ping-pong mode |
| **Chorus / Unison** | 2–8 detuned voices with LFO modulation and stereo spread |
| **EQ / Filter** | Biquad filter: Low/High/Band Pass, Notch, Peak, Low/High Shelf |
| **Granular Synthesis** | Grain scheduler with size, overlap, scatter, density, pitch, position |
| **Limiter** | Lookahead soft limiter for the master bus |

### 📊 Visualization
- **Waveform Editor** — Scrollable, zoomable waveform with selection, loop point markers, and edit tools
- **Spectrum Analyzer** — Real-time FFT display with log frequency scale (20 Hz – 20 kHz)
- **ADSR Display** — Interactive curve with drag handles
- **LED Meters** — Stereo peak/VU meters with green/yellow/red zones and peak hold

### 💾 Sample Management
- Load WAV files from disk
- Non-destructive editing: trim, normalize, reverse, fade in/out, loop crossfade
- Sample browser with search and sorting
- Export recordings and performances to WAV (44.1/48 kHz, 16/24-bit)

### 🎨 Professional Dark UI
- Custom dark theme with purple-blue accents
- Dockable, rearrangeable panel layout
- Custom rotary knobs, faders, and LED meters
- Keyboard shortcuts for all common operations

---

## Installation

### Requirements
- Python 3.10+
- Windows 10/11 (primary), macOS/Linux (untested but should work)
- Audio output device
- MIDI keyboard (optional but recommended)

### Install from source

```bash
git clone <repository-url>
cd vynth
pip install -e ".[dev]"
```

### Run

```bash
# As a module
python -m vynth

# Or directly
vynth
```

---

## Quick Start

1. **Launch Vynth** — `python -m vynth`
2. **Select your audio device** — Edit → Preferences → Audio tab
3. **Connect your MIDI keyboard** — It should be auto-detected (check the MIDI dropdown in the toolbar)
4. **Record your voice** — Click the red Record button, sing/speak, click Stop
5. **Select the recording** in the Sample Browser — it appears automatically
6. **Play it!** — Press keys on your MIDI keyboard. Each key plays your voice at a different pitch.
7. **Shape the sound** — Adjust effects in the Effects Rack (right panel):
   - Dial in reverb for choir-like spaciousness
   - Add chorus for thickness
   - Use the ADSR to control how notes start and fade
   - Enable Granular mode for ambient textures
8. **Export** — File → Export WAV to render your creation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        PyQt6 UI                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │ Waveform │ │ Spectrum │ │ Effects  │ │ Sample       │   │
│  │ Editor   │ │ Analyzer │ │ Rack     │ │ Browser      │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘   │
│       │             │            │               │           │
│  ┌────┴─────────────┴────────────┴───────────────┴─────┐    │
│  │              Signal / Slot + Command Queue           │    │
│  └──────────────────────┬──────────────────────────────┘    │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────┐
│                   Audio Engine                              │
│  ┌────────────┐  ┌──────┴───────┐  ┌────────────────────┐  │
│  │ MIDI       │  │ Voice        │  │ Master Bus         │  │
│  │ Engine     │──│ Allocator    │──│ Chorus→Delay→      │  │
│  │ (rtmidi)   │  │ (64 voices)  │  │ Reverb→Limiter     │  │
│  └────────────┘  └──────────────┘  └────────────────────┘  │
│                         │                                    │
│                  ┌──────┴──────┐                            │
│                  │   Voice     │  × 64                      │
│                  │ Sample Read │                            │
│                  │ ADSR→Pitch→ │                            │
│                  │ Formant→EQ  │                            │
│                  └─────────────┘                            │
│                         │                                    │
│                  ┌──────┴──────┐                            │
│                  │ sounddevice │                            │
│                  │ OutputStream│                            │
│                  └─────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### Thread Model
- **Audio Thread** — sounddevice callback, lock-free. Reads commands from queue, renders voices, writes output.
- **MIDI Thread** — Daemon thread polling rtmidi at 1ms intervals. Pushes commands to audio thread queue.
- **UI Thread** — PyQt6 main thread. Emits commands via queue. Reads visualization data from ring buffer.

---

## Project Structure

```
vynth/
├── pyproject.toml              # Package config & dependencies
├── requirements.txt
├── README.md
├── tests/                      # 42 tests, ~100% core coverage
│   ├── conftest.py
│   ├── test_audio_engine.py
│   ├── test_voice_allocator.py
│   ├── test_sample_manager.py
│   ├── test_export.py
│   └── test_dsp/
│       ├── test_adsr.py
│       ├── test_pitch_shift.py
│       ├── test_reverb.py
│       ├── test_chorus.py
│       ├── test_filter.py
│       └── test_granular.py
└── src/vynth/
    ├── app.py                  # Application entry point & wiring
    ├── config.py               # Constants & settings
    ├── engine/
    │   ├── audio_engine.py     # sounddevice output stream
    │   ├── midi_engine.py      # MIDI input polling
    │   ├── voice_allocator.py  # 64-voice polyphonic allocator
    │   ├── voice.py            # Single voice: sample reader + DSP
    │   ├── recorder.py         # Microphone input recording
    │   └── export.py           # WAV rendering & export
    ├── dsp/
    │   ├── base.py             # Abstract DSPEffect base class
    │   ├── adsr.py             # ADSR envelope generator
    │   ├── pitch_shift.py      # Phase vocoder pitch shifter
    │   ├── formant.py          # Formant preservation (LPC)
    │   ├── time_stretch.py     # WSOLA time stretcher
    │   ├── reverb.py           # Freeverb algorithm
    │   ├── delay.py            # Stereo delay with feedback
    │   ├── chorus.py           # Chorus / Unison
    │   ├── filter.py           # Biquad EQ filter
    │   ├── granular.py         # Granular synthesis engine
    │   └── limiter.py          # Lookahead soft limiter
    ├── sampler/
    │   ├── sample.py           # Sample data model
    │   ├── sample_manager.py   # Sample collection manager
    │   └── sample_editor.py    # Non-destructive editing ops
    ├── ui/
    │   ├── theme.py            # Dark theme & color palette
    │   ├── main_window.py      # Main window with docks
    │   ├── widgets/
    │   │   ├── knob.py         # Rotary knob
    │   │   ├── fader.py        # Vertical/horizontal fader
    │   │   ├── led_meter.py    # Stereo LED peak meter
    │   │   ├── midi_keyboard.py # On-screen piano
    │   │   ├── adsr_display.py # Interactive ADSR curve
    │   │   ├── waveform_view.py # Scrollable waveform
    │   │   ├── waveform_editor.py # Editable waveform
    │   │   └── spectrum_view.py # FFT spectrum analyzer
    │   ├── panels/
    │   │   ├── recorder_panel.py
    │   │   ├── sample_browser.py
    │   │   ├── effects_rack.py
    │   │   ├── mixer_panel.py
    │   │   └── export_panel.py
    │   └── dialogs/
    │       ├── settings_dialog.py
    │       └── about_dialog.py
    └── utils/
        ├── ring_buffer.py      # Lock-free ring buffer
        ├── thread_safe_queue.py # Audio-thread command queue
        └── audio_utils.py      # Mono/stereo, normalize, etc.
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=vynth --cov-report=term-missing

# Run specific test module
python -m pytest tests/test_dsp/test_adsr.py -v
```

### Test Coverage
- **DSP modules**: Frequency verification (FFT peak analysis), envelope timing, filter attenuation
- **Voice allocator**: Polyphony, voice stealing, note on/off lifecycle
- **Sample manager**: Load/save/organize samples, WAV file I/O
- **Export**: WAV rendering at 44.1/48 kHz, 16/24-bit verification
- **Audio engine**: Creation, command queue, volume control

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Toggle recording |
| `Ctrl+O` | Load sample file |
| `Ctrl+S` | Save session |
| `Ctrl+E` | Export WAV |
| `Ctrl+N` | New session |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |

---

## Technical Details

- **Sample Rate**: 48 kHz (configurable to 44.1 kHz)
- **Block Size**: 512 samples (~10.7 ms latency at 48 kHz)
- **Polyphony**: 64 voices with oldest-note-first voice stealing
- **Audio Backend**: PortAudio via sounddevice
- **MIDI Backend**: rtmidi with 1ms polling
- **DSP**: All numpy/scipy — no external C dependencies for effects
- **UI Framework**: PyQt6 with pyqtgraph for waveform/spectrum rendering

---

## License

MIT
