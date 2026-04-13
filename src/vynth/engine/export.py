"""Offline WAV renderer / exporter."""
from __future__ import annotations

import logging
import os
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from vynth.config import BLOCK_SIZE, EXPORT_BIT_DEPTHS, EXPORT_SAMPLE_RATES, SAMPLE_RATE
from vynth.engine.voice_allocator import VoiceAllocator
from vynth.sampler.sample import Sample

logger = logging.getLogger(__name__)

_SUBTYPE_MAP: dict[int, str] = {
    16: "PCM_16",
    24: "PCM_24",
}

# All routable DSP parameters with their VoiceAllocator prefixes.
_ALL_PARAMS: list[str] = [
    # Per-voice: ADSR
    "adsr_attack_ms", "adsr_decay_ms", "adsr_sustain", "adsr_release_ms",
    # Per-voice: Filter
    "filter_frequency", "filter_q", "filter_gain_db", "filter_mode",
    # Master: Gain
    "gain_gain_db",
    # Master: Chorus
    "chorus_num_voices", "chorus_detune_cents", "chorus_rate_hz",
    "chorus_depth", "chorus_mix", "chorus_spread",
    # Master: Delay
    "delay_time_ms", "delay_feedback", "delay_mix", "delay_ping_pong",
    # Master: Reverb
    "reverb_room_size", "reverb_damping", "reverb_wet", "reverb_dry", "reverb_width",
    # Master: Limiter
    "limiter_threshold_db", "limiter_release_ms", "limiter_lookahead_ms",
]

# Bypass-able DSP module prefixes.
_BYPASS_PREFIXES: list[str] = [
    "adsr", "filter", "gain", "chorus", "delay", "reverb", "limiter",
]


class Exporter:
    """Renders audio to WAV file."""

    # ── Single-sample export ─────────────────────────────────────────────

    @staticmethod
    def export_sample(sample: Sample, path: str, sample_rate: int = 48_000,
                      bit_depth: int = 24) -> None:
        """Write *sample* to a WAV file, resampling / re-quantizing as needed.

        Parameters
        ----------
        sample:
            Source audio.
        path:
            Destination file path (must end in ``.wav``).
        sample_rate:
            Target sample rate.  Must be in ``EXPORT_SAMPLE_RATES``.
        bit_depth:
            Target bit depth.  Must be in ``EXPORT_BIT_DEPTHS``.
        """
        _validate_path(path)

        if sample_rate not in EXPORT_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported sample rate {sample_rate}. "
                f"Choose from {EXPORT_SAMPLE_RATES}."
            )
        if bit_depth not in EXPORT_BIT_DEPTHS:
            raise ValueError(
                f"Unsupported bit depth {bit_depth}. "
                f"Choose from {EXPORT_BIT_DEPTHS}."
            )

        subtype = _SUBTYPE_MAP.get(bit_depth)
        if subtype is None:
            raise ValueError(f"No soundfile subtype for bit depth {bit_depth}.")

        data = sample.data
        if data.size == 0:
            logger.warning("Exporting empty sample to %s.", path)
            sf.write(path, data, sample_rate, subtype=subtype)
            return

        # Resample if rates differ
        if sample.sample_rate != sample_rate:
            data = _resample(data, sample.sample_rate, sample_rate)

        sf.write(path, data, sample_rate, subtype=subtype)
        logger.info("Exported %s (%d Hz, %d-bit).", path, sample_rate, bit_depth)

    # ── Rendered sample export (with effects) ────────────────────────────

    @staticmethod
    def render_sample(
        sample: Sample,
        source_allocator: VoiceAllocator,
        path: str,
        sample_rate: int = 48_000,
        bit_depth: int = 24,
        master_volume: float = 1.0,
        release_tail_s: float = 2.0,
    ) -> None:
        """Render *sample* through the full DSP chain and write to WAV.

        Creates an offline ``VoiceAllocator`` with the same DSP settings
        as *source_allocator*, triggers a note, processes the entire
        duration plus a release tail, then writes the result.

        Parameters
        ----------
        sample:
            Source audio.
        source_allocator:
            Live ``VoiceAllocator`` whose DSP parameters will be copied.
        path:
            Destination file path.
        sample_rate:
            Target sample rate.
        bit_depth:
            Target bit depth.
        master_volume:
            Master volume scalar applied after DSP chain.
        release_tail_s:
            Extra seconds to render after note-off for release / reverb tail.
        """
        _validate_path(path)

        if sample_rate not in EXPORT_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported sample rate {sample_rate}. "
                f"Choose from {EXPORT_SAMPLE_RATES}."
            )
        if bit_depth not in EXPORT_BIT_DEPTHS:
            raise ValueError(
                f"Unsupported bit depth {bit_depth}. "
                f"Choose from {EXPORT_BIT_DEPTHS}."
            )

        subtype = _SUBTYPE_MAP.get(bit_depth)
        if subtype is None:
            raise ValueError(f"No soundfile subtype for bit depth {bit_depth}.")

        if sample.data.size == 0:
            logger.warning("Exporting empty sample to %s.", path)
            sf.write(path, sample.data, sample_rate, subtype=subtype)
            return

        # Build an offline allocator and mirror DSP state
        offline = _clone_allocator_params(source_allocator)
        offline.set_sample(sample)

        # Trigger note at sample root pitch, full velocity
        offline.note_on(sample.root_note, 127)

        # Render: sample duration + release tail
        body_frames = sample.length
        tail_frames = int(release_tail_s * SAMPLE_RATE)
        total_frames = body_frames + tail_frames
        block = BLOCK_SIZE

        output_blocks: list[np.ndarray] = []
        rendered = 0
        note_released = False

        while rendered < total_frames:
            n = min(block, total_frames - rendered)

            # Release the note once the body has been rendered
            if not note_released and rendered >= body_frames:
                offline.note_on(sample.root_note, 0)  # velocity 0 won't re-trigger
                offline.note_off(sample.root_note)
                note_released = True

            audio = offline.process(n)
            audio = audio * master_volume
            output_blocks.append(audio)
            rendered += n

            # Early exit if the tail is silent
            if note_released and np.max(np.abs(audio)) < 1e-6:
                break

        output = np.concatenate(output_blocks, axis=0)

        # Resample if the internal rate differs from target
        if SAMPLE_RATE != sample_rate:
            output = _resample(output, SAMPLE_RATE, sample_rate)

        sf.write(path, output, sample_rate, subtype=subtype)
        logger.info(
            "Rendered sample with effects to %s (%d Hz, %d-bit).",
            path, sample_rate, bit_depth,
        )

    # ── Full performance render ──────────────────────────────────────────

    @staticmethod
    def render_performance(
        voice_allocator: VoiceAllocator,
        midi_events: list[dict],
        duration_s: float,
        sample_rate: int,
        path: str,
    ) -> None:
        """Offline-render a list of MIDI events to a WAV file.

        Parameters
        ----------
        voice_allocator:
            Pre-configured ``VoiceAllocator`` instance.
        midi_events:
            Sorted list of dicts with keys ``time_s``, ``type``
            (``"note_on"`` / ``"note_off"``), ``note``, ``velocity``.
        duration_s:
            Total render duration in seconds.
        sample_rate:
            Target sample rate for the output file.
        path:
            Destination WAV path.
        """
        _validate_path(path)

        total_frames = int(duration_s * sample_rate)
        block = BLOCK_SIZE
        n_blocks = (total_frames + block - 1) // block

        output_blocks: list[np.ndarray] = []
        event_idx = 0
        n_events = len(midi_events)

        for blk in range(n_blocks):
            block_start_s = blk * block / sample_rate
            block_end_s = (blk + 1) * block / sample_rate

            # Dispatch events falling within this block
            while event_idx < n_events:
                evt = midi_events[event_idx]
                if evt["time_s"] >= block_end_s:
                    break
                _dispatch_event(voice_allocator, evt)
                event_idx += 1

            audio = voice_allocator.process(block)
            output_blocks.append(audio)

        output = np.concatenate(output_blocks, axis=0)[:total_frames]
        sf.write(path, output, sample_rate, subtype="FLOAT")
        logger.info("Rendered %.2fs performance to %s.", duration_s, path)


# ── Private helpers ──────────────────────────────────────────────────────

def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample *data* from *sr_in* to *sr_out*."""
    divisor = gcd(sr_in, sr_out)
    up = sr_out // divisor
    down = sr_in // divisor
    if data.ndim == 1:
        return resample_poly(data, up, down).astype(np.float32)
    # Per-channel for multi-channel audio
    channels = [
        resample_poly(data[:, ch], up, down).astype(np.float32)
        for ch in range(data.shape[1])
    ]
    return np.column_stack(channels)


def _dispatch_event(va: VoiceAllocator, evt: dict) -> None:
    etype = evt.get("type", "")
    if etype == "note_on":
        va.note_on(evt["note"], evt.get("velocity", 100))
    elif etype == "note_off":
        va.note_off(evt["note"])


def _validate_path(path: str) -> None:
    """Guard against path-traversal and ensure parent directory exists."""
    resolved = Path(path).resolve()
    # Reject obvious traversal patterns in the raw input
    if ".." in Path(path).parts:
        raise ValueError("Path traversal detected — refusing to write.")
    resolved.parent.mkdir(parents=True, exist_ok=True)


def _clone_allocator_params(source: VoiceAllocator) -> VoiceAllocator:
    """Create a fresh ``VoiceAllocator`` with DSP params copied from *source*."""
    offline = VoiceAllocator(SAMPLE_RATE)

    # Copy all routable params
    for param in _ALL_PARAMS:
        try:
            value = source.get_param(param)
            offline.set_param(param, value)
        except (KeyError, ValueError):
            pass  # param may not exist in this version

    # Copy bypass states
    for prefix in _BYPASS_PREFIXES:
        bypass_name = f"{prefix}_bypass"
        if prefix == "adsr":
            bypassed = source.voices[0]._adsr.bypassed
        elif prefix == "filter":
            bypassed = source.voices[0]._filter.bypassed
        elif prefix == "gain":
            bypassed = source._gain.bypassed
        elif prefix == "chorus":
            bypassed = source._chorus.bypassed
        elif prefix == "delay":
            bypassed = source._delay.bypassed
        elif prefix == "reverb":
            bypassed = source._reverb.bypassed
        elif prefix == "limiter":
            bypassed = source._limiter.bypassed
        else:
            continue
        offline.set_param(bypass_name, 1.0 if bypassed else 0.0)

    return offline
