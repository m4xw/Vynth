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
