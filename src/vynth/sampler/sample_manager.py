"""Sample library management."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from vynth.sampler.sample import Sample

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


class SampleManager(QObject):
    """Manages a collection of samples."""

    sample_added = pyqtSignal(str)
    sample_removed = pyqtSignal(str)
    sample_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._samples: dict[str, Sample] = {}
        self._selected: str | None = None

    # ── Query ────────────────────────────────────────────────────────────

    def get_sample(self, name: str) -> Sample | None:
        return self._samples.get(name)

    def get_selected(self) -> Sample | None:
        if self._selected is None:
            return None
        return self._samples.get(self._selected)

    def get_names(self) -> list[str]:
        return list(self._samples.keys())

    # ── Mutation ─────────────────────────────────────────────────────────

    def load_sample(self, path: str) -> Sample:
        """Load a WAV (or supported format) file and add it to the collection."""
        sample = Sample.from_file(path)
        self.add_sample(sample)
        return sample

    def add_sample(self, sample: Sample) -> None:
        """Add a *Sample* (e.g. from the recorder) to the collection."""
        name = self._unique_name(sample.name)
        sample.name = name
        self._samples[name] = sample
        self.sample_added.emit(name)
        logger.info("Added sample '%s' (%d frames).", name, sample.length)

    def remove_sample(self, name: str) -> None:
        if name not in self._samples:
            logger.warning("Sample '%s' not found.", name)
            return
        del self._samples[name]
        if self._selected == name:
            self._selected = None
        self.sample_removed.emit(name)
        logger.info("Removed sample '%s'.", name)

    def select_sample(self, name: str) -> None:
        if name not in self._samples:
            logger.warning("Cannot select unknown sample '%s'.", name)
            return
        self._selected = name
        self.sample_selected.emit(name)

    # ── Bulk I/O ─────────────────────────────────────────────────────────

    def save_to_directory(self, directory: str) -> None:
        """Save every sample as a WAV in *directory*."""
        dest = Path(directory)
        dest.mkdir(parents=True, exist_ok=True)
        for name, sample in self._samples.items():
            safe_name = _safe_filename(name)
            sample.save(dest / f"{safe_name}.wav")
        logger.info("Saved %d samples to %s.", len(self._samples), directory)

    def load_directory(self, directory: str) -> None:
        """Load all supported audio files from *directory*."""
        src = Path(directory)
        if not src.is_dir():
            logger.warning("Directory does not exist: %s", directory)
            return
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in _SUPPORTED_EXTENSIONS:
                try:
                    self.load_sample(str(p))
                except Exception:
                    logger.exception("Failed to load %s.", p)

    # ── Internal ─────────────────────────────────────────────────────────

    def _unique_name(self, base: str) -> str:
        """Ensure *base* is unique within the collection by appending a number."""
        if base not in self._samples:
            return base
        idx = 2
        while f"{base} ({idx})" in self._samples:
            idx += 1
        return f"{base} ({idx})"


def _safe_filename(name: str) -> str:
    """Strip characters that are unsafe for file names."""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-")
    return "".join(c if c in keep else "_" for c in name).strip() or "untitled"
