"""Sample data model — holds audio buffer and metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from vynth.config import SAMPLE_RATE


@dataclass
class LoopRegion:
    start: int = 0
    end: int = 0
    crossfade: int = 256
    enabled: bool = False


@dataclass
class Sample:
    """Represents a single audio sample with metadata."""
    name: str = "Untitled"
    data: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sample_rate: int = SAMPLE_RATE
    channels: int = 1
    root_note: int = 60  # Middle C
    loop: LoopRegion = field(default_factory=LoopRegion)
    file_path: str = ""
    note_range: tuple[int, int] = (0, 127)
    velocity_range: tuple[int, int] = (0, 127)
    selection_range: tuple[int, int] = (0, 0)

    @property
    def length(self) -> int:
        return len(self.data) if self.data.ndim == 1 else self.data.shape[0]

    @property
    def duration_s(self) -> float:
        if self.length == 0:
            return 0.0
        return self.length / self.sample_rate

    @property
    def mono(self) -> np.ndarray:
        if self.data.ndim == 1:
            return self.data
        return self.data.mean(axis=1)

    def get_stereo(self) -> np.ndarray:
        if self.data.ndim == 2 and self.data.shape[1] == 2:
            return self.data
        mono = self.mono
        return np.column_stack([mono, mono])

    @classmethod
    def from_file(cls, path: str | Path) -> Sample:
        path = Path(path)
        data, sr = sf.read(str(path), dtype="float32")
        channels = 1 if data.ndim == 1 else data.shape[1]
        return cls(
            name=path.stem,
            data=data,
            sample_rate=sr,
            channels=channels,
            file_path=str(path),
        )

    @classmethod
    def from_buffer(cls, data: np.ndarray, sample_rate: int = SAMPLE_RATE,
                    name: str = "Recording") -> Sample:
        channels = 1 if data.ndim == 1 else data.shape[1]
        return cls(
            name=name,
            data=data.astype(np.float32),
            sample_rate=sample_rate,
            channels=channels,
        )

    def save(self, path: str | Path, subtype: str = "FLOAT"):
        sf.write(str(path), self.data, self.sample_rate, subtype=subtype)
