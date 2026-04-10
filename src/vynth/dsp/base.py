"""DSP effect base class — all effects inherit from this."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DSPEffect(ABC):
    """Abstract base for all DSP processors."""

    def __init__(self, sample_rate: int = 48_000):
        self._sample_rate = sample_rate
        self._bypassed = False
        self._params: dict[str, float] = {}

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def bypassed(self) -> bool:
        return self._bypassed

    @bypassed.setter
    def bypassed(self, value: bool):
        self._bypassed = value

    def get_param(self, name: str) -> float:
        return self._params.get(name, 0.0)

    def set_param(self, name: str, value: float):
        self._params[name] = value
        self._on_param_changed(name, value)

    def _on_param_changed(self, name: str, value: float):
        """Override to react to parameter changes (recalculate coefficients etc.)."""

    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        """Process a block of audio. Shape: (frames,) or (frames, channels)."""
        ...

    def process_maybe_bypass(self, data: np.ndarray) -> np.ndarray:
        if self._bypassed:
            return data
        return self.process(data)

    def reset(self):
        """Reset internal state (e.g., delay lines, envelopes)."""

    def get_params(self) -> dict[str, float]:
        return dict(self._params)

    def set_params(self, params: dict[str, float]):
        for k, v in params.items():
            self.set_param(k, v)
