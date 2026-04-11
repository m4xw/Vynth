"""Simple gain/boost DSP effect."""
from __future__ import annotations

import numpy as np

from vynth.dsp.base import DSPEffect


class GainEffect(DSPEffect):
    """Linear gain stage with dB control.

    Params
    ------
    gain_db : float
        Gain in decibels. Range typically -24 … +24 dB. Default 0 dB.
    """

    PARAM_DEFAULTS: dict[str, float] = {"gain_db": 0.0}

    def __init__(self, sample_rate: int = 48_000) -> None:
        super().__init__(sample_rate)
        self._params = dict(self.PARAM_DEFAULTS)
        self._linear: float = 1.0

    # ── DSPEffect interface ──────────────────────────────────────────

    def process(self, data: np.ndarray) -> np.ndarray:
        return data * self._linear

    def _on_param_changed(self, name: str, value: float) -> None:
        if name == "gain_db":
            self._linear = float(np.power(10.0, value / 20.0))
