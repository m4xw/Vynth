"""Shared test fixtures for Vynth."""
import numpy as np
import pytest

from vynth.config import SAMPLE_RATE, BLOCK_SIZE


@pytest.fixture
def sr():
    return SAMPLE_RATE


@pytest.fixture
def block_size():
    return BLOCK_SIZE


@pytest.fixture
def sine_440(sr):
    """1-second 440 Hz sine wave, mono float32."""
    t = np.arange(sr, dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)


@pytest.fixture
def sine_440_stereo(sine_440):
    """1-second 440 Hz sine, stereo float32."""
    return np.column_stack([sine_440, sine_440])


@pytest.fixture
def impulse(sr):
    """Unit impulse, mono."""
    buf = np.zeros(sr, dtype=np.float32)
    buf[0] = 1.0
    return buf


@pytest.fixture
def silence(sr):
    """1-second silence, mono."""
    return np.zeros(sr, dtype=np.float32)


@pytest.fixture
def noise_block(block_size):
    """One block of white noise, stereo."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((block_size, 2)).astype(np.float32) * 0.5
