"""Audio utility functions."""
import numpy as np


def mono_to_stereo(data: np.ndarray) -> np.ndarray:
    """Convert mono (N,) array to stereo (N, 2)."""
    if data.ndim == 2 and data.shape[1] == 2:
        return data
    if data.ndim == 2 and data.shape[1] == 1:
        return np.column_stack([data[:, 0], data[:, 0]])
    return np.column_stack([data, data])


def stereo_to_mono(data: np.ndarray) -> np.ndarray:
    """Mix stereo (N, 2) down to mono (N,)."""
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def normalize(data: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """Normalize to peak amplitude."""
    mx = np.abs(data).max()
    if mx < 1e-10:
        return data
    return data * (peak / mx)


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    if linear < 1e-10:
        return -120.0
    return 20.0 * np.log10(linear)


def note_to_freq(note: int, a4: float = 440.0) -> float:
    """MIDI note number to frequency in Hz."""
    return a4 * (2.0 ** ((note - 69) / 12.0))


def freq_to_note(freq: float, a4: float = 440.0) -> float:
    """Frequency to fractional MIDI note number."""
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / a4)


def crossfade(a: np.ndarray, b: np.ndarray, length: int) -> np.ndarray:
    """Crossfade between end of `a` and start of `b` over `length` samples."""
    length = min(length, len(a), len(b))
    if length <= 0:
        return np.concatenate([a, b])

    fade_out = np.linspace(1.0, 0.0, length, dtype=np.float32)
    fade_in = 1.0 - fade_out

    result = np.empty(len(a) + len(b) - length, dtype=a.dtype)
    result[:len(a) - length] = a[:len(a) - length]
    if a.ndim == 1:
        result[len(a) - length:len(a)] = a[-length:] * fade_out + b[:length] * fade_in
    else:
        result[len(a) - length:len(a)] = (
            a[-length:] * fade_out[:, None] + b[:length] * fade_in[:, None]
        )
    result[len(a):] = b[length:]
    return result


def resample_linear(data: np.ndarray, ratio: float) -> np.ndarray:
    """Simple linear-interpolation resampling. ratio > 1 = shorter/higher."""
    if abs(ratio - 1.0) < 1e-6:
        return data.copy()
    n_out = int(len(data) / ratio)
    if n_out <= 0:
        return np.zeros(1, dtype=data.dtype)
    indices = np.arange(n_out, dtype=np.float64) * ratio
    idx_int = indices.astype(np.int64)
    frac = (indices - idx_int).astype(np.float32)
    idx_int = np.clip(idx_int, 0, len(data) - 2)
    if data.ndim == 1:
        return data[idx_int] * (1.0 - frac) + data[idx_int + 1] * frac
    else:
        return data[idx_int] * (1.0 - frac)[:, None] + data[idx_int + 1] * frac[:, None]
