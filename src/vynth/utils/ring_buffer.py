"""Lock-free ring buffer for audio-thread communication."""
import numpy as np


class RingBuffer:
    """Single-producer single-consumer ring buffer backed by numpy.

    The audio callback writes blocks; the UI thread reads for visualization.
    Thread-safe via atomic index reads (numpy int64) — no locks needed.
    """

    def __init__(self, capacity: int, channels: int = 2, dtype=np.float32):
        self._capacity = capacity
        self._channels = channels
        self._buffer = np.zeros((capacity, channels), dtype=dtype)
        self._write_idx = 0  # only modified by writer
        self._read_idx = 0   # only modified by reader

    @property
    def capacity(self) -> int:
        return self._capacity

    def available_read(self) -> int:
        diff = self._write_idx - self._read_idx
        if diff < 0:
            diff += self._capacity
        return diff

    def available_write(self) -> int:
        return self._capacity - self.available_read() - 1

    def write(self, data: np.ndarray) -> int:
        """Write data into the buffer. Returns number of frames actually written."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            if self._channels == 2:
                data = np.column_stack([data, data])

        n = min(len(data), self.available_write())
        if n == 0:
            return 0

        idx = self._write_idx % self._capacity
        end = idx + n

        if end <= self._capacity:
            self._buffer[idx:end] = data[:n]
        else:
            first = self._capacity - idx
            self._buffer[idx:] = data[:first]
            self._buffer[:n - first] = data[first:n]

        self._write_idx = (self._write_idx + n) % self._capacity
        return n

    def read(self, n: int) -> np.ndarray:
        """Read up to n frames from the buffer."""
        available = self.available_read()
        n = min(n, available)
        if n == 0:
            return np.zeros((0, self._channels), dtype=self._buffer.dtype)

        idx = self._read_idx % self._capacity
        end = idx + n

        if end <= self._capacity:
            result = self._buffer[idx:end].copy()
        else:
            first = self._capacity - idx
            result = np.empty((n, self._channels), dtype=self._buffer.dtype)
            result[:first] = self._buffer[idx:]
            result[first:] = self._buffer[:n - first]

        self._read_idx = (self._read_idx + n) % self._capacity
        return result

    def peek(self, n: int) -> np.ndarray:
        """Read without advancing the read pointer."""
        available = self.available_read()
        n = min(n, available)
        if n == 0:
            return np.zeros((0, self._channels), dtype=self._buffer.dtype)

        idx = self._read_idx % self._capacity
        end = idx + n

        if end <= self._capacity:
            return self._buffer[idx:end].copy()
        else:
            first = self._capacity - idx
            result = np.empty((n, self._channels), dtype=self._buffer.dtype)
            result[:first] = self._buffer[idx:]
            result[first:] = self._buffer[:n - first]
            return result

    def clear(self):
        self._write_idx = 0
        self._read_idx = 0
