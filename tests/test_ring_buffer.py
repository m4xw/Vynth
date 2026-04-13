"""Exhaustive tests for the lock-free RingBuffer."""
import numpy as np
import pytest

from vynth.utils.ring_buffer import RingBuffer


class TestRingBufferInit:
    def test_capacity(self):
        rb = RingBuffer(1024, channels=2)
        assert rb.capacity == 1024

    def test_empty_initially(self):
        rb = RingBuffer(1024)
        assert rb.available_read() == 0

    def test_full_write_capacity(self):
        rb = RingBuffer(1024)
        # capacity - 1 because one slot is reserved
        assert rb.available_write() == 1023

    def test_custom_channels(self):
        rb = RingBuffer(512, channels=1)
        assert rb._channels == 1


class TestRingBufferWrite:
    def test_write_mono_to_stereo(self):
        rb = RingBuffer(100, channels=2)
        data = np.ones(10, dtype=np.float32) * 0.5
        written = rb.write(data)
        assert written == 10

    def test_write_stereo(self):
        rb = RingBuffer(100, channels=2)
        data = np.ones((10, 2), dtype=np.float32) * 0.3
        written = rb.write(data)
        assert written == 10

    def test_write_updates_available(self):
        rb = RingBuffer(100, channels=2)
        data = np.ones((10, 2), dtype=np.float32)
        rb.write(data)
        assert rb.available_read() == 10

    def test_write_more_than_capacity(self):
        rb = RingBuffer(10, channels=2)
        data = np.ones((20, 2), dtype=np.float32)
        written = rb.write(data)
        assert written <= rb.capacity

    def test_write_zero_when_full(self):
        rb = RingBuffer(10, channels=2)
        data = np.ones((9, 2), dtype=np.float32)  # fills it
        rb.write(data)
        written = rb.write(np.ones((1, 2), dtype=np.float32))
        assert written == 0  # capacity - 1 = 9, so 0 left


class TestRingBufferRead:
    def test_read_written_data(self):
        rb = RingBuffer(100, channels=2)
        data = np.ones((10, 2), dtype=np.float32) * 0.7
        rb.write(data)
        out = rb.read(10)
        assert out.shape == (10, 2)
        np.testing.assert_allclose(out, 0.7, atol=1e-6)

    def test_read_advances_pointer(self):
        rb = RingBuffer(100, channels=2)
        rb.write(np.ones((20, 2), dtype=np.float32))
        rb.read(10)
        assert rb.available_read() == 10

    def test_read_more_than_available(self):
        rb = RingBuffer(100, channels=2)
        rb.write(np.ones((5, 2), dtype=np.float32))
        out = rb.read(20)
        assert out.shape == (5, 2)

    def test_read_empty(self):
        rb = RingBuffer(100, channels=2)
        out = rb.read(10)
        assert out.shape == (0, 2)


class TestRingBufferPeek:
    def test_peek_does_not_advance(self):
        rb = RingBuffer(100, channels=2)
        rb.write(np.ones((10, 2), dtype=np.float32))
        rb.peek(10)
        assert rb.available_read() == 10

    def test_peek_returns_data(self):
        rb = RingBuffer(100, channels=2)
        data = np.ones((10, 2), dtype=np.float32) * 0.3
        rb.write(data)
        out = rb.peek(10)
        np.testing.assert_allclose(out, 0.3, atol=1e-6)

    def test_peek_empty(self):
        rb = RingBuffer(100, channels=2)
        out = rb.peek(10)
        assert out.shape == (0, 2)


class TestRingBufferWrap:
    def test_write_read_wrap(self):
        rb = RingBuffer(10, channels=2)
        # Write 8, read 8, write 8 (wraps around)
        data1 = np.ones((8, 2), dtype=np.float32) * 0.1
        rb.write(data1)
        rb.read(8)

        data2 = np.ones((8, 2), dtype=np.float32) * 0.9
        rb.write(data2)
        out = rb.read(8)
        np.testing.assert_allclose(out, 0.9, atol=1e-6)

    def test_peek_wrap(self):
        rb = RingBuffer(10, channels=2)
        rb.write(np.ones((8, 2), dtype=np.float32))
        rb.read(8)
        rb.write(np.ones((8, 2), dtype=np.float32) * 0.5)
        out = rb.peek(8)
        np.testing.assert_allclose(out, 0.5, atol=1e-6)


class TestRingBufferClear:
    def test_clear_resets_pointers(self):
        rb = RingBuffer(100, channels=2)
        rb.write(np.ones((50, 2), dtype=np.float32))
        rb.clear()
        assert rb.available_read() == 0
        assert rb.available_write() == 99

    def test_read_after_clear_empty(self):
        rb = RingBuffer(100, channels=2)
        rb.write(np.ones((50, 2), dtype=np.float32))
        rb.clear()
        out = rb.read(10)
        assert out.shape == (0, 2)


class TestRingBufferStress:
    def test_many_write_read_cycles(self):
        rb = RingBuffer(64, channels=2)
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = rng.integers(1, 30)
            data = rng.standard_normal((n, 2)).astype(np.float32)
            written = rb.write(data)
            # Read back what was written
            rb.read(written)
        assert rb.available_read() == 0
