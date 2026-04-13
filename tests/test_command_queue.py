"""Exhaustive tests for CommandQueue, Command, and CommandType."""
import pytest

from vynth.utils.thread_safe_queue import Command, CommandQueue, CommandType


class TestCommandType:
    def test_all_types_exist(self):
        expected = [
            "NOTE_ON", "NOTE_OFF", "ALL_NOTES_OFF", "PARAM_CHANGE",
            "PITCH_BEND", "MOD_WHEEL", "SUSTAIN_PEDAL", "SET_SAMPLE",
            "SET_PLAYBACK_MODE",
        ]
        for name in expected:
            assert hasattr(CommandType, name)

    def test_unique_values(self):
        values = [ct.value for ct in CommandType]
        assert len(values) == len(set(values))


class TestCommand:
    def test_default_fields(self):
        cmd = Command(type=CommandType.NOTE_ON)
        assert cmd.channel == 0
        assert cmd.note == 0
        assert cmd.velocity == 0
        assert cmd.param_name == ""
        assert cmd.param_value == 0.0
        assert cmd.data is None

    def test_note_on_command(self):
        cmd = Command(type=CommandType.NOTE_ON, note=60, velocity=100)
        assert cmd.type == CommandType.NOTE_ON
        assert cmd.note == 60
        assert cmd.velocity == 100

    def test_param_change_command(self):
        cmd = Command(
            type=CommandType.PARAM_CHANGE,
            param_name="chorus_rate",
            param_value=2.5,
        )
        assert cmd.param_name == "chorus_rate"
        assert cmd.param_value == 2.5

    def test_set_sample_with_data(self):
        cmd = Command(type=CommandType.SET_SAMPLE, data="test_data")
        assert cmd.data == "test_data"

    def test_slots(self):
        """Command uses __slots__ for memory efficiency."""
        cmd = Command(type=CommandType.NOTE_ON)
        assert hasattr(cmd, "__slots__") or hasattr(cmd.__class__, "__slots__")


class TestCommandQueue:
    def test_init_empty(self):
        q = CommandQueue()
        assert len(q) == 0

    def test_default_maxlen(self):
        q = CommandQueue()
        assert q._queue.maxlen == 4096

    def test_custom_maxlen(self):
        q = CommandQueue(maxlen=128)
        assert q._queue.maxlen == 128


class TestCommandQueuePush:
    def test_push_single(self):
        q = CommandQueue()
        cmd = Command(type=CommandType.NOTE_ON, note=60, velocity=100)
        assert q.push(cmd) is True
        assert len(q) == 1

    def test_push_multiple(self):
        q = CommandQueue()
        for i in range(10):
            q.push(Command(type=CommandType.NOTE_ON, note=60 + i, velocity=100))
        assert len(q) == 10

    def test_push_full_returns_false(self):
        q = CommandQueue(maxlen=2)
        q.push(Command(type=CommandType.NOTE_ON))
        q.push(Command(type=CommandType.NOTE_ON))
        result = q.push(Command(type=CommandType.NOTE_ON))
        assert result is False

    def test_dropped_count(self):
        q = CommandQueue(maxlen=2)
        q.push(Command(type=CommandType.NOTE_ON))
        q.push(Command(type=CommandType.NOTE_ON))
        q.push(Command(type=CommandType.NOTE_ON))
        assert q.dropped_count == 1


class TestCommandQueuePop:
    def test_pop_returns_command(self):
        q = CommandQueue()
        cmd = Command(type=CommandType.NOTE_ON, note=60, velocity=100)
        q.push(cmd)
        result = q.pop()
        assert result is not None
        assert result.note == 60

    def test_pop_empty_returns_none(self):
        q = CommandQueue()
        assert q.pop() is None

    def test_pop_fifo_order(self):
        q = CommandQueue()
        for i in range(5):
            q.push(Command(type=CommandType.NOTE_ON, note=60 + i))
        for i in range(5):
            cmd = q.pop()
            assert cmd.note == 60 + i


class TestCommandQueueDrain:
    def test_drain_all(self):
        q = CommandQueue()
        for i in range(5):
            q.push(Command(type=CommandType.NOTE_ON, note=60 + i))
        cmds = q.drain()
        assert len(cmds) == 5
        assert len(q) == 0

    def test_drain_with_limit(self):
        q = CommandQueue()
        for i in range(10):
            q.push(Command(type=CommandType.NOTE_ON))
        cmds = q.drain(max_count=3)
        assert len(cmds) == 3
        assert len(q) == 7

    def test_drain_empty(self):
        q = CommandQueue()
        cmds = q.drain()
        assert cmds == []

    def test_drain_preserves_order(self):
        q = CommandQueue()
        for i in range(5):
            q.push(Command(type=CommandType.NOTE_ON, note=i))
        cmds = q.drain()
        notes = [c.note for c in cmds]
        assert notes == [0, 1, 2, 3, 4]


class TestCommandQueueClear:
    def test_clear_empties_queue(self):
        q = CommandQueue()
        for i in range(10):
            q.push(Command(type=CommandType.NOTE_ON))
        q.clear()
        assert len(q) == 0
        assert q.pop() is None


class TestCommandQueueStress:
    def test_rapid_push_pop(self):
        q = CommandQueue()
        for _ in range(10000):
            q.push(Command(type=CommandType.NOTE_ON, note=60))
            q.pop()
        assert len(q) == 0

    def test_alternating_push_drain(self):
        q = CommandQueue()
        total_pushed = 0
        total_drained = 0
        for _ in range(100):
            for _ in range(10):
                q.push(Command(type=CommandType.NOTE_ON))
                total_pushed += 1
            cmds = q.drain(max_count=5)
            total_drained += len(cmds)
        # Drain remainder
        total_drained += len(q.drain(max_count=10000))
        assert total_pushed == total_drained
