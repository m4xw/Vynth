"""Exhaustive tests for SampleManager."""
import numpy as np
import pytest
import tempfile
from pathlib import Path

from vynth.sampler.sample import Sample
from vynth.sampler.sample_manager import SampleManager, _safe_filename


@pytest.fixture
def mgr(qapp):
    return SampleManager()


def _make_sample(name="test", sr=48000, duration_s=0.1):
    n = int(sr * duration_s)
    data = np.random.default_rng(42).standard_normal(n).astype(np.float32) * 0.5
    return Sample.from_buffer(data, sr, name)


class TestSampleManagerInit:
    def test_empty_initially(self, mgr):
        assert mgr.get_names() == []

    def test_no_selection(self, mgr):
        assert mgr.get_selected() is None


class TestSampleManagerAdd:
    def test_add_sample(self, mgr):
        s = _make_sample("Piano")
        mgr.add_sample(s)
        assert "Piano" in mgr.get_names()

    def test_add_emits_signal(self, mgr, qtbot):
        s = _make_sample("Piano")
        with qtbot.waitSignal(mgr.sample_added, timeout=1000):
            mgr.add_sample(s)

    def test_add_multiple(self, mgr):
        for name in ["A", "B", "C"]:
            mgr.add_sample(_make_sample(name))
        assert len(mgr.get_names()) == 3

    def test_unique_name_collision(self, mgr):
        mgr.add_sample(_make_sample("Piano"))
        mgr.add_sample(_make_sample("Piano"))
        names = mgr.get_names()
        assert len(names) == 2
        assert "Piano" in names
        assert "Piano (2)" in names


class TestSampleManagerRemove:
    def test_remove_sample(self, mgr):
        mgr.add_sample(_make_sample("Piano"))
        mgr.remove_sample("Piano")
        assert "Piano" not in mgr.get_names()

    def test_remove_emits_signal(self, mgr, qtbot):
        mgr.add_sample(_make_sample("Piano"))
        with qtbot.waitSignal(mgr.sample_removed, timeout=1000):
            mgr.remove_sample("Piano")

    def test_remove_nonexistent(self, mgr):
        mgr.remove_sample("Nonexistent")  # Should not raise

    def test_remove_clears_selection(self, mgr):
        mgr.add_sample(_make_sample("Piano"))
        mgr.select_sample("Piano")
        mgr.remove_sample("Piano")
        assert mgr.get_selected() is None


class TestSampleManagerGet:
    def test_get_sample_by_name(self, mgr):
        s = _make_sample("Piano")
        mgr.add_sample(s)
        result = mgr.get_sample("Piano")
        assert result is not None
        assert result.name == "Piano"

    def test_get_nonexistent_returns_none(self, mgr):
        assert mgr.get_sample("Nonexistent") is None


class TestSampleManagerSelect:
    def test_select_sample(self, mgr):
        mgr.add_sample(_make_sample("Piano"))
        mgr.select_sample("Piano")
        selected = mgr.get_selected()
        assert selected is not None
        assert selected.name == "Piano"

    def test_select_emits_signal(self, mgr, qtbot):
        mgr.add_sample(_make_sample("Piano"))
        with qtbot.waitSignal(mgr.sample_selected, timeout=1000):
            mgr.select_sample("Piano")

    def test_select_nonexistent(self, mgr):
        mgr.select_sample("Nonexistent")  # Should not raise
        assert mgr.get_selected() is None


class TestSampleManagerFileIO:
    def test_save_and_load_directory(self, mgr, tmp_path):
        mgr.add_sample(_make_sample("A"))
        mgr.add_sample(_make_sample("B"))
        save_dir = str(tmp_path / "samples")
        mgr.save_to_directory(save_dir)

        # Check files exist
        saved = list(Path(save_dir).glob("*.wav"))
        assert len(saved) == 2

        # Load back into new manager
        mgr2 = SampleManager()
        mgr2.load_directory(save_dir)
        assert len(mgr2.get_names()) == 2

    def test_load_nonexistent_directory(self, mgr):
        mgr.load_directory("/nonexistent/path")  # Should not raise
        assert len(mgr.get_names()) == 0


class TestSampleManagerUniqueName:
    def test_first_use_unchanged(self, mgr):
        name = mgr._unique_name("Piano")
        assert name == "Piano"

    def test_collision_appends_number(self, mgr):
        mgr.add_sample(_make_sample("Piano"))
        name = mgr._unique_name("Piano")
        assert name == "Piano (2)"

    def test_multiple_collisions(self, mgr):
        mgr.add_sample(_make_sample("Piano"))
        mgr.add_sample(_make_sample("Piano"))
        name = mgr._unique_name("Piano")
        assert name == "Piano (3)"


class TestSafeFilename:
    def test_normal_name(self):
        assert _safe_filename("Piano") == "Piano"

    def test_special_chars(self):
        result = _safe_filename("My/File:Name")
        assert "/" not in result
        assert ":" not in result

    def test_empty_returns_untitled(self):
        assert _safe_filename("") == "untitled"

    def test_all_special(self):
        assert _safe_filename("///") == "___"
