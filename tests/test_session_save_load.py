"""Tests for session save/load — sample persistence, effects state, round-trip."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vynth.config import SAMPLE_RATE
from vynth.sampler.sample import Sample, LoopRegion
from vynth.sampler.sample_manager import SampleManager


@pytest.fixture
def tmp_session(tmp_path):
    """Provide a temp dir and session file path."""
    return tmp_path / "test_session.json"


@pytest.fixture
def sample_wav(tmp_path):
    """Create a real WAV file on disk and return its path."""
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    wav_path = tmp_path / "sine440.wav"
    sf.write(str(wav_path), data, SAMPLE_RATE, subtype="FLOAT")
    return wav_path


@pytest.fixture
def recording_sample():
    """A sample with no file_path (like a recording)."""
    t = np.arange(SAMPLE_RATE // 2, dtype=np.float32) / SAMPLE_RATE
    data = (np.sin(2 * np.pi * 880 * t) * 0.3).astype(np.float32)
    s = Sample.from_buffer(data, SAMPLE_RATE, "My Recording")
    return s


def _build_session_data(sample_manager, effects_state=None, master_volume=0.8):
    """Replicate the save logic from VynthApp._on_save_session (without UI)."""
    sample_entries = []
    for name in sample_manager.get_names():
        s = sample_manager.get_sample(name)
        if s is None:
            continue
        if s.file_path and Path(s.file_path).exists():
            sample_entries.append({
                "path": s.file_path,
                "root_note": s.root_note,
            })
        else:
            sample_entries.append({
                "path": "",
                "root_note": s.root_note,
                "embedded": True,
            })

    return {
        "master_volume": master_volume,
        "sample_rate": SAMPLE_RATE,
        "block_size": 512,
        "samples": sample_entries,
        "effects": effects_state or {"params": {}, "bypass": {}, "enabled": {}},
    }


def _save_session(session_path, sample_manager, effects_state=None, master_volume=0.8):
    """Save session to disk, replicating the app logic."""
    session_dir = Path(session_path).parent
    samples_dir = session_dir / (Path(session_path).stem + "_samples")
    sample_entries = []

    for name in sample_manager.get_names():
        s = sample_manager.get_sample(name)
        if s is None:
            continue
        if s.file_path and Path(s.file_path).exists():
            sample_entries.append({"path": s.file_path, "root_note": s.root_note})
        else:
            samples_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
            wav_path = samples_dir / f"{safe}.wav"
            s.save(wav_path)
            sample_entries.append({"path": str(wav_path), "root_note": s.root_note})

    data = {
        "master_volume": master_volume,
        "sample_rate": SAMPLE_RATE,
        "block_size": 512,
        "samples": sample_entries,
        "effects": effects_state or {"params": {}, "bypass": {}, "enabled": {}},
    }
    Path(session_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _load_session(session_path):
    """Load session from disk."""
    return json.loads(Path(session_path).read_text(encoding="utf-8"))


# ── Sample persistence ───────────────────────────────────────────────────


class TestSampleSave:
    """Samples must be persisted correctly in session files."""

    def test_file_backed_sample_saves_path(self, tmp_session, sample_wav):
        mgr = SampleManager()
        mgr.load_sample(str(sample_wav))
        data = _save_session(tmp_session, mgr)
        assert len(data["samples"]) == 1
        assert data["samples"][0]["path"] == str(sample_wav)

    def test_file_backed_sample_preserves_root_note(self, tmp_session, sample_wav):
        mgr = SampleManager()
        s = mgr.load_sample(str(sample_wav))
        s.root_note = 72
        data = _save_session(tmp_session, mgr)
        assert data["samples"][0]["root_note"] == 72

    def test_recording_saved_to_samples_dir(self, tmp_session, recording_sample):
        mgr = SampleManager()
        mgr.add_sample(recording_sample)
        _save_session(tmp_session, mgr)

        samples_dir = tmp_session.parent / (tmp_session.stem + "_samples")
        assert samples_dir.exists()
        wav_files = list(samples_dir.glob("*.wav"))
        assert len(wav_files) == 1

    def test_recording_wav_is_valid(self, tmp_session, recording_sample):
        mgr = SampleManager()
        mgr.add_sample(recording_sample)
        _save_session(tmp_session, mgr)

        samples_dir = tmp_session.parent / (tmp_session.stem + "_samples")
        wav_file = list(samples_dir.glob("*.wav"))[0]
        loaded_data, sr = sf.read(str(wav_file), dtype="float32")
        assert sr == SAMPLE_RATE
        assert len(loaded_data) == recording_sample.length

    def test_recording_path_in_json(self, tmp_session, recording_sample):
        mgr = SampleManager()
        mgr.add_sample(recording_sample)
        _save_session(tmp_session, mgr)

        data = _load_session(tmp_session)
        entry = data["samples"][0]
        assert Path(entry["path"]).exists()

    def test_multiple_samples_saved(self, tmp_session, sample_wav, recording_sample):
        mgr = SampleManager()
        mgr.load_sample(str(sample_wav))
        mgr.add_sample(recording_sample)
        data = _save_session(tmp_session, mgr)
        assert len(data["samples"]) == 2

    def test_empty_session_no_samples(self, tmp_session):
        mgr = SampleManager()
        data = _save_session(tmp_session, mgr)
        assert data["samples"] == []


# ── Sample loading ───────────────────────────────────────────────────────


class TestSampleLoad:
    """Samples must be correctly restored from session files."""

    def test_load_file_backed_sample(self, tmp_session, sample_wav):
        mgr = SampleManager()
        mgr.load_sample(str(sample_wav))
        _save_session(tmp_session, mgr)

        # Load into fresh manager
        data = _load_session(tmp_session)
        mgr2 = SampleManager()
        for entry in data["samples"]:
            path = entry["path"] if isinstance(entry, dict) else entry
            root = entry.get("root_note", 60) if isinstance(entry, dict) else 60
            s = mgr2.load_sample(path)
            s.root_note = root

        assert len(mgr2.get_names()) == 1
        s = mgr2.get_sample(mgr2.get_names()[0])
        assert s is not None
        assert s.length == SAMPLE_RATE

    def test_load_recording_sample(self, tmp_session, recording_sample):
        mgr = SampleManager()
        mgr.add_sample(recording_sample)
        _save_session(tmp_session, mgr)

        data = _load_session(tmp_session)
        mgr2 = SampleManager()
        for entry in data["samples"]:
            path = entry["path"]
            root = entry.get("root_note", 60)
            s = mgr2.load_sample(path)
            s.root_note = root

        assert len(mgr2.get_names()) == 1
        s = mgr2.get_sample(mgr2.get_names()[0])
        assert s is not None
        assert s.length == recording_sample.length

    def test_load_preserves_root_note(self, tmp_session, sample_wav):
        mgr = SampleManager()
        s = mgr.load_sample(str(sample_wav))
        s.root_note = 48
        _save_session(tmp_session, mgr)

        data = _load_session(tmp_session)
        entry = data["samples"][0]
        assert entry["root_note"] == 48

    def test_missing_file_skipped(self, tmp_session):
        data = {
            "master_volume": 0.8,
            "sample_rate": SAMPLE_RATE,
            "block_size": 512,
            "samples": [{"path": "/nonexistent/sample.wav", "root_note": 60}],
            "effects": {"params": {}, "bypass": {}, "enabled": {}},
        }
        Path(tmp_session).write_text(json.dumps(data), encoding="utf-8")
        loaded = _load_session(tmp_session)

        mgr = SampleManager()
        loaded_count = 0
        for entry in loaded["samples"]:
            path = entry["path"]
            if Path(path).exists():
                mgr.load_sample(path)
                loaded_count += 1
        assert loaded_count == 0

    def test_old_format_string_entries(self, tmp_session, sample_wav):
        """Old session format used plain string paths."""
        data = {
            "master_volume": 0.8,
            "sample_rate": SAMPLE_RATE,
            "block_size": 512,
            "samples": [str(sample_wav)],
        }
        Path(tmp_session).write_text(json.dumps(data), encoding="utf-8")
        loaded = _load_session(tmp_session)

        mgr = SampleManager()
        for entry in loaded["samples"]:
            if isinstance(entry, str):
                path, root = entry, 60
            else:
                path, root = entry["path"], entry.get("root_note", 60)
            if Path(path).exists():
                s = mgr.load_sample(path)
                s.root_note = root

        assert len(mgr.get_names()) == 1


# ── Effects state ────────────────────────────────────────────────────────


class TestEffectsStateSave:
    """Effects params, bypass, and enabled states must survive round-trip."""

    def test_effects_state_in_json(self, tmp_session):
        mgr = SampleManager()
        effects = {
            "params": {"chorus_detune_cents": 20.0, "delay_time_ms": 500.0},
            "bypass": {"chorus": True, "delay": False},
            "enabled": {"chorus": True, "delay": True},
        }
        _save_session(tmp_session, mgr, effects_state=effects)

        loaded = _load_session(tmp_session)
        assert loaded["effects"]["params"]["chorus_detune_cents"] == 20.0
        assert loaded["effects"]["params"]["delay_time_ms"] == 500.0
        assert loaded["effects"]["bypass"]["chorus"] is True
        assert loaded["effects"]["bypass"]["delay"] is False

    def test_master_volume_saved(self, tmp_session):
        mgr = SampleManager()
        _save_session(tmp_session, mgr, master_volume=0.6)
        loaded = _load_session(tmp_session)
        assert loaded["master_volume"] == pytest.approx(0.6)

    def test_session_json_is_valid(self, tmp_session, sample_wav):
        mgr = SampleManager()
        mgr.load_sample(str(sample_wav))
        _save_session(tmp_session, mgr)
        # Must not raise
        data = json.loads(Path(tmp_session).read_text(encoding="utf-8"))
        assert "samples" in data
        assert "master_volume" in data
        assert "effects" in data


# ── Round-trip ───────────────────────────────────────────────────────────


class TestRoundTrip:
    """Full save → load → verify cycle."""

    def test_full_round_trip(self, tmp_session, sample_wav, recording_sample):
        # Save
        mgr = SampleManager()
        s1 = mgr.load_sample(str(sample_wav))
        s1.root_note = 48
        mgr.add_sample(recording_sample)

        effects = {
            "params": {"adsr_attack_ms": 25.0, "reverb_wet": 0.5},
            "bypass": {"reverb": True},
            "enabled": {"reverb": False},
        }
        _save_session(tmp_session, mgr, effects_state=effects, master_volume=0.65)

        # Load
        loaded = _load_session(tmp_session)

        assert loaded["master_volume"] == pytest.approx(0.65)
        assert len(loaded["samples"]) == 2

        # Restore samples
        mgr2 = SampleManager()
        for entry in loaded["samples"]:
            path = entry["path"]
            root = entry.get("root_note", 60)
            assert Path(path).exists(), f"Sample file missing: {path}"
            s = mgr2.load_sample(path)
            s.root_note = root

        names = mgr2.get_names()
        assert len(names) == 2

        # First sample root note preserved
        first = mgr2.get_sample(names[0])
        assert first.root_note == 48

        # Effects state preserved
        assert loaded["effects"]["params"]["adsr_attack_ms"] == pytest.approx(25.0)
        assert loaded["effects"]["bypass"]["reverb"] is True
        assert loaded["effects"]["enabled"]["reverb"] is False
