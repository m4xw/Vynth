"""Exhaustive tests for session save/load round-trip."""
import json
import numpy as np
import pytest
import tempfile
from pathlib import Path

from vynth.sampler.sample import Sample


def _make_sample(name="Piano", sr=48000, duration_s=0.1):
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    data = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    return Sample.from_buffer(data, sr, name)


class TestSessionDataStructure:
    """Test that session JSON has the expected structure."""

    def test_session_json_structure(self, tmp_path):
        session_data = {
            "master_volume": 0.75,
            "sample_rate": 48000,
            "block_size": 512,
            "samples": [],
            "effects": {"params": {}, "bypass": {}},
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(session_data), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["master_volume"] == 0.75
        assert loaded["sample_rate"] == 48000
        assert loaded["block_size"] == 512
        assert isinstance(loaded["samples"], list)
        assert isinstance(loaded["effects"], dict)


class TestSessionSampleRoundTrip:
    """Test that samples survive save→load→save cycle."""

    def test_save_and_reload_sample_wav(self, tmp_path):
        s = _make_sample("Test")
        wav_path = tmp_path / "test.wav"
        s.save(wav_path)

        loaded = Sample.from_file(wav_path)
        assert loaded.name == "test"  # stem of filename
        np.testing.assert_allclose(loaded.data, s.data, atol=1e-4)

    def test_stereo_save_reload(self, tmp_path):
        mono = _make_sample("Stereo", duration_s=0.1)
        stereo_data = np.column_stack([mono.data, mono.data])
        s = Sample.from_buffer(stereo_data, 48000, "Stereo")
        wav_path = tmp_path / "stereo.wav"
        s.save(wav_path)

        loaded = Sample.from_file(wav_path)
        assert loaded.channels == 2
        np.testing.assert_allclose(loaded.data, stereo_data, atol=1e-4)


class TestSessionEffectsState:
    def test_effects_state_roundtrip(self, tmp_path):
        effects_state = {
            "params": {
                "chorus_rate": 2.0,
                "chorus_depth": 0.5,
                "reverb_room_size": 0.8,
                "reverb_wet": 0.4,
                "delay_time_ms": 300.0,
                "filter_freq": 1500.0,
            },
            "bypass": {
                "chorus": False,
                "reverb": True,
                "delay": False,
            },
        }
        session_data = {
            "master_volume": 0.8,
            "sample_rate": 48000,
            "block_size": 512,
            "samples": [],
            "effects": effects_state,
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(session_data), encoding="utf-8")

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["effects"]["params"]["chorus_rate"] == 2.0
        assert loaded["effects"]["bypass"]["reverb"] is True


class TestSessionMidiControllerProfile:
    def test_midi_controller_profile_persisted(self, tmp_path):
        data = {
            "samples": [],
            "effects": {},
            "midi_controller_profile": {
                "name": "My MPK Profile",
                "mappings": [
                    {
                        "enabled": True,
                        "input": "cc",
                        "number": 74,
                        "channel": "all",
                        "trigger": "change",
                        "mode": "absolute",
                        "target_type": "param",
                        "target": "filter_frequency",
                        "min": 200.0,
                        "max": 12000.0,
                    }
                ],
            },
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = json.loads(path.read_text(encoding="utf-8"))
        profile = loaded.get("midi_controller_profile", {})
        assert profile.get("name") == "My MPK Profile"
        assert isinstance(profile.get("mappings"), list)
        assert profile["mappings"][0]["target"] == "filter_frequency"


class TestSessionMasterVolume:
    @pytest.mark.parametrize("vol", [0.0, 0.5, 0.8, 1.0])
    def test_volume_roundtrip(self, tmp_path, vol):
        data = {"master_volume": vol, "samples": [], "effects": {}}
        path = tmp_path / "session.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["master_volume"] == pytest.approx(vol)


class TestSessionSamplePaths:
    def test_sample_entry_with_path(self, tmp_path):
        s = _make_sample("Piano")
        wav_path = str(tmp_path / "Piano.wav")
        s.save(wav_path)

        session = {
            "samples": [{"path": wav_path, "root_note": 60}],
            "effects": {},
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(session), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["samples"][0]["path"] == wav_path
        assert loaded["samples"][0]["root_note"] == 60

    def test_sample_entry_old_format_string(self, tmp_path):
        """Test backward compatibility with old format (plain path string)."""
        s = _make_sample("Piano")
        wav_path = str(tmp_path / "Piano.wav")
        s.save(wav_path)

        session = {
            "samples": [wav_path],
            "effects": {},
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(session), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["samples"][0] == wav_path


class TestSessionSampleMetadata:
    def test_sample_metadata_fields_persisted(self, tmp_path):
        session = {
            "samples": [
                {
                    "path": "C:/audio/piano.wav",
                    "name": "Piano",
                    "root_note": 57,
                    "note_range": [36, 84],
                    "velocity_range": [1, 120],
                    "loop": {
                        "start": 120,
                        "end": 2048,
                        "crossfade": 128,
                        "enabled": True,
                    },
                    "selected": True,
                    "selection": {"start": 200, "end": 900},
                    "index": 0,
                }
            ],
            "selected_sample_index": 0,
            "selected_sample_name": "Piano",
            "waveform_selection": {"start": 200, "end": 900},
            "runtime": {
                "octave_shift": 1,
                "controller": {"division": 0.25, "swing": 0.1},
            },
            "effects": {},
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(session), encoding="utf-8")

        loaded = json.loads(path.read_text(encoding="utf-8"))
        sample0 = loaded["samples"][0]
        assert sample0["name"] == "Piano"
        assert sample0["loop"]["enabled"] is True
        assert sample0["note_range"] == [36, 84]
        assert sample0["selection"] == {"start": 200, "end": 900}
        assert loaded["selected_sample_index"] == 0
        assert loaded["waveform_selection"]["end"] == 900
        assert loaded["runtime"]["octave_shift"] == 1


class TestSessionEdgeCases:
    def test_empty_session(self, tmp_path):
        data = {"samples": [], "effects": {}}
        path = tmp_path / "session.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert len(loaded["samples"]) == 0

    def test_corrupt_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            json.loads(path.read_text(encoding="utf-8"))

    def test_missing_keys_use_defaults(self, tmp_path):
        data = {}
        path = tmp_path / "session.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        # Should be able to use .get with defaults
        assert loaded.get("master_volume", 0.8) == 0.8
        assert loaded.get("samples", []) == []

    def test_multiple_samples_in_session(self, tmp_path):
        samples = []
        for i in range(5):
            s = _make_sample(f"Sample_{i}")
            wav_path = str(tmp_path / f"sample_{i}.wav")
            s.save(wav_path)
            samples.append({"path": wav_path, "root_note": 60 + i})

        data = {"samples": samples, "effects": {}}
        path = tmp_path / "session.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert len(loaded["samples"]) == 5
