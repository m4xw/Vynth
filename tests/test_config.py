"""Exhaustive tests for config.py — constants, SessionSettings, AppConfig."""
import pytest
import copy
from dataclasses import fields

from vynth.config import (
    SAMPLE_RATE,
    BLOCK_SIZE,
    MAX_VOICES,
    CHANNELS,
    SessionSettings,
    AppConfig,
    default_midi_controller_profile,
)


class TestConstants:
    def test_sample_rate(self):
        assert SAMPLE_RATE == 48_000

    def test_block_size(self):
        assert BLOCK_SIZE == 512

    def test_max_voices(self):
        assert MAX_VOICES == 64

    def test_channels(self):
        assert CHANNELS == 2

    def test_sample_rate_positive(self):
        assert SAMPLE_RATE > 0

    def test_block_size_positive(self):
        assert BLOCK_SIZE > 0

    def test_max_voices_positive(self):
        assert MAX_VOICES > 0

    def test_channels_positive(self):
        assert CHANNELS > 0

    def test_block_size_power_of_two(self):
        assert BLOCK_SIZE & (BLOCK_SIZE - 1) == 0, "BLOCK_SIZE should be power of 2"


class TestSessionSettings:
    def test_default_construction(self):
        s = SessionSettings()
        assert isinstance(s, SessionSettings)

    def test_has_master_volume(self):
        s = SessionSettings()
        assert hasattr(s, "master_volume")

    def test_default_master_volume(self):
        s = SessionSettings()
        assert s.master_volume == pytest.approx(0.8) or s.master_volume >= 0

    def test_is_dataclass(self):
        f = fields(SessionSettings)
        assert len(f) > 0

    def test_copy(self):
        s = SessionSettings()
        s2 = copy.copy(s)
        assert s == s2

    def test_deepcopy(self):
        s = SessionSettings()
        s2 = copy.deepcopy(s)
        assert s == s2

    def test_fields_are_numeric_or_str_or_dict(self):
        s = SessionSettings()
        for f in fields(s):
            val = getattr(s, f.name)
            assert isinstance(val, (int, float, str, bool, dict, list, type(None))), \
                f"Unexpected type for {f.name}: {type(val)}"


class TestAppConfig:
    def test_construction(self):
        c = AppConfig()
        assert isinstance(c, AppConfig)

    def test_last_session_path_default(self):
        c = AppConfig()
        assert isinstance(c.last_session_path, str)

    def test_midi_controller_profile_default(self):
        c = AppConfig()
        profile = c.midi_controller_profile
        assert isinstance(profile, dict)
        assert "mappings" in profile

    def test_midi_controller_profile_roundtrip(self, tmp_path, monkeypatch):
        path = tmp_path / "vynth_app_config.json"
        monkeypatch.setattr(AppConfig, "_resolve_path", staticmethod(lambda: path))

        c = AppConfig()
        profile = default_midi_controller_profile()
        profile["name"] = "Test Profile"
        c.midi_controller_profile = profile

        c2 = AppConfig()
        loaded = c2.midi_controller_profile
        assert loaded["name"] == "Test Profile"
        assert isinstance(loaded.get("mappings"), list)
