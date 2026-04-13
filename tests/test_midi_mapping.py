"""Tests for MIDI controller mapping resolution."""

from vynth.engine.midi_mapping import normalize_event, resolve_mapping_events, sanitize_profile


class TestMidiMappingSanitize:
    def test_sanitize_profile_defaults(self):
        profile = sanitize_profile(None)
        assert profile["name"] == "Controller Profile"
        assert profile["mappings"] == []

    def test_sanitize_keeps_dict_mappings(self):
        profile = sanitize_profile({"name": "X", "mappings": [{"input": "cc", "number": 7}]})
        assert profile["name"] == "X"
        assert len(profile["mappings"]) == 1


class TestMidiMappingResolve:
    def test_cc_absolute_param_mapping(self):
        event = normalize_event("cc", 74, 0, 64, True)
        mappings = [
            {
                "enabled": True,
                "input": "cc",
                "number": 74,
                "channel": "all",
                "trigger": "change",
                "mode": "absolute",
                "target_type": "param",
                "target": "filter_frequency",
                "min": 0.0,
                "max": 1.0,
            }
        ]
        out, _state = resolve_mapping_events(event, mappings, {})
        assert len(out) == 1
        assert out[0]["type"] == "param"
        assert out[0]["target"] == "filter_frequency"
        assert 0.49 < out[0]["value"] < 0.51

    def test_note_momentary_param_mapping(self):
        mappings = [
            {
                "enabled": True,
                "input": "note",
                "number": 36,
                "channel": "all",
                "trigger": "change",
                "mode": "momentary",
                "target_type": "param",
                "target": "delay_bypass",
                "min": 0.0,
                "max": 1.0,
            }
        ]
        pressed_event = normalize_event("note", 36, 0, 100, True)
        released_event = normalize_event("note", 36, 0, 0, False)

        out_press, state = resolve_mapping_events(pressed_event, mappings, {})
        out_release, _state = resolve_mapping_events(released_event, mappings, state)

        assert out_press[0]["value"] == 1.0
        assert out_release[0]["value"] == 0.0

    def test_note_toggle_param_mapping(self):
        mappings = [
            {
                "enabled": True,
                "input": "note",
                "number": 38,
                "channel": "all",
                "trigger": "press",
                "mode": "toggle",
                "target_type": "param",
                "target": "reverb_bypass",
                "min": 0.0,
                "max": 1.0,
            }
        ]
        press = normalize_event("note", 38, 0, 127, True)

        out_a, state = resolve_mapping_events(press, mappings, {})
        out_b, _state = resolve_mapping_events(press, mappings, state)

        assert out_a[0]["value"] == 1.0
        assert out_b[0]["value"] == 0.0

    def test_action_mapping_emits_action(self):
        event = normalize_event("note", 37, 0, 100, True)
        mappings = [
            {
                "enabled": True,
                "input": "note",
                "number": 37,
                "channel": "all",
                "trigger": "press",
                "mode": "momentary",
                "target_type": "action",
                "target": "stop",
            }
        ]
        out, _state = resolve_mapping_events(event, mappings, {})
        assert out == [{"type": "action", "target": "stop"}]
