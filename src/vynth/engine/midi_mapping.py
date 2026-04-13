"""Helpers for resolving MIDI controller mappings."""
from __future__ import annotations

from typing import Any


def normalize_event(
    event_type: str,
    number: int,
    channel: int,
    value: int,
    pressed: bool,
) -> dict[str, Any]:
    """Build a normalized event payload used by the mapper."""
    return {
        "input": event_type,
        "number": int(number),
        "channel": int(channel),
        "value": int(max(0, min(127, value))),
        "pressed": bool(pressed),
    }


def sanitize_profile(profile: dict | None) -> dict:
    """Return profile with normalized shape and defaults."""
    if not isinstance(profile, dict):
        return {"name": "Controller Profile", "mappings": []}
    name = str(profile.get("name", "Controller Profile"))
    mappings = profile.get("mappings", [])
    if not isinstance(mappings, list):
        mappings = []
    out: list[dict] = []
    for mapping in mappings:
        if isinstance(mapping, dict):
            out.append(_normalize_mapping(mapping))
    return {"name": name, "mappings": out}


def resolve_mapping_events(
    event: dict[str, Any],
    mappings: list[dict],
    toggle_states: dict[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Resolve one MIDI event into parameter/action outputs.

    Returns a tuple of (resolved events, updated toggle state map).
    """
    resolved: list[dict[str, Any]] = []
    for raw_mapping in mappings:
        mapping = _normalize_mapping(raw_mapping)
        if not mapping.get("enabled", True):
            continue
        if mapping["input"] != event.get("input"):
            continue
        if mapping["number"] != event.get("number"):
            continue
        if not _channel_matches(mapping.get("channel", "all"), event.get("channel", 0)):
            continue
        if not _trigger_matches(mapping.get("trigger", "change"), event):
            continue

        target_type = mapping.get("target_type", "param")
        target = str(mapping.get("target", "")).strip()
        if not target:
            continue

        mode = mapping.get("mode", "absolute")
        if target_type == "param":
            value = _resolve_param_value(mapping, event, mode, toggle_states)
            if value is None:
                continue
            resolved.append({
                "type": "param",
                "target": target,
                "value": value,
            })
        elif target_type == "action":
            if _action_should_fire(mode, event):
                _update_toggle_if_needed(mapping, event, mode, toggle_states)
                resolved.append({
                    "type": "action",
                    "target": target,
                })
    return resolved, toggle_states


def _normalize_mapping(mapping: dict) -> dict:
    out = {
        "enabled": bool(mapping.get("enabled", True)),
        "input": str(mapping.get("input", "cc")).lower(),
        "number": int(mapping.get("number", 0)),
        "channel": mapping.get("channel", "all"),
        "trigger": str(mapping.get("trigger", "change")).lower(),
        "mode": str(mapping.get("mode", "absolute")).lower(),
        "target_type": str(mapping.get("target_type", "param")).lower(),
        "target": str(mapping.get("target", "")),
        "min": float(mapping.get("min", 0.0)),
        "max": float(mapping.get("max", 1.0)),
    }
    if out["input"] not in ("cc", "note"):
        out["input"] = "cc"
    if out["target_type"] not in ("param", "action"):
        out["target_type"] = "param"
    if out["mode"] not in ("absolute", "toggle", "momentary"):
        out["mode"] = "absolute"
    if out["trigger"] not in ("change", "press", "release"):
        out["trigger"] = "change"
    return out


def _channel_matches(mapping_channel: Any, event_channel: int) -> bool:
    if mapping_channel in ("all", "*", None):
        return True
    try:
        mc = int(mapping_channel)
    except (TypeError, ValueError):
        return True
    # UI channel is 1..16, MIDI event channel is 0..15
    return (mc - 1) == int(event_channel)


def _trigger_matches(trigger: str, event: dict[str, Any]) -> bool:
    if event.get("input") == "cc":
        return trigger == "change"
    pressed = bool(event.get("pressed", False))
    if trigger == "press":
        return pressed
    if trigger == "release":
        return not pressed
    return True


def _resolve_param_value(
    mapping: dict,
    event: dict[str, Any],
    mode: str,
    toggle_states: dict[str, bool],
) -> float | None:
    min_v = float(mapping.get("min", 0.0))
    max_v = float(mapping.get("max", 1.0))

    if mode == "absolute":
        norm = float(event.get("value", 0)) / 127.0
        return min_v + (max_v - min_v) * norm

    if mode == "momentary":
        return max_v if bool(event.get("pressed", False)) else min_v

    if mode == "toggle":
        if not bool(event.get("pressed", False)):
            return None
        key = _toggle_key(mapping)
        new_state = not toggle_states.get(key, False)
        toggle_states[key] = new_state
        return max_v if new_state else min_v

    return None


def _action_should_fire(mode: str, event: dict[str, Any]) -> bool:
    if mode == "absolute":
        if event.get("input") == "cc":
            return True
        return bool(event.get("pressed", False))
    if mode in ("momentary", "toggle"):
        return bool(event.get("pressed", False))
    return False


def _update_toggle_if_needed(
    mapping: dict,
    event: dict[str, Any],
    mode: str,
    toggle_states: dict[str, bool],
) -> None:
    if mode != "toggle":
        return
    if not bool(event.get("pressed", False)):
        return
    key = _toggle_key(mapping)
    toggle_states[key] = not toggle_states.get(key, False)


def _toggle_key(mapping: dict) -> str:
    return f"{mapping.get('input')}:{mapping.get('number')}:{mapping.get('target_type')}:{mapping.get('target')}"
