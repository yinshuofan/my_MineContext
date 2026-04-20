#!/usr/bin/env python3
"""Offline validator for BaseEventsRequest JSON.

Mirrors the server-side validator (_validate_base_event_tree in
opencontext/server/routes/agent_base_events.py) so you can verify a
generated JSON file locally before POSTing it to the agents base-events
endpoint. Error messages match the server's 400 response verbatim.

Usage:
    python3 validate_base_events.py path/to/file.json [more.json ...]

Exit codes:
    0 — all files valid
    1 — at least one file had errors
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

_MAX_TOTAL_EVENTS = 500
_VALID_LEVELS = {0, 1, 2, 3}


class ValidationError(Exception):
    pass


def _parse_time(value, node_path: str, field_name: str) -> datetime.datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(
            f"{node_path}: {field_name} must be a string, got {type(value).__name__}"
        )
    try:
        return datetime.datetime.fromisoformat(value)
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"{node_path}: invalid ISO 8601 format for {field_name}: '{value}'"
        ) from e


def _require_str(obj: dict, name: str, node_path: str) -> None:
    if name not in obj:
        raise ValidationError(f"{node_path}: missing required field '{name}'")
    if not isinstance(obj[name], str):
        raise ValidationError(f"{node_path}: '{name}' must be str, got {type(obj[name]).__name__}")


def _validate_tree(events, path: str = "events") -> int:
    if not isinstance(events, list):
        raise ValidationError(f"{path}: must be a list")

    total = 0
    for i, event in enumerate(events):
        node_path = f"{path}[{i}]"
        if not isinstance(event, dict):
            raise ValidationError(f"{node_path}: must be an object")

        _require_str(event, "title", node_path)
        _require_str(event, "summary", node_path)

        level = event.get("hierarchy_level", 0)
        if not isinstance(level, int) or isinstance(level, bool):
            raise ValidationError(
                f"{node_path}: hierarchy_level must be int, got {type(level).__name__}"
            )
        if level not in _VALID_LEVELS:
            raise ValidationError(f"{node_path}: hierarchy_level must be 0-3, got {level}")

        ets = _parse_time(event.get("event_time_start"), node_path, "event_time_start")
        children = event.get("children")

        if level > 0:
            if not event.get("event_time_end"):
                raise ValidationError(
                    f"{node_path}: event_time_end is required for hierarchy_level > 0"
                )
            if not children:
                raise ValidationError(f"{node_path}: children is required for hierarchy_level > 0")
            if not isinstance(children, list):
                raise ValidationError(f"{node_path}: children must be a list")

            ete = _parse_time(event["event_time_end"], node_path, "event_time_end")
            if ets and ete and ets > ete:
                raise ValidationError(f"{node_path}: event_time_start must be <= event_time_end")

            # child hierarchy_level must be parent - 1
            for j, child in enumerate(children):
                if not isinstance(child, dict):
                    raise ValidationError(f"{node_path}.children[{j}]: must be an object")
                child_level = child.get("hierarchy_level", 0)
                if child_level != level - 1:
                    raise ValidationError(
                        f"{node_path}.children[{j}]: hierarchy_level must be "
                        f"{level - 1} (parent is {level}), got {child_level}"
                    )

            # Parent time range must cover all direct children
            child_starts: list[datetime.datetime] = []
            child_ends: list[datetime.datetime] = []
            for j, child in enumerate(children):
                child_path = f"{node_path}.children[{j}]"
                cs = _parse_time(child.get("event_time_start"), child_path, "event_time_start")
                if cs:
                    child_starts.append(cs)
                child_ete_raw = child.get("event_time_end")
                if child_ete_raw:
                    ce = _parse_time(child_ete_raw, child_path, "event_time_end")
                    if ce:
                        child_ends.append(ce)
                elif cs:
                    # L0: event_time_end defaults to event_time_start
                    child_ends.append(cs)

            if ets and child_starts and ets > min(child_starts):
                raise ValidationError(
                    f"{node_path}: event_time_start ({ets.isoformat()}) must be <= "
                    f"min child event_time_start ({min(child_starts).isoformat()})"
                )
            if ete and child_ends and ete < max(child_ends):
                raise ValidationError(
                    f"{node_path}: event_time_end ({ete.isoformat()}) must be >= "
                    f"max child event_time_end ({max(child_ends).isoformat()})"
                )

            total += _validate_tree(children, f"{node_path}.children")
        else:
            if children:
                raise ValidationError(f"{node_path}: hierarchy_level 0 cannot have children")

        total += 1

    return total


def validate_file(path: str | Path) -> int:
    """Validate a single BaseEventsRequest JSON file. Returns total node count."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "events" not in data:
        raise ValidationError('Top level must be {"events": [...]}')
    events = data["events"]
    if not isinstance(events, list) or not events:
        raise ValidationError("'events' must be a non-empty list")

    total = _validate_tree(events)
    if total > _MAX_TOTAL_EVENTS:
        raise ValidationError(f"Total event count {total} exceeds maximum of {_MAX_TOTAL_EVENTS}")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline validator for BaseEventsRequest JSON.")
    parser.add_argument("files", nargs="+", help="JSON file(s) to validate")
    args = parser.parse_args()

    had_error = False
    for file_path in args.files:
        try:
            total = validate_file(file_path)
            print(f"OK  {file_path} — {total} events")
        except (json.JSONDecodeError, ValidationError, OSError) as e:
            print(f"FAIL {file_path}: {e}", file=sys.stderr)
            had_error = True
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
