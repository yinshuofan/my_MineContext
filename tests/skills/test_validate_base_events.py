# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the offline validator shipped with the
narrative-to-base-events skill.

These tests load the validator script by absolute path (it lives under
.claude/skills/... not under the opencontext package) and exercise each
rule mirrored from opencontext/server/routes/agent_base_events.py.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALIDATOR_PATH = (
    _REPO_ROOT
    / ".claude"
    / "skills"
    / "narrative-to-base-events"
    / "scripts"
    / "validate_base_events.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_base_events", _VALIDATOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def validator():
    return _load_module()


def _write(tmp_path: Path, payload: dict) -> Path:
    f = tmp_path / "events.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    return f


def _minimal_l0() -> dict:
    return {
        "title": "t",
        "summary": "s",
        "event_time_start": "2024-01-01T00:00:00+00:00",
        "hierarchy_level": 0,
    }


def _wrap(events: list[dict]) -> dict:
    return {"events": events}


@pytest.mark.unit
class TestValidatorHappyPath:
    def test_full_l3_l2_l1_l0_tree_passes(self, validator, tmp_path):
        tree = {
            "title": "L3",
            "summary": "L3 summary",
            "event_time_start": "2024-01-01T00:00:00+00:00",
            "event_time_end": "2024-12-31T23:59:59+00:00",
            "hierarchy_level": 3,
            "children": [
                {
                    "title": "L2",
                    "summary": "L2 summary",
                    "event_time_start": "2024-01-01T00:00:00+00:00",
                    "event_time_end": "2024-03-31T23:59:59+00:00",
                    "hierarchy_level": 2,
                    "children": [
                        {
                            "title": "L1",
                            "summary": "L1 summary",
                            "event_time_start": "2024-01-01T00:00:00+00:00",
                            "event_time_end": "2024-01-31T23:59:59+00:00",
                            "hierarchy_level": 1,
                            "children": [_minimal_l0()],
                        }
                    ],
                }
            ],
        }
        f = _write(tmp_path, _wrap([tree]))
        total = validator.validate_file(f)
        assert total == 4


@pytest.mark.unit
class TestValidatorRuleViolations:
    def test_missing_title(self, validator, tmp_path):
        bad = {k: v for k, v in _minimal_l0().items() if k != "title"}
        f = _write(tmp_path, _wrap([bad]))
        with pytest.raises(validator.ValidationError, match="missing required field 'title'"):
            validator.validate_file(f)

    def test_invalid_hierarchy_level(self, validator, tmp_path):
        bad = {**_minimal_l0(), "hierarchy_level": 4}
        f = _write(tmp_path, _wrap([bad]))
        with pytest.raises(validator.ValidationError, match=r"hierarchy_level must be 0-3, got 4"):
            validator.validate_file(f)

    def test_l1_missing_event_time_end(self, validator, tmp_path):
        bad = {
            "title": "L1",
            "summary": "s",
            "event_time_start": "2024-01-01T00:00:00+00:00",
            "hierarchy_level": 1,
            "children": [_minimal_l0()],
        }
        f = _write(tmp_path, _wrap([bad]))
        with pytest.raises(
            validator.ValidationError, match="event_time_end is required for hierarchy_level > 0"
        ):
            validator.validate_file(f)

    def test_l1_with_empty_children(self, validator, tmp_path):
        bad = {
            "title": "L1",
            "summary": "s",
            "event_time_start": "2024-01-01T00:00:00+00:00",
            "event_time_end": "2024-01-01T23:59:59+00:00",
            "hierarchy_level": 1,
            "children": [],
        }
        f = _write(tmp_path, _wrap([bad]))
        with pytest.raises(
            validator.ValidationError, match="children is required for hierarchy_level > 0"
        ):
            validator.validate_file(f)

    def test_level_skipping(self, validator, tmp_path):
        # L3 parent with L0 child directly — child should be L2
        bad = {
            "title": "L3",
            "summary": "s",
            "event_time_start": "2024-01-01T00:00:00+00:00",
            "event_time_end": "2024-12-31T00:00:00+00:00",
            "hierarchy_level": 3,
            "children": [_minimal_l0()],
        }
        f = _write(tmp_path, _wrap([bad]))
        with pytest.raises(
            validator.ValidationError,
            match=r"hierarchy_level must be 2 \(parent is 3\), got 0",
        ):
            validator.validate_file(f)

    def test_parent_start_after_min_child_start(self, validator, tmp_path):
        bad = {
            "title": "L1",
            "summary": "s",
            "event_time_start": "2024-06-01T00:00:00+00:00",  # AFTER child starts
            "event_time_end": "2024-12-31T23:59:59+00:00",
            "hierarchy_level": 1,
            "children": [
                {
                    **_minimal_l0(),
                    "event_time_start": "2024-01-01T00:00:00+00:00",
                }
            ],
        }
        f = _write(tmp_path, _wrap([bad]))
        with pytest.raises(
            validator.ValidationError,
            match="event_time_start .* must be <= min child event_time_start",
        ):
            validator.validate_file(f)

    def test_l0_with_children(self, validator, tmp_path):
        bad = {**_minimal_l0(), "children": [_minimal_l0()]}
        f = _write(tmp_path, _wrap([bad]))
        with pytest.raises(
            validator.ValidationError, match="hierarchy_level 0 cannot have children"
        ):
            validator.validate_file(f)

    def test_total_exceeds_500(self, validator, tmp_path):
        # 501 L0 nodes
        events = [_minimal_l0() for _ in range(501)]
        f = _write(tmp_path, _wrap(events))
        with pytest.raises(
            validator.ValidationError, match="Total event count 501 exceeds maximum of 500"
        ):
            validator.validate_file(f)

    def test_invalid_iso8601(self, validator, tmp_path):
        bad = {**_minimal_l0(), "event_time_start": "not-a-timestamp"}
        f = _write(tmp_path, _wrap([bad]))
        with pytest.raises(
            validator.ValidationError,
            match=r"invalid ISO 8601 format for event_time_start: 'not-a-timestamp'",
        ):
            validator.validate_file(f)

    def test_empty_events_list(self, validator, tmp_path):
        f = _write(tmp_path, _wrap([]))
        with pytest.raises(validator.ValidationError, match="non-empty list"):
            validator.validate_file(f)
