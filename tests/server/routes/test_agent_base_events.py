# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent base-event routes."""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import pytest

from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    Vectorize,
)
from opencontext.models.enums import ContentFormat, ContextType

TZ = ZoneInfo("Asia/Shanghai")


def _make_base_ctx(
    ctx_id: str,
    context_type: ContextType,
    hierarchy_level: int,
    refs: dict[str, list[str]] | None = None,
    agent_id: str = "agent-x",
) -> ProcessedContext:
    """Build a minimal AGENT_BASE_* ProcessedContext."""
    ct = datetime.datetime(2026, 4, 10, 12, 0, 0, tzinfo=TZ)
    return ProcessedContext(
        id=ctx_id,
        properties=ContextProperties(
            raw_properties=[],
            create_time=ct,
            update_time=ct,
            event_time_start=ct,
            event_time_end=ct,
            is_processed=True,
            user_id="__base__",
            agent_id=agent_id,
            hierarchy_level=hierarchy_level,
            refs=refs or {},
        ),
        extracted_data=ExtractedData(
            title=f"t-{ctx_id}",
            summary=f"s-{ctx_id}",
            context_type=context_type,
            confidence=10,
            importance=5,
        ),
        vectorize=Vectorize(vector=[0.1], content_format=ContentFormat.TEXT),
    )


@pytest.mark.unit
class TestCollectSubtreeIds:
    """Tests for _collect_subtree_ids."""

    def test_single_leaf_returns_only_root(self):
        from opencontext.server.routes.agent_base_events import _collect_subtree_ids

        leaf = _make_base_ctx("leaf-1", ContextType.AGENT_BASE_EVENT, 0)
        ctx_by_id = {"leaf-1": leaf}

        result = _collect_subtree_ids(ctx_by_id, "leaf-1")

        assert result == {"leaf-1"}

    def test_collects_all_descendants_of_l1(self):
        from opencontext.server.routes.agent_base_events import _collect_subtree_ids

        # L1 summary with two L0 children
        l1 = _make_base_ctx(
            "l1-1",
            ContextType.AGENT_BASE_L1_SUMMARY,
            1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["l0-a", "l0-b"]},
        )
        l0_a = _make_base_ctx(
            "l0-a",
            ContextType.AGENT_BASE_EVENT,
            0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-1"]},
        )
        l0_b = _make_base_ctx(
            "l0-b",
            ContextType.AGENT_BASE_EVENT,
            0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-1"]},
        )
        ctx_by_id = {"l1-1": l1, "l0-a": l0_a, "l0-b": l0_b}

        result = _collect_subtree_ids(ctx_by_id, "l1-1")

        assert result == {"l1-1", "l0-a", "l0-b"}

    def test_ignores_upward_refs(self):
        from opencontext.server.routes.agent_base_events import _collect_subtree_ids

        # L0 has only an upward ref — BFS should not follow upward
        l0 = _make_base_ctx(
            "l0-x",
            ContextType.AGENT_BASE_EVENT,
            0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-parent"]},
        )
        l1 = _make_base_ctx("l1-parent", ContextType.AGENT_BASE_L1_SUMMARY, 1)
        ctx_by_id = {"l0-x": l0, "l1-parent": l1}

        result = _collect_subtree_ids(ctx_by_id, "l0-x")

        assert result == {"l0-x"}

    def test_cycle_terminates(self):
        from opencontext.server.routes.agent_base_events import _collect_subtree_ids

        # Synthetic cycle — guard against pathological data
        a = _make_base_ctx(
            "a",
            ContextType.AGENT_BASE_L1_SUMMARY,
            1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["b"]},
        )
        b = _make_base_ctx(
            "b",
            ContextType.AGENT_BASE_EVENT,
            0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["a"]},
        )
        # Force a downward-looking cycle by also adding b's refs to a:
        b.properties.refs[ContextType.AGENT_BASE_L1_SUMMARY.value] = ["a"]
        # Pretend a also has a downward ref to b (already set); b now has a
        # matching-level ref back to a — the function should not loop.
        ctx_by_id = {"a": a, "b": b}

        result = _collect_subtree_ids(ctx_by_id, "a")

        assert "a" in result and "b" in result
        assert len(result) == 2  # terminated, no duplicates


@pytest.mark.unit
class TestFindParentId:
    """Tests for _find_parent_id."""

    def test_returns_parent_for_l0_with_upward_ref(self):
        from opencontext.server.routes.agent_base_events import _find_parent_id

        l0 = _make_base_ctx(
            "l0-1",
            ContextType.AGENT_BASE_EVENT,
            0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-parent"]},
        )
        l1 = _make_base_ctx("l1-parent", ContextType.AGENT_BASE_L1_SUMMARY, 1)
        ctx_by_id = {"l0-1": l0, "l1-parent": l1}

        assert _find_parent_id(ctx_by_id, "l0-1") == "l1-parent"

    def test_returns_none_for_root(self):
        from opencontext.server.routes.agent_base_events import _find_parent_id

        # L3 at the top of the tree has no parent
        l3 = _make_base_ctx("l3-1", ContextType.AGENT_BASE_L3_SUMMARY, 3)
        ctx_by_id = {"l3-1": l3}

        assert _find_parent_id(ctx_by_id, "l3-1") is None

    def test_returns_none_for_unknown_id(self):
        from opencontext.server.routes.agent_base_events import _find_parent_id

        assert _find_parent_id({}, "does-not-exist") is None

    def test_ignores_downward_refs(self):
        from opencontext.server.routes.agent_base_events import _find_parent_id

        # L1 that has only downward refs (to its L0 children) but no upward ref
        l1 = _make_base_ctx(
            "l1-top",
            ContextType.AGENT_BASE_L1_SUMMARY,
            1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["l0-a"]},
        )
        l0 = _make_base_ctx("l0-a", ContextType.AGENT_BASE_EVENT, 0)
        ctx_by_id = {"l1-top": l1, "l0-a": l0}

        assert _find_parent_id(ctx_by_id, "l1-top") is None


@pytest.mark.unit
class TestScrubParentRefs:
    """Tests for _scrub_parent_refs."""

    def test_removes_child_id_from_parent_refs(self):
        from opencontext.server.routes.agent_base_events import _scrub_parent_refs

        parent = _make_base_ctx(
            "p",
            ContextType.AGENT_BASE_L1_SUMMARY,
            1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["c1", "c2", "c3"]},
        )

        _scrub_parent_refs(parent, "c2", ContextType.AGENT_BASE_EVENT)

        assert parent.properties.refs[ContextType.AGENT_BASE_EVENT.value] == ["c1", "c3"]

    def test_missing_id_is_no_op(self):
        from opencontext.server.routes.agent_base_events import _scrub_parent_refs

        parent = _make_base_ctx(
            "p",
            ContextType.AGENT_BASE_L1_SUMMARY,
            1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["c1"]},
        )

        _scrub_parent_refs(parent, "c999", ContextType.AGENT_BASE_EVENT)

        assert parent.properties.refs[ContextType.AGENT_BASE_EVENT.value] == ["c1"]

    def test_missing_key_is_no_op(self):
        from opencontext.server.routes.agent_base_events import _scrub_parent_refs

        parent = _make_base_ctx("p", ContextType.AGENT_BASE_L1_SUMMARY, 1, refs={})

        _scrub_parent_refs(parent, "c1", ContextType.AGENT_BASE_EVENT)

        assert parent.properties.refs == {}
