"""Tests: EventNode.context_type is populated by node builders from ProcessedContext."""

from datetime import UTC, datetime

import pytest

from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    Vectorize,
)
from opencontext.models.enums import ContextType
from opencontext.server.routes.search import (
    _to_context_node,
    _to_search_hit_node,
)


def _make_ctx(ct: ContextType) -> ProcessedContext:
    ts = datetime(2026, 4, 15, tzinfo=UTC)
    return ProcessedContext(
        id=f"id-{ct.value}",
        properties=ContextProperties(
            create_time=ts,
            event_time_start=ts,
            event_time_end=ts,
            update_time=ts,
            hierarchy_level=0,
            refs={},
        ),
        extracted_data=ExtractedData(
            title="t",
            summary="s",
            context_type=ct,
            keywords=[],
            entities=[],
        ),
        vectorize=Vectorize(),
    )


@pytest.mark.unit
def test_search_hit_node_carries_event_context_type():
    ctx = _make_ctx(ContextType.EVENT)
    node = _to_search_hit_node(ctx, score=0.9)
    assert node.context_type == "event"


@pytest.mark.unit
def test_search_hit_node_carries_agent_base_event_context_type():
    ctx = _make_ctx(ContextType.AGENT_BASE_EVENT)
    node = _to_search_hit_node(ctx, score=0.8)
    assert node.context_type == "agent_base_event"


@pytest.mark.unit
def test_context_node_for_ancestor_carries_daily_summary_type():
    ctx = _make_ctx(ContextType.DAILY_SUMMARY)
    node = _to_context_node(ctx)
    assert node.context_type == "daily_summary"


@pytest.mark.unit
def test_context_node_for_agent_base_summary_type():
    ctx = _make_ctx(ContextType.AGENT_BASE_L1_SUMMARY)
    node = _to_context_node(ctx)
    assert node.context_type == "agent_base_l1_summary"
