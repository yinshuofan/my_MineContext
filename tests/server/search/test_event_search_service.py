"""Unit tests for EventSearchService."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from opencontext.server.search.event_search_service import EventSearchService


def _ctx(ctx_id: str, level: int, refs: dict | None = None):
    """Minimal ProcessedContext-like object."""
    return SimpleNamespace(
        id=ctx_id,
        properties=SimpleNamespace(hierarchy_level=level, refs=refs or {}),
    )


MOCK_STORAGE = "opencontext.server.search.event_search_service.get_storage"


@pytest.mark.unit
async def test_collect_descendants_single_level():
    """L1 hit with refs to two L0 children -- both returned as descendants."""
    child_a = _ctx("c-a", level=0)
    child_b = _ctx("c-b", level=0)
    parent = _ctx("p-1", level=1, refs={"event": ["c-a", "c-b"]})

    storage = AsyncMock()
    storage.get_contexts_by_ids = AsyncMock(return_value=[child_a, child_b])

    with patch(MOCK_STORAGE, return_value=storage):
        service = EventSearchService()
        result = await service.collect_descendants([(parent, 0.9)], min_level=0)

    assert set(result.keys()) == {"c-a", "c-b"}


@pytest.mark.unit
async def test_collect_descendants_multi_level():
    """L2 hit -> L1 children -> L0 grandchildren. All non-hit descendants returned."""
    l1_child = _ctx("l1-1", level=1, refs={"event": ["l0-1", "l0-2"]})
    l0_a = _ctx("l0-1", level=0)
    l0_b = _ctx("l0-2", level=0)
    l2_parent = _ctx("l2-1", level=2, refs={"daily_summary": ["l1-1"]})

    storage = AsyncMock()
    storage.get_contexts_by_ids = AsyncMock(side_effect=[[l1_child], [l0_a, l0_b]])

    with patch(MOCK_STORAGE, return_value=storage):
        service = EventSearchService()
        result = await service.collect_descendants([(l2_parent, 0.8)], min_level=0)

    assert set(result.keys()) == {"l1-1", "l0-1", "l0-2"}


@pytest.mark.unit
async def test_collect_descendants_skips_upward_refs():
    """Refs pointing to higher levels are ignored (not followed as children)."""
    parent_ref = _ctx("l2-1", level=2)
    hit = _ctx(
        "l1-1",
        level=1,
        refs={
            "event": ["l0-1"],
            "weekly_summary": ["l2-1"],
        },
    )
    l0_child = _ctx("l0-1", level=0)

    storage = AsyncMock()
    storage.get_contexts_by_ids = AsyncMock(return_value=[l0_child, parent_ref])

    with patch(MOCK_STORAGE, return_value=storage):
        service = EventSearchService()
        result = await service.collect_descendants([(hit, 0.9)], min_level=0)

    assert set(result.keys()) == {"l0-1"}
    assert "l2-1" not in result


@pytest.mark.unit
async def test_collect_descendants_respects_min_level():
    """min_level=1 stops at L1, does not drill to L0."""
    l1_child = _ctx("l1-1", level=1, refs={"event": ["l0-1"]})
    l2_parent = _ctx("l2-1", level=2, refs={"daily_summary": ["l1-1"]})

    storage = AsyncMock()
    storage.get_contexts_by_ids = AsyncMock(return_value=[l1_child])

    with patch(MOCK_STORAGE, return_value=storage):
        service = EventSearchService()
        result = await service.collect_descendants([(l2_parent, 0.8)], min_level=1)

    assert set(result.keys()) == {"l1-1"}
    assert storage.get_contexts_by_ids.await_count == 1


@pytest.mark.unit
async def test_collect_descendants_empty_refs():
    """Hit with no refs -> empty descendants dict."""
    hit = _ctx("l0-1", level=0, refs={})

    storage = AsyncMock()
    with patch(MOCK_STORAGE, return_value=storage):
        service = EventSearchService()
        result = await service.collect_descendants([(hit, 0.9)], min_level=0)

    assert result == {}
    storage.get_contexts_by_ids.assert_not_awaited()


@pytest.mark.unit
async def test_collect_descendants_dedup():
    """Two hits sharing a child -> child appears once."""
    child = _ctx("shared", level=0)
    hit_a = _ctx("h-a", level=1, refs={"event": ["shared"]})
    hit_b = _ctx("h-b", level=1, refs={"event": ["shared"]})

    storage = AsyncMock()
    storage.get_contexts_by_ids = AsyncMock(return_value=[child])

    with patch(MOCK_STORAGE, return_value=storage):
        service = EventSearchService()
        result = await service.collect_descendants([(hit_a, 0.9), (hit_b, 0.8)], min_level=0)

    assert result == {"shared": child}
