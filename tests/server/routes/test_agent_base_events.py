# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent base-event routes."""

from __future__ import annotations

import contextlib
import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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


@pytest.mark.unit
def _ok_bulk_mock():
    """AsyncMock whose default behavior is per-type success for any ids_by_type."""
    return AsyncMock(side_effect=lambda ids_by_type: {ct: True for ct in ids_by_type})


class TestReplaceBaseEventsImpl:
    """Tests for _replace_base_events_impl (shared replace logic)."""

    @pytest.fixture
    def mock_storage(self):
        from unittest.mock import AsyncMock, MagicMock

        storage = MagicMock()
        storage.get_all_processed_contexts = AsyncMock(return_value={})
        storage.batch_upsert_processed_context = AsyncMock(return_value=["id-1"])
        storage.delete_batch_by_type = _ok_bulk_mock()
        return storage

    @pytest.fixture
    def loguru_capture(self):
        """Capture WARNING-and-above loguru messages into a list.

        Inline copy of the scheduler-tests fixture since there is no shared
        conftest at tests/server/routes/.
        """
        from loguru import logger

        messages: list[str] = []
        sink_id = logger.add(
            lambda msg: messages.append(str(msg)),
            level="WARNING",
            format="{message}",
        )
        try:
            yield messages
        finally:
            logger.remove(sink_id)

    async def test_fresh_agent_upserts_all_no_delete(self, mock_storage):
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        new_contexts = [_make_base_ctx("new-1", ContextType.AGENT_BASE_EVENT, 0)]
        mock_storage.batch_upsert_processed_context.return_value = ["new-1"]

        result = await _replace_base_events_impl(mock_storage, "agent-x", new_contexts)

        assert result["upserted"] == 1
        assert result["deleted"] == 0
        assert result["stragglers"] == 0
        mock_storage.batch_upsert_processed_context.assert_awaited_once_with(new_contexts)
        mock_storage.delete_batch_by_type.assert_not_awaited()

    async def test_diff_deletes_removed_ids(self, mock_storage):
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        # Existing: a, b, c. New: a, d. Should delete: b, c.
        existing = {
            ContextType.AGENT_BASE_EVENT.value: [
                _make_base_ctx("a", ContextType.AGENT_BASE_EVENT, 0),
                _make_base_ctx("b", ContextType.AGENT_BASE_EVENT, 0),
                _make_base_ctx("c", ContextType.AGENT_BASE_EVENT, 0),
            ],
        }
        mock_storage.get_all_processed_contexts.return_value = existing
        mock_storage.batch_upsert_processed_context.return_value = ["a", "d"]

        new_contexts = [
            _make_base_ctx("a", ContextType.AGENT_BASE_EVENT, 0),
            _make_base_ctx("d", ContextType.AGENT_BASE_EVENT, 0),
        ]

        result = await _replace_base_events_impl(mock_storage, "agent-x", new_contexts)

        assert result["upserted"] == 2
        assert result["deleted"] == 2
        ids_by_type_arg = mock_storage.delete_batch_by_type.call_args[0][0]
        all_deleted = {i for ids in ids_by_type_arg.values() for i in ids}
        assert all_deleted == {"b", "c"}

    async def test_empty_new_deletes_all_existing(self, mock_storage):
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        existing = {
            ContextType.AGENT_BASE_EVENT.value: [
                _make_base_ctx("a", ContextType.AGENT_BASE_EVENT, 0),
            ],
        }
        mock_storage.get_all_processed_contexts.return_value = existing
        mock_storage.batch_upsert_processed_context.return_value = []

        result = await _replace_base_events_impl(mock_storage, "agent-x", [])

        assert result["upserted"] == 0
        assert result["deleted"] == 1
        ids_by_type_arg = mock_storage.delete_batch_by_type.call_args[0][0]
        all_deleted = {i for ids in ids_by_type_arg.values() for i in ids}
        assert all_deleted == {"a"}

    async def test_upsert_failure_propagates(self, mock_storage):
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        mock_storage.batch_upsert_processed_context.return_value = None  # indicates failure

        with pytest.raises(RuntimeError, match="Failed to upsert"):
            await _replace_base_events_impl(
                mock_storage,
                "agent-x",
                [_make_base_ctx("x", ContextType.AGENT_BASE_EVENT, 0)],
            )
        mock_storage.delete_batch_by_type.assert_not_awaited()

    async def test_delete_failure_logged_returns_stragglers(self, mock_storage, loguru_capture):
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        existing = {
            ContextType.AGENT_BASE_EVENT.value: [
                _make_base_ctx("old", ContextType.AGENT_BASE_EVENT, 0),
            ],
        }
        mock_storage.get_all_processed_contexts.return_value = existing
        mock_storage.batch_upsert_processed_context.return_value = ["new"]
        mock_storage.delete_batch_by_type = AsyncMock(side_effect=Exception("viking down"))

        result = await _replace_base_events_impl(
            mock_storage,
            "agent-x",
            [_make_base_ctx("new", ContextType.AGENT_BASE_EVENT, 0)],
        )

        assert result["upserted"] == 1
        assert result["deleted"] == 0
        assert result["stragglers"] == 1
        assert any("delete_batch_by_type raised" in m for m in loguru_capture)

    async def test_delete_passes_grouped_ids_in_single_bulk_call(self, mock_storage):
        """Verify delete_batch_by_type is invoked once with ids grouped by context_type."""
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        # Existing tree: 1 L1 summary + 2 L0 events, all going away
        existing = {
            ContextType.AGENT_BASE_L1_SUMMARY.value: [
                _make_base_ctx("l1", ContextType.AGENT_BASE_L1_SUMMARY, 1),
            ],
            ContextType.AGENT_BASE_EVENT.value: [
                _make_base_ctx("l0-a", ContextType.AGENT_BASE_EVENT, 0),
                _make_base_ctx("l0-b", ContextType.AGENT_BASE_EVENT, 0),
            ],
        }
        mock_storage.get_all_processed_contexts.return_value = existing
        mock_storage.batch_upsert_processed_context.return_value = []

        result = await _replace_base_events_impl(mock_storage, "agent-x", [])

        assert result["deleted"] == 3
        assert result["stragglers"] == 0

        # Single bulk call, dict argument grouped by context_type
        assert mock_storage.delete_batch_by_type.await_count == 1
        ids_by_type_arg = mock_storage.delete_batch_by_type.call_args[0][0]
        grouped = {ct: set(ids) for ct, ids in ids_by_type_arg.items()}
        assert grouped == {
            ContextType.AGENT_BASE_L1_SUMMARY.value: {"l1"},
            ContextType.AGENT_BASE_EVENT.value: {"l0-a", "l0-b"},
        }

    async def test_delete_backend_returns_false_counts_as_straggler(
        self, mock_storage, loguru_capture
    ):
        """Verify that a False return from the backend is treated as failure."""
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        existing = {
            ContextType.AGENT_BASE_EVENT.value: [
                _make_base_ctx("old", ContextType.AGENT_BASE_EVENT, 0),
            ],
        }
        mock_storage.get_all_processed_contexts.return_value = existing
        mock_storage.batch_upsert_processed_context.return_value = ["new"]
        mock_storage.delete_batch_by_type = AsyncMock(
            side_effect=lambda ids_by_type: {ct: False for ct in ids_by_type}
        )

        result = await _replace_base_events_impl(
            mock_storage,
            "agent-x",
            [_make_base_ctx("new", ContextType.AGENT_BASE_EVENT, 0)],
        )

        assert result["deleted"] == 0
        assert result["stragglers"] == 1
        assert any("delete failed for type=" in msg for msg in loguru_capture)


def _build_app():
    from opencontext.server.routes.agent_base_events import router

    app = FastAPI()
    app.include_router(router)
    return app


@contextlib.contextmanager
def _patched_client(mock_storage, mock_cache=None):
    """TestClient with get_storage + get_cache + auth patched."""
    app = _build_app()

    if mock_cache is None:
        mock_cache = MagicMock()
        mock_cache.acquire_lock = AsyncMock(return_value="lock-token-123")
        mock_cache.release_lock = AsyncMock(return_value=True)

    with (
        patch(
            "opencontext.server.routes.agent_base_events.get_storage",
            return_value=mock_storage,
        ),
        patch(
            "opencontext.server.routes.agent_base_events.get_cache",
            new=AsyncMock(return_value=mock_cache),
        ),
        patch(
            "opencontext.server.routes.agent_base_events.auth_dependency",
            return_value="test-token",
        ),
    ):
        yield TestClient(app), mock_cache


@pytest.mark.unit
class TestPushBaseEventsReplace:
    """Tests for POST /api/agents/{agent_id}/base/events — replace semantics."""

    def _mock_storage_with_existing(self, existing_ids):
        storage = MagicMock()
        storage.get_agent = AsyncMock(return_value={"agent_id": "a1", "name": "A1"})
        existing_dict = {
            ContextType.AGENT_BASE_EVENT.value: [
                _make_base_ctx(eid, ContextType.AGENT_BASE_EVENT, 0) for eid in existing_ids
            ],
        }
        storage.get_all_processed_contexts = AsyncMock(return_value=existing_dict)
        storage.batch_upsert_processed_context = AsyncMock(return_value=["new-id"])
        storage.delete_batch_by_type = _ok_bulk_mock()
        return storage

    def test_404_when_agent_missing(self):
        storage = self._mock_storage_with_existing([])
        storage.get_agent = AsyncMock(return_value=None)

        with _patched_client(storage) as (client, _):
            resp = client.post(
                "/api/agents/a1/base/events",
                json={"events": [{"title": "t", "summary": "s"}]},
            )
            assert resp.status_code == 404

    def test_replace_deletes_ids_missing_from_payload(self):
        storage = self._mock_storage_with_existing(["old-1", "old-2"])

        with _patched_client(storage) as (client, _):
            resp = client.post(
                "/api/agents/a1/base/events",
                json={"events": [{"title": "t", "summary": "s"}]},  # one new event
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["data"]["upserted"] == 1
            assert body["data"]["deleted"] == 2
            ids_by_type_arg = storage.delete_batch_by_type.call_args[0][0]
            all_deleted_ids = {i for ids in ids_by_type_arg.values() for i in ids}
            assert all_deleted_ids == {"old-1", "old-2"}

    def test_acquires_and_releases_lock(self):
        storage = self._mock_storage_with_existing([])

        with _patched_client(storage) as (client, mock_cache):
            client.post(
                "/api/agents/a1/base/events",
                json={"events": [{"title": "t", "summary": "s"}]},
            )
            mock_cache.acquire_lock.assert_awaited_once()
            lock_args = mock_cache.acquire_lock.call_args
            assert lock_args[0][0] == "agent_base_edit:a1"
            mock_cache.release_lock.assert_awaited_once_with("agent_base_edit:a1", "lock-token-123")

    def test_returns_503_when_lock_unavailable(self):
        storage = self._mock_storage_with_existing([])

        mock_cache = MagicMock()
        mock_cache.acquire_lock = AsyncMock(return_value=None)  # acquisition failed
        mock_cache.release_lock = AsyncMock(return_value=True)

        with _patched_client(storage, mock_cache) as (client, _):
            resp = client.post(
                "/api/agents/a1/base/events",
                json={"events": [{"title": "t", "summary": "s"}]},
            )
            assert resp.status_code == 503


@pytest.mark.unit
class TestDeleteBaseEvent:
    """Tests for DELETE /api/agents/{agent_id}/base/events/{event_id}."""

    def _tree_storage(self):
        """Build mock storage with a small L0/L1 tree."""
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
        existing_dict = {
            ContextType.AGENT_BASE_L1_SUMMARY.value: [l1],
            ContextType.AGENT_BASE_EVENT.value: [l0_a, l0_b],
        }
        storage = MagicMock()
        storage.get_all_processed_contexts = AsyncMock(return_value=existing_dict)
        storage.batch_upsert_processed_context = AsyncMock(
            return_value=["l1-1", "l0-b"],
        )
        storage.delete_batch_by_type = _ok_bulk_mock()
        return storage

    def test_404_when_event_not_found(self):
        storage = self._tree_storage()

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events/does-not-exist")
            assert resp.status_code == 404

    def test_delete_leaf_prunes_single_id_and_scrubs_parent(self):
        storage = self._tree_storage()

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events/l0-a")
            assert resp.status_code == 200
            body = resp.json()
            assert body["data"]["deleted_ids"] == ["l0-a"]
            assert body["data"]["updated_parent_id"] == "l1-1"

            # Verify parent ctx passed to upsert has l0-a removed from refs
            upserted = storage.batch_upsert_processed_context.call_args[0][0]
            parent_ctx = next(c for c in upserted if c.id == "l1-1")
            assert parent_ctx.properties.refs[ContextType.AGENT_BASE_EVENT.value] == ["l0-b"]

            # Verify delete_batch_by_type called with l0-a
            ids_by_type_arg = storage.delete_batch_by_type.call_args[0][0]
            all_deleted = {i for ids in ids_by_type_arg.values() for i in ids}
            assert all_deleted == {"l0-a"}

    def test_delete_l1_prunes_entire_subtree(self):
        storage = self._tree_storage()

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events/l1-1")
            assert resp.status_code == 200
            body = resp.json()
            assert set(body["data"]["deleted_ids"]) == {"l1-1", "l0-a", "l0-b"}
            assert body["data"]["updated_parent_id"] is None  # L1 had no parent

            # When the entire tree is pruned, upsert is skipped (no surviving contexts).
            # The guard in _replace_base_events_impl avoids calling the backend with [].
            assert storage.batch_upsert_processed_context.call_args is None

            ids_by_type_arg = storage.delete_batch_by_type.call_args[0][0]
            all_deleted = {i for ids in ids_by_type_arg.values() for i in ids}
            assert all_deleted == {"l1-1", "l0-a", "l0-b"}

    def test_delete_leaf_preserves_empty_parent_summary(self):
        """Regression guard: deleting the only child of an L1 leaves L1 intact."""
        # Build tree with single L0 child under L1
        l1 = _make_base_ctx(
            "l1-only",
            ContextType.AGENT_BASE_L1_SUMMARY,
            1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["l0-only"]},
        )
        l0 = _make_base_ctx(
            "l0-only",
            ContextType.AGENT_BASE_EVENT,
            0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-only"]},
        )
        existing = {
            ContextType.AGENT_BASE_L1_SUMMARY.value: [l1],
            ContextType.AGENT_BASE_EVENT.value: [l0],
        }
        storage = MagicMock()
        storage.get_all_processed_contexts = AsyncMock(return_value=existing)
        storage.batch_upsert_processed_context = AsyncMock(return_value=["l1-only"])
        storage.delete_batch_by_type = _ok_bulk_mock()

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events/l0-only")
            assert resp.status_code == 200

            # L1 still in the upsert list (preserved) with empty downward ref list
            upserted = storage.batch_upsert_processed_context.call_args[0][0]
            assert any(c.id == "l1-only" for c in upserted)
            l1_after = next(c for c in upserted if c.id == "l1-only")
            assert l1_after.properties.refs[ContextType.AGENT_BASE_EVENT.value] == []

    def test_delete_acquires_and_releases_lock(self):
        storage = self._tree_storage()

        with _patched_client(storage) as (client, mock_cache):
            client.delete("/api/agents/a1/base/events/l0-a")
            mock_cache.acquire_lock.assert_awaited_once()
            lock_args = mock_cache.acquire_lock.call_args
            assert lock_args[0][0] == "agent_base_edit:a1"
            mock_cache.release_lock.assert_awaited_once()

    def test_delete_returns_503_when_lock_unavailable(self):
        storage = self._tree_storage()
        mock_cache = MagicMock()
        mock_cache.acquire_lock = AsyncMock(return_value=None)
        mock_cache.release_lock = AsyncMock(return_value=True)

        with _patched_client(storage, mock_cache) as (client, _):
            resp = client.delete("/api/agents/a1/base/events/l0-a")
            assert resp.status_code == 503


@pytest.mark.unit
class TestDeleteAllBaseEvents:
    """Tests for DELETE /api/agents/{agent_id}/base/events — clear entire tree."""

    def _tree_storage(self, events_present: bool = True):
        """Build mock storage with an agent + (optional) L1/L0 tree."""
        if events_present:
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
            existing_dict = {
                ContextType.AGENT_BASE_L1_SUMMARY.value: [l1],
                ContextType.AGENT_BASE_EVENT.value: [l0_a, l0_b],
            }
        else:
            existing_dict = {}
        storage = MagicMock()
        storage.get_agent = AsyncMock(return_value={"agent_id": "a1", "name": "A1"})
        storage.get_all_processed_contexts = AsyncMock(return_value=existing_dict)
        storage.batch_upsert_processed_context = AsyncMock(return_value=[])
        storage.delete_batch_by_type = _ok_bulk_mock()
        return storage

    def test_deletes_all_events(self):
        storage = self._tree_storage()

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events")
            assert resp.status_code == 200
            body = resp.json()
            assert set(body["data"]["deleted_ids"]) == {"l1-1", "l0-a", "l0-b"}
            assert body["data"]["deleted"] == 3
            assert body["data"]["stragglers"] == 0

            # Upsert is NOT called (empty list path in _replace_base_events_impl)
            assert storage.batch_upsert_processed_context.call_args is None

            # Single bulk call, all existing ids grouped by context_type
            assert storage.delete_batch_by_type.await_count == 1
            ids_by_type_arg = storage.delete_batch_by_type.call_args[0][0]
            all_deleted = {i for ids in ids_by_type_arg.values() for i in ids}
            assert all_deleted == {"l1-1", "l0-a", "l0-b"}

    def test_empty_tree_is_noop(self):
        storage = self._tree_storage(events_present=False)

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events")
            assert resp.status_code == 200
            body = resp.json()
            assert body["data"]["deleted_ids"] == []
            assert body["data"]["deleted"] == 0
            assert body["data"]["stragglers"] == 0

            storage.batch_upsert_processed_context.assert_not_called()
            storage.delete_batch_by_type.assert_not_called()

    def test_404_when_agent_missing(self):
        storage = self._tree_storage()
        storage.get_agent = AsyncMock(return_value=None)

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events")
            assert resp.status_code == 404

    def test_acquires_and_releases_lock(self):
        storage = self._tree_storage()

        with _patched_client(storage) as (client, mock_cache):
            client.delete("/api/agents/a1/base/events")
            mock_cache.acquire_lock.assert_awaited_once()
            lock_args = mock_cache.acquire_lock.call_args
            assert lock_args[0][0] == "agent_base_edit:a1"
            mock_cache.release_lock.assert_awaited_once_with("agent_base_edit:a1", "lock-token-123")

    def test_returns_503_when_lock_unavailable(self):
        storage = self._tree_storage()
        mock_cache = MagicMock()
        mock_cache.acquire_lock = AsyncMock(return_value=None)
        mock_cache.release_lock = AsyncMock(return_value=True)

        with _patched_client(storage, mock_cache) as (client, _):
            resp = client.delete("/api/agents/a1/base/events")
            assert resp.status_code == 503
