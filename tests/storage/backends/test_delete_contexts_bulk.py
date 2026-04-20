"""Unit tests for `delete_contexts_bulk` on VikingDB + Qdrant backends.

VikingDB collapses all ids across types into a single HTTP call.
Qdrant fans out per-collection via asyncio.gather (parallel).
"""

from unittest.mock import AsyncMock

import pytest

from opencontext.storage.backends.qdrant_backend import QdrantBackend
from opencontext.storage.backends.vikingdb_backend import VikingDBBackend


def _make_vikingdb_backend() -> VikingDBBackend:
    backend = VikingDBBackend.__new__(VikingDBBackend)
    backend._initialized = True
    backend._collection_name = "test_collection"
    backend._index_name = "test_index"
    backend._client = AsyncMock()
    return backend


def _make_qdrant_backend() -> QdrantBackend:
    backend = QdrantBackend.__new__(QdrantBackend)
    backend._initialized = True
    backend._collections = {
        "agent_base_event": "col_l0",
        "agent_base_l1_summary": "col_l1",
        "agent_base_l2_summary": "col_l2",
    }
    backend._client = AsyncMock()
    return backend


@pytest.mark.unit
class TestVikingDBDeleteContextsBulk:
    async def test_empty_input_is_noop(self):
        backend = _make_vikingdb_backend()
        result = await backend.delete_contexts_bulk({})
        assert result == {}
        backend._client.async_data_request.assert_not_awaited()

    async def test_flattens_all_ids_into_single_http_call(self):
        backend = _make_vikingdb_backend()
        backend._client.async_data_request = AsyncMock(return_value={"code": "Success"})

        ids_by_type = {
            "agent_base_event": ["l0-a", "l0-b"],
            "agent_base_l1_summary": ["l1-1"],
        }
        result = await backend.delete_contexts_bulk(ids_by_type)

        assert result == {
            "agent_base_event": True,
            "agent_base_l1_summary": True,
        }
        # Exactly one HTTP call
        assert backend._client.async_data_request.await_count == 1
        call = backend._client.async_data_request.call_args
        assert call.kwargs["path"] == "/api/vikingdb/data/delete"
        assert call.kwargs["data"]["collection_name"] == "test_collection"
        assert set(call.kwargs["data"]["ids"]) == {"l0-a", "l0-b", "l1-1"}

    async def test_single_api_failure_marks_all_types_failed(self):
        backend = _make_vikingdb_backend()
        backend._client.async_data_request = AsyncMock(
            return_value={"code": "InternalError", "message": "boom"}
        )

        ids_by_type = {"agent_base_event": ["x"], "agent_base_l1_summary": ["y"]}
        result = await backend.delete_contexts_bulk(ids_by_type)

        assert result == {"agent_base_event": False, "agent_base_l1_summary": False}

    async def test_exception_marks_all_types_failed(self):
        backend = _make_vikingdb_backend()
        backend._client.async_data_request = AsyncMock(side_effect=Exception("network"))

        ids_by_type = {"agent_base_event": ["x"]}
        result = await backend.delete_contexts_bulk(ids_by_type)

        assert result == {"agent_base_event": False}

    async def test_uninitialized_returns_false_for_all_types(self):
        backend = _make_vikingdb_backend()
        backend._initialized = False

        ids_by_type = {"agent_base_event": ["x"], "agent_base_l1_summary": ["y"]}
        result = await backend.delete_contexts_bulk(ids_by_type)

        assert result == {"agent_base_event": False, "agent_base_l1_summary": False}
        backend._client.async_data_request.assert_not_awaited()


@pytest.mark.unit
class TestQdrantDeleteContextsBulk:
    async def test_empty_input_is_noop(self):
        backend = _make_qdrant_backend()
        result = await backend.delete_contexts_bulk({})
        assert result == {}
        backend._client.delete.assert_not_awaited()

    async def test_fans_out_one_delete_per_collection(self):
        backend = _make_qdrant_backend()
        backend._client.delete = AsyncMock(return_value=None)

        ids_by_type = {
            "agent_base_event": ["l0-a", "l0-b"],
            "agent_base_l1_summary": ["l1-1"],
        }
        result = await backend.delete_contexts_bulk(ids_by_type)

        assert result == {
            "agent_base_event": True,
            "agent_base_l1_summary": True,
        }
        # One Qdrant delete call per context_type (collection)
        assert backend._client.delete.await_count == 2
        called_collections = {
            call.kwargs["collection_name"] for call in backend._client.delete.call_args_list
        }
        assert called_collections == {"col_l0", "col_l1"}

    async def test_partial_failure_reports_per_type_success(self):
        """One collection raises → only that type marked False."""
        backend = _make_qdrant_backend()

        async def delete_side_effect(collection_name, points_selector):
            if collection_name == "col_l1":
                raise RuntimeError("qdrant down")
            return None

        backend._client.delete = AsyncMock(side_effect=delete_side_effect)

        ids_by_type = {
            "agent_base_event": ["l0-a"],
            "agent_base_l1_summary": ["l1-1"],
        }
        result = await backend.delete_contexts_bulk(ids_by_type)

        assert result == {
            "agent_base_event": True,
            "agent_base_l1_summary": False,
        }

    async def test_uninitialized_returns_false_for_all_types(self):
        backend = _make_qdrant_backend()
        backend._initialized = False

        ids_by_type = {"agent_base_event": ["x"]}
        result = await backend.delete_contexts_bulk(ids_by_type)

        assert result == {"agent_base_event": False}

    async def test_unknown_context_type_marked_false(self):
        """Qdrant's delete_contexts returns False for unknown types; bulk preserves that."""
        backend = _make_qdrant_backend()
        backend._client.delete = AsyncMock(return_value=None)

        ids_by_type = {
            "agent_base_event": ["l0-a"],
            "unknown_type": ["nope"],
        }
        result = await backend.delete_contexts_bulk(ids_by_type)

        assert result == {
            "agent_base_event": True,
            "unknown_type": False,
        }
