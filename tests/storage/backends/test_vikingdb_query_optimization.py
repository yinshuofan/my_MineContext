"""Unit tests for VikingDB query optimization (combined context-type queries)."""

from unittest.mock import AsyncMock

import pytest

from opencontext.storage.backends.vikingdb_backend import VikingDBBackend


@pytest.mark.unit
class TestSplitTypesByUserScope:
    """Tests for _split_types_by_user_scope helper."""

    def test_no_user_id_returns_single_group(self):
        types = ["document", "event", "knowledge", "agent_base_event"]
        result = VikingDBBackend._split_types_by_user_scope(types, user_id=None, device_id=None)
        assert len(result) == 1
        assert result[0] == (types, False)

    def test_with_user_id_splits_into_two_groups(self):
        types = ["document", "event", "agent_base_event", "agent_base_l1_summary"]
        result = VikingDBBackend._split_types_by_user_scope(types, user_id="u1", device_id=None)
        assert len(result) == 2
        regular_group, base_group = result
        assert regular_group == (["document", "event"], False)
        assert base_group == (["agent_base_event", "agent_base_l1_summary"], True)

    def test_with_device_id_only_splits(self):
        types = ["event", "agent_base_event"]
        result = VikingDBBackend._split_types_by_user_scope(types, user_id=None, device_id="d1")
        assert len(result) == 2

    def test_only_regular_types_returns_single_group(self):
        types = ["document", "event", "knowledge"]
        result = VikingDBBackend._split_types_by_user_scope(types, user_id="u1", device_id=None)
        assert len(result) == 1
        assert result[0] == (types, False)

    def test_only_base_types_returns_single_group(self):
        types = ["agent_base_event", "agent_base_l1_summary"]
        result = VikingDBBackend._split_types_by_user_scope(types, user_id="u1", device_id=None)
        assert len(result) == 1
        assert result[0] == (types, True)

    def test_empty_types_returns_empty(self):
        result = VikingDBBackend._split_types_by_user_scope([], user_id="u1", device_id=None)
        assert result == []


def _make_vikingdb_backend() -> VikingDBBackend:
    """Create a VikingDBBackend with mocked client for testing."""
    backend = VikingDBBackend.__new__(VikingDBBackend)
    backend._initialized = True
    backend._collection_name = "test_collection"
    backend._index_name = "test_index"
    backend._client = AsyncMock()
    return backend


def _make_scalar_response(items: list[dict]) -> dict:
    """Build a VikingDB scalar search success response."""
    return {
        "code": "Success",
        "result": {"data": items, "total_return_count": len(items)},
    }


def _make_vikingdb_item(id: str, context_type: str, create_time_ts: float) -> dict:
    """Build a single VikingDB item with minimal fields for _doc_to_context."""
    return {
        "id": id,
        "fields": {
            "context_type": context_type,
            "create_time": "2026-01-01T00:00:00+08:00",
            "create_time_ts": create_time_ts,
            "event_time_start": "2026-01-01T00:00:00+08:00",
            "event_time_end": "2026-01-01T01:00:00+08:00",
            "update_time": "2026-01-01T00:00:00+08:00",
            "data_type": "context",
            "document": f"Content for {context_type} {id}",
            "title": f"Test {context_type} {id}",
            "summary": f"Summary for {id}",
        },
    }


@pytest.mark.unit
class TestGetAllProcessedContextsCombined:
    """Tests for combined-query optimization in get_all_processed_contexts."""

    @pytest.mark.asyncio
    async def test_multi_type_no_user_id_issues_single_api_call(self):
        """When no user_id, all types combine into one VikingDB API call."""
        backend = _make_vikingdb_backend()
        backend._client.async_data_request.return_value = _make_scalar_response(
            [
                _make_vikingdb_item("1", "document", 1000.0),
                _make_vikingdb_item("2", "event", 999.0),
            ]
        )

        await backend.get_all_processed_contexts(
            context_types=["document", "event", "knowledge"],
            limit=10,
        )

        # Should issue exactly 1 API call (not 3)
        assert backend._client.async_data_request.call_count == 1
        call_data = backend._client.async_data_request.call_args[1]["data"]
        # Filter should contain context_types with all 3 types
        filter_conds = call_data["filter"]["conds"]
        ctx_type_cond = [c for c in filter_conds if c.get("field") == "context_type"][0]
        assert set(ctx_type_cond["conds"]) == {"document", "event", "knowledge"}

    @pytest.mark.asyncio
    async def test_multi_type_with_user_id_issues_two_api_calls(self):
        """When user_id is provided, regular and base types become separate calls."""
        backend = _make_vikingdb_backend()
        backend._client.async_data_request.return_value = _make_scalar_response(
            [
                _make_vikingdb_item("1", "event", 1000.0),
            ]
        )

        await backend.get_all_processed_contexts(
            context_types=["event", "agent_base_event"],
            limit=10,
            user_id="user1",
        )

        # Should issue exactly 2 API calls (regular + base)
        assert backend._client.async_data_request.call_count == 2

    @pytest.mark.asyncio
    async def test_results_grouped_by_context_type(self):
        """Combined query results are correctly grouped by context_type."""
        backend = _make_vikingdb_backend()
        backend._client.async_data_request.return_value = _make_scalar_response(
            [
                _make_vikingdb_item("1", "document", 1000.0),
                _make_vikingdb_item("2", "event", 999.0),
                _make_vikingdb_item("3", "document", 998.0),
            ]
        )

        result = await backend.get_all_processed_contexts(
            context_types=["document", "event"],
            limit=10,
        )

        assert "document" in result
        assert "event" in result
        assert len(result["document"]) == 2
        assert len(result["event"]) == 1

    @pytest.mark.asyncio
    async def test_single_type_uses_direct_query(self):
        """Single context_type still uses the direct (non-combined) path."""
        backend = _make_vikingdb_backend()
        backend._client.async_data_request.return_value = _make_scalar_response(
            [
                _make_vikingdb_item("1", "event", 1000.0),
            ]
        )

        await backend.get_all_processed_contexts(
            context_types=["event"],
            limit=10,
        )

        assert backend._client.async_data_request.call_count == 1
        call_data = backend._client.async_data_request.call_args[1]["data"]
        filter_conds = call_data["filter"]["conds"]
        ctx_type_cond = [c for c in filter_conds if c.get("field") == "context_type"][0]
        # Single type: should use single-value conds, not list
        assert ctx_type_cond["conds"] == ["event"]

    @pytest.mark.asyncio
    async def test_skip_slice_true_uses_combined_limit(self):
        """With skip_slice=True, limit+offset is the total fetch limit across all types."""
        backend = _make_vikingdb_backend()
        backend._client.async_data_request.return_value = _make_scalar_response([])

        await backend.get_all_processed_contexts(
            context_types=["document", "event"],
            limit=15,
            offset=30,
            skip_slice=True,
        )

        call_data = backend._client.async_data_request.call_args[1]["data"]
        # skip_slice=True: fetch_limit = limit + offset = 45
        assert call_data["limit"] == 45

    @pytest.mark.asyncio
    async def test_skip_slice_false_uses_per_type_limit(self):
        """With skip_slice=False, limit is per context type (over-fetches for combined query)."""
        backend = _make_vikingdb_backend()
        backend._client.async_data_request.return_value = _make_scalar_response([])

        await backend.get_all_processed_contexts(
            context_types=["document", "event", "knowledge"],
            limit=10,
            offset=0,
            skip_slice=False,
        )

        call_data = backend._client.async_data_request.call_args[1]["data"]
        # skip_slice=False: fetch limit * len(group_types) = 10 * 3 = 30
        assert call_data["limit"] == 30

    @pytest.mark.asyncio
    async def test_skip_slice_false_limits_per_type(self):
        """With skip_slice=False, each type's results are capped at limit."""
        backend = _make_vikingdb_backend()
        # Return 5 document items -- but limit is 2
        backend._client.async_data_request.return_value = _make_scalar_response(
            [_make_vikingdb_item(f"d{i}", "document", 1000.0 - i) for i in range(5)]
            + [
                _make_vikingdb_item("e1", "event", 500.0),
            ]
        )

        result = await backend.get_all_processed_contexts(
            context_types=["document", "event"],
            limit=2,
            offset=0,
            skip_slice=False,
        )

        # Each type capped at limit=2
        assert len(result.get("document", [])) == 2
        assert len(result.get("event", [])) == 1  # only 1 exists


@pytest.mark.unit
class TestGetFilteredContextCount:
    """Tests for combined count query."""

    @pytest.mark.asyncio
    async def test_no_user_id_issues_single_api_call(self):
        backend = _make_vikingdb_backend()
        backend._client.async_data_request.return_value = {
            "code": "Success",
            "result": {"filter_matched_count": 42, "total_return_count": 0},
        }

        count = await backend.get_filtered_context_count(
            context_types=["document", "event", "knowledge"],
        )

        assert count == 42
        assert backend._client.async_data_request.call_count == 1

    @pytest.mark.asyncio
    async def test_with_user_id_issues_two_api_calls_and_sums(self):
        backend = _make_vikingdb_backend()
        backend._client.async_data_request.side_effect = [
            {"code": "Success", "result": {"filter_matched_count": 30}},
            {"code": "Success", "result": {"filter_matched_count": 12}},
        ]

        count = await backend.get_filtered_context_count(
            context_types=["event", "knowledge", "agent_base_event"],
            user_id="user1",
        )

        assert count == 42
        assert backend._client.async_data_request.call_count == 2

    @pytest.mark.asyncio
    async def test_single_type_still_works(self):
        """Single type should still work (no grouping needed)."""
        backend = _make_vikingdb_backend()
        backend._client.async_data_request.return_value = {
            "code": "Success",
            "result": {"filter_matched_count": 7},
        }

        count = await backend.get_filtered_context_count(
            context_types=["event"],
        )

        assert count == 7
        assert backend._client.async_data_request.call_count == 1
