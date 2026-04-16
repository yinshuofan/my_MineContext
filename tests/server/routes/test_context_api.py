# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GET /api/contexts endpoint."""

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
from opencontext.server.routes.context import router

TZ = ZoneInfo("Asia/Shanghai")


def _make_context(
    context_id: str = "ctx-1",
    context_type: ContextType = ContextType.EVENT,
    create_time: datetime.datetime | None = None,
    user_id: str | None = None,
    device_id: str | None = None,
    agent_id: str | None = None,
    hierarchy_level: int = 0,
) -> ProcessedContext:
    """Build a minimal ProcessedContext for testing."""
    ct = create_time or datetime.datetime(2026, 4, 10, 12, 0, 0, tzinfo=TZ)
    return ProcessedContext(
        id=context_id,
        properties=ContextProperties(
            create_time=ct,
            event_time_start=ct,
            event_time_end=ct,
            update_time=ct,
            is_processed=True,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            hierarchy_level=hierarchy_level,
        ),
        extracted_data=ExtractedData(
            title=f"Title {context_id}",
            summary=f"Summary for {context_id}",
            context_type=context_type,
            confidence=5,
            importance=5,
        ),
        vectorize=Vectorize(vector=[0.1, 0.2, 0.3], content_format=ContentFormat.TEXT),
    )


def _build_app() -> FastAPI:
    """Build a minimal FastAPI app with the context router."""
    app = FastAPI()
    app.include_router(router)
    return app


@contextlib.contextmanager
def _patched_client(
    mock_storage: MagicMock | None,
    mock_opencontext: MagicMock | None = None,
):
    """Context manager that yields a TestClient with patched storage and auth."""
    app = _build_app()

    if mock_opencontext is None:
        mock_opencontext = MagicMock()

    app.state.context_lab_instance = mock_opencontext

    with (
        patch("opencontext.server.routes.context.get_storage", return_value=mock_storage),
        patch(
            "opencontext.server.routes.context.auth_dependency",
            return_value="test-token",
        ),
    ):
        yield TestClient(app)


def _make_mock_storage(
    contexts: list[ProcessedContext] | None = None,
    total_count: int | None = None,
    available_types: list[str] | None = None,
) -> MagicMock:
    """Create a mock UnifiedStorage with preset return values."""
    storage = MagicMock()

    if available_types is None:
        available_types = [ct.value for ct in ContextType]
    storage.get_available_context_types.return_value = available_types

    ctx_list = contexts or []
    if total_count is None:
        total_count = len(ctx_list)
    storage.get_filtered_context_count = AsyncMock(return_value=total_count)

    # get_all_processed_contexts returns dict[str, list[ProcessedContext]]
    # Group by context_type
    contexts_dict: dict[str, list[ProcessedContext]] = {}
    for c in ctx_list:
        ct_val = c.extracted_data.context_type.value
        contexts_dict.setdefault(ct_val, []).append(c)
    storage.get_all_processed_contexts = AsyncMock(return_value=contexts_dict)

    return storage


@pytest.mark.unit
class TestGetContextsAPI:
    """Tests for GET /api/contexts."""

    def test_returns_empty_list_when_no_contexts(self):
        storage = _make_mock_storage(contexts=[], total_count=0)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 0
            data = body["data"]
            assert data["contexts"] == []
            assert data["total"] == 0
            assert data["page"] == 1
            assert data["total_pages"] == 1

    def test_returns_contexts_with_default_pagination(self):
        ctx1 = _make_context(
            "ctx-1",
            create_time=datetime.datetime(2026, 4, 10, 12, 0, 0, tzinfo=TZ),
        )
        ctx2 = _make_context(
            "ctx-2",
            context_type=ContextType.KNOWLEDGE,
            create_time=datetime.datetime(2026, 4, 11, 12, 0, 0, tzinfo=TZ),
        )
        storage = _make_mock_storage(contexts=[ctx1, ctx2], total_count=2)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert len(data["contexts"]) == 2
            assert data["total"] == 2
            assert data["page"] == 1
            assert data["limit"] == 15
            # sorted desc by create_time: ctx2 first
            assert data["contexts"][0]["id"] == "ctx-2"
            assert data["contexts"][1]["id"] == "ctx-1"

    def test_excludes_embedding_and_raw_contexts(self):
        ctx = _make_context("ctx-embed")
        storage = _make_mock_storage(contexts=[ctx], total_count=1)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts")
            assert resp.status_code == 200
            context_data = resp.json()["data"]["contexts"][0]
            assert "embedding" not in context_data
            assert "raw_contexts" not in context_data

    def test_pagination_params(self):
        # 3 contexts, limit=2, page=2 -> should get 1 context
        ctxs = [
            _make_context(
                f"ctx-{i}",
                create_time=datetime.datetime(2026, 4, i + 1, 12, 0, 0, tzinfo=TZ),
            )
            for i in range(3)
        ]
        storage = _make_mock_storage(contexts=ctxs, total_count=3)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts?page=2&limit=2")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["page"] == 2
            assert data["limit"] == 2
            assert data["total"] == 3
            assert data["total_pages"] == 2
            assert len(data["contexts"]) == 1

    def test_limit_clamped_to_max_100(self):
        storage = _make_mock_storage(contexts=[], total_count=0)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts?limit=200")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["limit"] == 100

    def test_type_filter_passed_to_storage(self):
        storage = _make_mock_storage(contexts=[], total_count=0)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts?type=event")
            assert resp.status_code == 200
            # Verify storage was called with event type
            call_kwargs = storage.get_filtered_context_count.call_args.kwargs
            assert call_kwargs["context_types"] == ["event"]

    def test_user_id_filter_passed_to_storage(self):
        storage = _make_mock_storage(contexts=[], total_count=0)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts?user_id=u1&device_id=d1&agent_id=a1")
            assert resp.status_code == 200
            call_kwargs = storage.get_filtered_context_count.call_args.kwargs
            assert call_kwargs["user_id"] == "u1"
            assert call_kwargs["device_id"] == "d1"
            assert call_kwargs["agent_id"] == "a1"

    def test_hierarchy_level_filter(self):
        storage = _make_mock_storage(contexts=[], total_count=0)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts?hierarchy_level=1")
            assert resp.status_code == 200
            call_kwargs = storage.get_filtered_context_count.call_args.kwargs
            assert call_kwargs["filter"]["hierarchy_level"] == 1

    def test_date_filter_date_only_format(self):
        storage = _make_mock_storage(contexts=[], total_count=0)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts?start_date=2026-04-01&end_date=2026-04-10")
            assert resp.status_code == 200
            call_kwargs = storage.get_filtered_context_count.call_args.kwargs
            filt = call_kwargs["filter"]
            assert "create_time_ts" in filt
            assert "$gte" in filt["create_time_ts"]
            assert "$lt" in filt["create_time_ts"]

    def test_date_filter_datetime_format(self):
        storage = _make_mock_storage(contexts=[], total_count=0)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts?start_date=2026-04-01T10:00")
            assert resp.status_code == 200
            call_kwargs = storage.get_filtered_context_count.call_args.kwargs
            filt = call_kwargs["filter"]
            assert "$gte" in filt["create_time_ts"]

    def test_context_types_excludes_profile_types(self):
        storage = _make_mock_storage(contexts=[], total_count=0)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts")
            assert resp.status_code == 200
            data = resp.json()["data"]
            ct_list = data["context_types"]
            assert "profile" not in ct_list
            assert "agent_profile" not in ct_list
            assert "agent_base_profile" not in ct_list

    def test_storage_unavailable_returns_503(self):
        with _patched_client(mock_storage=None) as client:
            resp = client.get("/api/contexts")
            assert resp.status_code == 503
            body = resp.json()
            assert body["code"] == 503

    def test_page_clamped_to_total_pages(self):
        # Only 1 context, page=100 -> should be clamped to page 1
        ctx = _make_context("ctx-1")
        storage = _make_mock_storage(contexts=[ctx], total_count=1)
        with _patched_client(storage) as client:
            resp = client.get("/api/contexts?page=100&limit=15")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["page"] == 1

    def test_route_order_literal_before_parameterized(self):
        """Verify /api/contexts is matched before /api/contexts/{context_id}."""
        app = _build_app()
        # Collect routes in order
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        contexts_idx = None
        context_id_idx = None
        for i, p in enumerate(paths):
            if p == "/api/contexts":
                contexts_idx = i
            elif p == "/api/contexts/{context_id}":
                context_id_idx = i
        assert contexts_idx is not None, "/api/contexts route not found"
        assert context_id_idx is not None, "/api/contexts/{context_id} route not found"
        assert contexts_idx < context_id_idx, (
            "/api/contexts must come before /api/contexts/{context_id}"
        )
