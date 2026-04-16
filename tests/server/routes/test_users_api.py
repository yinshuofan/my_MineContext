# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GET /api/users endpoint."""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opencontext.server.routes.users_api import router


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@contextlib.contextmanager
def _patched_client(mock_storage: MagicMock | None):
    """Yield a TestClient with patched storage and auth."""
    app = _build_app()
    with (
        patch("opencontext.server.routes.users_api.get_storage", return_value=mock_storage),
        patch(
            "opencontext.server.routes.users_api.auth_dependency",
            return_value="test-token",
        ),
    ):
        yield TestClient(app)


def _make_mock_storage(users: list[dict] | None = None) -> MagicMock:
    storage = MagicMock()
    storage.list_distinct_users = AsyncMock(return_value=users or [])
    return storage


@pytest.mark.unit
class TestListUsersAPI:
    """Tests for GET /api/users."""

    def test_returns_empty_list_when_no_users(self):
        storage = _make_mock_storage(users=[])
        with _patched_client(storage) as client:
            resp = client.get("/api/users")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 0
            data = body["data"]
            assert data["users"] == []
            assert data["total"] == 0

    def test_returns_users(self):
        users = [
            {"user_id": "alice", "device_id": "default", "agent_id": "default"},
            {"user_id": "bob", "device_id": "phone", "agent_id": "assistant"},
        ]
        storage = _make_mock_storage(users=users)
        with _patched_client(storage) as client:
            resp = client.get("/api/users")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert len(data["users"]) == 2
            assert data["total"] == 2
            assert data["users"][0]["user_id"] == "alice"
            assert data["users"][1]["user_id"] == "bob"

    def test_storage_unavailable_returns_503(self):
        with _patched_client(mock_storage=None) as client:
            resp = client.get("/api/users")
            assert resp.status_code == 503
            body = resp.json()
            assert body["code"] == 503

    def test_calls_storage_list_distinct_users(self):
        storage = _make_mock_storage(users=[])
        with _patched_client(storage) as client:
            client.get("/api/users")
            storage.list_distinct_users.assert_awaited_once()
