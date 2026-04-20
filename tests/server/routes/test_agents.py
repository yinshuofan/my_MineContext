# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for /api/agents endpoints."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opencontext.server.routes.agents import router


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.unit
class TestDownloadNarrativeSkill:
    """Tests for GET /api/agents/skills/narrative-to-base-events/download."""

    def test_returns_zip_with_skill_files(self):
        client = _build_client()
        resp = client.get("/api/agents/skills/narrative-to-base-events/download")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert "narrative-to-base-events.zip" in resp.headers.get("content-disposition", "")

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = set(zf.namelist())
        assert "narrative-to-base-events/SKILL.md" in names
        assert "narrative-to-base-events/references/base-event-schema.md" in names
        assert "narrative-to-base-events/references/roleplay-prompt-guide.md" in names

    def test_returns_404_when_bundle_missing(self, monkeypatch, tmp_path):
        from opencontext.server.routes import agents as agents_module

        monkeypatch.setattr(agents_module, "_NARRATIVE_SKILL_ZIP", tmp_path / "missing.zip")
        client = _build_client()
        resp = client.get("/api/agents/skills/narrative-to-base-events/download")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
