"""Tests for MemoryCacheManager snapshot construction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opencontext.server.cache.memory_cache_manager import MemoryCacheManager


@pytest.mark.unit
async def test_build_snapshot_falls_back_to_default_device_for_agent_base_profile():
    """Agent base profile is agent-level data, stored under device_id='default'."""
    manager = MemoryCacheManager()
    fake_storage = MagicMock()

    async def fake_get_profile(user_id, device_id, agent_id, *, context_type):
        if context_type == "profile":
            return None
        if context_type == "agent_profile":
            return None
        if (
            user_id == "__base__"
            and device_id == "default"
            and agent_id == "Voxi"
            and context_type == "agent_base_profile"
        ):
            return {"factual_profile": "base Voxi prompt", "behavioral_profile": None}
        return None

    fake_storage.get_profile = AsyncMock(side_effect=fake_get_profile)
    fake_storage.get_all_processed_contexts = AsyncMock(return_value={})
    fake_storage.search_hierarchy = AsyncMock(return_value=[])

    with patch(
        "opencontext.server.cache.memory_cache_manager.get_storage",
        return_value=fake_storage,
    ):
        snapshot = await manager._build_snapshot(
            user_id="admin",
            device_id="web-studio-1",
            agent_id="Voxi",
            recent_days=3,
            max_today_events=5,
        )

    assert snapshot["agent_prompt"]["factual_profile"] == "base Voxi prompt"
    fake_storage.get_profile.assert_any_await(
        "__base__",
        "default",
        "Voxi",
        context_type="agent_base_profile",
    )
