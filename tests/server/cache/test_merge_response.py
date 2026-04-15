"""Tests for MemoryCacheManager._merge_response — profile + agent_prompt assembly."""

import pytest

from opencontext.server.cache.memory_cache_manager import MemoryCacheManager


def _make_manager() -> MemoryCacheManager:
    return MemoryCacheManager()


def _snapshot(
    *,
    profile: dict | None = None,
    agent_prompt: dict | None = None,
    today_events: list | None = None,
    daily_summaries: list | None = None,
) -> dict:
    return {
        "user_id": "user_001",
        "device_id": "default",
        "agent_id": "kiki",
        "profile": profile,
        "agent_prompt": agent_prompt,
        "recent_memories": {
            "today_events": today_events or [],
            "daily_summaries": daily_summaries or [],
        },
    }


@pytest.mark.unit
def test_profile_and_agent_prompt_gate_independently_both_requested():
    mgr = _make_manager()
    snap = _snapshot(
        profile={"factual_profile": "user facts", "behavioral_profile": None, "metadata": {}},
        agent_prompt={
            "factual_profile": "I am kiki",
            "behavioral_profile": "concise",
            "metadata": {"v": 1},
        },
    )

    resp = mgr._merge_response(
        snap,
        accessed=[],
        cache_hit=True,
        ttl_remaining=0,
        include_sections={"profile", "agent_prompt"},
    )

    assert resp.profile is not None
    assert resp.profile.factual_profile == "user facts"
    assert resp.agent_prompt is not None
    assert resp.agent_prompt.factual_profile == "I am kiki"
    assert resp.agent_prompt.behavioral_profile == "concise"
    assert resp.today_events is None
    assert resp.daily_summaries is None
    assert resp.recently_accessed is None


@pytest.mark.unit
def test_only_profile_requested_hides_agent_prompt_even_if_in_snapshot():
    mgr = _make_manager()
    snap = _snapshot(
        profile={"factual_profile": "user facts", "behavioral_profile": None, "metadata": {}},
        agent_prompt={"factual_profile": "I am kiki", "behavioral_profile": None, "metadata": {}},
    )

    resp = mgr._merge_response(
        snap,
        accessed=[],
        cache_hit=True,
        ttl_remaining=0,
        include_sections={"profile"},
    )

    assert resp.profile is not None
    assert resp.agent_prompt is None


@pytest.mark.unit
def test_only_agent_prompt_requested_hides_profile_even_if_in_snapshot():
    mgr = _make_manager()
    snap = _snapshot(
        profile={"factual_profile": "user facts", "behavioral_profile": None, "metadata": {}},
        agent_prompt={"factual_profile": "I am kiki", "behavioral_profile": None, "metadata": {}},
    )

    resp = mgr._merge_response(
        snap,
        accessed=[],
        cache_hit=True,
        ttl_remaining=0,
        include_sections={"agent_prompt"},
    )

    assert resp.profile is None
    assert resp.agent_prompt is not None


@pytest.mark.unit
def test_neither_profile_nor_agent_prompt_when_only_events_requested():
    mgr = _make_manager()
    snap = _snapshot(
        profile={"factual_profile": "u", "behavioral_profile": None, "metadata": {}},
        agent_prompt={"factual_profile": "a", "behavioral_profile": None, "metadata": {}},
    )

    resp = mgr._merge_response(
        snap,
        accessed=[],
        cache_hit=True,
        ttl_remaining=0,
        include_sections={"events"},
    )

    assert resp.profile is None
    assert resp.agent_prompt is None


@pytest.mark.unit
def test_empty_profile_and_agent_prompt_both_none():
    mgr = _make_manager()
    snap = _snapshot(profile=None, agent_prompt=None)

    resp = mgr._merge_response(
        snap,
        accessed=[],
        cache_hit=True,
        ttl_remaining=0,
        include_sections={"profile", "agent_prompt"},
    )

    assert resp.profile is None
    assert resp.agent_prompt is None
