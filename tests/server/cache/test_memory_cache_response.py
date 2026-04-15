"""Tests for UserMemoryCacheResponse — focus on the agent_prompt field."""

import pytest

from opencontext.server.cache.models import (
    SimpleProfile,
    UserMemoryCacheResponse,
)


@pytest.mark.unit
def test_agent_prompt_defaults_to_none():
    resp = UserMemoryCacheResponse(
        success=True,
        user_id="user_001",
        agent_id="default",
    )
    assert resp.agent_prompt is None


@pytest.mark.unit
def test_agent_prompt_accepts_simple_profile():
    agent_prompt = SimpleProfile(
        factual_profile="I am kiki, an assistant.",
        behavioral_profile="Reply concisely in English.",
        metadata={"version": 3},
    )
    resp = UserMemoryCacheResponse(
        success=True,
        user_id="user_001",
        agent_id="kiki",
        agent_prompt=agent_prompt,
    )
    assert resp.agent_prompt is not None
    assert resp.agent_prompt.factual_profile == "I am kiki, an assistant."
    assert resp.agent_prompt.behavioral_profile == "Reply concisely in English."
    assert resp.agent_prompt.metadata == {"version": 3}


@pytest.mark.unit
def test_agent_prompt_serializes_as_dict_in_model_dump():
    resp = UserMemoryCacheResponse(
        success=True,
        user_id="user_001",
        agent_id="kiki",
        agent_prompt=SimpleProfile(factual_profile="x"),
    )
    dumped = resp.model_dump()
    assert "agent_prompt" in dumped
    assert dumped["agent_prompt"]["factual_profile"] == "x"
    assert dumped["agent_prompt"]["behavioral_profile"] is None
