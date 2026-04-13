"""Unit tests for RecallAgent."""

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from opencontext.context_processing.processor.recall_agent import RecallAgent


def _make_ctx(ctx_id: str, title: str, summary: str = "", date: str = "2026-04-01"):
    """Build a minimal ProcessedContext-like object for formatting assertions."""
    return SimpleNamespace(
        id=ctx_id,
        extracted_data=SimpleNamespace(title=title, summary=summary),
        properties=SimpleNamespace(event_time_start=datetime.datetime.fromisoformat(date)),
    )


_FAKE_CONFIG = {"max_turns": 3, "model": ""}

_FAKE_PROMPT_GROUP = {
    "system": "You are {agent_name}. Persona: {agent_persona}. Max turns: {max_turns}.",
    "user": "Chat: {chat_history}\nPrev: {previous_actions}\nBrief: {accumulated_brief}",
}


async def _run_recall(agent: RecallAgent) -> str:
    return await agent.recall(
        chat_content="hello",
        agent_name="Bot",
        agent_persona="helpful",
        user_id="u1",
        device_id="d1",
        agent_id="a1",
    )


@pytest.mark.unit
async def test_first_turn_done_returns_empty():
    """Agent says 'done' on turn 1 → recall() returns empty string, no search called."""
    agent = RecallAgent()

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value='{"action": "done", "reason": "not relevant"}'),
        ) as llm,
        patch.object(RecallAgent, "_execute_search", new=AsyncMock(return_value=[])) as search,
    ):
        result = await _run_recall(agent)

    assert result == ""
    assert llm.await_count == 1
    assert search.await_count == 0


@pytest.mark.unit
async def test_single_search_then_done():
    """Turn 1 searches and finds 2 hits, turn 2 says done → formatted memories."""
    agent = RecallAgent()

    ctx1 = _make_ctx("c1", "meeting with alice", "discussed Q2 plans", "2026-04-01")
    ctx2 = _make_ctx("c2", "demo feedback", "team liked the UI", "2026-04-03")

    llm_responses = [
        '{"action": "search", "query": "alice demo", "reason": "new keywords"}',
        '{"action": "done", "reason": "got what I need"}',
    ]

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(side_effect=llm_responses),
        ) as llm,
        patch.object(
            RecallAgent, "_execute_search", new=AsyncMock(return_value=[ctx1, ctx2])
        ) as search,
    ):
        result = await _run_recall(agent)

    assert llm.await_count == 2
    assert search.await_count == 1
    assert "meeting with alice" in result
    assert "demo feedback" in result
    assert "2026-04-01" in result
    assert "2026-04-03" in result
    # Memories are sorted ascending by date, so alice appears before demo
    assert result.index("meeting with alice") < result.index("demo feedback")
