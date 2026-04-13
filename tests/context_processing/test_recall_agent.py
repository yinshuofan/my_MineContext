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


@pytest.mark.unit
async def test_max_turns_hard_cap():
    """Agent never says done → loop stops at max_turns (default 3)."""
    agent = RecallAgent()
    always_search = '{"action": "search", "query": "q", "reason": "more"}'
    ctxs_per_turn = [[_make_ctx(f"c{i}", f"title{i}", "sum", "2026-04-01")] for i in range(10)]

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
            new=AsyncMock(return_value=always_search),
        ) as llm,
        patch.object(
            RecallAgent,
            "_execute_search",
            new=AsyncMock(side_effect=ctxs_per_turn),
        ) as search,
    ):
        await _run_recall(agent)

    assert llm.await_count == 3
    assert search.await_count == 3


@pytest.mark.unit
async def test_two_consecutive_empty_searches_stops():
    """Two empty searches in a row → safety brake breaks the loop."""
    agent = RecallAgent()
    config_five_turns = {"max_turns": 5, "model": ""}
    always_search = '{"action": "search", "query": "q", "reason": "more"}'

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=config_five_turns,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value=always_search),
        ) as llm,
        patch.object(
            RecallAgent,
            "_execute_search",
            new=AsyncMock(side_effect=[[], []]),
        ) as search,
    ):
        result = await _run_recall(agent)

    assert search.await_count == 2
    assert llm.await_count == 2
    assert result == ""


@pytest.mark.unit
async def test_dedup_across_turns():
    """Turn 1 and turn 2 return overlapping ctx IDs → final output dedupes."""
    agent = RecallAgent()
    ctx_a = _make_ctx("c1", "shared event", "sum", "2026-04-01")
    ctx_b = _make_ctx("c2", "turn 1 extra", "sum", "2026-04-02")
    ctx_c = _make_ctx("c3", "turn 2 extra", "sum", "2026-04-03")

    llm_responses = [
        '{"action": "search", "query": "first", "reason": "r"}',
        '{"action": "search", "query": "second", "reason": "r"}',
        '{"action": "done", "reason": "enough"}',
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
        ),
        patch.object(
            RecallAgent,
            "_execute_search",
            new=AsyncMock(side_effect=[[ctx_a, ctx_b], [ctx_a, ctx_c]]),
        ),
    ):
        result = await _run_recall(agent)

    assert result.count("shared event") == 1
    assert "turn 1 extra" in result
    assert "turn 2 extra" in result


@pytest.mark.unit
async def test_turn_one_invalid_json_returns_empty():
    """Unparseable LLM output on turn 1 → abort with empty string."""
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
            new=AsyncMock(return_value="not json at all"),
        ),
        patch.object(RecallAgent, "_execute_search", new=AsyncMock()) as search,
    ):
        result = await _run_recall(agent)
    assert result == ""
    assert search.await_count == 0


@pytest.mark.unit
async def test_unknown_action_breaks():
    """Unknown action type → break, return accumulated (empty here)."""
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
            new=AsyncMock(return_value='{"action": "fetch", "query": "x"}'),
        ),
        patch.object(RecallAgent, "_execute_search", new=AsyncMock()) as search,
    ):
        result = await _run_recall(agent)
    assert result == ""
    assert search.await_count == 0


@pytest.mark.unit
async def test_search_raises_returns_accumulated():
    """Turn 1 search ok, turn 2 search raises → return turn 1's memories."""
    agent = RecallAgent()
    ctx1 = _make_ctx("c1", "first hit", "sum", "2026-04-01")
    llm_responses = [
        '{"action": "search", "query": "q1", "reason": "r"}',
        '{"action": "search", "query": "q2", "reason": "r"}',
    ]
    call_count = {"n": 0}

    async def search_side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [ctx1]
        raise RuntimeError("search down")

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
        ),
        patch.object(RecallAgent, "_execute_search", new=search_side_effect),
    ):
        result = await _run_recall(agent)
    assert "first hit" in result


@pytest.mark.unit
async def test_turn_one_llm_raises_returns_empty():
    """Turn 1 LLM raises → return empty string."""
    agent = RecallAgent()

    async def boom(*args, **kwargs):
        raise RuntimeError("LLM outage")

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
            new=boom,
        ),
        patch.object(RecallAgent, "_execute_search", new=AsyncMock()) as search,
    ):
        result = await _run_recall(agent)
    assert result == ""
    assert search.await_count == 0


@pytest.mark.unit
async def test_model_override_passed_when_configured():
    """Config.model set → generate_with_messages called with model kwarg."""
    agent = RecallAgent()
    config_with_model = {"max_turns": 3, "model": "cheap-model"}
    done_once = '{"action": "done", "reason": "quick"}'

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=config_with_model,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value=done_once),
        ) as llm,
        patch.object(RecallAgent, "_execute_search", new=AsyncMock(return_value=[])),
    ):
        await _run_recall(agent)

    assert llm.await_count == 1
    call_kwargs = llm.await_args.kwargs
    assert call_kwargs.get("model") == "cheap-model"
    assert call_kwargs.get("enable_executor") is False


@pytest.mark.unit
async def test_model_override_omitted_when_not_configured():
    """Config.model empty → generate_with_messages called WITHOUT model kwarg."""
    agent = RecallAgent()
    done_once = '{"action": "done", "reason": "quick"}'

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
            new=AsyncMock(return_value=done_once),
        ) as llm,
        patch.object(RecallAgent, "_execute_search", new=AsyncMock(return_value=[])),
    ):
        await _run_recall(agent)

    assert llm.await_count == 1
    assert "model" not in llm.await_args.kwargs


@pytest.mark.unit
async def test_config_hot_reload():
    """Changing max_turns in config affects the next recall() call."""
    agent = RecallAgent()
    always_search = '{"action": "search", "query": "q", "reason": "r"}'

    config_2_turns = {"max_turns": 2, "model": ""}
    config_1_turn = {"max_turns": 1, "model": ""}

    # Run with max_turns=2
    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=config_2_turns,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value=always_search),
        ) as llm2,
        patch.object(
            RecallAgent,
            "_execute_search",
            new=AsyncMock(side_effect=[[_make_ctx(f"a{i}", f"t{i}")] for i in range(5)]),
        ),
    ):
        await _run_recall(agent)
    assert llm2.await_count == 2

    # Run with max_turns=1 — same agent instance, different config
    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=config_1_turn,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value=always_search),
        ) as llm1,
        patch.object(
            RecallAgent,
            "_execute_search",
            new=AsyncMock(side_effect=[[_make_ctx(f"b{i}", f"t{i}")] for i in range(5)]),
        ),
    ):
        await _run_recall(agent)
    assert llm1.await_count == 1


@pytest.mark.unit
async def test_search_action_empty_query_breaks():
    """Search action with empty query → break immediately, no search executed."""
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
            new=AsyncMock(return_value='{"action": "search", "query": "", "reason": "empty"}'),
        ),
        patch.object(RecallAgent, "_execute_search", new=AsyncMock()) as search,
    ):
        result = await _run_recall(agent)

    assert result == ""
    assert search.await_count == 0


@pytest.mark.unit
async def test_format_memories_with_tz_aware_datetimes():
    """_format_memories handles timezone-aware datetimes without TypeError."""
    from opencontext.context_processing.processor.recall_agent import RecallAgent

    tz_ctx = SimpleNamespace(
        id="tz1",
        extracted_data=SimpleNamespace(title="tz event", summary="tz sum"),
        properties=SimpleNamespace(
            event_time_start=datetime.datetime(2026, 4, 1, tzinfo=datetime.UTC)
        ),
    )
    no_time_ctx = SimpleNamespace(
        id="nt1",
        extracted_data=SimpleNamespace(title="no time", summary=""),
        properties=SimpleNamespace(event_time_start=None),
    )

    result = RecallAgent._format_memories([tz_ctx, no_time_ctx])
    assert "tz event" in result
    assert "no time" in result
    # no-time context should sort first (min datetime)
    assert result.index("no time") < result.index("tz event")
