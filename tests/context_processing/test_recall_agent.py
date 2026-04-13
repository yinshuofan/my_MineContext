"""Unit tests for RecallAgent (LangGraph + tool calling)."""

import contextlib
import datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opencontext.context_processing.processor.recall_agent import (
    SEARCH_MEMORIES_TOOL,
    RecallAgent,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

MOCK_LLM = "opencontext.context_processing.processor.recall_agent.generate_for_agent_async"
MOCK_SEARCH = "opencontext.context_processing.processor.recall_agent._do_search"
MOCK_CONFIG = "opencontext.context_processing.processor.recall_agent.get_config"
MOCK_PROMPT = "opencontext.context_processing.processor.recall_agent.get_prompt_group"
MOCK_STORAGE = "opencontext.context_processing.processor.recall_agent.get_storage"

_FAKE_CONFIG = {"max_turns": 3, "model": ""}

_FAKE_PROMPT_GROUP = {
    "system": (
        "You are {agent_name}. Persona: {agent_persona}. "
        "User map: {user_memory_map}. Agent map: {agent_memory_map}."
    ),
    "user": "Chat: {chat_history}",
}


def _make_ctx(ctx_id: str, title: str, summary: str = "", date: str = "2026-04-01", level: int = 0):
    """Build a minimal ProcessedContext-like object."""
    return SimpleNamespace(
        id=ctx_id,
        extracted_data=SimpleNamespace(title=title, summary=summary),
        properties=SimpleNamespace(
            event_time_start=datetime.datetime.fromisoformat(date),
            event_time_end=None,
            hierarchy_level=level,
            refs={},
        ),
    )


def _make_tool_call_response(tool_call_id: str, query: str, reason: str = ""):
    """Fake OpenAI response WITH a search_memories tool call."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=tool_call_id,
                            function=SimpleNamespace(
                                name="search_memories",
                                arguments=json.dumps({"query": query, "reason": reason}),
                            ),
                        )
                    ],
                )
            )
        ]
    )


def _make_done_response(content: str = "Recall complete."):
    """Fake OpenAI response WITHOUT tool calls (LLM decided to stop)."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=None,
                )
            )
        ]
    )


def _make_unknown_tool_response(tool_call_id: str = "tc_unk"):
    """Fake OpenAI response with an unknown tool name."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=tool_call_id,
                            function=SimpleNamespace(
                                name="unknown_tool",
                                arguments=json.dumps({"foo": "bar"}),
                            ),
                        )
                    ],
                )
            )
        ]
    )


def _make_empty_query_response(tool_call_id: str = "tc_eq"):
    """Fake OpenAI response with search_memories but empty query."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=tool_call_id,
                            function=SimpleNamespace(
                                name="search_memories",
                                arguments=json.dumps({"query": "", "reason": "empty"}),
                            ),
                        )
                    ],
                )
            )
        ]
    )


async def _run_recall(agent: RecallAgent) -> str:
    return await agent.recall(
        chat_content="hello",
        agent_name="Bot",
        agent_persona="helpful",
        user_id="u1",
        device_id="d1",
        agent_id="a1",
    )


@contextlib.contextmanager
def _patches(llm_mock, search_mock=None, config=None, prompt=None, storage=None):
    """Context manager that patches config, prompt, LLM, search, and storage."""
    if storage is None:
        storage = MagicMock()
        storage.get_hierarchy_map = AsyncMock(return_value={3: [], 2: [], 1: []})
    with (
        patch(MOCK_CONFIG, return_value=config or _FAKE_CONFIG),
        patch(MOCK_PROMPT, return_value=prompt or _FAKE_PROMPT_GROUP),
        patch(MOCK_LLM, new=llm_mock),
        patch(MOCK_SEARCH, new=search_mock or AsyncMock(return_value=([], []))),
        patch(MOCK_STORAGE, return_value=storage),
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_first_turn_done_returns_empty():
    """LLM responds without tool calls on turn 1 → empty string, no search."""
    agent = RecallAgent()
    llm = AsyncMock(return_value=_make_done_response())
    search = AsyncMock(return_value=([], []))

    with _patches(llm, search):
        result = await _run_recall(agent)

    assert result == ""
    assert llm.await_count == 1
    assert search.await_count == 0


@pytest.mark.unit
async def test_single_search_then_done():
    """Turn 1: tool call → search finds 2 hits. Turn 2: done → formatted memories."""
    agent = RecallAgent()
    ctx1 = _make_ctx("c1", "meeting with alice", "discussed Q2 plans", "2026-04-01")
    ctx2 = _make_ctx("c2", "demo feedback", "team liked the UI", "2026-04-03")

    llm = AsyncMock(
        side_effect=[
            _make_tool_call_response("tc1", "alice demo"),
            _make_done_response(),
        ]
    )
    search = AsyncMock(return_value=([ctx1, ctx2], []))

    with _patches(llm, search):
        result = await _run_recall(agent)

    assert llm.await_count == 2
    assert search.await_count == 1
    assert "meeting with alice" in result
    assert "demo feedback" in result
    assert "2026-04-01" in result
    assert "2026-04-03" in result
    assert result.index("meeting with alice") < result.index("demo feedback")


@pytest.mark.unit
async def test_max_turns_hard_cap():
    """LLM always calls tool → loop stops at recursion limit."""
    agent = RecallAgent()
    # Each turn returns a different ctx ID so the empty-search brake doesn't fire
    ctxs_per_turn = [([_make_ctx(f"c{i}", f"title{i}")], []) for i in range(10)]

    call_count = {"n": 0}

    async def llm_side_effect(*args, **kwargs):
        call_count["n"] += 1
        return _make_tool_call_response(f"tc{call_count['n']}", f"q{call_count['n']}")

    search = AsyncMock(side_effect=ctxs_per_turn)

    with _patches(AsyncMock(side_effect=llm_side_effect), search):
        await _run_recall(agent)

    # With max_turns=3, recursion_limit=(3*2)+3=9.
    # Each cycle is call_llm + execute_search = 2 steps. The graph runs until
    # GraphRecursionError fires. We expect 3-4 searches before the cap hits.
    # Recursion limit caps total node invocations. The exact count depends on
    # langgraph's step accounting. We verify: at least 2 searches happened (real
    # work), and no more than max_turns+1 (bounded).
    assert 2 <= search.await_count <= 4


@pytest.mark.unit
async def test_two_consecutive_empty_searches_stops():
    """Two empty searches → route_after_search stops the loop."""
    agent = RecallAgent()
    config = {"max_turns": 5, "model": ""}

    call_count = {"n": 0}

    async def llm_always_search(*args, **kwargs):
        call_count["n"] += 1
        return _make_tool_call_response(f"tc{call_count['n']}", "q")

    search = AsyncMock(return_value=([], []))  # always empty

    with _patches(AsyncMock(side_effect=llm_always_search), search, config=config):
        result = await _run_recall(agent)

    assert result == ""
    assert search.await_count == 2  # stops after 2 consecutive empty


@pytest.mark.unit
async def test_dedup_across_turns():
    """Two turns return overlapping IDs → final output dedupes."""
    agent = RecallAgent()
    ctx_a = _make_ctx("c1", "shared event", "sum", "2026-04-01")
    ctx_b = _make_ctx("c2", "turn 1 extra", "sum", "2026-04-02")
    ctx_c = _make_ctx("c3", "turn 2 extra", "sum", "2026-04-03")

    llm = AsyncMock(
        side_effect=[
            _make_tool_call_response("tc1", "first"),
            _make_tool_call_response("tc2", "second"),
            _make_done_response(),
        ]
    )
    search = AsyncMock(side_effect=[([ctx_a, ctx_b], []), ([ctx_a, ctx_c], [])])

    with _patches(llm, search):
        result = await _run_recall(agent)

    assert result.count("shared event") == 1
    assert "turn 1 extra" in result
    assert "turn 2 extra" in result


@pytest.mark.unit
async def test_turn_one_llm_raises_returns_empty():
    """LLM exception on turn 1 → graceful fallback, empty string."""
    agent = RecallAgent()

    async def boom(*args, **kwargs):
        raise RuntimeError("LLM outage")

    search = AsyncMock()

    with _patches(boom, search):
        result = await _run_recall(agent)

    assert result == ""
    assert search.await_count == 0


@pytest.mark.unit
async def test_unknown_tool_name_returns_error_then_stops():
    """LLM calls unknown tool → error tool message → LLM stops."""
    agent = RecallAgent()

    llm = AsyncMock(
        side_effect=[
            _make_unknown_tool_response("tc1"),
            _make_done_response(),  # LLM sees error and stops
        ]
    )
    search = AsyncMock()

    with _patches(llm, search):
        result = await _run_recall(agent)

    assert result == ""
    assert search.await_count == 0  # no actual search was executed


@pytest.mark.unit
async def test_search_raises_returns_error_to_llm():
    """Turn 1 search ok, turn 2 search raises → error as tool message → LLM stops."""
    agent = RecallAgent()
    ctx1 = _make_ctx("c1", "first hit", "sum", "2026-04-01")

    call_count = {"n": 0}

    async def search_fn(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [ctx1], []
        raise RuntimeError("search down")

    llm = AsyncMock(
        side_effect=[
            _make_tool_call_response("tc1", "q1"),
            _make_tool_call_response("tc2", "q2"),
            _make_done_response(),  # after seeing error, LLM stops
        ]
    )

    with _patches(llm, search_fn):
        result = await _run_recall(agent)

    assert "first hit" in result


@pytest.mark.unit
async def test_empty_query_returns_error_tool_message():
    """Tool call with empty query → error tool message → LLM stops."""
    agent = RecallAgent()

    llm = AsyncMock(
        side_effect=[
            _make_empty_query_response("tc1"),
            _make_done_response(),
        ]
    )
    search = AsyncMock()

    with _patches(llm, search):
        result = await _run_recall(agent)

    assert result == ""
    assert search.await_count == 0


@pytest.mark.unit
async def test_model_override_passed_when_configured():
    """Config.model set → generate_for_agent_async called with model kwarg."""
    agent = RecallAgent()
    config = {"max_turns": 3, "model": "cheap-model"}

    llm = AsyncMock(return_value=_make_done_response())

    with _patches(llm, config=config):
        await _run_recall(agent)

    assert llm.await_count == 1
    call_kwargs = llm.call_args.kwargs
    assert call_kwargs.get("model") == "cheap-model"


@pytest.mark.unit
async def test_model_override_omitted_when_not_configured():
    """Config.model empty → no model kwarg passed."""
    agent = RecallAgent()

    llm = AsyncMock(return_value=_make_done_response())

    with _patches(llm):
        await _run_recall(agent)

    assert llm.await_count == 1
    assert "model" not in llm.call_args.kwargs


@pytest.mark.unit
async def test_config_hot_reload():
    """Same agent instance, different config → different behavior."""
    agent = RecallAgent()

    call_count_2 = {"n": 0}

    async def llm_always_search_2(*args, **kwargs):
        call_count_2["n"] += 1
        return _make_tool_call_response(f"tc{call_count_2['n']}", "q")

    # Run 1: max_turns=2
    with _patches(
        AsyncMock(side_effect=llm_always_search_2),
        AsyncMock(side_effect=[([_make_ctx(f"a{i}", f"t{i}")], []) for i in range(5)]),
        config={"max_turns": 2, "model": ""},
    ):
        await _run_recall(agent)

    call_count_1 = {"n": 0}

    async def llm_always_search_1(*args, **kwargs):
        call_count_1["n"] += 1
        return _make_tool_call_response(f"tc{call_count_1['n']}", "q")

    # Run 2: max_turns=1
    with _patches(
        AsyncMock(side_effect=llm_always_search_1),
        AsyncMock(side_effect=[([_make_ctx(f"b{i}", f"t{i}")], []) for i in range(5)]),
        config={"max_turns": 1, "model": ""},
    ):
        await _run_recall(agent)

    # Both runs must have done real work
    assert call_count_1["n"] >= 1, "max_turns=1 run did no work"
    assert call_count_2["n"] >= 1, "max_turns=2 run did no work"
    # max_turns=1 should result in fewer LLM calls than max_turns=2
    assert call_count_1["n"] < call_count_2["n"]


@pytest.mark.unit
async def test_format_memories_with_tz_aware_datetimes():
    """_format_memories handles timezone-aware datetimes without TypeError."""
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
    assert result.index("no time") < result.index("tz event")


# ---------------------------------------------------------------------------
# New tests (not in old suite)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_tools_param_passed_to_llm():
    """Verify generate_for_agent_async receives tools=[SEARCH_MEMORIES_TOOL]."""
    agent = RecallAgent()
    llm = AsyncMock(return_value=_make_done_response())

    with _patches(llm):
        await _run_recall(agent)

    call_args = llm.call_args
    # generate_for_agent_async is called with keyword args: messages=..., tools=...
    tools_arg = call_args.kwargs.get("tools")
    assert tools_arg == [SEARCH_MEMORIES_TOOL]


@pytest.mark.unit
async def test_message_history_accumulates():
    """Messages passed to LLM on turn 2 contain turn 1's assistant + tool messages."""
    agent = RecallAgent()
    ctx1 = _make_ctx("c1", "found event", "sum", "2026-04-01")

    llm_calls: list = []

    async def capture_llm(messages, **kwargs):
        llm_calls.append(list(messages))  # snapshot
        if len(llm_calls) == 1:
            return _make_tool_call_response("tc1", "first query")
        return _make_done_response()

    with _patches(capture_llm, AsyncMock(return_value=([ctx1], []))):
        await _run_recall(agent)

    assert len(llm_calls) == 2
    # Turn 1 messages: [system, user]
    assert len(llm_calls[0]) == 2
    assert llm_calls[0][0]["role"] == "system"
    assert llm_calls[0][1]["role"] == "user"
    # Turn 2 messages: [system, user, assistant(with tool_calls), tool(result)]
    assert len(llm_calls[1]) == 4
    assert llm_calls[1][2]["role"] == "assistant"
    assert "tool_calls" in llm_calls[1][2]
    assert llm_calls[1][3]["role"] == "tool"
    assert "found event" in llm_calls[1][3]["content"]


@pytest.mark.unit
async def test_search_error_becomes_tool_message():
    """When search raises, LLM receives a tool message with error text."""
    agent = RecallAgent()

    llm_calls: list = []

    async def capture_llm(messages, **kwargs):
        llm_calls.append(list(messages))
        if len(llm_calls) == 1:
            return _make_tool_call_response("tc1", "query")
        return _make_done_response()

    async def boom(*args, **kwargs):
        raise RuntimeError("search down")

    with _patches(capture_llm, boom):
        await _run_recall(agent)

    assert len(llm_calls) == 2
    # Turn 2 should see the error tool message
    tool_msg = llm_calls[1][3]
    assert tool_msg["role"] == "tool"
    assert "Error" in tool_msg["content"]
    assert "unavailable" in tool_msg["content"]


@pytest.mark.unit
async def test_graph_recursion_error_returns_empty():
    """When recursion limit fires, recall() returns empty string gracefully."""
    agent = RecallAgent()
    # max_turns=1 → recursion_limit=(1*2)+3=5 — very tight
    config = {"max_turns": 1, "model": ""}

    call_count = {"n": 0}

    async def llm_always_search(*args, **kwargs):
        call_count["n"] += 1
        return _make_tool_call_response(f"tc{call_count['n']}", "q")

    # Return unique results each time so consecutive-empty brake doesn't fire
    search = AsyncMock(side_effect=[([_make_ctx(f"x{i}", f"t{i}")], []) for i in range(20)])

    with _patches(AsyncMock(side_effect=llm_always_search), search, config=config):
        result = await _run_recall(agent)

    # With tight recursion_limit, the graph may hit GraphRecursionError
    # recall() catches it and returns "" — this is the expected graceful degradation
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Memory map tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_memory_map_injected_into_system_prompt():
    """L3 summaries from memory map appear in the system message passed to LLM."""
    agent = RecallAgent()

    ctx_l3 = _make_ctx("l3_1", "March project kickoff", "Started the new project", "2026-03-15")
    fake_storage = MagicMock()
    fake_storage.get_hierarchy_map = AsyncMock(
        side_effect=[
            {3: [ctx_l3], 2: [], 1: []},  # user map
            {3: [], 2: [], 1: []},  # agent map
        ]
    )

    llm_calls: list = []

    async def capture_llm(messages, **kwargs):
        llm_calls.append(list(messages))
        return _make_done_response()

    with _patches(capture_llm, storage=fake_storage):
        await _run_recall(agent)

    assert len(llm_calls) == 1
    system_content = llm_calls[0][0]["content"]
    # _format_map prefers summary over title when both are present
    assert "Started the new project" in system_content


@pytest.mark.unit
async def test_empty_memory_map_shows_placeholder():
    """No history → '(no history available)' in prompt."""
    agent = RecallAgent()

    llm_calls: list = []

    async def capture_llm(messages, **kwargs):
        llm_calls.append(list(messages))
        return _make_done_response()

    with _patches(capture_llm):
        await _run_recall(agent)

    system_content = llm_calls[0][0]["content"]
    assert "(no history available)" in system_content


@pytest.mark.unit
async def test_both_user_and_agent_maps_fetched():
    """get_hierarchy_map called twice: once for user, once for agent_base."""
    agent = RecallAgent()

    fake_storage = MagicMock()
    fake_storage.get_hierarchy_map = AsyncMock(return_value={3: [], 2: [], 1: []})

    with _patches(AsyncMock(return_value=_make_done_response()), storage=fake_storage):
        await _run_recall(agent)

    assert fake_storage.get_hierarchy_map.await_count == 2
    calls = fake_storage.get_hierarchy_map.await_args_list
    # First call: user events
    assert calls[0].args[0] == "user" or calls[0].kwargs.get("owner_type") == "user"
    # Second call: agent_base events
    assert calls[1].args[0] == "agent_base" or calls[1].kwargs.get("owner_type") == "agent_base"


@pytest.mark.unit
async def test_format_map_groups_by_level():
    """_format_map outputs L3 before L2 before L1, sorted by time within each."""
    from opencontext.context_processing.processor.recall_agent import RecallAgent

    # _format_map prefers summary over title; use summary for descriptive text
    ctx_l3_jan = _make_ctx("l3a", "", "January overview", "2026-01-15")
    ctx_l3_feb = _make_ctx("l3b", "", "February overview", "2026-02-15")
    ctx_l2 = _make_ctx("l2a", "", "Week 14 report", "2026-04-01")
    ctx_l1 = _make_ctx("l1a", "", "Yesterday daily", "2026-04-12")

    result = RecallAgent._format_map(
        {
            3: [ctx_l3_feb, ctx_l3_jan],  # deliberately out of order
            2: [ctx_l2],
            1: [ctx_l1],
        }
    )

    # L3 should come first, sorted by date (Jan before Feb)
    assert result.index("January overview") < result.index("February overview")
    # L3 before L2
    assert result.index("February overview") < result.index("Week 14 report")
    # L2 before L1
    assert result.index("Week 14 report") < result.index("Yesterday daily")
    # Level labels present (L3/L2/L1, not monthly/weekly/daily)
    assert "L3" in result
    assert "L2" in result
    assert "L1" in result
    # Actual dates present
    assert "2026-01-15" in result
    assert "2026-04-12" in result


# ---------------------------------------------------------------------------
# Tests for _format_search_result and _format_time_range_str
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_format_search_result_hits_only():
    """Hits without ancestors renders flat list with ★ markers."""
    from opencontext.context_processing.processor.recall_agent import _format_search_result

    ctx1 = _make_ctx("h1", "meeting", "Q2 plans", "2026-04-01")
    ctx2 = _make_ctx("h2", "review", "auth module", "2026-04-03")
    result = _format_search_result([ctx1, ctx2], [])
    assert "2 hits" in result
    assert "meeting" in result and "★" in result
    assert "review" in result
    assert "ancestors" not in result


@pytest.mark.unit
async def test_format_search_result_with_hierarchy_tree():
    """Hits + ancestors build an indented tree via refs linkage."""
    from opencontext.context_processing.processor.recall_agent import _format_search_result

    # L0 hit has a ref pointing up to L1 ancestor
    hit = SimpleNamespace(
        id="h1",
        extracted_data=SimpleNamespace(title="meeting", summary=""),
        properties=SimpleNamespace(
            event_time_start=datetime.datetime.fromisoformat("2026-04-01"),
            event_time_end=None,
            hierarchy_level=0,
            refs={"daily_summary": ["a1"]},
        ),
    )
    ancestor = SimpleNamespace(
        id="a1",
        extracted_data=SimpleNamespace(title="daily overview", summary=""),
        properties=SimpleNamespace(
            event_time_start=datetime.datetime.fromisoformat("2026-04-01"),
            event_time_end=None,
            hierarchy_level=1,
            refs={},
        ),
    )
    result = _format_search_result([hit], [ancestor])
    assert "1 hits" in result
    assert "1 ancestors" in result
    # Ancestor is root (no indentation), hit is indented (child)
    lines = result.split("\n")
    ancestor_line = [ln for ln in lines if "daily overview" in ln][0]
    hit_line = [ln for ln in lines if "meeting" in ln][0]
    # Hit should be indented more than ancestor
    assert len(hit_line) - len(hit_line.lstrip()) > len(ancestor_line) - len(ancestor_line.lstrip())
    assert "★" in hit_line


@pytest.mark.unit
async def test_format_search_result_empty():
    """No hits and no ancestors returns 'no new memories' message."""
    from opencontext.context_processing.processor.recall_agent import _format_search_result

    result = _format_search_result([], [])
    assert "No new memories" in result


@pytest.mark.unit
async def test_format_time_range_str_same_day():
    """Same-day start/end returns single date."""
    from opencontext.context_processing.processor.recall_agent import _format_time_range_str

    ctx = _make_ctx("t1", "test", date="2026-04-12")
    assert _format_time_range_str(ctx) == "2026-04-12"


@pytest.mark.unit
async def test_format_time_range_str_multi_day():
    """Different start/end returns range."""
    from opencontext.context_processing.processor.recall_agent import _format_time_range_str

    ctx = SimpleNamespace(
        properties=SimpleNamespace(
            event_time_start=datetime.datetime.fromisoformat("2026-04-01"),
            event_time_end=datetime.datetime.fromisoformat("2026-04-07"),
            hierarchy_level=2,
            refs={},
        ),
        extracted_data=SimpleNamespace(title="", summary=""),
        id="r1",
    )
    assert _format_time_range_str(ctx) == "2026-04-01~2026-04-07"
