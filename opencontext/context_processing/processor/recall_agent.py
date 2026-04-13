"""Recall Agent — multi-turn LLM-driven memory recall using LangGraph StateGraph."""

import asyncio
import contextlib
import datetime
from typing import Any, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from opencontext.config.global_config import get_config, get_prompt_group
from opencontext.llm.global_vlm_client import generate_for_agent_async
from opencontext.models.context import ProcessedContext
from opencontext.server.search.event_search_service import EventSearchService
from opencontext.storage.global_storage import get_storage
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class RecallState(TypedDict):
    messages: list[dict[str, Any]]  # OpenAI-format dicts, manually accumulated
    accumulated: list[Any]  # list[ProcessedContext]
    seen_ids: list[str]  # list (not set) for serializability
    consecutive_empty: int  # safety brake counter
    search_params: dict[str, str]  # {user_id, device_id, agent_id}
    recall_model: str  # "" = use VLM default
    done: bool  # True when LLM responds without tool_calls


# ---------------------------------------------------------------------------
# Tool definition (module-level constant)
# ---------------------------------------------------------------------------

SEARCH_MEMORIES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_memories",
        "description": (
            "Search past memories/events by semantic query. Results include "
            "matching events and their hierarchy ancestors automatically. "
            "Events are organized in a 4-level hierarchy: "
            "L0 (raw events), L1 (first-level summaries), "
            "L2 (second-level summaries), L3 (top-level summaries) — "
            "matching the levels shown in the memory maps. "
            "Use filters to narrow results by time range or hierarchy level."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short keyword query to search for in past memories",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this search is needed",
                },
                "time_start": {
                    "type": "string",
                    "description": (
                        "Filter: only events starting at or after this date "
                        "(ISO 8601, e.g. '2026-03-01'). Optional."
                    ),
                },
                "time_end": {
                    "type": "string",
                    "description": (
                        "Filter: only events starting at or before this date "
                        "(ISO 8601, e.g. '2026-03-31'). Optional."
                    ),
                },
                "hierarchy_levels": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Filter: hierarchy levels to search. "
                        "0=L0 raw events, 1=L1 summaries, "
                        "2=L2 summaries, 3=L3 summaries "
                        "(same L-levels as shown in the memory maps). "
                        "Default: [0] (raw events only). Optional."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

# ---------------------------------------------------------------------------
# Helper: _do_search
# ---------------------------------------------------------------------------


async def _do_search(
    query: str,
    user_id: str,
    device_id: str,
    agent_id: str,
    time_range: Any | None = None,
    hierarchy_levels: list[int] | None = None,
) -> tuple[list[ProcessedContext], list[ProcessedContext]]:
    """Execute one search with optional filters. Exceptions propagate to caller.

    Returns (hits, ancestors) — both deduplicated, ancestors exclude any hit IDs.
    """
    from opencontext.server.search.event_search_service import SearchResult

    service = EventSearchService()
    result: SearchResult | None = await service.search(
        query=[{"type": "text", "text": query}],
        user_id=user_id,
        device_id=device_id,
        agent_id=agent_id,
        time_range=time_range,
        hierarchy_levels=hierarchy_levels,
        drill_up=True,
    )
    if not result:
        return [], []
    hits: list[ProcessedContext] = []
    hit_ids: set[str] = set()
    for ctx, _score in result.hits:
        if ctx.id not in hit_ids:
            hits.append(ctx)
            hit_ids.add(ctx.id)
    ancestors: list[ProcessedContext] = []
    for ctx_id, ctx in result.ancestors.items():
        if ctx_id not in hit_ids:
            ancestors.append(ctx)
    return hits, ancestors


def _format_ctx_entry(ctx: ProcessedContext) -> str:
    """Format a single context as '[time_range L{level}] title\\n  summary'."""
    level = ctx.properties.hierarchy_level if ctx.properties else 0
    start = ctx.properties.event_time_start if ctx.properties else None
    end = ctx.properties.event_time_end if ctx.properties else None
    if start:
        start_str = start.strftime("%Y-%m-%d")
        if end and end.date() != start.date():
            time_str = f"{start_str}~{end.strftime('%Y-%m-%d')}"
        else:
            time_str = start_str
    else:
        time_str = ""
    title = (ctx.extracted_data.title if ctx.extracted_data else "") or ""
    summary = (ctx.extracted_data.summary if ctx.extracted_data else "") or ""
    entry = f"[{time_str} L{level}] {title}"
    if summary:
        entry += f"\n  {summary}"
    return entry


def _format_search_result(hits: list[ProcessedContext], ancestors: list[ProcessedContext]) -> str:
    """Format search results separating direct hits from drill-up ancestors."""
    if not hits and not ancestors:
        return "No new memories found for this query."
    parts: list[str] = []
    if hits:
        lines = [_format_ctx_entry(ctx) for ctx in hits]
        parts.append(f"Found {len(hits)} direct hits:\n" + "\n".join(lines))
    if ancestors:
        lines = [_format_ctx_entry(ctx) for ctx in ancestors]
        parts.append(f"Hierarchy context ({len(ancestors)} ancestors):\n" + "\n".join(lines))
    if not parts:
        return "No new memories found for this query."
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def call_llm(state: RecallState) -> dict:
    """Call the LLM with the current message history and the search tool."""
    kwargs: dict[str, Any] = {}
    if state["recall_model"]:
        kwargs["model"] = state["recall_model"]

    try:
        response = await generate_for_agent_async(
            messages=state["messages"],
            tools=[SEARCH_MEMORIES_TOOL],
            **kwargs,
        )
    except Exception as exc:
        logger.warning(f"[recall_agent] LLM call failed: {exc}")
        return {"done": True}

    message = response.choices[0].message
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]

    new_messages = state["messages"] + [assistant_msg]

    if not message.tool_calls:
        return {"messages": new_messages, "done": True}
    return {"messages": new_messages, "done": False}


async def execute_search(state: RecallState) -> dict:
    """Execute search_memories tool calls from the last assistant message."""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.get("tool_calls", [])

    new_messages = list(state["messages"])
    new_accumulated = list(state["accumulated"])
    seen = set(state["seen_ids"])
    consecutive_empty = state["consecutive_empty"]
    total_new_hits = 0

    for tc in tool_calls:
        func = tc["function"]
        tool_call_id = tc["id"]

        if func["name"] != "search_memories":
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error: unknown tool '{func['name']}'",
                }
            )
            continue

        args = parse_json_from_response(func["arguments"])
        query = (args.get("query", "") if isinstance(args, dict) else "") or ""

        if not query:
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "Error: query parameter is required and must be non-empty.",
                }
            )
            continue

        # Parse optional filters
        time_range = None
        if isinstance(args, dict) and (args.get("time_start") or args.get("time_end")):
            from opencontext.server.search.models import TimeRange

            tr_start = None
            tr_end = None
            try:
                if args.get("time_start"):
                    tr_start = int(
                        datetime.datetime.fromisoformat(args["time_start"])
                        .replace(tzinfo=datetime.UTC)
                        .timestamp()
                    )
                if args.get("time_end"):
                    tr_end = int(
                        datetime.datetime.fromisoformat(args["time_end"])
                        .replace(tzinfo=datetime.UTC)
                        .timestamp()
                    )
            except (ValueError, TypeError):
                pass  # ignore bad dates, search without time filter
            if tr_start is not None or tr_end is not None:
                time_range = TimeRange(start=tr_start, end=tr_end)

        hierarchy_levels = None
        if isinstance(args, dict) and args.get("hierarchy_levels"):
            with contextlib.suppress(ValueError, TypeError):
                hierarchy_levels = [int(lv) for lv in args["hierarchy_levels"]]

        try:
            search_hits, search_ancestors = await _do_search(
                query=query,
                user_id=state["search_params"]["user_id"],
                device_id=state["search_params"]["device_id"],
                agent_id=state["search_params"]["agent_id"],
                time_range=time_range,
                hierarchy_levels=hierarchy_levels,
            )
        except Exception:
            logger.warning("[recall_agent] search raised, returning error to LLM")
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "Error: search service temporarily unavailable.",
                }
            )
            continue

        # Dedup hits against previously seen
        new_hits = [c for c in search_hits if c.id not in seen]
        new_ancestor_hits = [c for c in search_ancestors if c.id not in seen]
        seen.update(c.id for c in new_hits)
        seen.update(c.id for c in new_ancestor_hits)
        new_accumulated.extend(new_hits)
        new_accumulated.extend(new_ancestor_hits)
        total_new_hits += len(new_hits) + len(new_ancestor_hits)

        # Format result text for the LLM
        result_text = _format_search_result(new_hits, new_ancestor_hits)

        new_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_text,
            }
        )

    if total_new_hits == 0:
        consecutive_empty += 1
    else:
        consecutive_empty = 0

    return {
        "messages": new_messages,
        "accumulated": new_accumulated,
        "seen_ids": list(seen),
        "consecutive_empty": consecutive_empty,
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_llm(state: RecallState) -> str:
    return END if state["done"] else "execute_search"


def route_after_search(state: RecallState) -> str:
    return END if state["consecutive_empty"] >= 2 else "call_llm"


# ---------------------------------------------------------------------------
# RecallAgent class
# ---------------------------------------------------------------------------


class RecallAgent:
    """Multi-turn memory recall using LangGraph StateGraph with tool calling."""

    def __init__(self) -> None:
        self._graph = self._build_graph()

    @staticmethod
    def _build_graph():
        builder = StateGraph(RecallState)
        builder.add_node("call_llm", call_llm)
        builder.add_node("execute_search", execute_search)
        builder.add_edge(START, "call_llm")
        builder.add_conditional_edges("call_llm", route_after_llm)
        builder.add_conditional_edges("execute_search", route_after_search)
        return builder.compile()

    async def recall(
        self,
        chat_content: str,
        agent_name: str,
        agent_persona: str,
        user_id: str,
        device_id: str,
        agent_id: str,
    ) -> str:
        """Run the recall loop. Returns formatted related_memories text."""
        cfg = get_config("processing.agent_memory_processor.recall_agent") or {}
        max_turns = int(cfg.get("max_turns", 3))
        recall_model = cfg.get("model") or ""

        # Fetch memory maps for both user events and agent base events
        storage = get_storage()
        if storage:
            user_map, agent_map = await asyncio.gather(
                storage.get_hierarchy_map("user", user_id, device_id, agent_id),
                storage.get_hierarchy_map("agent_base", "__base__", None, agent_id),
            )
        else:
            user_map, agent_map = {3: [], 2: [], 1: []}, {3: [], 2: [], 1: []}

        user_map_text = self._format_map(user_map)
        agent_map_text = self._format_map(agent_map)

        messages = self._build_initial_messages(
            chat_content=chat_content,
            agent_name=agent_name,
            agent_persona=agent_persona,
            user_memory_map=user_map_text,
            agent_memory_map=agent_map_text,
        )
        if messages is None:
            logger.warning("[recall_agent] Prompt missing — aborting")
            return ""

        initial_state: RecallState = {
            "messages": messages,
            "accumulated": [],
            "seen_ids": [],
            "consecutive_empty": 0,
            "search_params": {
                "user_id": user_id,
                "device_id": device_id,
                "agent_id": agent_id,
            },
            "recall_model": recall_model,
            "done": False,
        }

        # Each search turn = 2 supersteps (call_llm + execute_search).
        # +3 headroom covers the final call_llm that terminates, plus buffer.
        # If GraphRecursionError fires, accumulated memories are lost (ainvoke
        # does not return partial state). This is acceptable: with +3 headroom
        # the normal exit paths (route_after_llm / route_after_search) fire
        # before the limit in all expected scenarios.
        recursion_limit = (max_turns * 2) + 3

        try:
            final_state = await self._graph.ainvoke(
                initial_state,
                {"recursion_limit": recursion_limit},
            )
        except GraphRecursionError:
            logger.warning("[recall_agent] Hit recursion limit, returning empty")
            return ""
        except Exception as exc:
            logger.warning(f"[recall_agent] Graph execution failed: {exc}")
            return ""

        return self._format_memories(final_state["accumulated"])

    @staticmethod
    def _build_initial_messages(
        chat_content: str,
        agent_name: str,
        agent_persona: str,
        user_memory_map: str = "",
        agent_memory_map: str = "",
    ) -> list[dict[str, Any]] | None:
        prompt_group = get_prompt_group("processing.extraction.agent_memory_recall")
        if not prompt_group:
            return None
        system_template: str = prompt_group.get("system", "")
        system = (
            system_template.replace("{agent_name}", agent_name)
            .replace("{agent_persona}", agent_persona)
            .replace("{user_memory_map}", user_memory_map)
            .replace("{agent_memory_map}", agent_memory_map)
        )
        user_template: str = prompt_group.get("user", "")
        user = user_template.replace("{chat_history}", chat_content)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _format_map(hierarchy_map: dict[int, list]) -> str:
        """Format a hierarchy map into readable text for the system prompt."""
        if not hierarchy_map or not any(hierarchy_map.values()):
            return "(no history available)"

        lines: list[str] = []

        for level in [3, 2, 1]:
            contexts = hierarchy_map.get(level, [])
            sorted_ctxs = sorted(
                contexts,
                key=lambda c: (
                    c.properties.event_time_start
                    if c.properties and c.properties.event_time_start
                    else datetime.datetime.min.replace(tzinfo=datetime.UTC)
                ),
            )
            for ctx in sorted_ctxs:
                time_range = RecallAgent._format_time_range(ctx)
                title = (ctx.extracted_data.title if ctx.extracted_data else "") or ""
                summary = (ctx.extracted_data.summary if ctx.extracted_data else "") or ""
                if not title and not summary:
                    continue
                entry = f"[{time_range} L{level}] {title}"
                if summary:
                    entry += f"\n  {summary}"
                lines.append(entry)

        return "\n".join(lines) if lines else "(no history available)"

    @staticmethod
    def _format_time_range(ctx: Any) -> str:
        """Format event time range as 'YYYY-MM-DD' or 'YYYY-MM-DD~YYYY-MM-DD'."""
        if not ctx.properties or not ctx.properties.event_time_start:
            return ""
        start = ctx.properties.event_time_start
        end = ctx.properties.event_time_end
        start_str = start.strftime("%Y-%m-%d")
        if not end or end.date() == start.date():
            return start_str
        return f"{start_str}~{end.strftime('%Y-%m-%d')}"

    @staticmethod
    def _format_memories(contexts: list[ProcessedContext]) -> str:
        """Final output — full [date] title + summary text for commentary prompt."""
        if not contexts:
            return ""
        sorted_ctxs = sorted(
            contexts,
            key=lambda c: (
                c.properties.event_time_start
                if c.properties and c.properties.event_time_start
                else datetime.datetime.min.replace(tzinfo=datetime.UTC)
            ),
        )
        lines = []
        for ctx in sorted_ctxs:
            title = (ctx.extracted_data.title if ctx.extracted_data else "") or ""
            summary = (ctx.extracted_data.summary if ctx.extracted_data else "") or ""
            event_time = (
                ctx.properties.event_time_start.strftime("%Y-%m-%d")
                if ctx.properties and ctx.properties.event_time_start
                else ""
            )
            lines.append(f"[{event_time}] {title}")
            if summary:
                lines.append(summary)
            lines.append("")
        return "\n".join(lines).strip()
