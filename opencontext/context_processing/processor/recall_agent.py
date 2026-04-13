"""Recall Agent — multi-turn LLM-driven memory recall using LangGraph StateGraph."""

import datetime
from typing import Any, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from opencontext.config.global_config import get_config, get_prompt_group
from opencontext.llm.global_vlm_client import generate_for_agent_async
from opencontext.models.context import ProcessedContext
from opencontext.server.search.event_search_service import EventSearchService
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
            "Search past memories/events for the given query. "
            "Returns a summary of matching memories."
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
            },
            "required": ["query"],
        },
    },
}

# ---------------------------------------------------------------------------
# Helper: _do_search
# ---------------------------------------------------------------------------


async def _do_search(
    query: str, user_id: str, device_id: str, agent_id: str
) -> list[ProcessedContext]:
    """Execute one search. Exceptions propagate to caller."""
    service = EventSearchService()
    result = await service.search(
        query=[{"type": "text", "text": query}],
        user_id=user_id,
        device_id=device_id,
        agent_id=agent_id,
    )
    if not result:
        return []
    collected: dict[str, ProcessedContext] = {}
    for ctx, _score in result.hits:
        collected[ctx.id] = ctx
    for ctx_id, ctx in result.ancestors.items():
        if ctx_id not in collected:
            collected[ctx_id] = ctx
    return list(collected.values())


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

        try:
            new_contexts = await _do_search(
                query=query,
                user_id=state["search_params"]["user_id"],
                device_id=state["search_params"]["device_id"],
                agent_id=state["search_params"]["agent_id"],
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

        new_hits = [c for c in new_contexts if c.id not in seen]
        seen.update(c.id for c in new_hits)
        new_accumulated.extend(new_hits)
        total_new_hits += len(new_hits)

        if new_hits:
            brief_lines = []
            for ctx in new_hits:
                title = (ctx.extracted_data.title if ctx.extracted_data else "") or "Untitled"
                event_time = (
                    ctx.properties.event_time_start.strftime("%Y-%m-%d")
                    if ctx.properties and ctx.properties.event_time_start
                    else ""
                )
                brief_lines.append(f"[{event_time}] {title}")
            result_text = f"Found {len(new_hits)} new memories:\n" + "\n".join(brief_lines)
        else:
            result_text = "No new memories found for this query."

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

        messages = self._build_initial_messages(
            chat_content=chat_content,
            agent_name=agent_name,
            agent_persona=agent_persona,
            max_turns=max_turns,
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
        max_turns: int,
    ) -> list[dict[str, Any]] | None:
        prompt_group = get_prompt_group("processing.extraction.agent_memory_recall")
        if not prompt_group:
            return None
        system_template: str = prompt_group.get("system", "")
        system = (
            system_template.replace("{agent_name}", agent_name)
            .replace("{agent_persona}", agent_persona)
            .replace("{max_turns}", str(max_turns))
        )
        user_template: str = prompt_group.get("user", "")
        user = user_template.replace("{chat_history}", chat_content)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

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
