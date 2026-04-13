"""Recall Agent — multi-turn LLM-driven loop that gathers past memories for commentary."""

import datetime
from dataclasses import dataclass, field
from typing import Any

from opencontext.config.global_config import get_config, get_prompt_group
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.context import ProcessedContext
from opencontext.server.search.event_search_service import EventSearchService
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class _RecallState:
    """Per-call state for the recall loop."""

    max_turns: int
    recall_model: str | None
    turn: int = 0
    seen_ids: set[str] = field(default_factory=set)
    accumulated: list[ProcessedContext] = field(default_factory=list)
    previous_actions: list[dict[str, Any]] = field(default_factory=list)
    consecutive_empty: int = 0


class RecallAgent:
    """Multi-turn memory recall loop driven by an LLM-selected search action."""

    def __init__(self) -> None:
        # No config cached — read fresh in recall() for hot reload support.
        pass

    async def recall(
        self,
        chat_content: str,
        agent_name: str,
        agent_persona: str,
        user_id: str,
        device_id: str,
        agent_id: str,
    ) -> str:
        """Run the recall loop. Returns formatted related_memories text.

        Returns an empty string when nothing was recalled (including first-turn
        aborts). Downstream commentary generation handles empty gracefully.
        """
        state = self._read_config()
        while state.turn < state.max_turns:
            messages = self._build_turn_messages(
                chat_content=chat_content,
                agent_name=agent_name,
                agent_persona=agent_persona,
                max_turns=state.max_turns,
                previous_actions=state.previous_actions,
                accumulated=state.accumulated,
            )
            if messages is None:
                logger.warning("[recall_agent] Prompt missing — aborting")
                break

            action = await self._decide_action(messages, state.recall_model)
            if action is None:
                logger.warning(f"[recall_agent] turn={state.turn}: action parse failed, breaking")
                break

            action_type = action.get("action")
            if action_type == "done":
                logger.info(
                    f"[recall_agent] turn={state.turn}: agent stopped "
                    f"(reason={action.get('reason')!r})"
                )
                break
            if action_type != "search":
                logger.warning(
                    f"[recall_agent] turn={state.turn}: unknown action {action_type!r}, breaking"
                )
                break

            query = action.get("query") or ""
            if not query:
                logger.warning(
                    f"[recall_agent] turn={state.turn}: search action missing query, breaking"
                )
                break

            try:
                new_contexts = await self._execute_search(
                    query=query,
                    user_id=user_id,
                    device_id=device_id,
                    agent_id=agent_id,
                )
            except Exception:
                logger.warning(f"[recall_agent] turn={state.turn}: search raised, breaking")
                break

            new_hits = [c for c in new_contexts if c.id not in state.seen_ids]
            state.seen_ids.update(c.id for c in new_hits)
            state.accumulated.extend(new_hits)

            state.previous_actions.append(
                {
                    "turn": state.turn,
                    "query": query,
                    "reason": action.get("reason", ""),
                    "new_hits": len(new_hits),
                }
            )

            state.turn += 1

        return self._format_memories(state.accumulated)

    def _read_config(self) -> _RecallState:
        cfg = get_config("processing.agent_memory_processor.recall_agent") or {}
        return _RecallState(
            max_turns=int(cfg.get("max_turns", 3)),
            recall_model=(cfg.get("model") or None),
        )

    async def _decide_action(
        self, messages: list[dict[str, Any]], recall_model: str | None
    ) -> dict[str, Any] | None:
        """One LLM call. Returns parsed action dict or None on parse failure."""
        kwargs: dict[str, Any] = {"enable_executor": False}
        if recall_model:
            kwargs["model"] = recall_model
        try:
            response = await generate_with_messages(messages, **kwargs)
        except Exception as exc:
            logger.warning(f"[recall_agent] LLM call failed: {exc}")
            return None
        if not response:
            return None
        parsed = parse_json_from_response(response)
        if not isinstance(parsed, dict):
            return None
        return parsed

    async def _execute_search(
        self, query: str, user_id: str, device_id: str, agent_id: str
    ) -> list[ProcessedContext]:
        """One search call. Returns flat list of hits + ancestors."""
        try:
            service = EventSearchService()
            result = await service.search(
                query=[{"type": "text", "text": query}],
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            )
        except Exception as exc:
            logger.warning(f"[recall_agent] Search failed: {exc}")
            raise
        if not result:
            return []
        collected: dict[str, ProcessedContext] = {}
        for ctx, _score in result.hits:
            collected[ctx.id] = ctx
        for ctx_id, ctx in result.ancestors.items():
            if ctx_id not in collected:
                collected[ctx_id] = ctx
        return list(collected.values())

    def _build_turn_messages(
        self,
        chat_content: str,
        agent_name: str,
        agent_persona: str,
        max_turns: int,
        previous_actions: list[dict[str, Any]],
        accumulated: list[ProcessedContext],
    ) -> list[dict[str, Any]] | None:
        prompt_group = get_prompt_group("processing.extraction.agent_memory_recall")
        if not prompt_group:
            logger.warning("[recall_agent] agent_memory_recall prompt not found")
            return None
        system_template: str = prompt_group.get("system", "")
        system = (
            system_template.replace("{agent_name}", agent_name)
            .replace("{agent_persona}", agent_persona)
            .replace("{max_turns}", str(max_turns))
        )
        user_template: str = prompt_group.get("user", "")
        user = user_template.format(
            chat_history=chat_content,
            previous_actions=self._format_previous_actions(previous_actions),
            accumulated_brief=self._format_brief(accumulated),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _format_previous_actions(previous_actions: list[dict[str, Any]]) -> str:
        if not previous_actions:
            return "(none yet)"
        lines = []
        for entry in previous_actions:
            lines.append(
                f"[turn {entry['turn']}] query={entry['query']!r}, new_hits={entry['new_hits']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_brief(accumulated: list[ProcessedContext]) -> str:
        if not accumulated:
            return "(none yet)"
        lines = []
        for ctx in accumulated:
            title = (ctx.extracted_data.title if ctx.extracted_data else "") or "Untitled"
            event_time = (
                ctx.properties.event_time_start.strftime("%Y-%m-%d")
                if ctx.properties and ctx.properties.event_time_start
                else ""
            )
            lines.append(f"[{event_time}] {title}")
        return "\n".join(lines)

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
                else datetime.datetime.min
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
