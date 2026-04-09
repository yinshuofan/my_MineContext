"""Agent Memory Processor — post-processor that annotates events with agent commentary."""

import asyncio
import datetime
from typing import Any, List, Optional

from opencontext.config.global_config import get_prompt_group
from opencontext.context_processing.processor.base_processor import BaseContextProcessor
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.context import ProcessedContext, RawContextProperties
from opencontext.models.enums import ContextSource, ContextType
from opencontext.server.search.event_search_service import EventSearchService
from opencontext.storage.global_storage import get_storage
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.time_utils import now as tz_now

logger = get_logger(__name__)


class AgentMemoryProcessor(BaseContextProcessor):
    """Post-processor that writes agent commentary onto events from prior_results."""

    def __init__(self):
        super().__init__({})

    def get_name(self) -> str:
        return "agent_memory_processor"

    def get_description(self) -> str:
        return "Annotates events with agent's subjective commentary."

    def can_process(self, context: Any) -> bool:
        if not isinstance(context, RawContextProperties):
            return False
        return context.source == ContextSource.CHAT_LOG

    async def process(
        self,
        context: RawContextProperties,
        prior_results: Optional[List[ProcessedContext]] = None,
    ) -> List[ProcessedContext]:
        """Annotate events from prior_results with agent commentary.

        Returns the modified event contexts (same IDs, with agent_commentary populated).
        Does NOT produce new contexts.
        """
        try:
            return await self._process_async(context, prior_results or [])
        except Exception as e:
            logger.error(f"Agent memory processing failed: {e}")
            return []

    async def _process_async(
        self,
        raw_context: RawContextProperties,
        prior_results: List[ProcessedContext],
    ) -> List[ProcessedContext]:
        # 0. Validate agent_id
        agent_id = raw_context.agent_id
        if not agent_id or agent_id == "default":
            logger.debug("No agent_id in context, skipping agent memory processing")
            return []

        storage = get_storage()
        agent = await storage.get_agent(agent_id) if storage else None
        if not agent:
            logger.debug(f"Agent {agent_id} not registered, skipping")
            return []

        # 1. Filter events from prior_results
        events = [
            ctx for ctx in prior_results if ctx.extracted_data.context_type == ContextType.EVENT
        ]
        if not events:
            logger.debug("[agent_memory_processor] No events in prior_results, skipping")
            return []

        agent_name = agent.get("name", agent_id)
        chat_content = raw_context.content_text or ""

        # 2. Parallel: fetch persona + extract search query
        profile_task = storage.get_profile(
            user_id=raw_context.user_id,
            device_id=raw_context.device_id or "default",
            agent_id=raw_context.agent_id,
            context_type="agent_profile",
        )
        query_task = self._extract_search_query(chat_content)

        profile_result, query_text = await asyncio.gather(profile_task, query_task)

        # Fallback to base profile if per-user profile doesn't exist
        if not profile_result:
            profile_result = await storage.get_profile(
                user_id="__base__",
                device_id=raw_context.device_id or "default",
                agent_id=raw_context.agent_id,
                context_type="agent_base_profile",
            )

        if not profile_result:
            logger.error(
                f"[agent_memory_processor] Agent profile not found for "
                f"user={raw_context.user_id}, agent={agent_id} (also checked __base__). "
                f"Agent must have a profile set up before agent memory processing."
            )
            return []

        agent_persona = profile_result.get("factual_profile", "")

        # 3. Search related past memories
        related_memories_text = ""
        if query_text:
            search_service = EventSearchService()
            search_result = await search_service.semantic_search(
                query=[{"type": "text", "text": query_text}],
                user_id=raw_context.user_id,
                device_id=raw_context.device_id or "default",
                agent_id=raw_context.agent_id,
            )
            if search_result:
                related_memories_text = self._format_related_memories(search_result)

        # 4. Build event list for prompt
        event_list = self._format_event_list(events)

        # 5. Load prompt and build LLM messages
        prompt_group = get_prompt_group("processing.extraction.agent_memory_analyze")
        if not prompt_group:
            logger.warning("agent_memory_analyze prompt not found")
            return []

        logger.debug(
            f"[agent_memory_processor] Processing: user={raw_context.user_id}, "
            f"agent={raw_context.agent_id}, events={len(events)}"
        )

        system_prompt = prompt_group.get("system", "")
        system_prompt = system_prompt.replace("{agent_name}", agent_name)
        system_prompt = system_prompt.replace("{agent_persona}", agent_persona)
        system_prompt = system_prompt.replace("{related_memories}", related_memories_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt_group.get("user", "").format(
                    current_time=tz_now().isoformat(),
                    event_list=event_list,
                    chat_history=chat_content,
                ),
            },
        ]

        # 6. Call LLM
        response = await generate_with_messages(messages, enable_executor=False)
        logger.debug(f"[agent_memory_processor] LLM response: {response}")
        if not response:
            return []

        # 7. Parse and apply commentaries
        analysis = parse_json_from_response(response)
        if not analysis:
            return []

        commentaries = analysis.get("commentaries", {})
        if not commentaries:
            logger.info("[agent_memory_processor] No commentaries from LLM")
            return []

        modified = []
        for idx_str, commentary in commentaries.items():
            try:
                idx = int(idx_str)
            except (ValueError, TypeError):
                continue
            if idx < 0 or idx >= len(events):
                continue
            if commentary and commentary != "null":
                events[idx].extracted_data.agent_commentary = str(commentary).strip()
                modified.append(events[idx])

        logger.info(f"[agent_memory_processor] Annotated {len(modified)}/{len(events)} events")
        return modified

    async def _extract_search_query(self, chat_content: str) -> Optional[str]:
        """Use LLM to extract a search query from chat content (from AI's perspective)."""
        try:
            prompt_group = get_prompt_group("processing.extraction.agent_memory_query")
            if not prompt_group:
                return None
            messages = [
                {"role": "system", "content": prompt_group.get("system", "")},
                {
                    "role": "user",
                    "content": prompt_group.get("user", "").format(chat_history=chat_content),
                },
            ]
            response = await generate_with_messages(messages, enable_executor=False)
            if not response or not response.strip():
                return None
            return response.strip()
        except Exception as e:
            logger.warning(f"[agent_memory_processor] Query extraction failed: {e}")
            return None

    @staticmethod
    def _format_related_memories(search_result) -> str:
        """Format search results into text for the LLM prompt."""

        all_contexts = {}
        for ctx, score in search_result.hits:
            all_contexts[ctx.id] = ctx
        for ctx_id, ctx in search_result.ancestors.items():
            if ctx_id not in all_contexts:
                all_contexts[ctx_id] = ctx

        if not all_contexts:
            return ""

        sorted_contexts = sorted(
            all_contexts.values(),
            key=lambda c: c.properties.event_time_start if c.properties else datetime.datetime.min,
        )

        lines = []
        for ctx in sorted_contexts:
            title = ctx.extracted_data.title if ctx.extracted_data else ""
            summary = ctx.extracted_data.summary if ctx.extracted_data else ""
            event_time_str = (
                ctx.properties.event_time_start.strftime("%Y-%m-%d") if ctx.properties else ""
            )
            lines.append(f"[{event_time_str}] {title}")
            if summary:
                lines.append(summary)
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _format_event_list(events: List[ProcessedContext]) -> str:
        """Format events into a numbered list for the LLM prompt."""
        lines = []
        for i, ctx in enumerate(events):
            title = ctx.extracted_data.title or "Untitled"
            summary = ctx.extracted_data.summary or ""
            lines.append(f"[Event {i}] {title}")
            if summary:
                lines.append(summary)
            lines.append("")
        return "\n".join(lines).strip()
