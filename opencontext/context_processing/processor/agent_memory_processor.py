"""Agent Memory Processor — extracts memories from the agent's perspective."""

import asyncio
import datetime
from typing import Any, Dict, List, Optional

from opencontext.config.global_config import get_prompt_group
from opencontext.context_processing.processor.base_processor import BaseContextProcessor
from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    RawContextProperties,
    Vectorize,
)
from opencontext.models.enums import ContentFormat, ContextSource, ContextType
from opencontext.storage.global_storage import get_storage
from opencontext.server.search.event_search_service import EventSearchService
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AgentMemoryProcessor(BaseContextProcessor):
    def __init__(self):
        super().__init__({})

    def get_name(self) -> str:
        return "agent_memory_processor"

    def get_description(self) -> str:
        return "Extracts memories from the agent's subjective perspective."

    def can_process(self, context: Any) -> bool:
        if not isinstance(context, RawContextProperties):
            return False
        return context.source == ContextSource.CHAT_LOG

    async def process(self, context: RawContextProperties) -> List[ProcessedContext]:
        """Process chat context from the agent's perspective."""
        try:
            return await self._process_async(context)
        except Exception as e:
            logger.error(f"Agent memory processing failed: {e}")
            return []

    async def _process_async(self, raw_context: RawContextProperties) -> List[ProcessedContext]:
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

        agent_name = agent.get("name", agent_id)
        chat_content = raw_context.content_text or ""

        # 1. Parallel: fetch persona + extract search query
        profile_task = storage.get_profile(
            user_id=raw_context.user_id,
            device_id=raw_context.device_id or "default",
            agent_id=raw_context.agent_id,
            context_type="agent_profile",
        )
        query_task = self._extract_search_query(chat_content)

        profile_result, query_text = await asyncio.gather(profile_task, query_task)

        if not profile_result:
            logger.error(
                f"[agent_memory_processor] Agent profile not found for "
                f"user={raw_context.user_id}, agent={agent_id}. "
                f"Agent must have a profile set up before agent memory processing."
            )
            return []

        agent_persona = profile_result.get("factual_profile", "")

        # 2. Search related past agent memories
        search_result = None
        if query_text:
            search_service = EventSearchService()
            search_result = await search_service.semantic_search(
                query=[{"type": "text", "text": query_text}],
                user_id=raw_context.user_id,
                device_id=raw_context.device_id or "default",
                agent_id=raw_context.agent_id,
                memory_owner="agent",
                drill_up=True,
            )

        # 3. Format related memories
        related_memories_text = ""
        if search_result:
            related_memories_text = self._format_related_memories(search_result)

        # 4. Load prompt and build LLM messages
        prompt_group = get_prompt_group("processing.extraction.agent_memory_analyze")
        if not prompt_group:
            logger.warning("agent_memory_analyze prompt not found")
            return []

        logger.debug(
            f"[agent_memory_processor] Processing: user={raw_context.user_id}, "
            f"agent={raw_context.agent_id}, agent_name={agent_name}, "
            f"related_memories={len(related_memories_text)} chars"
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
                    chat_history=chat_content,
                    current_time=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
                ),
            },
        ]

        # 5. Call LLM
        response = await generate_with_messages(messages, enable_executor=False)
        logger.debug(f"[agent_memory_processor] LLM response: {response}")
        if not response:
            return []

        # 6. Parse and build contexts (unchanged)
        analysis = parse_json_from_response(response)
        if not analysis:
            return []

        memories = analysis.get("memories", [])
        if not memories:
            logger.info("[agent_memory_processor] No memories extracted from chat analysis")
            return []

        batch_id = (raw_context.additional_info or {}).get("batch_id")
        results = []
        for memory in memories:
            ctx = self._build_agent_context(memory, raw_context, batch_id)
            if ctx:
                await do_vectorize(ctx.vectorize)
                results.append(ctx)

        type_counts = {}
        for ctx in results:
            t = ctx.extracted_data.context_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info(f"[agent_memory_processor] Extracted {len(results)} memories: {type_counts}")
        return results

    async def _extract_search_query(self, chat_content: str) -> Optional[str]:
        """Use LLM to extract a search query from chat content (from AI's perspective)."""
        prompt_group = get_prompt_group("processing.extraction.agent_memory_query")
        if not prompt_group:
            logger.warning("agent_memory_query prompt not found")
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

    @staticmethod
    def _format_related_memories(search_result) -> str:
        """Format search results (hits + ancestors) into text for the LLM prompt."""
        all_contexts = {}

        # Merge hits and ancestors, deduplicate by ID
        for ctx, score in search_result.hits:
            all_contexts[ctx.id] = ctx
        for ctx_id, ctx in search_result.ancestors.items():
            if ctx_id not in all_contexts:
                all_contexts[ctx_id] = ctx

        if not all_contexts:
            return ""

        # Sort by time_bucket ascending
        sorted_contexts = sorted(
            all_contexts.values(),
            key=lambda c: (c.properties.time_bucket or "") if c.properties else "",
        )

        lines = []
        for ctx in sorted_contexts:
            title = ctx.extracted_data.title if ctx.extracted_data else ""
            summary = ctx.extracted_data.summary if ctx.extracted_data else ""
            time_bucket = ctx.properties.time_bucket if ctx.properties else ""
            lines.append(f"[{time_bucket}] {title}")
            if summary:
                lines.append(summary)
            lines.append("")  # blank line between entries

        return "\n".join(lines).strip()

    def _build_agent_context(
        self, memory: Dict, raw_context: RawContextProperties, batch_id: Optional[str]
    ) -> Optional[ProcessedContext]:
        """Build ProcessedContext for a single agent memory with input validation."""
        # Validate memory is a dict
        if not isinstance(memory, dict):
            logger.warning(f"Invalid memory format: expected dict, got {type(memory).__name__}")
            return None

        try:
            # Validate context_type — read both "type" and "context_type" fields
            mem_type = memory.get("type") or memory.get("context_type") or "agent_event"
            if not isinstance(mem_type, str):
                mem_type = "agent_event"
            mem_type = mem_type.lower().strip()

            if mem_type == "agent_profile":
                context_type = ContextType.AGENT_PROFILE
            elif mem_type == "agent_event":
                context_type = ContextType.AGENT_EVENT
            else:
                logger.warning(f"[agent_memory_processor] Unknown type: {mem_type}, skipping")
                return None

            # Validate and sanitize title
            title = memory.get("title", "")
            if not isinstance(title, str) or not title.strip():
                title = "Untitled"
            title = title.strip()[:500]

            # Validate and sanitize summary
            summary = memory.get("summary", "")
            if not isinstance(summary, str):
                summary = str(summary) if summary else ""

            # Validate and sanitize keywords
            keywords = memory.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(k).strip()[:100] for k in keywords if k and str(k).strip()][:20]

            # Validate and sanitize entities
            raw_entities = memory.get("entities", [])
            entity_names = []
            if isinstance(raw_entities, list):
                for e in raw_entities:
                    if isinstance(e, str):
                        name = e.strip()[:255]
                        if name:
                            entity_names.append(name)
                    elif isinstance(e, dict):
                        name = str(e.get("name", "")).strip()[:255]
                        if name:
                            entity_names.append(name)

            # Validate importance (0-10 range)
            importance = memory.get("importance", 5)
            try:
                importance = int(importance)
            except (ValueError, TypeError):
                importance = 5
            importance = max(0, min(10, importance))

            # Validate confidence (0-10 range)
            confidence = memory.get("confidence", 7)
            try:
                confidence = int(confidence)
            except (ValueError, TypeError):
                confidence = 7
            confidence = max(0, min(10, confidence))

            # Parse event_time, fall back to raw_context.create_time
            event_time = raw_context.create_time
            event_time_str = memory.get("event_time")
            if event_time_str:
                try:
                    parsed = datetime.datetime.fromisoformat(event_time_str)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
                    event_time = parsed
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse event_time: {event_time_str}")

            if event_time is None:
                event_time = datetime.datetime.now(tz=datetime.timezone.utc)

            # Only knowledge type enables merge
            enable_merge = context_type == ContextType.KNOWLEDGE

            extracted_data = ExtractedData(
                title=title,
                summary=summary,
                keywords=keywords,
                entities=entity_names,
                context_type=context_type,
                importance=importance,
                confidence=confidence,
            )

            # L0 time_bucket: ISO datetime for per-event granularity
            time_bucket = event_time.strftime("%Y-%m-%dT%H:%M:%S")

            properties = ContextProperties(
                raw_properties=[raw_context],
                create_time=raw_context.create_time or datetime.datetime.now(tz=datetime.timezone.utc),
                update_time=datetime.datetime.now(tz=datetime.timezone.utc),
                event_time=event_time,
                time_bucket=time_bucket,
                is_processed=True,
                enable_merge=enable_merge,
                user_id=raw_context.user_id,
                device_id=raw_context.device_id,
                agent_id=raw_context.agent_id,
                raw_type="chat_batch" if batch_id else None,
                raw_id=batch_id,
            )

            text = f"{title}\n{summary}\n{', '.join(keywords)}"
            vectorize = Vectorize(
                input=[{"type": "text", "text": text}],
                content_format=ContentFormat.TEXT,
            )

            return ProcessedContext(
                properties=properties,
                extracted_data=extracted_data,
                vectorize=vectorize,
                metadata={},
            )

        except Exception as e:
            logger.warning(f"Failed to build agent context: {e}")
            return None
