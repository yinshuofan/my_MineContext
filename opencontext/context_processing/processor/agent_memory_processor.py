"""Agent Memory Processor — extracts memories from the agent's perspective."""

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
        # 1. Load agent info
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
        agent_description = agent.get("description", "")

        # 2. Load prompt
        prompt_group = get_prompt_group("processing.extraction.agent_memory_analyze")
        if not prompt_group:
            logger.warning("agent_memory_analyze prompt not found")
            return []

        # 3. Build LLM messages
        system_prompt = prompt_group.get("system", "")
        system_prompt = system_prompt.replace("{agent_name}", agent_name)
        system_prompt = system_prompt.replace("{agent_description}", agent_description)

        chat_history = raw_context.content_text or ""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt_group.get("user", "").format(
                    chat_history=chat_history,
                    current_time=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
                ),
            },
        ]

        # 4. Call LLM (disable tool executor — pure extraction)
        response = await generate_with_messages(messages, enable_executor=False)
        if not response:
            return []

        # 5. Parse response
        analysis = parse_json_from_response(response)
        if not analysis:
            return []

        memories = analysis.get("memories", [])
        if not memories:
            logger.info("No agent memories extracted from chat analysis")
            return []

        # 6. Build ProcessedContext for each memory
        batch_id = (raw_context.additional_info or {}).get("batch_id")
        results = []
        for memory in memories:
            ctx = self._build_agent_context(memory, raw_context, batch_id)
            if ctx:
                await do_vectorize(ctx.vectorize)
                results.append(ctx)

        logger.debug(f"Extracted {len(results)} agent memories from chat")
        return results

    def _build_agent_context(
        self, memory: Dict, raw_context: RawContextProperties, batch_id: Optional[str]
    ) -> Optional[ProcessedContext]:
        """Build ProcessedContext for a single agent memory with input validation."""
        # Validate memory is a dict
        if not isinstance(memory, dict):
            logger.warning(f"Invalid memory format: expected dict, got {type(memory).__name__}")
            return None

        try:
            # Validate context_type
            mem_type = memory.get("type", "agent_event")
            if not isinstance(mem_type, str):
                mem_type = "agent_event"
            mem_type = mem_type.lower().strip()

            if mem_type == "profile":
                context_type = ContextType.PROFILE
            else:
                context_type = ContextType.AGENT_EVENT

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
                metadata={"owner_type": "agent"},
            )

        except Exception as e:
            logger.warning(f"Failed to build agent context: {e}")
            return None
