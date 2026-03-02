import datetime
from typing import Any, Dict, List, Optional

from opencontext.config.global_config import get_prompt_group
from opencontext.context_processing.processor.base_processor import BaseContextProcessor
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    RawContextProperties,
    Vectorize,
)
from opencontext.models.enums import (
    ContentFormat,
    ContextSource,
    ContextType,
    get_context_type_for_analysis,
)
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TextChatProcessor(BaseContextProcessor):
    def __init__(self):
        super().__init__({})

    def get_name(self) -> str:
        return "text_chat_processor"

    def get_description(self) -> str:
        return "Processes text chat history to extract structured context information."

    def can_process(self, context: RawContextProperties) -> bool:
        return (
            isinstance(context, RawContextProperties)
            and context.source == ContextSource.CHAT_LOG
            and context.content_format == ContentFormat.TEXT
        )

    async def process(self, context: RawContextProperties) -> List[ProcessedContext]:
        """Process chat context asynchronously."""
        logger.debug(f"Processing chat context: {context}")
        try:
            processed_list = await self._process_async(context)
            return processed_list or []
        except Exception as e:
            logger.error(f"Failed to process chat context: {e}")
            return []

    async def _process_async(self, raw_context: RawContextProperties) -> List[ProcessedContext]:
        # 1. 获取 Prompt
        prompt_group = get_prompt_group("processing.extraction.chat_analyze")
        if not prompt_group:
            logger.warning("Prompt 'chat_analyze' not found, using default fallback.")
            return []

        # 2. 准备 LLM 输入
        chat_history_str = raw_context.content_text

        messages = [
            {"role": "system", "content": prompt_group.get("system", "")},
            {
                "role": "user",
                "content": prompt_group.get("user", "").format(
                    chat_history=chat_history_str, current_time=datetime.datetime.now().isoformat()
                ),
            },
        ]
        logger.debug(f"LLM messages: {messages}")

        # 3. 调用 LLM
        response = await generate_with_messages(messages)
        logger.debug(f"LLM response: {response}")

        # 4. 解析结果
        analysis = parse_json_from_response(response)
        if not analysis:
            return []

        # 5. 提取 memories 数组
        memories = analysis.get("memories", [])
        if not memories:
            logger.info("No memories extracted from chat analysis")
            return []

        # 6. 为每条 memory 构建 ProcessedContext
        processed_list = []
        all_structured_entities = []

        for memory in memories:
            try:
                pc = self._build_processed_context(memory, raw_context)
                if pc:
                    processed_list.append(pc)
                    raw_entities = memory.get("entities", [])
                    if isinstance(raw_entities, list):
                        all_structured_entities.extend(
                            e for e in raw_entities if isinstance(e, dict) and "name" in e
                        )
            except Exception as e:
                logger.warning(f"Failed to build ProcessedContext for memory: {e}")

        # 7. 跨记忆实体持久化：确保非 entity 类型记忆中的实体也能进入关系型 DB
        if all_structured_entities and raw_context.user_id:
            await self._persist_entities(all_structured_entities, raw_context)

        logger.debug(f"Extracted {len(processed_list)} memories from chat")
        return processed_list

    def _build_processed_context(
        self, memory: Dict, raw_context: RawContextProperties
    ) -> Optional[ProcessedContext]:
        """Build ProcessedContext for a single memory with input validation."""
        # Validate memory is a dict
        if not isinstance(memory, dict):
            logger.warning(f"Invalid memory format: expected dict, got {type(memory).__name__}")
            return None

        # Validate and sanitize entity names
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

        # Validate context_type
        raw_type = memory.get("context_type", "event")
        if not isinstance(raw_type, str):
            raw_type = "event"
        from opencontext.models.enums import validate_context_type

        if not validate_context_type(raw_type.lower().strip()):
            logger.warning(
                f"LLM returned unrecognized context_type: '{raw_type}', falling back to event"
            )
        context_type = get_context_type_for_analysis(raw_type)

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

        # Validate importance (0-10 range)
        importance = memory.get("importance", 5)
        try:
            importance = int(importance)
        except (ValueError, TypeError):
            importance = 5
        importance = max(0, min(10, importance))

        # Validate confidence (0-10 range)
        confidence = memory.get("confidence", 0)
        try:
            confidence = int(confidence)
        except (ValueError, TypeError):
            confidence = 0
        confidence = max(0, min(10, confidence))

        # 解析 LLM 返回的 event_time，回退到 raw_context.create_time
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

        # 仅 knowledge 类型启用合并
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

        return ProcessedContext(
            properties=ContextProperties(
                raw_properties=[raw_context],
                create_time=raw_context.create_time,
                update_time=datetime.datetime.now(),
                event_time=event_time,
                is_processed=True,
                enable_merge=enable_merge,
                user_id=raw_context.user_id,
                device_id=raw_context.device_id,
                agent_id=raw_context.agent_id,
            ),
            extracted_data=extracted_data,
            vectorize=Vectorize(
                content_format=ContentFormat.TEXT,
                text=f"{extracted_data.title}\n{extracted_data.summary}\n{' '.join(extracted_data.keywords)}",
                metadata={"structured_entities": raw_entities},
            ),
        )

    async def _persist_entities(
        self, structured_entities: List[Dict], raw_context: RawContextProperties
    ) -> None:
        """
        将所有记忆中提取的实体持久化到关系型 DB。
        确保非 entity 类型记忆中提到的实体也能进入实体表。
        """
        try:
            from opencontext.context_processing.processor.entity_processor import (
                refresh_entities,
                validate_and_clean_entities,
            )

            entities_info = validate_and_clean_entities(structured_entities)
            if entities_info:
                await refresh_entities(
                    entities_info=entities_info,
                    context_text=raw_context.content_text or "",
                    user_id=raw_context.user_id or "default",
                    device_id=raw_context.device_id or "default",
                    agent_id=raw_context.agent_id or "default",
                )
                logger.debug(f"Persisted {len(entities_info)} entities from chat analysis")
        except Exception as e:
            logger.warning(f"Entity persistence failed (non-fatal): {e}")
