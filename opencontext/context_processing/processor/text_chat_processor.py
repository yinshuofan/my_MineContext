import json
from typing import Any

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
from opencontext.utils.time_utils import now as tz_now

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
            and context.content_format in (ContentFormat.TEXT, ContentFormat.MULTIMODAL)
        )

    async def process(
        self, context: RawContextProperties, prior_results=None
    ) -> list[ProcessedContext]:
        """Process chat context asynchronously."""
        logger.debug(
            f"[text_chat_processor] Processing: user={context.user_id}, "
            f"agent={context.agent_id}, source={context.source}, "
            f"content_length={len(context.content_text or '')}"
        )
        try:
            processed_list = await self._process_async(context)
            return processed_list or []
        except Exception as e:
            logger.error(f"Failed to process chat context: {e}")
            return []

    async def _process_async(self, raw_context: RawContextProperties) -> list[ProcessedContext]:
        # 1. 获取 Prompt
        prompt_group = get_prompt_group("processing.extraction.chat_analyze")
        if not prompt_group:
            logger.warning("Prompt 'chat_analyze' not found, using default fallback.")
            return []

        # 2. 准备 LLM 输入
        chat_history_str = raw_context.content_text
        is_multimodal = raw_context.content_format == ContentFormat.MULTIMODAL

        # Build media index for mapping LLM-returned related_media indices back to URLs
        media_index = []  # Ordered list of {"type": "image"|"video", "url": "..."}
        if is_multimodal:
            media_index = self._build_media_index(chat_history_str)

        if is_multimodal:
            # For multimodal messages, pass the original messages directly to the LLM
            # so it can see images and videos, producing more accurate memory extraction
            messages = self._build_multimodal_llm_messages(prompt_group, chat_history_str)
        else:
            messages = [
                {"role": "system", "content": prompt_group.get("system", "")},
                {
                    "role": "user",
                    "content": prompt_group.get("user", "").format(
                        chat_history=chat_history_str,
                        current_time=tz_now().isoformat(),
                    ),
                },
            ]
        logger.debug(f"[text_chat_processor] LLM messages prepared (multimodal={is_multimodal})")

        # 3. 调用 LLM
        response = await generate_with_messages(messages)
        logger.debug(f"[text_chat_processor] LLM response: {response}")

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
        batch_id = (raw_context.additional_info or {}).get("batch_id")
        processed_list = []

        for memory in memories:
            try:
                pc = self._build_processed_context(
                    memory, raw_context, media_index=media_index, batch_id=batch_id
                )
                if pc:
                    processed_list.append(pc)
            except Exception as e:
                logger.warning(f"Failed to build ProcessedContext for memory: {e}")

        logger.debug(f"[text_chat_processor] Extracted {len(processed_list)} memories from chat")
        return processed_list

    @staticmethod
    def _build_media_index(chat_history_str: str) -> list[dict[str, str]]:
        """
        Build an ordered index of all media items across all messages.

        Scans through all messages' content parts, collecting image_url and video_url
        items in order. Returns a list of {"type": "image"|"video", "url": "..."} dicts.
        The index positions correspond to what the LLM returns in related_media.
        """
        media_index = []
        try:
            chat_messages = json.loads(chat_history_str)
            for msg in chat_messages:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    part_type = part.get("type", "")
                    if part_type == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url:
                            media_index.append({"type": "image", "url": url})
                    elif part_type == "video_url":
                        url = part.get("video_url", {}).get("url", "")
                        if url:
                            media_index.append({"type": "video", "url": url})
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse chat_history for media index: {e}")
        return media_index

    @staticmethod
    def _build_multimodal_llm_messages(
        prompt_group: dict[str, str], chat_history_str: str
    ) -> list[dict[str, Any]]:
        """
        Build LLM messages for multimodal chat analysis.

        The system prompt is sent as the first message. Then, the original multimodal
        chat messages are included directly so the LLM can see images/videos.
        Finally, a user message with the analysis instruction is appended.

        Local file paths in media URLs are converted to base64 data URIs so the
        remote LLM API can access the content.
        """

        llm_messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt_group.get("system", "")},
        ]

        # Include the original multimodal chat messages directly
        try:
            chat_messages = json.loads(chat_history_str)
            for msg in chat_messages:
                content = msg.get("content", "")
                # Convert local file paths in content parts to base64 data URIs
                if isinstance(content, list):
                    content = _convert_local_paths_in_content(content)
                llm_messages.append({"role": msg.get("role", "user"), "content": content})
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse multimodal messages, falling back to text: {e}")
            llm_messages.append({"role": "user", "content": chat_history_str})

        # Append the analysis instruction
        user_prompt = prompt_group.get("user", "").format(
            chat_history="[Messages included above]",
            current_time=tz_now().isoformat(),
        )
        llm_messages.append({"role": "user", "content": user_prompt})

        return llm_messages

    def _build_processed_context(
        self,
        memory: dict,
        raw_context: RawContextProperties,
        media_index: list[dict[str, str]] | None = None,
        batch_id: str | None = None,
    ) -> ProcessedContext | None:
        """Build ProcessedContext for a single memory with input validation.

        Args:
            memory: Extracted memory dict from LLM.
            raw_context: The original raw context.
            media_index: Ordered list of media items from multimodal messages.
                         Each item is {"type": "image"|"video", "url": "..."}.
                         Used to resolve related_media indices from LLM output.
            batch_id: Optional chat batch ID for traceability.
        """
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

        event_time_start = raw_context.create_time

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

        # Resolve related media references from LLM output
        vectorize_images = None
        vectorize_videos = None
        vectorize_format = ContentFormat.TEXT
        media_refs = []

        related_media = memory.get("related_media")
        if related_media and isinstance(related_media, dict) and media_index:
            image_indices = related_media.get("images", [])
            video_indices = related_media.get("videos", [])

            # Resolve image indices to URLs
            if isinstance(image_indices, list):
                resolved_images = []
                for idx in image_indices:
                    if isinstance(idx, int) and 0 <= idx < len(media_index):
                        item = media_index[idx]
                        if item["type"] == "image":
                            resolved_images.append(item["url"])
                            media_refs.append({"type": "image", "url": item["url"]})
                if resolved_images:
                    vectorize_images = resolved_images

            # Resolve video indices to URLs
            if isinstance(video_indices, list):
                resolved_videos = []
                for idx in video_indices:
                    if isinstance(idx, int) and 0 <= idx < len(media_index):
                        item = media_index[idx]
                        if item["type"] == "video":
                            resolved_videos.append(item["url"])
                            media_refs.append({"type": "video", "url": item["url"]})
                if resolved_videos:
                    vectorize_videos = resolved_videos

            if vectorize_images or vectorize_videos:
                vectorize_format = ContentFormat.MULTIMODAL

        # Build metadata
        metadata = {}
        if media_refs:
            metadata["media_refs"] = media_refs
            modalities = ["text"]
            if vectorize_images:
                modalities.append("image")
            if vectorize_videos:
                modalities.append("video")
            metadata["content_modalities"] = ",".join(modalities)

        # Build content parts list for Vectorize
        ark_input = [
            {
                "type": "text",
                "text": (
                    f"{extracted_data.title}\n{extracted_data.summary}\n"
                    f"{' '.join(extracted_data.keywords)}"
                ),
            }
        ]
        if vectorize_images:
            for img_url in vectorize_images:
                ark_input.append({"type": "image_url", "image_url": {"url": img_url}})
        if vectorize_videos:
            for vid_url in vectorize_videos:
                ark_input.append({"type": "video_url", "video_url": {"url": vid_url, "fps": 1.0}})

        return ProcessedContext(
            properties=ContextProperties(
                raw_properties=[raw_context],
                create_time=raw_context.create_time,
                update_time=tz_now(),
                event_time_start=event_time_start,
                is_processed=True,
                enable_merge=enable_merge,
                user_id=raw_context.user_id,
                device_id=raw_context.device_id,
                agent_id=raw_context.agent_id,
                raw_type="chat_batch" if batch_id else None,
                raw_id=batch_id,
            ),
            extracted_data=extracted_data,
            vectorize=Vectorize(
                input=ark_input,
                content_format=vectorize_format,
            ),
            metadata=metadata if metadata else {},
        )


def _convert_local_paths_in_content(content_parts: list[dict]) -> list[dict]:
    """Convert local file paths in multimodal content parts to base64 data URIs."""
    from opencontext.models.context import _file_to_data_uri, _is_local_path

    result = []
    for part in content_parts:
        part_type = part.get("type", "")
        if part_type == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if _is_local_path(url):
                part = {"type": "image_url", "image_url": {"url": _file_to_data_uri(url)}}
        elif part_type == "video_url":
            url = part.get("video_url", {}).get("url", "")
            if _is_local_path(url):
                new_video = {"url": _file_to_data_uri(url)}
                if "fps" in part.get("video_url", {}):
                    new_video["fps"] = part["video_url"]["fps"]
                part = {"type": "video_url", "video_url": new_video}
        result.append(part)
    return result
