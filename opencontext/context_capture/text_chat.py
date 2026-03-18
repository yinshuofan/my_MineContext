import datetime
import json
from typing import Any, Dict, List, Optional

from opencontext.context_capture.base import BaseCaptureComponent
from opencontext.models.context import RawContextProperties
from opencontext.models.enums import ContentFormat, ContextSource
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TextChatCapture(BaseCaptureComponent):
    """
    文本聊天捕获组件。
    接收聊天消息并直接发送到处理管道。
    支持多用户环境，通过 user_id, device_id, agent_id 区分不同用户的上下文。
    """

    def __init__(self):
        super().__init__(
            name="TextChatCapture",
            description="Captures text chat history",
            source_type=ContextSource.CHAT_LOG,
        )

    def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        return True

    def _start_impl(self) -> bool:
        logger.info("TextChatCapture started.")
        return True

    def _stop_impl(self, graceful: bool = True) -> bool:
        return True

    def _capture_impl(self) -> List[RawContextProperties]:
        return []

    async def _create_and_send_context(
        self,
        messages: List[Dict[str, Any]],
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str],
    ):
        """创建 RawContext 并发送到处理管道"""
        if not messages:
            return

        now = datetime.datetime.now()
        # 将消息列表序列化为字符串存储
        chat_content_str = json.dumps(messages, ensure_ascii=False)

        # Detect if any message contains multimodal content
        has_images = False
        has_videos = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    part_type = part.get("type", "")
                    if part_type == "image_url":
                        has_images = True
                    elif part_type == "video_url":
                        has_videos = True

        is_multimodal = has_images or has_videos
        content_format = ContentFormat.MULTIMODAL if is_multimodal else ContentFormat.TEXT

        # Build modalities list for additional_info
        modalities = ["text"]
        if has_images:
            modalities.append("image")
        if has_videos:
            modalities.append("video")

        raw_context = RawContextProperties(
            source=ContextSource.CHAT_LOG,
            content_format=content_format,
            create_time=now,
            content_text=chat_content_str,
            additional_info={
                "message_count": len(messages),
                "roles": list(set(m.get("role", "user") for m in messages)),
                "modalities": modalities,
            },
            # 多用户支持字段
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
        )

        # 通过回调上报数据
        if self._callback:
            await self._callback([raw_context])

    async def process_messages_directly(
        self,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        """
        直接处理聊天消息，立即发送到处理管道。

        Args:
            messages: 消息列表，每条消息包含 role 和 content
            user_id: 用户标识符
            device_id: 设备标识符
            agent_id: Agent标识符
        """
        await self._create_and_send_context(messages, user_id, device_id, agent_id)
