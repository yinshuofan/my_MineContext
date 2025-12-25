import json
import datetime
import threading
from typing import Any, Dict, List, Optional, Tuple

from opencontext.context_capture.base import BaseCaptureComponent
from opencontext.models.context import RawContextProperties
from opencontext.models.enums import ContentFormat, ContextSource
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

class TextChatCapture(BaseCaptureComponent):
    """
    文本聊天捕获组件。
    用于缓存聊天机器人的对话记录，并在满足条件（如条数）时打包成上下文。
    支持多用户环境，通过 user_id, device_id, agent_id 区分不同用户的上下文。
    """

    def __init__(self):
        super().__init__(
            name="TextChatCapture",
            description="Captures text chat history",
            source_type=ContextSource.CHAT_LOG, # 使用 INPUT 或自定义枚举
        )
        # 使用 (user_id, device_id, agent_id) 作为 key 来区分不同用户的 buffer
        self._buffers: Dict[tuple, List[Dict[str, Any]]] = {}
        self._buffer_size = 2  # 默认2条总结一次
        self._lock = threading.RLock()

    def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        # 从配置中读取缓存大小
        self._buffer_size = config.get("buffer_size", 4)
        return True

    def _start_impl(self) -> bool:
        logger.info("TextChatCapture started.")
        return True

    def _stop_impl(self, graceful: bool = True) -> bool:
        # 停止时，如果有剩余未处理的消息，强制处理所有用户的 buffer
        if graceful:
            with self._lock:
                for user_key in list(self._buffers.keys()):
                    if self._buffers.get(user_key):
                        self._flush_buffer(user_key)
        return True

    def _capture_impl(self) -> List[RawContextProperties]:
        # 对于被动组件，此方法可能不常被主动调用，主要靠 push_message 触发
        return []

    def push_message(
        self,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """
        供外部聊天机器人调用的接口，推入新消息。
        格式遵循 OpenAI: {"role": "user", "content": "..."}

        Args:
            role: 消息角色 (user/assistant)
            content: 消息内容
            user_id: 用户标识符
            device_id: 设备标识符
            agent_id: Agent标识符
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # 使用 (user_id, device_id, agent_id) 作为 key
        user_key = (user_id, device_id, agent_id)

        with self._lock:
            if user_key not in self._buffers:
                self._buffers[user_key] = []

            self._buffers[user_key].append(message)
            if len(self._buffers[user_key]) >= self._buffer_size:
                logger.info(f"Buffer size reached {self._buffer_size} for user {user_key}, flushing messages.")
                self._flush_buffer(user_key)

    def _flush_buffer(self, user_key: tuple):
        """
        将缓存的消息打包成 RawContext 并通过回调发送给 Processor

        Args:
            user_key: (user_id, device_id, agent_id) 元组
        """
        buffer = self._buffers.get(user_key)
        if not buffer:
            return

        user_id, device_id, agent_id = user_key

        # 1. 创建 RawContext
        now = datetime.datetime.now()
        # 将消息列表序列化为字符串存储
        chat_content_str = json.dumps(buffer, ensure_ascii=False)

        raw_context = RawContextProperties(
            source=ContextSource.CHAT_LOG,
            content_format=ContentFormat.TEXT, # 标记为纯文本
            create_time=now,
            content_text=chat_content_str, # 核心内容在这里
            additional_info={
                "message_count": len(buffer),
                "roles": list(set(m["role"] for m in buffer))
            },
            # 多用户支持字段
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
        )

        # 2. 通过 BaseCaptureComponent 的机制上报数据
        # Manager 会接收这个数据并传给 Processor
        if self._callback:
            self._callback([raw_context])

        # 3. 清空该用户的缓存
        self._buffers[user_key] = []
        logger.info(f"Flushed {raw_context.additional_info['message_count']} chat messages for user {user_key} to pipeline.")

    def flush_user_buffer(
        self,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """
        手动刷新指定用户的缓冲区（外部接口）。
        用于在需要时立即处理用户的聊天历史。

        Args:
            user_id: 用户标识符
            device_id: 设备标识符
            agent_id: Agent标识符
        """
        user_key = (user_id, device_id, agent_id)
        with self._lock:
            if self._buffers.get(user_key):
                self._flush_buffer(user_key)