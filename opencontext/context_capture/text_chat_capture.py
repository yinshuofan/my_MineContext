import json
import datetime
import threading
from typing import Any, Dict, List, Optional

from opencontext.context_capture.base import BaseCaptureComponent
from opencontext.models.context import RawContextProperties
from opencontext.models.enums import ContentFormat, ContextSource
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

class TextChatCapture(BaseCaptureComponent):
    """
    文本聊天捕获组件。
    用于缓存聊天机器人的对话记录，并在满足条件（如条数）时打包成上下文。
    """

    def __init__(self):
        super().__init__(
            name="TextChatCapture",
            description="Captures text chat history",
            source_type=ContextSource.INPUT, # 使用 INPUT 或自定义枚举
        )
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_size = 4  # 默认4条总结一次
        self._lock = threading.RLock()

    def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        # 从配置中读取缓存大小
        self._buffer_size = config.get("buffer_size", 4)
        return True

    def _start_impl(self) -> bool:
        logger.info("TextChatCapture started.")
        return True

    def _stop_impl(self, graceful: bool = True) -> bool:
        # 停止时，如果有剩余未处理的消息，强制处理
        if graceful and self._buffer:
            self._flush_buffer()
        return True

    def _capture_impl(self) -> List[RawContextProperties]:
        # 对于被动组件，此方法可能不常被主动调用，主要靠 push_message 触发
        return []

    def push_message(self, role: str, content: str):
        """
        供外部聊天机器人调用的接口，推入新消息。
        格式遵循 OpenAI: {"role": "user", "content": "..."}
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with self._lock:
            self._buffer.append(message)
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()

    def _flush_buffer(self):
        """将缓存的消息打包成 RawContext 并通过回调发送给 Processor"""
        if not self._buffer:
            return

        # 1. 创建 RawContext
        now = datetime.datetime.now()
        # 将消息列表序列化为字符串存储
        chat_content_str = json.dumps(self._buffer, ensure_ascii=False)
        
        raw_context = RawContextProperties(
            source=ContextSource.INPUT,
            content_format=ContentFormat.TEXT, # 标记为纯文本
            create_time=now,
            content_text=chat_content_str, # 核心内容在这里
            additional_info={
                "message_count": len(self._buffer),
                "roles": list(set(m["role"] for m in self._buffer))
            }
        )

        # 2. 通过 BaseCaptureComponent 的机制上报数据
        # Manager 会接收这个数据并传给 Processor
        if self._callback:
            self._callback([raw_context])
        
        # 3. 清空缓存
        self._buffer = []
        logger.info(f"Flushed {raw_context.additional_info['message_count']} chat messages to pipeline.")