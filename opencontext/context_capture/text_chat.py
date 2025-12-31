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
    文本聊天捕获组件（无状态版本）。
    用于缓存聊天机器人的对话记录，并在满足条件（如条数）时打包成上下文。
    支持多用户环境，通过 user_id, device_id, agent_id 区分不同用户的上下文。
    
    所有状态存储在 Redis 中，支持多服务实例共享用户数据。
    服务模式下必须配置 Redis，否则将抛出错误。
    """

    # Redis key 前缀
    BUFFER_KEY_PREFIX = "chat:buffer:"
    
    def __init__(self):
        super().__init__(
            name="TextChatCapture",
            description="Captures text chat history (stateless with Redis)",
            source_type=ContextSource.CHAT_LOG,
        )
        self._buffer_size = 2  # 默认2条总结一次
        self._buffer_ttl = 3600 * 24  # 缓冲区 TTL：24小时
        
        # Redis 缓存
        self._redis_cache = None
        self._initialized = False

    def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        # 从配置中读取缓存大小
        self._buffer_size = config.get("buffer_size", 4)
        self._buffer_ttl = config.get("buffer_ttl", 3600 * 24)
        
        # 初始化 Redis 缓存（必须）
        redis_config = config.get("redis", {})
        
        try:
            from opencontext.storage.redis_cache import get_redis_cache, RedisCacheConfig
            
            redis_cfg = RedisCacheConfig(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                password=redis_config.get("password"),
                db=redis_config.get("db", 0),
                key_prefix=redis_config.get("key_prefix", "opencontext:"),
                default_ttl=self._buffer_ttl,
            )
            
            self._redis_cache = get_redis_cache(redis_cfg)
            
            if not self._redis_cache.is_connected():
                logger.error("TextChatCapture: Redis connection failed. Redis is required for stateless mode.")
                raise RuntimeError("Redis connection required for TextChatCapture in stateless mode")
            
            logger.info("TextChatCapture: Redis cache connected for stateless operation")
            self._initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"TextChatCapture: Redis module not available: {e}")
            raise RuntimeError("Redis module required for TextChatCapture") from e
        except Exception as e:
            logger.error(f"TextChatCapture: Failed to initialize Redis: {e}")
            raise RuntimeError(f"Redis initialization failed: {e}") from e

    def _ensure_redis(self):
        """确保 Redis 连接可用"""
        if not self._redis_cache or not self._redis_cache.is_connected():
            raise RuntimeError("Redis connection not available. TextChatCapture requires Redis in stateless mode.")

    def _start_impl(self) -> bool:
        logger.info("TextChatCapture started (stateless mode).")
        return True

    def _stop_impl(self, graceful: bool = True) -> bool:
        """停止组件，刷新所有缓冲区"""
        if graceful:
            self._flush_all_buffers()
        return True

    def _capture_impl(self) -> List[RawContextProperties]:
        # 对于被动组件，此方法可能不常被主动调用，主要靠 push_message 触发
        return []

    def _make_buffer_key(self, user_id: Optional[str], device_id: Optional[str], agent_id: Optional[str]) -> str:
        """生成缓冲区的 key"""
        # 使用 : 作为分隔符，处理 None 值
        parts = [
            user_id or "_",
            device_id or "_",
            agent_id or "_",
        ]
        return f"{self.BUFFER_KEY_PREFIX}{':'.join(parts)}"

    def _parse_buffer_key(self, key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """解析缓冲区 key 为 (user_id, device_id, agent_id)"""
        if key.startswith(self.BUFFER_KEY_PREFIX):
            key = key[len(self.BUFFER_KEY_PREFIX):]
        parts = key.split(":")
        if len(parts) >= 3:
            return (
                parts[0] if parts[0] != "_" else None,
                parts[1] if parts[1] != "_" else None,
                parts[2] if parts[2] != "_" else None,
            )
        return (None, None, None)

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
        self._ensure_redis()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }

        buffer_key = self._make_buffer_key(user_id, device_id, agent_id)
        
        try:
            # 将消息推入 Redis 列表
            self._redis_cache.rpush_json(buffer_key, message)
            
            # 设置/刷新 TTL
            self._redis_cache.expire(buffer_key, self._buffer_ttl)
            
            # 检查缓冲区大小
            buffer_len = self._redis_cache.llen(buffer_key)
            
            if buffer_len >= self._buffer_size:
                logger.info(f"Buffer size reached {self._buffer_size} for {buffer_key}, flushing messages.")
                self._flush_buffer(buffer_key, user_id, device_id, agent_id)
                
        except Exception as e:
            logger.error(f"Redis push_message error: {e}")
            raise RuntimeError(f"Failed to push message to Redis: {e}") from e

    def _flush_buffer(
        self,
        buffer_key: str,
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str]
    ):
        """刷新 Redis 中的缓冲区"""
        self._ensure_redis()
        
        try:
            # 使用分布式锁确保原子操作
            lock_token = self._redis_cache.acquire_lock(
                f"flush:{buffer_key}",
                timeout=30,
                blocking=True,
                blocking_timeout=5.0
            )
            
            if not lock_token:
                logger.warning(f"Failed to acquire lock for flushing {buffer_key}")
                return
            
            try:
                # 获取所有消息
                messages = self._redis_cache.lrange_json(buffer_key, 0, -1)
                
                if not messages:
                    return
                
                # 创建 RawContext
                self._create_and_send_context(messages, user_id, device_id, agent_id)
                
                # 清空缓冲区
                self._redis_cache.delete(buffer_key)
                
                logger.info(f"Flushed {len(messages)} chat messages for {buffer_key} to pipeline.")
                
            finally:
                # 释放锁
                self._redis_cache.release_lock(f"flush:{buffer_key}", lock_token)
                
        except Exception as e:
            logger.error(f"Redis flush_buffer error: {e}")
            raise RuntimeError(f"Failed to flush buffer: {e}") from e

    def _create_and_send_context(
        self,
        messages: List[Dict[str, Any]],
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str]
    ):
        """创建 RawContext 并发送到处理管道"""
        if not messages:
            return
        
        now = datetime.datetime.now()
        # 将消息列表序列化为字符串存储
        chat_content_str = json.dumps(messages, ensure_ascii=False)

        raw_context = RawContextProperties(
            source=ContextSource.CHAT_LOG,
            content_format=ContentFormat.TEXT,
            create_time=now,
            content_text=chat_content_str,
            additional_info={
                "message_count": len(messages),
                "roles": list(set(m["role"] for m in messages))
            },
            # 多用户支持字段
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
        )

        # 通过 BaseCaptureComponent 的机制上报数据
        if self._callback:
            self._callback([raw_context])

    def process_messages_directly(
        self,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """
        直接处理聊天消息，跳过 Redis 缓冲区，立即发送到处理管道。

        Args:
            messages: 消息列表，每条消息包含 role 和 content
            user_id: 用户标识符
            device_id: 设备标识符
            agent_id: Agent标识符
        """
        self._create_and_send_context(messages, user_id, device_id, agent_id)

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
        self._ensure_redis()
        
        buffer_key = self._make_buffer_key(user_id, device_id, agent_id)
        
        # 检查缓冲区是否有数据
        buffer_len = self._redis_cache.llen(buffer_key)
        if buffer_len > 0:
            self._flush_buffer(buffer_key, user_id, device_id, agent_id)

    def _flush_all_buffers(self):
        """刷新所有缓冲区（停止时调用）"""
        if not self._redis_cache or not self._redis_cache.is_connected():
            logger.warning("Redis not available, skipping flush_all_buffers")
            return
        
        try:
            # 获取所有缓冲区 key
            keys = self._redis_cache.keys(f"{self.BUFFER_KEY_PREFIX}*")
            for key in keys:
                user_id, device_id, agent_id = self._parse_buffer_key(key)
                try:
                    self._flush_buffer(key, user_id, device_id, agent_id)
                except Exception as e:
                    logger.error(f"Error flushing buffer {key}: {e}")
        except Exception as e:
            logger.error(f"Error flushing all Redis buffers: {e}")

    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        stats = {
            "mode": "stateless_redis",
            "buffer_size": self._buffer_size,
            "buffer_ttl": self._buffer_ttl,
            "redis_connected": False,
        }
        
        if self._redis_cache and self._redis_cache.is_connected():
            stats["redis_connected"] = True
            try:
                keys = self._redis_cache.keys(f"{self.BUFFER_KEY_PREFIX}*")
                stats["buffer_count"] = len(keys)
                stats["total_messages"] = sum(
                    self._redis_cache.llen(key) for key in keys
                )
            except Exception as e:
                stats["redis_error"] = str(e)
        
        return stats

    def get_user_buffer_length(
        self,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> int:
        """
        获取指定用户缓冲区中的消息数量。
        
        Args:
            user_id: 用户标识符
            device_id: 设备标识符
            agent_id: Agent标识符
            
        Returns:
            缓冲区中的消息数量
        """
        if not self._redis_cache or not self._redis_cache.is_connected():
            return 0
        
        buffer_key = self._make_buffer_key(user_id, device_id, agent_id)
        return self._redis_cache.llen(buffer_key)
