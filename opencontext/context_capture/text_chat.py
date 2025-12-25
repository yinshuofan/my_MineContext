import json
import datetime
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

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
    
    支持两种缓存模式：
    1. Redis 模式（推荐）：支持多服务实例共享用户数据
    2. 内存模式（降级）：当 Redis 不可用时使用本地内存
    """

    # Redis key 前缀
    BUFFER_KEY_PREFIX = "chat:buffer:"
    
    def __init__(self):
        super().__init__(
            name="TextChatCapture",
            description="Captures text chat history",
            source_type=ContextSource.CHAT_LOG,
        )
        # 本地内存缓存（作为降级方案）
        self._local_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._buffer_size = 2  # 默认2条总结一次
        self._lock = threading.RLock()
        
        # Redis 缓存
        self._redis_cache = None
        self._use_redis = False
        self._buffer_ttl = 3600 * 24  # 缓冲区 TTL：24小时

    def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        # 从配置中读取缓存大小
        self._buffer_size = config.get("buffer_size", 4)
        self._buffer_ttl = config.get("buffer_ttl", 3600 * 24)
        
        # 尝试初始化 Redis 缓存
        redis_config = config.get("redis", {})
        if redis_config.get("enabled", True):
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
                self._use_redis = self._redis_cache.is_connected()
                
                if self._use_redis:
                    logger.info("TextChatCapture: Using Redis cache for multi-instance support")
                else:
                    logger.warning("TextChatCapture: Redis unavailable, falling back to local memory cache")
            except Exception as e:
                logger.warning(f"TextChatCapture: Failed to initialize Redis: {e}, using local memory cache")
                self._use_redis = False
        else:
            logger.info("TextChatCapture: Redis disabled in config, using local memory cache")
            self._use_redis = False
        
        return True

    def _start_impl(self) -> bool:
        logger.info("TextChatCapture started.")
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
        # 使用 | 作为分隔符，处理 None 值
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
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }

        buffer_key = self._make_buffer_key(user_id, device_id, agent_id)

        if self._use_redis and self._redis_cache and self._redis_cache.is_connected():
            self._push_message_redis(buffer_key, message, user_id, device_id, agent_id)
        else:
            self._push_message_local(buffer_key, message, user_id, device_id, agent_id)

    def _push_message_redis(
        self,
        buffer_key: str,
        message: Dict[str, Any],
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str]
    ):
        """使用 Redis 存储消息"""
        try:
            # 将消息推入 Redis 列表
            self._redis_cache.rpush_json(buffer_key, message)
            
            # 设置/刷新 TTL
            self._redis_cache.expire(buffer_key, self._buffer_ttl)
            
            # 检查缓冲区大小
            buffer_len = self._redis_cache.llen(buffer_key)
            
            if buffer_len >= self._buffer_size:
                logger.info(f"Buffer size reached {self._buffer_size} for {buffer_key}, flushing messages.")
                self._flush_buffer_redis(buffer_key, user_id, device_id, agent_id)
                
        except Exception as e:
            logger.error(f"Redis push_message error: {e}, falling back to local cache")
            # 降级到本地缓存
            self._push_message_local(buffer_key, message, user_id, device_id, agent_id)

    def _push_message_local(
        self,
        buffer_key: str,
        message: Dict[str, Any],
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str]
    ):
        """使用本地内存存储消息"""
        with self._lock:
            if buffer_key not in self._local_buffers:
                self._local_buffers[buffer_key] = []

            self._local_buffers[buffer_key].append(message)
            
            if len(self._local_buffers[buffer_key]) >= self._buffer_size:
                logger.info(f"Buffer size reached {self._buffer_size} for {buffer_key}, flushing messages.")
                self._flush_buffer_local(buffer_key, user_id, device_id, agent_id)

    def _flush_buffer_redis(
        self,
        buffer_key: str,
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str]
    ):
        """刷新 Redis 中的缓冲区"""
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
                
                logger.info(f"Flushed {len(messages)} chat messages for {buffer_key} to pipeline (Redis).")
                
            finally:
                # 释放锁
                self._redis_cache.release_lock(f"flush:{buffer_key}", lock_token)
                
        except Exception as e:
            logger.error(f"Redis flush_buffer error: {e}")

    def _flush_buffer_local(
        self,
        buffer_key: str,
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str]
    ):
        """刷新本地内存中的缓冲区"""
        buffer = self._local_buffers.get(buffer_key)
        if not buffer:
            return

        # 创建 RawContext
        self._create_and_send_context(buffer, user_id, device_id, agent_id)

        # 清空该用户的缓存
        self._local_buffers[buffer_key] = []
        
        logger.info(f"Flushed {len(buffer)} chat messages for {buffer_key} to pipeline (local).")

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
        buffer_key = self._make_buffer_key(user_id, device_id, agent_id)
        
        if self._use_redis and self._redis_cache and self._redis_cache.is_connected():
            self._flush_buffer_redis(buffer_key, user_id, device_id, agent_id)
        else:
            with self._lock:
                if self._local_buffers.get(buffer_key):
                    self._flush_buffer_local(buffer_key, user_id, device_id, agent_id)

    def _flush_all_buffers(self):
        """刷新所有缓冲区（停止时调用）"""
        if self._use_redis and self._redis_cache and self._redis_cache.is_connected():
            try:
                # 获取所有缓冲区 key
                keys = self._redis_cache.keys(f"{self.BUFFER_KEY_PREFIX}*")
                for key in keys:
                    user_id, device_id, agent_id = self._parse_buffer_key(key)
                    self._flush_buffer_redis(key, user_id, device_id, agent_id)
            except Exception as e:
                logger.error(f"Error flushing all Redis buffers: {e}")
        
        # 也刷新本地缓存（以防有降级数据）
        with self._lock:
            for buffer_key in list(self._local_buffers.keys()):
                if self._local_buffers.get(buffer_key):
                    user_id, device_id, agent_id = self._parse_buffer_key(buffer_key)
                    self._flush_buffer_local(buffer_key, user_id, device_id, agent_id)

    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        stats = {
            "use_redis": self._use_redis,
            "buffer_size": self._buffer_size,
            "buffer_ttl": self._buffer_ttl,
        }
        
        if self._use_redis and self._redis_cache and self._redis_cache.is_connected():
            try:
                keys = self._redis_cache.keys(f"{self.BUFFER_KEY_PREFIX}*")
                stats["redis_buffer_count"] = len(keys)
                stats["redis_total_messages"] = sum(
                    self._redis_cache.llen(key) for key in keys
                )
            except Exception as e:
                stats["redis_error"] = str(e)
        
        with self._lock:
            stats["local_buffer_count"] = len(self._local_buffers)
            stats["local_total_messages"] = sum(
                len(buf) for buf in self._local_buffers.values()
            )
        
        return stats
