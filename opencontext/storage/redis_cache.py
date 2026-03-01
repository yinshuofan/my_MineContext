#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Redis Cache Manager (Async Only)

Provides a unified Redis caching layer for multi-instance deployment.
All operations are asynchronous (non-blocking).

Usage:
    await cache.get("key")
    await cache.set("key", "value")
    await cache.hgetall("key")
"""

import asyncio
import json
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union

from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Global Redis client instance
_redis_client: Optional["RedisCache"] = None
_redis_lock = threading.Lock()

# Lua script for atomic RPUSH + EXPIRE + LLEN in a single round-trip.
# Used by text_chat push_message to reduce 3 Redis calls to 1.
_RPUSH_EXPIRE_LLEN_LUA = """
redis.call('RPUSH', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
return redis.call('LLEN', KEYS[1])
"""

# Lua script for atomic lock release: only DEL if the caller still owns the lock.
# Prevents releasing a lock that expired and was re-acquired by another instance.
_RELEASE_LOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


@dataclass
class RedisCacheConfig:
    """Redis cache configuration"""

    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    key_prefix: str = "opencontext:"
    default_ttl: int = 3600  # 1 hour
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    decode_responses: bool = True


class _PrefixedPipeline:
    """Minimal pipeline wrapper that auto-prefixes keys."""

    def __init__(self, pipe, prefix: str):
        self._pipe = pipe
        self._prefix = prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def hset(self, key: str, field=None, value=None, mapping=None):
        if mapping is not None:
            self._pipe.hset(self._k(key), mapping=mapping)
        elif field is not None and value is not None:
            self._pipe.hset(self._k(key), field, value)
        return self

    def expire(self, key: str, ttl: int):
        self._pipe.expire(self._k(key), ttl)
        return self

    def zadd(self, key: str, mapping: dict):
        self._pipe.zadd(self._k(key), mapping)
        return self

    async def execute(self):
        return await self._pipe.execute()


class RedisCache:
    """
    Redis Cache Manager (Async Only)

    Provides a unified interface for Redis operations with support for:
    - Key-value storage with TTL
    - List operations (for message buffers)
    - Hash operations (for structured data)
    - Set operations (for deduplication)
    - Atomic operations for multi-instance safety

    All methods are asynchronous (non-blocking).
    """

    def __init__(self, config: Optional[RedisCacheConfig] = None):
        """
        Initialize Redis cache manager.

        Args:
            config: Redis configuration. If None, uses default config.
        """
        self.config = config or RedisCacheConfig()
        self._async_client = None
        self._async_lock = asyncio.Lock()
        self._async_connected = False

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def _connect_async(self) -> bool:
        """
        Establish asynchronous connection to Redis server.

        Returns:
            True if connection successful, False otherwise.
        """
        if self._async_connected and self._async_client:
            return True

        try:
            import redis.asyncio as aioredis

            pool = aioredis.BlockingConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                decode_responses=self.config.decode_responses,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                max_connections=self.config.max_connections,
                timeout=5,  # seconds to wait for an idle connection
            )
            self._async_client = aioredis.Redis(connection_pool=pool)

            # Test connection
            await self._async_client.ping()
            self._async_connected = True
            logger.info(
                f"Redis async client connected: {self.config.host}:{self.config.port}/{self.config.db}"
            )
            return True

        except ImportError:
            logger.error("Redis asyncio package not available. Ensure redis>=4.2.0")
            self._async_connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect async Redis client: {e}")
            self._async_connected = False
            return False

    async def _ensure_async_client(self) -> bool:
        """Ensure async client is connected."""
        if self._async_connected:
            return True
        async with self._async_lock:
            if self._async_connected:
                return True
            return await self._connect_async()

    async def is_connected(self) -> bool:
        """Check if async Redis is connected."""
        if not self._async_connected or not self._async_client:
            return await self._connect_async()
        try:
            await self._async_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis async ping failed: {e}")
            self._async_connected = False
            return False

    def _make_key(self, key: str) -> str:
        """Generate full Redis key with prefix."""
        return f"{self.config.key_prefix}{key}"

    # =========================================================================
    # Basic Key-Value Operations
    # =========================================================================

    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.get(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set key-value pair."""
        if not await self._ensure_async_client():
            return False
        try:
            ttl = ttl if ttl is not None else self.config.default_ttl
            return bool(
                await self._async_client.set(
                    self._make_key(key),
                    value,
                    ex=ttl if ttl > 0 else None,
                    nx=nx,
                    xx=xx,
                )
            )
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        if not await self._ensure_async_client() or not keys:
            return 0
        try:
            full_keys = [self._make_key(k) for k in keys]
            return await self._async_client.delete(*full_keys)
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not await self._ensure_async_client():
            return False
        try:
            return bool(await self._async_client.exists(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for a key."""
        if not await self._ensure_async_client():
            return False
        try:
            return bool(await self._async_client.expire(self._make_key(key), ttl))
        except Exception as e:
            logger.error(f"Redis EXPIRE error: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """Get remaining TTL for a key."""
        if not await self._ensure_async_client():
            return -2
        try:
            return await self._async_client.ttl(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis TTL error: {e}")
            return -2

    # =========================================================================
    # JSON Operations
    # =========================================================================

    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value by key."""
        value = await self.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for key {key}: {e}")
            return None

    async def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set JSON value."""
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            return await self.set(key, json_str, ttl=ttl, nx=nx, xx=xx)
        except Exception as e:
            logger.error(f"JSON encode error for key {key}: {e}")
            return False

    # =========================================================================
    # List Operations
    # =========================================================================

    async def lpush(self, key: str, *values: str) -> int:
        """Push values to the left of a list."""
        if not await self._ensure_async_client() or not values:
            return 0
        try:
            return await self._async_client.lpush(self._make_key(key), *values)
        except Exception as e:
            logger.error(f"Redis LPUSH error: {e}")
            return 0

    async def rpush(self, key: str, *values: str) -> int:
        """Push values to the right of a list."""
        if not await self._ensure_async_client() or not values:
            return 0
        try:
            return await self._async_client.rpush(self._make_key(key), *values)
        except Exception as e:
            logger.error(f"Redis RPUSH error: {e}")
            return 0

    async def lpop(self, key: str, count: int = 1) -> Optional[Union[str, List[str]]]:
        """Pop values from the left of a list."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.lpop(self._make_key(key), count)
        except Exception as e:
            logger.error(f"Redis LPOP error: {e}")
            return None

    async def rpop(self, key: str, count: int = 1) -> Optional[Union[str, List[str]]]:
        """Pop values from the right of a list."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.rpop(self._make_key(key), count)
        except Exception as e:
            logger.error(f"Redis RPOP error: {e}")
            return None

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get range of elements from a list."""
        if not await self._ensure_async_client():
            return []
        try:
            return await self._async_client.lrange(self._make_key(key), start, end)
        except Exception as e:
            logger.error(f"Redis LRANGE error: {e}")
            return []

    async def llen(self, key: str) -> int:
        """Get length of a list."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.llen(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis LLEN error: {e}")
            return 0

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim a list to specified range."""
        if not await self._ensure_async_client():
            return False
        try:
            return bool(await self._async_client.ltrim(self._make_key(key), start, end))
        except Exception as e:
            logger.error(f"Redis LTRIM error: {e}")
            return False

    async def lrange_json(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range of JSON elements from a list."""
        items = await self.lrange(key, start, end)
        result = []
        for item in items:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                result.append(item)
        return result

    async def rpush_json(self, key: str, *values: Any) -> int:
        """Push JSON values to the right of a list."""
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception as e:
                logger.error(f"JSON encode error: {e}")
        if not json_values:
            return 0
        return await self.rpush(key, *json_values)

    async def rpush_expire_llen(self, key: str, value: str, ttl: int) -> int:
        """Atomic RPUSH + EXPIRE + LLEN in a single round-trip via Lua script."""
        if not await self._ensure_async_client():
            return 0
        try:
            full_key = self._make_key(key)
            result = await self._async_client.eval(
                _RPUSH_EXPIRE_LLEN_LUA, 1, full_key, value, str(ttl)
            )
            return int(result)
        except Exception as e:
            logger.error(f"Redis rpush_expire_llen error: {e}")
            return 0

    async def lpush_json(self, key: str, *values: Any) -> int:
        """Push JSON values to the left of a list."""
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception as e:
                logger.error(f"JSON encode error: {e}")
        if not json_values:
            return 0
        return await self.lpush(key, *json_values)

    # =========================================================================
    # Hash Operations
    # =========================================================================

    async def hget(self, key: str, field: str) -> Optional[str]:
        """Get a field from a hash."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.hget(self._make_key(key), field)
        except Exception as e:
            logger.error(f"Redis HGET error: {e}")
            return None

    async def hset(self, key: str, field: str, value: str) -> int:
        """Set a field in a hash."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.hset(self._make_key(key), field, value)
        except Exception as e:
            logger.error(f"Redis HSET error: {e}")
            return 0

    async def hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        """Set multiple fields in a hash."""
        if not await self._ensure_async_client() or not mapping:
            return False
        try:
            return bool(await self._async_client.hset(self._make_key(key), mapping=mapping))
        except Exception as e:
            logger.error(f"Redis HMSET error: {e}")
            return False

    async def hgetall(self, key: str) -> Dict[str, str]:
        """Get all fields and values from a hash."""
        if not await self._ensure_async_client():
            return {}
        try:
            return await self._async_client.hgetall(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis HGETALL error: {e}")
            return {}

    async def hdel(self, key: str, *fields: str) -> int:
        """Delete fields from a hash."""
        if not await self._ensure_async_client() or not fields:
            return 0
        try:
            return await self._async_client.hdel(self._make_key(key), *fields)
        except Exception as e:
            logger.error(f"Redis HDEL error: {e}")
            return 0

    async def hlen(self, key: str) -> int:
        """Get number of fields in a hash."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.hlen(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis HLEN error: {e}")
            return 0

    async def hexists(self, key: str, field: str) -> bool:
        """Check if a field exists in a hash."""
        if not await self._ensure_async_client():
            return False
        try:
            return bool(await self._async_client.hexists(self._make_key(key), field))
        except Exception as e:
            logger.error(f"Redis HEXISTS error: {e}")
            return False

    async def hget_json(self, key: str, field: str) -> Optional[Any]:
        """Get a JSON field from a hash."""
        value = await self.hget(key, field)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    async def hset_json(self, key: str, field: str, value: Any) -> int:
        """Set a JSON field in a hash."""
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            return await self.hset(key, field, json_str)
        except Exception as e:
            logger.error(f"Redis HSET_JSON error: {e}")
            return 0

    async def hmset_json(self, key: str, mapping: Dict[str, Any]) -> bool:
        """Set multiple JSON fields in a hash."""
        if not mapping:
            return False
        try:
            json_mapping = {}
            for k, v in mapping.items():
                json_mapping[k] = json.dumps(v, ensure_ascii=False, default=str)
            return await self.hmset(key, json_mapping)
        except Exception as e:
            logger.error(f"Redis HMSET_JSON error: {e}")
            return False

    async def hgetall_json(self, key: str) -> Dict[str, Any]:
        """Get all fields from a hash as JSON objects."""
        data = await self.hgetall(key)
        result = {}
        for k, v in data.items():
            try:
                result[k] = json.loads(v)
            except json.JSONDecodeError:
                result[k] = v
        return result

    async def hkeys(self, key: str) -> List[str]:
        """Get all field names from a hash."""
        if not await self._ensure_async_client():
            return []
        try:
            return list(await self._async_client.hkeys(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis HKEYS error: {e}")
            return []

    async def hvals(self, key: str) -> List[str]:
        """Get all values from a hash."""
        if not await self._ensure_async_client():
            return []
        try:
            return list(await self._async_client.hvals(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis HVALS error: {e}")
            return []

    async def hvals_json(self, key: str) -> List[Any]:
        """Get all values from a hash as JSON objects."""
        values = await self.hvals(key)
        result = []
        for v in values:
            try:
                result.append(json.loads(v))
            except json.JSONDecodeError:
                result.append(v)
        return result

    # =========================================================================
    # Set Operations
    # =========================================================================

    async def sadd(self, key: str, *members: str) -> int:
        """Add members to a set."""
        if not await self._ensure_async_client() or not members:
            return 0
        try:
            return await self._async_client.sadd(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis SADD error: {e}")
            return 0

    async def srem(self, key: str, *members: str) -> int:
        """Remove members from a set."""
        if not await self._ensure_async_client() or not members:
            return 0
        try:
            return await self._async_client.srem(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis SREM error: {e}")
            return 0

    async def sismember(self, key: str, member: str) -> bool:
        """Check if a member is in a set."""
        if not await self._ensure_async_client():
            return False
        try:
            return bool(await self._async_client.sismember(self._make_key(key), member))
        except Exception as e:
            logger.error(f"Redis SISMEMBER error: {e}")
            return False

    async def smembers(self, key: str) -> Set[str]:
        """Get all members of a set."""
        if not await self._ensure_async_client():
            return set()
        try:
            return await self._async_client.smembers(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis SMEMBERS error: {e}")
            return set()

    async def scard(self, key: str) -> int:
        """Get number of members in a set."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.scard(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis SCARD error: {e}")
            return 0

    # =========================================================================
    # Sorted Set Operations
    # =========================================================================

    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Add members to a sorted set with their scores."""
        if not await self._ensure_async_client() or not mapping:
            return 0
        try:
            return await self._async_client.zadd(self._make_key(key), mapping)
        except Exception as e:
            logger.error(f"Redis ZADD error: {e}")
            return 0

    async def zrangebyscore(
        self, key: str, min_score: float, max_score: float, withscores: bool = False
    ) -> List[str]:
        """Get members from a sorted set within a score range."""
        if not await self._ensure_async_client():
            return []
        try:
            full_key = self._make_key(key)
            if withscores:
                return await self._async_client.zrangebyscore(
                    full_key, min_score, max_score, withscores=True
                )
            else:
                return await self._async_client.zrangebyscore(full_key, min_score, max_score)
        except Exception as e:
            logger.error(f"Redis ZRANGEBYSCORE error: {e}")
            return []

    async def eval_lua(self, script: str, keys: List[str], args: List = None) -> Any:
        """Execute a Lua script on Redis server.

        Keys are automatically prefixed. Scripts run atomically (Redis single-threaded).
        """
        if not await self._ensure_async_client():
            return None
        try:
            prefixed_keys = [self._make_key(k) for k in keys]
            return await self._async_client.eval(
                script, len(prefixed_keys), *prefixed_keys, *(args or [])
            )
        except Exception as e:
            logger.error(f"Redis eval_lua error: {e}")
            return None

    async def zrem(self, key: str, *members: str) -> int:
        """Remove members from a sorted set."""
        if not await self._ensure_async_client() or not members:
            return 0
        try:
            return await self._async_client.zrem(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis ZREM error: {e}")
            return 0

    async def zscore(self, key: str, member: str) -> Optional[float]:
        """Get the score of a member in a sorted set."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.zscore(self._make_key(key), member)
        except Exception as e:
            logger.error(f"Redis ZSCORE error: {e}")
            return None

    async def zcard(self, key: str) -> int:
        """Get number of members in a sorted set."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.zcard(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis ZCARD error: {e}")
            return 0

    # =========================================================================
    # Atomic Operations
    # =========================================================================

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment a key's value atomically."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.incr(self._make_key(key), amount)
        except Exception as e:
            logger.error(f"Redis INCR error: {e}")
            return 0

    async def decr(self, key: str, amount: int = 1) -> int:
        """Decrement a key's value atomically."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.decr(self._make_key(key), amount)
        except Exception as e:
            logger.error(f"Redis DECR error: {e}")
            return 0

    async def getset(self, key: str, value: str) -> Optional[str]:
        """Set a new value and return the old value atomically."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.getset(self._make_key(key), value)
        except Exception as e:
            logger.error(f"Redis GETSET error: {e}")
            return None

    # =========================================================================
    # Lock Operations (Non-blocking)
    # =========================================================================

    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: float = 5.0,
    ) -> Optional[str]:
        """
        Acquire a distributed lock (non-blocking).

        Uses asyncio.sleep() instead of time.sleep(), won't block the event loop.

        Args:
            lock_name: Name of the lock
            timeout: Lock expiration time in seconds
            blocking: Whether to block until lock is acquired
            blocking_timeout: Maximum time to wait for lock

        Returns:
            Lock token if acquired, None otherwise
        """
        if not await self._ensure_async_client():
            return None
        try:
            import time

            lock_key = f"lock:{lock_name}"
            token = str(uuid.uuid4())

            if blocking:
                start_time = time.time()
                while time.time() - start_time < blocking_timeout:
                    if await self.set(lock_key, token, ttl=timeout, nx=True):
                        return token
                    await asyncio.sleep(0.1)  # Non-blocking sleep!
                return None
            else:
                if await self.set(lock_key, token, ttl=timeout, nx=True):
                    return token
                return None

        except Exception as e:
            logger.error(f"Redis acquire_lock error: {e}")
            return None

    async def release_lock(self, lock_name: str, token: str) -> bool:
        """
        Release a distributed lock atomically.

        Uses a Lua script to ensure the lock is only deleted if the caller
        still owns it (token matches). Prevents releasing a lock that expired
        and was re-acquired by another instance.

        Args:
            lock_name: Name of the lock
            token: Lock token returned by acquire_lock

        Returns:
            True if lock was released
        """
        if not await self._ensure_async_client():
            return False
        try:
            lock_key = self._make_key(f"lock:{lock_name}")
            result = await self._async_client.eval(_RELEASE_LOCK_LUA, 1, lock_key, token)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis release_lock error: {e}")
            return False

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern."""
        if not await self._ensure_async_client():
            return []
        try:
            full_pattern = self._make_key(pattern)
            keys = await self._async_client.keys(full_pattern)
            prefix_len = len(self.config.key_prefix)
            return [k[prefix_len:] if isinstance(k, str) else k.decode()[prefix_len:] for k in keys]
        except Exception as e:
            logger.error(f"Redis KEYS error: {e}")
            return []

    async def flush_db(self) -> bool:
        """Flush the current database. Use with caution!"""
        if not await self._ensure_async_client():
            return False
        try:
            await self._async_client.flushdb()
            logger.warning("Redis database flushed!")
            return True
        except Exception as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            return False

    async def info(self) -> Dict[str, Any]:
        """Get Redis server info."""
        if not await self._ensure_async_client():
            return {}
        try:
            return await self._async_client.info()
        except Exception as e:
            logger.error(f"Redis INFO error: {e}")
            return {}

    # =========================================================================
    # Pub/Sub Operations
    # =========================================================================

    async def publish(self, channel: str, message: str) -> int:
        """Publish a message to a Redis channel. Channel is auto-prefixed."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.publish(self._make_key(channel), message)
        except Exception as e:
            logger.error(f"Redis PUBLISH error: {e}")
            return 0

    async def create_pubsub(self):
        """Create a raw redis.asyncio PubSub instance. Caller manages lifecycle.
        Returns None if not connected."""
        if not await self._ensure_async_client():
            return None
        try:
            return self._async_client.pubsub()
        except Exception as e:
            logger.error(f"Redis create_pubsub error: {e}")
            return None

    # =========================================================================
    # Pipeline Support
    # =========================================================================

    @asynccontextmanager
    async def pipeline(self, transaction: bool = False):
        """Get a Redis pipeline with auto-prefixed keys.

        Usage:
            async with cache.pipeline() as pipe:
                pipe.hset("key", mapping={...})
                pipe.expire("key", 3600)
                pipe.zadd("queue", {"member": 1.0})
                results = await pipe.execute()
        """
        if not await self._ensure_async_client():
            raise RuntimeError("Redis not connected")
        async with self._async_client.pipeline(transaction=transaction) as pipe:
            yield _PrefixedPipeline(pipe, self.config.key_prefix)

    # =========================================================================
    # Connection Cleanup
    # =========================================================================

    async def close(self):
        """Close async Redis connection."""
        if self._async_client:
            try:
                await self._async_client.close()
            except Exception as e:
                logger.debug(f"Error closing Redis async client: {e}")
            self._async_client = None
            self._async_connected = False
            logger.info("Redis async connection closed")


# =============================================================================
# Global Redis Cache Instance
# =============================================================================


def get_redis_cache(config: Optional[RedisCacheConfig] = None) -> RedisCache:
    """
    Get or create the global Redis cache instance.

    Args:
        config: Redis configuration (only used on first call)

    Returns:
        RedisCache instance
    """
    global _redis_client

    with _redis_lock:
        if _redis_client is None:
            _redis_client = RedisCache(config)
        return _redis_client


def init_redis_cache(config: RedisCacheConfig) -> RedisCache:
    """
    Initialize the global Redis cache with specific configuration.

    Args:
        config: Redis configuration

    Returns:
        RedisCache instance
    """
    global _redis_client

    with _redis_lock:
        _redis_client = RedisCache(config)
        return _redis_client


async def close_redis_cache():
    """Close the global Redis cache connection."""
    global _redis_client

    with _redis_lock:
        if _redis_client is not None:
            await _redis_client.close()
            _redis_client = None


# =============================================================================
# Fallback In-Memory Cache (when Redis is unavailable)
# =============================================================================


class InMemoryCache:
    """
    In-memory cache fallback when Redis is unavailable.
    Provides the same async interface as RedisCache but stores data locally.
    Note: This does NOT support multi-instance sharing.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lists: Dict[str, List[str]] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._sets: Dict[str, Set[str]] = {}
        self._sorted_sets: Dict[str, Dict[str, float]] = {}
        self._expiry: Dict[str, datetime] = {}
        self._async_lock = asyncio.Lock()
        logger.warning("Using in-memory cache fallback. Multi-instance sharing is NOT supported!")

    async def is_connected(self) -> bool:
        return True

    def _check_expiry(self, key: str) -> bool:
        """Check if key is expired and remove if so."""
        if key in self._expiry:
            if datetime.now() > self._expiry[key]:
                # Remove expired key from all storages
                self._data.pop(key, None)
                self._lists.pop(key, None)
                self._hashes.pop(key, None)
                self._sets.pop(key, None)
                self._sorted_sets.pop(key, None)
                del self._expiry[key]
                return True
        return False

    # Basic Key-Value Operations
    async def get(self, key: str) -> Optional[str]:
        async with self._async_lock:
            if self._check_expiry(key):
                return None
            return self._data.get(key)

    async def set(
        self, key: str, value: str, ttl: Optional[int] = None, nx: bool = False, xx: bool = False
    ) -> bool:
        async with self._async_lock:
            exists = key in self._data
            if nx and exists:
                return False
            if xx and not exists:
                return False
            self._data[key] = value
            if ttl and ttl > 0:
                self._expiry[key] = datetime.now() + timedelta(seconds=ttl)
            return True

    async def delete(self, *keys: str) -> int:
        count = 0
        async with self._async_lock:
            for key in keys:
                if key in self._data:
                    del self._data[key]
                    count += 1
                self._lists.pop(key, None)
                self._hashes.pop(key, None)
                self._sets.pop(key, None)
                self._sorted_sets.pop(key, None)
                self._expiry.pop(key, None)
        return count

    async def exists(self, key: str) -> bool:
        async with self._async_lock:
            if self._check_expiry(key):
                return False
            return (
                key in self._data
                or key in self._lists
                or key in self._hashes
                or key in self._sets
                or key in self._sorted_sets
            )

    async def expire(self, key: str, ttl: int) -> bool:
        async with self._async_lock:
            if (
                key in self._data
                or key in self._lists
                or key in self._hashes
                or key in self._sets
                or key in self._sorted_sets
            ):
                self._expiry[key] = datetime.now() + timedelta(seconds=ttl)
                return True
            return False

    async def ttl(self, key: str) -> int:
        async with self._async_lock:
            if key in self._expiry:
                remaining = (self._expiry[key] - datetime.now()).total_seconds()
                return int(remaining) if remaining > 0 else -2
            return -1

    # JSON Operations
    async def get_json(self, key: str) -> Optional[Any]:
        value = await self.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    async def set_json(
        self, key: str, value: Any, ttl: Optional[int] = None, nx: bool = False, xx: bool = False
    ) -> bool:
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            return await self.set(key, json_str, ttl=ttl, nx=nx, xx=xx)
        except Exception as e:
            logger.debug(f"Redis set_json failed: {e}")
            return False

    # List Operations
    async def rpush(self, key: str, *values: str) -> int:
        async with self._async_lock:
            if key not in self._lists:
                self._lists[key] = []
            self._lists[key].extend(values)
            return len(self._lists[key])

    async def lpush(self, key: str, *values: str) -> int:
        async with self._async_lock:
            if key not in self._lists:
                self._lists[key] = []
            for v in reversed(values):
                self._lists[key].insert(0, v)
            return len(self._lists[key])

    async def lpop(self, key: str, count: int = 1) -> Optional[Union[str, List[str]]]:
        async with self._async_lock:
            if key not in self._lists or not self._lists[key]:
                return None
            if count == 1:
                return self._lists[key].pop(0)
            result = self._lists[key][:count]
            self._lists[key] = self._lists[key][count:]
            return result

    async def rpop(self, key: str, count: int = 1) -> Optional[Union[str, List[str]]]:
        async with self._async_lock:
            if key not in self._lists or not self._lists[key]:
                return None
            if count == 1:
                return self._lists[key].pop()
            result = self._lists[key][-count:]
            self._lists[key] = self._lists[key][:-count]
            return result

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        async with self._async_lock:
            if key not in self._lists:
                return []
            lst = self._lists[key]
            if end == -1:
                return lst[start:]
            return lst[start : end + 1]

    async def llen(self, key: str) -> int:
        async with self._async_lock:
            return len(self._lists.get(key, []))

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        async with self._async_lock:
            if key not in self._lists:
                return True
            lst = self._lists[key]
            if end == -1:
                self._lists[key] = lst[start:]
            else:
                self._lists[key] = lst[start : end + 1]
            return True

    async def lrange_json(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        items = await self.lrange(key, start, end)
        result = []
        for item in items:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                result.append(item)
        return result

    async def rpush_json(self, key: str, *values: Any) -> int:
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception as e:
                logger.debug(f"JSON serialization failed for list push: {e}")
        if not json_values:
            return 0
        return await self.rpush(key, *json_values)

    async def lpush_json(self, key: str, *values: Any) -> int:
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception as e:
                logger.debug(f"JSON serialization failed for list push: {e}")
        if not json_values:
            return 0
        return await self.lpush(key, *json_values)

    # Hash Operations
    async def hget(self, key: str, field: str) -> Optional[str]:
        async with self._async_lock:
            return self._hashes.get(key, {}).get(field)

    async def hset(self, key: str, field: str, value: str) -> int:
        async with self._async_lock:
            if key not in self._hashes:
                self._hashes[key] = {}
            is_new = field not in self._hashes[key]
            self._hashes[key][field] = value
            return 1 if is_new else 0

    async def hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        async with self._async_lock:
            if key not in self._hashes:
                self._hashes[key] = {}
            self._hashes[key].update(mapping)
            return True

    async def hgetall(self, key: str) -> Dict[str, str]:
        async with self._async_lock:
            return dict(self._hashes.get(key, {}))

    async def hdel(self, key: str, *fields: str) -> int:
        async with self._async_lock:
            if key not in self._hashes:
                return 0
            count = 0
            for field in fields:
                if field in self._hashes[key]:
                    del self._hashes[key][field]
                    count += 1
            return count

    async def hlen(self, key: str) -> int:
        async with self._async_lock:
            return len(self._hashes.get(key, {}))

    async def hexists(self, key: str, field: str) -> bool:
        async with self._async_lock:
            return field in self._hashes.get(key, {})

    async def hget_json(self, key: str, field: str) -> Optional[Any]:
        value = await self.hget(key, field)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    async def hset_json(self, key: str, field: str, value: Any) -> int:
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            return await self.hset(key, field, json_str)
        except Exception as e:
            logger.debug(f"Redis hset_json failed: {e}")
            return 0

    async def hmset_json(self, key: str, mapping: Dict[str, Any]) -> bool:
        if not mapping:
            return False
        try:
            json_mapping = {}
            for k, v in mapping.items():
                json_mapping[k] = json.dumps(v, ensure_ascii=False, default=str)
            return await self.hmset(key, json_mapping)
        except Exception as e:
            logger.debug(f"Redis hmset_json failed: {e}")
            return False

    async def hgetall_json(self, key: str) -> Dict[str, Any]:
        data = await self.hgetall(key)
        result = {}
        for k, v in data.items():
            try:
                result[k] = json.loads(v)
            except json.JSONDecodeError:
                result[k] = v
        return result

    async def hkeys(self, key: str) -> List[str]:
        async with self._async_lock:
            return list(self._hashes.get(key, {}).keys())

    async def hvals(self, key: str) -> List[str]:
        async with self._async_lock:
            return list(self._hashes.get(key, {}).values())

    async def hvals_json(self, key: str) -> List[Any]:
        values = await self.hvals(key)
        result = []
        for v in values:
            try:
                result.append(json.loads(v))
            except json.JSONDecodeError:
                result.append(v)
        return result

    # Set Operations
    async def sadd(self, key: str, *members: str) -> int:
        async with self._async_lock:
            if key not in self._sets:
                self._sets[key] = set()
            before = len(self._sets[key])
            self._sets[key].update(members)
            return len(self._sets[key]) - before

    async def srem(self, key: str, *members: str) -> int:
        async with self._async_lock:
            if key not in self._sets:
                return 0
            count = 0
            for m in members:
                if m in self._sets[key]:
                    self._sets[key].remove(m)
                    count += 1
            return count

    async def sismember(self, key: str, member: str) -> bool:
        async with self._async_lock:
            return member in self._sets.get(key, set())

    async def smembers(self, key: str) -> Set[str]:
        async with self._async_lock:
            return set(self._sets.get(key, set()))

    async def scard(self, key: str) -> int:
        async with self._async_lock:
            return len(self._sets.get(key, set()))

    # Sorted Set Operations
    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        async with self._async_lock:
            if key not in self._sorted_sets:
                self._sorted_sets[key] = {}
            before = len(self._sorted_sets[key])
            self._sorted_sets[key].update(mapping)
            return len(self._sorted_sets[key]) - before

    async def zrangebyscore(
        self, key: str, min_score: float, max_score: float, withscores: bool = False
    ) -> List:
        async with self._async_lock:
            if key not in self._sorted_sets:
                return []
            items = [
                (m, s) for m, s in self._sorted_sets[key].items() if min_score <= s <= max_score
            ]
            items.sort(key=lambda x: x[1])
            if withscores:
                return items
            return [m for m, s in items]

    async def zrem(self, key: str, *members: str) -> int:
        async with self._async_lock:
            if key not in self._sorted_sets:
                return 0
            count = 0
            for m in members:
                if m in self._sorted_sets[key]:
                    del self._sorted_sets[key][m]
                    count += 1
            return count

    async def zscore(self, key: str, member: str) -> Optional[float]:
        async with self._async_lock:
            return self._sorted_sets.get(key, {}).get(member)

    async def zcard(self, key: str) -> int:
        async with self._async_lock:
            return len(self._sorted_sets.get(key, {}))

    # Atomic Operations
    async def incr(self, key: str, amount: int = 1) -> int:
        async with self._async_lock:
            current = int(self._data.get(key, 0))
            new_value = current + amount
            self._data[key] = str(new_value)
            return new_value

    async def decr(self, key: str, amount: int = 1) -> int:
        async with self._async_lock:
            current = int(self._data.get(key, 0))
            new_value = current - amount
            self._data[key] = str(new_value)
            return new_value

    async def getset(self, key: str, value: str) -> Optional[str]:
        async with self._async_lock:
            old_value = self._data.get(key)
            self._data[key] = value
            return old_value

    # Lock Operations
    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: float = 5.0,
    ) -> Optional[str]:
        import time

        lock_key = f"lock:{lock_name}"
        token = str(uuid.uuid4())

        if blocking:
            start_time = time.time()
            while time.time() - start_time < blocking_timeout:
                if await self.set(lock_key, token, ttl=timeout, nx=True):
                    return token
                await asyncio.sleep(0.1)  # Non-blocking
            return None
        else:
            if await self.set(lock_key, token, ttl=timeout, nx=True):
                return token
            return None

    async def release_lock(self, lock_name: str, token: str) -> bool:
        lock_key = f"lock:{lock_name}"
        current_token = await self.get(lock_key)
        if current_token == token:
            await self.delete(lock_key)
            return True
        return False

    # Utility Methods
    async def keys(self, pattern: str = "*") -> List[str]:
        async with self._async_lock:
            import fnmatch

            all_keys = (
                set(self._data.keys())
                | set(self._lists.keys())
                | set(self._hashes.keys())
                | set(self._sets.keys())
                | set(self._sorted_sets.keys())
            )
            if pattern == "*":
                return list(all_keys)
            return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]

    async def flush_db(self) -> bool:
        async with self._async_lock:
            self._data.clear()
            self._lists.clear()
            self._hashes.clear()
            self._sets.clear()
            self._sorted_sets.clear()
            self._expiry.clear()
            return True

    async def info(self) -> Dict[str, Any]:
        async with self._async_lock:
            return {
                "keys": len(self._data)
                + len(self._lists)
                + len(self._hashes)
                + len(self._sets)
                + len(self._sorted_sets),
                "type": "in_memory",
            }

    async def close(self):
        pass


async def get_cache(config: Optional[RedisCacheConfig] = None) -> Union[RedisCache, InMemoryCache]:
    """
    Get the best available cache implementation.
    Returns RedisCache if Redis is available, otherwise InMemoryCache.

    Args:
        config: Redis configuration

    Returns:
        Cache instance (RedisCache or InMemoryCache)
    """
    redis_cache = get_redis_cache(config)
    if await redis_cache.is_connected():
        return redis_cache
    else:
        logger.warning("Redis unavailable, falling back to in-memory cache")
        return InMemoryCache()
