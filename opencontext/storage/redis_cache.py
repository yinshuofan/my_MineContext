#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Redis Cache Manager (Async + Sync)

Provides a unified Redis caching layer for multi-instance deployment.
Supports both synchronous and asynchronous operations.

Usage:
    # Sync (backward compatible)
    cache.get("key")
    cache.set("key", "value")

    # Async (new, non-blocking)
    await cache.async_get("key")
    await cache.async_set("key", "value")
"""

import asyncio
import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union

from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Global Redis client instance
_redis_client: Optional["RedisCache"] = None
_redis_lock = threading.Lock()


@dataclass
class RedisCacheConfig:
    """Redis cache configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    key_prefix: str = "opencontext:"
    default_ttl: int = 3600  # 1 hour
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    decode_responses: bool = True


class RedisCache:
    """
    Redis Cache Manager (Async + Sync)

    Provides a unified interface for Redis operations with support for:
    - Key-value storage with TTL
    - List operations (for message buffers)
    - Hash operations (for structured data)
    - Set operations (for deduplication)
    - Atomic operations for multi-instance safety

    All methods have both sync and async versions:
    - Sync: get(), set(), rpush(), etc. (backward compatible)
    - Async: async_get(), async_set(), async_rpush(), etc. (non-blocking)
    """

    def __init__(self, config: Optional[RedisCacheConfig] = None):
        """
        Initialize Redis cache manager.

        Args:
            config: Redis configuration. If None, uses default config.
        """
        self.config = config or RedisCacheConfig()
        self._sync_client = None
        self._async_client = None
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._connected = False
        self._async_connected = False

        # Try to connect sync client
        self._connect_sync()

    # =========================================================================
    # Connection Management
    # =========================================================================

    def _connect_sync(self) -> bool:
        """
        Establish synchronous connection to Redis server.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            import redis

            self._sync_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                decode_responses=self.config.decode_responses,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                max_connections=self.config.max_connections,
            )

            # Test connection
            self._sync_client.ping()
            self._connected = True
            logger.info(f"Redis sync client connected: {self.config.host}:{self.config.port}/{self.config.db}")
            return True

        except ImportError:
            logger.error("Redis package not installed. Run: pip install redis")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect sync Redis client: {e}")
            self._connected = False
            return False

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

            self._async_client = aioredis.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                decode_responses=self.config.decode_responses,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                max_connections=self.config.max_connections,
            )

            # Test connection
            await self._async_client.ping()
            self._async_connected = True
            logger.info(f"Redis async client connected: {self.config.host}:{self.config.port}/{self.config.db}")
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
        if not self._async_connected:
            return await self._connect_async()
        return True

    def is_connected(self) -> bool:
        """Check if sync Redis is connected."""
        if not self._connected or not self._sync_client:
            return False
        try:
            self._sync_client.ping()
            return True
        except Exception:
            self._connected = False
            return False

    async def async_is_connected(self) -> bool:
        """Check if async Redis is connected."""
        if not self._async_connected or not self._async_client:
            return await self._connect_async()
        try:
            await self._async_client.ping()
            return True
        except Exception:
            self._async_connected = False
            return False

    def _make_key(self, key: str) -> str:
        """Generate full Redis key with prefix."""
        return f"{self.config.key_prefix}{key}"

    # =========================================================================
    # Basic Key-Value Operations (Sync)
    # =========================================================================

    def get(self, key: str) -> Optional[str]:
        """Get value by key (sync)."""
        if not self.is_connected():
            return None
        try:
            return self._sync_client.get(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None

    def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set key-value pair (sync)."""
        if not self.is_connected():
            return False
        try:
            ttl = ttl if ttl is not None else self.config.default_ttl
            return bool(self._sync_client.set(
                self._make_key(key),
                value,
                ex=ttl if ttl > 0 else None,
                nx=nx,
                xx=xx,
            ))
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False

    def delete(self, *keys: str) -> int:
        """Delete one or more keys (sync)."""
        if not self.is_connected() or not keys:
            return 0
        try:
            full_keys = [self._make_key(k) for k in keys]
            return self._sync_client.delete(*full_keys)
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return 0

    def exists(self, key: str) -> bool:
        """Check if key exists (sync)."""
        if not self.is_connected():
            return False
        try:
            return bool(self._sync_client.exists(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for a key (sync)."""
        if not self.is_connected():
            return False
        try:
            return bool(self._sync_client.expire(self._make_key(key), ttl))
        except Exception as e:
            logger.error(f"Redis EXPIRE error: {e}")
            return False

    def ttl(self, key: str) -> int:
        """Get remaining TTL for a key (sync)."""
        if not self.is_connected():
            return -2
        try:
            return self._sync_client.ttl(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis TTL error: {e}")
            return -2

    # =========================================================================
    # Basic Key-Value Operations (Async)
    # =========================================================================

    async def async_get(self, key: str) -> Optional[str]:
        """Get value by key (async, non-blocking)."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.get(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis async GET error: {e}")
            return None

    async def async_set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set key-value pair (async, non-blocking)."""
        if not await self._ensure_async_client():
            return False
        try:
            ttl = ttl if ttl is not None else self.config.default_ttl
            return bool(await self._async_client.set(
                self._make_key(key),
                value,
                ex=ttl if ttl > 0 else None,
                nx=nx,
                xx=xx,
            ))
        except Exception as e:
            logger.error(f"Redis async SET error: {e}")
            return False

    async def async_delete(self, *keys: str) -> int:
        """Delete one or more keys (async, non-blocking)."""
        if not await self._ensure_async_client() or not keys:
            return 0
        try:
            full_keys = [self._make_key(k) for k in keys]
            return await self._async_client.delete(*full_keys)
        except Exception as e:
            logger.error(f"Redis async DELETE error: {e}")
            return 0

    async def async_exists(self, key: str) -> bool:
        """Check if key exists (async, non-blocking)."""
        if not await self._ensure_async_client():
            return False
        try:
            return bool(await self._async_client.exists(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis async EXISTS error: {e}")
            return False

    async def async_expire(self, key: str, ttl: int) -> bool:
        """Set TTL for a key (async, non-blocking)."""
        if not await self._ensure_async_client():
            return False
        try:
            return bool(await self._async_client.expire(self._make_key(key), ttl))
        except Exception as e:
            logger.error(f"Redis async EXPIRE error: {e}")
            return False

    async def async_ttl(self, key: str) -> int:
        """Get remaining TTL for a key (async, non-blocking)."""
        if not await self._ensure_async_client():
            return -2
        try:
            return await self._async_client.ttl(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis async TTL error: {e}")
            return -2

    # =========================================================================
    # JSON Operations (Sync)
    # =========================================================================

    def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value by key (sync)."""
        value = self.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for key {key}: {e}")
            return None

    def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set JSON value (sync)."""
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            return self.set(key, json_str, ttl=ttl, nx=nx, xx=xx)
        except Exception as e:
            logger.error(f"JSON encode error for key {key}: {e}")
            return False

    # =========================================================================
    # JSON Operations (Async)
    # =========================================================================

    async def async_get_json(self, key: str) -> Optional[Any]:
        """Get JSON value by key (async, non-blocking)."""
        value = await self.async_get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for key {key}: {e}")
            return None

    async def async_set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set JSON value (async, non-blocking)."""
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            return await self.async_set(key, json_str, ttl=ttl, nx=nx, xx=xx)
        except Exception as e:
            logger.error(f"JSON encode error for key {key}: {e}")
            return False

    # =========================================================================
    # List Operations (Sync)
    # =========================================================================

    def lpush(self, key: str, *values: str) -> int:
        """Push values to the left of a list (sync)."""
        if not self.is_connected() or not values:
            return 0
        try:
            return self._sync_client.lpush(self._make_key(key), *values)
        except Exception as e:
            logger.error(f"Redis LPUSH error: {e}")
            return 0

    def rpush(self, key: str, *values: str) -> int:
        """Push values to the right of a list (sync)."""
        if not self.is_connected() or not values:
            return 0
        try:
            return self._sync_client.rpush(self._make_key(key), *values)
        except Exception as e:
            logger.error(f"Redis RPUSH error: {e}")
            return 0

    def lpop(self, key: str, count: int = 1) -> Optional[Union[str, List[str]]]:
        """Pop values from the left of a list (sync)."""
        if not self.is_connected():
            return None
        try:
            return self._sync_client.lpop(self._make_key(key), count)
        except Exception as e:
            logger.error(f"Redis LPOP error: {e}")
            return None

    def rpop(self, key: str, count: int = 1) -> Optional[Union[str, List[str]]]:
        """Pop values from the right of a list (sync)."""
        if not self.is_connected():
            return None
        try:
            return self._sync_client.rpop(self._make_key(key), count)
        except Exception as e:
            logger.error(f"Redis RPOP error: {e}")
            return None

    def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get range of elements from a list (sync)."""
        if not self.is_connected():
            return []
        try:
            return self._sync_client.lrange(self._make_key(key), start, end)
        except Exception as e:
            logger.error(f"Redis LRANGE error: {e}")
            return []

    def llen(self, key: str) -> int:
        """Get length of a list (sync)."""
        if not self.is_connected():
            return 0
        try:
            return self._sync_client.llen(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis LLEN error: {e}")
            return 0

    def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim a list to specified range (sync)."""
        if not self.is_connected():
            return False
        try:
            return bool(self._sync_client.ltrim(self._make_key(key), start, end))
        except Exception as e:
            logger.error(f"Redis LTRIM error: {e}")
            return False

    def lrange_json(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range of JSON elements from a list (sync)."""
        items = self.lrange(key, start, end)
        result = []
        for item in items:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                result.append(item)
        return result

    def rpush_json(self, key: str, *values: Any) -> int:
        """Push JSON values to the right of a list (sync)."""
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception as e:
                logger.error(f"JSON encode error: {e}")
        if not json_values:
            return 0
        return self.rpush(key, *json_values)

    def lpush_json(self, key: str, *values: Any) -> int:
        """Push JSON values to the left of a list (sync)."""
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception as e:
                logger.error(f"JSON encode error: {e}")
        if not json_values:
            return 0
        return self.lpush(key, *json_values)

    # =========================================================================
    # List Operations (Async)
    # =========================================================================

    async def async_lpush(self, key: str, *values: str) -> int:
        """Push values to the left of a list (async, non-blocking)."""
        if not await self._ensure_async_client() or not values:
            return 0
        try:
            return await self._async_client.lpush(self._make_key(key), *values)
        except Exception as e:
            logger.error(f"Redis async LPUSH error: {e}")
            return 0

    async def async_rpush(self, key: str, *values: str) -> int:
        """Push values to the right of a list (async, non-blocking)."""
        if not await self._ensure_async_client() or not values:
            return 0
        try:
            return await self._async_client.rpush(self._make_key(key), *values)
        except Exception as e:
            logger.error(f"Redis async RPUSH error: {e}")
            return 0

    async def async_lpop(self, key: str, count: int = 1) -> Optional[Union[str, List[str]]]:
        """Pop values from the left of a list (async, non-blocking)."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.lpop(self._make_key(key), count)
        except Exception as e:
            logger.error(f"Redis async LPOP error: {e}")
            return None

    async def async_rpop(self, key: str, count: int = 1) -> Optional[Union[str, List[str]]]:
        """Pop values from the right of a list (async, non-blocking)."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.rpop(self._make_key(key), count)
        except Exception as e:
            logger.error(f"Redis async RPOP error: {e}")
            return None

    async def async_lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get range of elements from a list (async, non-blocking)."""
        if not await self._ensure_async_client():
            return []
        try:
            return await self._async_client.lrange(self._make_key(key), start, end)
        except Exception as e:
            logger.error(f"Redis async LRANGE error: {e}")
            return []

    async def async_llen(self, key: str) -> int:
        """Get length of a list (async, non-blocking)."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.llen(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis async LLEN error: {e}")
            return 0

    async def async_ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim a list to specified range (async, non-blocking)."""
        if not await self._ensure_async_client():
            return False
        try:
            return bool(await self._async_client.ltrim(self._make_key(key), start, end))
        except Exception as e:
            logger.error(f"Redis async LTRIM error: {e}")
            return False

    async def async_lrange_json(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range of JSON elements from a list (async, non-blocking)."""
        items = await self.async_lrange(key, start, end)
        result = []
        for item in items:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                result.append(item)
        return result

    async def async_rpush_json(self, key: str, *values: Any) -> int:
        """Push JSON values to the right of a list (async, non-blocking)."""
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception as e:
                logger.error(f"JSON encode error: {e}")
        if not json_values:
            return 0
        return await self.async_rpush(key, *json_values)

    async def async_lpush_json(self, key: str, *values: Any) -> int:
        """Push JSON values to the left of a list (async, non-blocking)."""
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception as e:
                logger.error(f"JSON encode error: {e}")
        if not json_values:
            return 0
        return await self.async_lpush(key, *json_values)

    # =========================================================================
    # Hash Operations (Sync)
    # =========================================================================

    def hget(self, key: str, field: str) -> Optional[str]:
        """Get a field from a hash (sync)."""
        if not self.is_connected():
            return None
        try:
            return self._sync_client.hget(self._make_key(key), field)
        except Exception as e:
            logger.error(f"Redis HGET error: {e}")
            return None

    def hset(self, key: str, field: str, value: str) -> int:
        """Set a field in a hash (sync)."""
        if not self.is_connected():
            return 0
        try:
            return self._sync_client.hset(self._make_key(key), field, value)
        except Exception as e:
            logger.error(f"Redis HSET error: {e}")
            return 0

    def hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        """Set multiple fields in a hash (sync)."""
        if not self.is_connected() or not mapping:
            return False
        try:
            return bool(self._sync_client.hset(self._make_key(key), mapping=mapping))
        except Exception as e:
            logger.error(f"Redis HMSET error: {e}")
            return False

    def hgetall(self, key: str) -> Dict[str, str]:
        """Get all fields and values from a hash (sync)."""
        if not self.is_connected():
            return {}
        try:
            return self._sync_client.hgetall(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis HGETALL error: {e}")
            return {}

    def hdel(self, key: str, *fields: str) -> int:
        """Delete fields from a hash (sync)."""
        if not self.is_connected() or not fields:
            return 0
        try:
            return self._sync_client.hdel(self._make_key(key), *fields)
        except Exception as e:
            logger.error(f"Redis HDEL error: {e}")
            return 0

    def hlen(self, key: str) -> int:
        """Get number of fields in a hash (sync)."""
        if not self.is_connected():
            return 0
        try:
            return self._sync_client.hlen(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis HLEN error: {e}")
            return 0

    def hexists(self, key: str, field: str) -> bool:
        """Check if a field exists in a hash (sync)."""
        if not self.is_connected():
            return False
        try:
            return bool(self._sync_client.hexists(self._make_key(key), field))
        except Exception as e:
            logger.error(f"Redis HEXISTS error: {e}")
            return False

    def hget_json(self, key: str, field: str) -> Optional[Any]:
        """Get a JSON field from a hash (sync)."""
        value = self.hget(key, field)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def hset_json(self, key: str, field: str, value: Any) -> int:
        """Set a JSON field in a hash (sync)."""
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            return self.hset(key, field, json_str)
        except Exception as e:
            logger.error(f"Redis HSET_JSON error: {e}")
            return 0

    def hmset_json(self, key: str, mapping: Dict[str, Any]) -> bool:
        """Set multiple JSON fields in a hash (sync)."""
        if not mapping:
            return False
        try:
            json_mapping = {}
            for k, v in mapping.items():
                json_mapping[k] = json.dumps(v, ensure_ascii=False, default=str)
            return self.hmset(key, json_mapping)
        except Exception as e:
            logger.error(f"Redis HMSET_JSON error: {e}")
            return False

    def hgetall_json(self, key: str) -> Dict[str, Any]:
        """Get all fields from a hash as JSON objects (sync)."""
        data = self.hgetall(key)
        result = {}
        for k, v in data.items():
            try:
                result[k] = json.loads(v)
            except json.JSONDecodeError:
                result[k] = v
        return result

    def hkeys(self, key: str) -> List[str]:
        """Get all field names from a hash (sync)."""
        if not self.is_connected():
            return []
        try:
            return list(self._sync_client.hkeys(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis HKEYS error: {e}")
            return []

    def hvals(self, key: str) -> List[str]:
        """Get all values from a hash (sync)."""
        if not self.is_connected():
            return []
        try:
            return list(self._sync_client.hvals(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis HVALS error: {e}")
            return []

    def hvals_json(self, key: str) -> List[Any]:
        """Get all values from a hash as JSON objects (sync)."""
        values = self.hvals(key)
        result = []
        for v in values:
            try:
                result.append(json.loads(v))
            except json.JSONDecodeError:
                result.append(v)
        return result

    # =========================================================================
    # Hash Operations (Async)
    # =========================================================================

    async def async_hget(self, key: str, field: str) -> Optional[str]:
        """Get a field from a hash (async, non-blocking)."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.hget(self._make_key(key), field)
        except Exception as e:
            logger.error(f"Redis async HGET error: {e}")
            return None

    async def async_hset(self, key: str, field: str, value: str) -> int:
        """Set a field in a hash (async, non-blocking)."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.hset(self._make_key(key), field, value)
        except Exception as e:
            logger.error(f"Redis async HSET error: {e}")
            return 0

    async def async_hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        """Set multiple fields in a hash (async, non-blocking)."""
        if not await self._ensure_async_client() or not mapping:
            return False
        try:
            return bool(await self._async_client.hset(self._make_key(key), mapping=mapping))
        except Exception as e:
            logger.error(f"Redis async HMSET error: {e}")
            return False

    async def async_hgetall(self, key: str) -> Dict[str, str]:
        """Get all fields and values from a hash (async, non-blocking)."""
        if not await self._ensure_async_client():
            return {}
        try:
            return await self._async_client.hgetall(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis async HGETALL error: {e}")
            return {}

    async def async_hdel(self, key: str, *fields: str) -> int:
        """Delete fields from a hash (async, non-blocking)."""
        if not await self._ensure_async_client() or not fields:
            return 0
        try:
            return await self._async_client.hdel(self._make_key(key), *fields)
        except Exception as e:
            logger.error(f"Redis async HDEL error: {e}")
            return 0

    async def async_hlen(self, key: str) -> int:
        """Get number of fields in a hash (async, non-blocking)."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.hlen(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis async HLEN error: {e}")
            return 0

    async def async_hexists(self, key: str, field: str) -> bool:
        """Check if a field exists in a hash (async, non-blocking)."""
        if not await self._ensure_async_client():
            return False
        try:
            return bool(await self._async_client.hexists(self._make_key(key), field))
        except Exception as e:
            logger.error(f"Redis async HEXISTS error: {e}")
            return False

    async def async_hget_json(self, key: str, field: str) -> Optional[Any]:
        """Get a JSON field from a hash (async, non-blocking)."""
        value = await self.async_hget(key, field)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    async def async_hset_json(self, key: str, field: str, value: Any) -> int:
        """Set a JSON field in a hash (async, non-blocking)."""
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            return await self.async_hset(key, field, json_str)
        except Exception as e:
            logger.error(f"Redis async HSET_JSON error: {e}")
            return 0

    async def async_hmset_json(self, key: str, mapping: Dict[str, Any]) -> bool:
        """Set multiple JSON fields in a hash (async, non-blocking)."""
        if not mapping:
            return False
        try:
            json_mapping = {}
            for k, v in mapping.items():
                json_mapping[k] = json.dumps(v, ensure_ascii=False, default=str)
            return await self.async_hmset(key, json_mapping)
        except Exception as e:
            logger.error(f"Redis async HMSET_JSON error: {e}")
            return False

    async def async_hgetall_json(self, key: str) -> Dict[str, Any]:
        """Get all fields from a hash as JSON objects (async, non-blocking)."""
        data = await self.async_hgetall(key)
        result = {}
        for k, v in data.items():
            try:
                result[k] = json.loads(v)
            except json.JSONDecodeError:
                result[k] = v
        return result

    async def async_hkeys(self, key: str) -> List[str]:
        """Get all field names from a hash (async, non-blocking)."""
        if not await self._ensure_async_client():
            return []
        try:
            return list(await self._async_client.hkeys(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis async HKEYS error: {e}")
            return []

    async def async_hvals(self, key: str) -> List[str]:
        """Get all values from a hash (async, non-blocking)."""
        if not await self._ensure_async_client():
            return []
        try:
            return list(await self._async_client.hvals(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis async HVALS error: {e}")
            return []

    async def async_hvals_json(self, key: str) -> List[Any]:
        """Get all values from a hash as JSON objects (async, non-blocking)."""
        values = await self.async_hvals(key)
        result = []
        for v in values:
            try:
                result.append(json.loads(v))
            except json.JSONDecodeError:
                result.append(v)
        return result

    # =========================================================================
    # Set Operations (Sync)
    # =========================================================================

    def sadd(self, key: str, *members: str) -> int:
        """Add members to a set (sync)."""
        if not self.is_connected() or not members:
            return 0
        try:
            return self._sync_client.sadd(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis SADD error: {e}")
            return 0

    def srem(self, key: str, *members: str) -> int:
        """Remove members from a set (sync)."""
        if not self.is_connected() or not members:
            return 0
        try:
            return self._sync_client.srem(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis SREM error: {e}")
            return 0

    def sismember(self, key: str, member: str) -> bool:
        """Check if a member is in a set (sync)."""
        if not self.is_connected():
            return False
        try:
            return bool(self._sync_client.sismember(self._make_key(key), member))
        except Exception as e:
            logger.error(f"Redis SISMEMBER error: {e}")
            return False

    def smembers(self, key: str) -> Set[str]:
        """Get all members of a set (sync)."""
        if not self.is_connected():
            return set()
        try:
            return self._sync_client.smembers(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis SMEMBERS error: {e}")
            return set()

    def scard(self, key: str) -> int:
        """Get number of members in a set (sync)."""
        if not self.is_connected():
            return 0
        try:
            return self._sync_client.scard(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis SCARD error: {e}")
            return 0

    # =========================================================================
    # Set Operations (Async)
    # =========================================================================

    async def async_sadd(self, key: str, *members: str) -> int:
        """Add members to a set (async, non-blocking)."""
        if not await self._ensure_async_client() or not members:
            return 0
        try:
            return await self._async_client.sadd(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis async SADD error: {e}")
            return 0

    async def async_srem(self, key: str, *members: str) -> int:
        """Remove members from a set (async, non-blocking)."""
        if not await self._ensure_async_client() or not members:
            return 0
        try:
            return await self._async_client.srem(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis async SREM error: {e}")
            return 0

    async def async_sismember(self, key: str, member: str) -> bool:
        """Check if a member is in a set (async, non-blocking)."""
        if not await self._ensure_async_client():
            return False
        try:
            return bool(await self._async_client.sismember(self._make_key(key), member))
        except Exception as e:
            logger.error(f"Redis async SISMEMBER error: {e}")
            return False

    async def async_smembers(self, key: str) -> Set[str]:
        """Get all members of a set (async, non-blocking)."""
        if not await self._ensure_async_client():
            return set()
        try:
            return await self._async_client.smembers(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis async SMEMBERS error: {e}")
            return set()

    async def async_scard(self, key: str) -> int:
        """Get number of members in a set (async, non-blocking)."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.scard(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis async SCARD error: {e}")
            return 0

    # =========================================================================
    # Sorted Set Operations (Sync)
    # =========================================================================

    def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Add members to a sorted set with their scores (sync)."""
        if not self.is_connected() or not mapping:
            return 0
        try:
            return self._sync_client.zadd(self._make_key(key), mapping)
        except Exception as e:
            logger.error(f"Redis ZADD error: {e}")
            return 0

    def zrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
        withscores: bool = False
    ) -> List[str]:
        """Get members from a sorted set within a score range (sync)."""
        if not self.is_connected():
            return []
        try:
            full_key = self._make_key(key)
            if withscores:
                return self._sync_client.zrangebyscore(full_key, min_score, max_score, withscores=True)
            else:
                return self._sync_client.zrangebyscore(full_key, min_score, max_score)
        except Exception as e:
            logger.error(f"Redis ZRANGEBYSCORE error: {e}")
            return []

    def zrem(self, key: str, *members: str) -> int:
        """Remove members from a sorted set (sync)."""
        if not self.is_connected() or not members:
            return 0
        try:
            return self._sync_client.zrem(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis ZREM error: {e}")
            return 0

    def zscore(self, key: str, member: str) -> Optional[float]:
        """Get the score of a member in a sorted set (sync)."""
        if not self.is_connected():
            return None
        try:
            return self._sync_client.zscore(self._make_key(key), member)
        except Exception as e:
            logger.error(f"Redis ZSCORE error: {e}")
            return None

    def zcard(self, key: str) -> int:
        """Get number of members in a sorted set (sync)."""
        if not self.is_connected():
            return 0
        try:
            return self._sync_client.zcard(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis ZCARD error: {e}")
            return 0

    # =========================================================================
    # Sorted Set Operations (Async)
    # =========================================================================

    async def async_zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Add members to a sorted set with their scores (async, non-blocking)."""
        if not await self._ensure_async_client() or not mapping:
            return 0
        try:
            return await self._async_client.zadd(self._make_key(key), mapping)
        except Exception as e:
            logger.error(f"Redis async ZADD error: {e}")
            return 0

    async def async_zrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
        withscores: bool = False
    ) -> List[str]:
        """Get members from a sorted set within a score range (async, non-blocking)."""
        if not await self._ensure_async_client():
            return []
        try:
            full_key = self._make_key(key)
            if withscores:
                return await self._async_client.zrangebyscore(full_key, min_score, max_score, withscores=True)
            else:
                return await self._async_client.zrangebyscore(full_key, min_score, max_score)
        except Exception as e:
            logger.error(f"Redis async ZRANGEBYSCORE error: {e}")
            return []

    async def async_zrem(self, key: str, *members: str) -> int:
        """Remove members from a sorted set (async, non-blocking)."""
        if not await self._ensure_async_client() or not members:
            return 0
        try:
            return await self._async_client.zrem(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis async ZREM error: {e}")
            return 0

    async def async_zscore(self, key: str, member: str) -> Optional[float]:
        """Get the score of a member in a sorted set (async, non-blocking)."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.zscore(self._make_key(key), member)
        except Exception as e:
            logger.error(f"Redis async ZSCORE error: {e}")
            return None

    async def async_zcard(self, key: str) -> int:
        """Get number of members in a sorted set (async, non-blocking)."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.zcard(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis async ZCARD error: {e}")
            return 0

    # =========================================================================
    # Atomic Operations (Sync)
    # =========================================================================

    def incr(self, key: str, amount: int = 1) -> int:
        """Increment a key's value atomically (sync)."""
        if not self.is_connected():
            return 0
        try:
            return self._sync_client.incr(self._make_key(key), amount)
        except Exception as e:
            logger.error(f"Redis INCR error: {e}")
            return 0

    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement a key's value atomically (sync)."""
        if not self.is_connected():
            return 0
        try:
            return self._sync_client.decr(self._make_key(key), amount)
        except Exception as e:
            logger.error(f"Redis DECR error: {e}")
            return 0

    def getset(self, key: str, value: str) -> Optional[str]:
        """Set a new value and return the old value atomically (sync)."""
        if not self.is_connected():
            return None
        try:
            return self._sync_client.getset(self._make_key(key), value)
        except Exception as e:
            logger.error(f"Redis GETSET error: {e}")
            return None

    # =========================================================================
    # Atomic Operations (Async)
    # =========================================================================

    async def async_incr(self, key: str, amount: int = 1) -> int:
        """Increment a key's value atomically (async, non-blocking)."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.incr(self._make_key(key), amount)
        except Exception as e:
            logger.error(f"Redis async INCR error: {e}")
            return 0

    async def async_decr(self, key: str, amount: int = 1) -> int:
        """Decrement a key's value atomically (async, non-blocking)."""
        if not await self._ensure_async_client():
            return 0
        try:
            return await self._async_client.decr(self._make_key(key), amount)
        except Exception as e:
            logger.error(f"Redis async DECR error: {e}")
            return 0

    async def async_getset(self, key: str, value: str) -> Optional[str]:
        """Set a new value and return the old value atomically (async, non-blocking)."""
        if not await self._ensure_async_client():
            return None
        try:
            return await self._async_client.getset(self._make_key(key), value)
        except Exception as e:
            logger.error(f"Redis async GETSET error: {e}")
            return None

    # =========================================================================
    # Lock Operations (Sync) - Uses blocking time.sleep()
    # =========================================================================

    def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: float = 5.0,
    ) -> Optional[str]:
        """
        Acquire a distributed lock (sync, may block).

        WARNING: This method uses time.sleep() which blocks the event loop.
        Use async_acquire_lock() in async contexts.

        Args:
            lock_name: Name of the lock
            timeout: Lock expiration time in seconds
            blocking: Whether to block until lock is acquired
            blocking_timeout: Maximum time to wait for lock

        Returns:
            Lock token if acquired, None otherwise
        """
        if not self.is_connected():
            return None
        try:
            import time

            lock_key = f"lock:{lock_name}"
            token = str(uuid.uuid4())

            if blocking:
                start_time = time.time()
                while time.time() - start_time < blocking_timeout:
                    if self.set(lock_key, token, ttl=timeout, nx=True):
                        return token
                    time.sleep(0.1)  # Blocking sleep
                return None
            else:
                if self.set(lock_key, token, ttl=timeout, nx=True):
                    return token
                return None

        except Exception as e:
            logger.error(f"Redis acquire_lock error: {e}")
            return None

    def release_lock(self, lock_name: str, token: str) -> bool:
        """
        Release a distributed lock (sync).

        Args:
            lock_name: Name of the lock
            token: Lock token returned by acquire_lock

        Returns:
            True if lock was released
        """
        if not self.is_connected():
            return False
        try:
            lock_key = f"lock:{lock_name}"
            # Only release if we own the lock
            current_token = self.get(lock_key)
            if current_token == token:
                self.delete(lock_key)
                return True
            return False
        except Exception as e:
            logger.error(f"Redis release_lock error: {e}")
            return False

    # =========================================================================
    # Lock Operations (Async) - Uses non-blocking asyncio.sleep()
    # =========================================================================

    async def async_acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: float = 5.0,
    ) -> Optional[str]:
        """
        Acquire a distributed lock (async, non-blocking).

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
                    if await self.async_set(lock_key, token, ttl=timeout, nx=True):
                        return token
                    await asyncio.sleep(0.1)  # Non-blocking sleep!
                return None
            else:
                if await self.async_set(lock_key, token, ttl=timeout, nx=True):
                    return token
                return None

        except Exception as e:
            logger.error(f"Redis async_acquire_lock error: {e}")
            return None

    async def async_release_lock(self, lock_name: str, token: str) -> bool:
        """
        Release a distributed lock (async, non-blocking).

        Args:
            lock_name: Name of the lock
            token: Lock token returned by async_acquire_lock

        Returns:
            True if lock was released
        """
        if not await self._ensure_async_client():
            return False
        try:
            lock_key = f"lock:{lock_name}"
            # Only release if we own the lock
            current_token = await self.async_get(lock_key)
            if current_token == token:
                await self.async_delete(lock_key)
                return True
            return False
        except Exception as e:
            logger.error(f"Redis async_release_lock error: {e}")
            return False

    # =========================================================================
    # Utility Methods (Sync)
    # =========================================================================

    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern (sync)."""
        if not self.is_connected():
            return []
        try:
            full_pattern = self._make_key(pattern)
            keys = self._sync_client.keys(full_pattern)
            prefix_len = len(self.config.key_prefix)
            return [k[prefix_len:] if isinstance(k, str) else k.decode()[prefix_len:] for k in keys]
        except Exception as e:
            logger.error(f"Redis KEYS error: {e}")
            return []

    def flush_db(self) -> bool:
        """Flush the current database (sync). Use with caution!"""
        if not self.is_connected():
            return False
        try:
            self._sync_client.flushdb()
            logger.warning("Redis database flushed!")
            return True
        except Exception as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            return False

    def info(self) -> Dict[str, Any]:
        """Get Redis server info (sync)."""
        if not self.is_connected():
            return {}
        try:
            return self._sync_client.info()
        except Exception as e:
            logger.error(f"Redis INFO error: {e}")
            return {}

    # =========================================================================
    # Utility Methods (Async)
    # =========================================================================

    async def async_keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern (async, non-blocking)."""
        if not await self._ensure_async_client():
            return []
        try:
            full_pattern = self._make_key(pattern)
            keys = await self._async_client.keys(full_pattern)
            prefix_len = len(self.config.key_prefix)
            return [k[prefix_len:] if isinstance(k, str) else k.decode()[prefix_len:] for k in keys]
        except Exception as e:
            logger.error(f"Redis async KEYS error: {e}")
            return []

    async def async_flush_db(self) -> bool:
        """Flush the current database (async). Use with caution!"""
        if not await self._ensure_async_client():
            return False
        try:
            await self._async_client.flushdb()
            logger.warning("Redis database flushed!")
            return True
        except Exception as e:
            logger.error(f"Redis async FLUSHDB error: {e}")
            return False

    async def async_info(self) -> Dict[str, Any]:
        """Get Redis server info (async, non-blocking)."""
        if not await self._ensure_async_client():
            return {}
        try:
            return await self._async_client.info()
        except Exception as e:
            logger.error(f"Redis async INFO error: {e}")
            return {}

    # =========================================================================
    # Connection Cleanup
    # =========================================================================

    def close(self):
        """Close sync Redis connection."""
        if self._sync_client:
            try:
                self._sync_client.close()
            except Exception:
                pass
            self._sync_client = None
            self._connected = False
            logger.info("Redis sync connection closed")

    async def async_close(self):
        """Close async Redis connection."""
        if self._async_client:
            try:
                await self._async_client.close()
            except Exception:
                pass
            self._async_client = None
            self._async_connected = False
            logger.info("Redis async connection closed")

    async def close_all(self):
        """Close both sync and async Redis connections."""
        self.close()
        await self.async_close()


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
        if _redis_client is not None:
            _redis_client.close()
        _redis_client = RedisCache(config)
        return _redis_client


def close_redis_cache():
    """Close the global Redis cache connection."""
    global _redis_client

    with _redis_lock:
        if _redis_client is not None:
            _redis_client.close()
            _redis_client = None


async def async_close_redis_cache():
    """Close the global Redis cache connections (both sync and async)."""
    global _redis_client

    with _redis_lock:
        if _redis_client is not None:
            await _redis_client.close_all()
            _redis_client = None


# =============================================================================
# Fallback In-Memory Cache (when Redis is unavailable)
# =============================================================================

class InMemoryCache:
    """
    In-memory cache fallback when Redis is unavailable.
    Provides the same interface as RedisCache but stores data locally.
    Note: This does NOT support multi-instance sharing.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lists: Dict[str, List[str]] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._sets: Dict[str, Set[str]] = {}
        self._expiry: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        logger.warning("Using in-memory cache fallback. Multi-instance sharing is NOT supported!")

    def is_connected(self) -> bool:
        return True

    async def async_is_connected(self) -> bool:
        return True

    def _check_expiry(self, key: str) -> bool:
        """Check if key is expired and remove if so."""
        if key in self._expiry:
            if datetime.now() > self._expiry[key]:
                self.delete(key)
                return True
        return False

    # Sync methods
    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if self._check_expiry(key):
                return None
            return self._data.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None, nx: bool = False, xx: bool = False) -> bool:
        with self._lock:
            exists = key in self._data
            if nx and exists:
                return False
            if xx and not exists:
                return False
            self._data[key] = value
            if ttl and ttl > 0:
                self._expiry[key] = datetime.now() + timedelta(seconds=ttl)
            return True

    def delete(self, *keys: str) -> int:
        count = 0
        with self._lock:
            for key in keys:
                if key in self._data:
                    del self._data[key]
                    count += 1
                if key in self._lists:
                    del self._lists[key]
                if key in self._hashes:
                    del self._hashes[key]
                if key in self._sets:
                    del self._sets[key]
                if key in self._expiry:
                    del self._expiry[key]
        return count

    def exists(self, key: str) -> bool:
        with self._lock:
            if self._check_expiry(key):
                return False
            return key in self._data or key in self._lists or key in self._hashes or key in self._sets

    def expire(self, key: str, ttl: int) -> bool:
        with self._lock:
            if key in self._data or key in self._lists or key in self._hashes or key in self._sets:
                self._expiry[key] = datetime.now() + timedelta(seconds=ttl)
                return True
            return False

    def rpush(self, key: str, *values: str) -> int:
        with self._lock:
            if key not in self._lists:
                self._lists[key] = []
            self._lists[key].extend(values)
            return len(self._lists[key])

    def lpush(self, key: str, *values: str) -> int:
        with self._lock:
            if key not in self._lists:
                self._lists[key] = []
            for v in reversed(values):
                self._lists[key].insert(0, v)
            return len(self._lists[key])

    def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        with self._lock:
            if key not in self._lists:
                return []
            lst = self._lists[key]
            if end == -1:
                return lst[start:]
            return lst[start:end + 1]

    def llen(self, key: str) -> int:
        with self._lock:
            return len(self._lists.get(key, []))

    def ltrim(self, key: str, start: int, end: int) -> bool:
        with self._lock:
            if key not in self._lists:
                return True
            lst = self._lists[key]
            if end == -1:
                self._lists[key] = lst[start:]
            else:
                self._lists[key] = lst[start:end + 1]
            return True

    def hget(self, key: str, field: str) -> Optional[str]:
        with self._lock:
            return self._hashes.get(key, {}).get(field)

    def hset(self, key: str, field: str, value: str) -> int:
        with self._lock:
            if key not in self._hashes:
                self._hashes[key] = {}
            is_new = field not in self._hashes[key]
            self._hashes[key][field] = value
            return 1 if is_new else 0

    def hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        with self._lock:
            if key not in self._hashes:
                self._hashes[key] = {}
            self._hashes[key].update(mapping)
            return True

    def hgetall(self, key: str) -> Dict[str, str]:
        with self._lock:
            return dict(self._hashes.get(key, {}))

    def hdel(self, key: str, *fields: str) -> int:
        with self._lock:
            if key not in self._hashes:
                return 0
            count = 0
            for field in fields:
                if field in self._hashes[key]:
                    del self._hashes[key][field]
                    count += 1
            return count

    def hlen(self, key: str) -> int:
        with self._lock:
            return len(self._hashes.get(key, {}))

    def keys(self, pattern: str = "*") -> List[str]:
        with self._lock:
            import fnmatch
            all_keys = list(self._data.keys()) + list(self._lists.keys()) + list(self._hashes.keys()) + list(self._sets.keys())
            if pattern == "*":
                return list(set(all_keys))
            return list(set(k for k in all_keys if fnmatch.fnmatch(k, pattern)))

    # JSON helpers
    def get_json(self, key: str) -> Optional[Any]:
        value = self.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None, nx: bool = False, xx: bool = False) -> bool:
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            return self.set(key, json_str, ttl=ttl, nx=nx, xx=xx)
        except Exception:
            return False

    def lrange_json(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        items = self.lrange(key, start, end)
        result = []
        for item in items:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                result.append(item)
        return result

    def rpush_json(self, key: str, *values: Any) -> int:
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception:
                pass
        if not json_values:
            return 0
        return self.rpush(key, *json_values)

    def lpush_json(self, key: str, *values: Any) -> int:
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception:
                pass
        if not json_values:
            return 0
        return self.lpush(key, *json_values)

    # Lock operations (simple in-memory implementation)
    def acquire_lock(self, lock_name: str, timeout: int = 10, blocking: bool = True, blocking_timeout: float = 5.0) -> Optional[str]:
        import time
        lock_key = f"lock:{lock_name}"
        token = str(uuid.uuid4())

        if blocking:
            start_time = time.time()
            while time.time() - start_time < blocking_timeout:
                if self.set(lock_key, token, ttl=timeout, nx=True):
                    return token
                time.sleep(0.1)
            return None
        else:
            if self.set(lock_key, token, ttl=timeout, nx=True):
                return token
            return None

    def release_lock(self, lock_name: str, token: str) -> bool:
        lock_key = f"lock:{lock_name}"
        current_token = self.get(lock_key)
        if current_token == token:
            self.delete(lock_key)
            return True
        return False

    # Async methods (delegate to sync with locks for simplicity)
    async def async_get(self, key: str) -> Optional[str]:
        async with self._async_lock:
            return self.get(key)

    async def async_set(self, key: str, value: str, ttl: Optional[int] = None, nx: bool = False, xx: bool = False) -> bool:
        async with self._async_lock:
            return self.set(key, value, ttl=ttl, nx=nx, xx=xx)

    async def async_delete(self, *keys: str) -> int:
        async with self._async_lock:
            return self.delete(*keys)

    async def async_rpush(self, key: str, *values: str) -> int:
        async with self._async_lock:
            return self.rpush(key, *values)

    async def async_lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        async with self._async_lock:
            return self.lrange(key, start, end)

    async def async_llen(self, key: str) -> int:
        async with self._async_lock:
            return self.llen(key)

    async def async_rpush_json(self, key: str, *values: Any) -> int:
        async with self._async_lock:
            return self.rpush_json(key, *values)

    async def async_lrange_json(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        async with self._async_lock:
            return self.lrange_json(key, start, end)

    async def async_expire(self, key: str, ttl: int) -> bool:
        async with self._async_lock:
            return self.expire(key, ttl)

    async def async_acquire_lock(self, lock_name: str, timeout: int = 10, blocking: bool = True, blocking_timeout: float = 5.0) -> Optional[str]:
        import time
        lock_key = f"lock:{lock_name}"
        token = str(uuid.uuid4())

        if blocking:
            start_time = time.time()
            while time.time() - start_time < blocking_timeout:
                if await self.async_set(lock_key, token, ttl=timeout, nx=True):
                    return token
                await asyncio.sleep(0.1)  # Non-blocking
            return None
        else:
            if await self.async_set(lock_key, token, ttl=timeout, nx=True):
                return token
            return None

    async def async_release_lock(self, lock_name: str, token: str) -> bool:
        async with self._async_lock:
            return self.release_lock(lock_name, token)

    def close(self):
        pass

    async def async_close(self):
        pass

    async def close_all(self):
        pass


def get_cache(config: Optional[RedisCacheConfig] = None) -> Union[RedisCache, InMemoryCache]:
    """
    Get the best available cache implementation.
    Returns RedisCache if Redis is available, otherwise InMemoryCache.

    Args:
        config: Redis configuration

    Returns:
        Cache instance (RedisCache or InMemoryCache)
    """
    redis_cache = get_redis_cache(config)
    if redis_cache.is_connected():
        return redis_cache
    else:
        logger.warning("Redis unavailable, falling back to in-memory cache")
        return InMemoryCache()
