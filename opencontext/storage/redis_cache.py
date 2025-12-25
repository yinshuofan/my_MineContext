#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Redis Cache Manager
Provides a unified Redis caching layer for multi-instance deployment.
Supports various data structures: strings, lists, hashes, and sets.
"""

import json
import threading
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
    Redis Cache Manager
    
    Provides a unified interface for Redis operations with support for:
    - Key-value storage with TTL
    - List operations (for message buffers)
    - Hash operations (for structured data)
    - Set operations (for deduplication)
    - Atomic operations for multi-instance safety
    """
    
    def __init__(self, config: Optional[RedisCacheConfig] = None):
        """
        Initialize Redis cache manager.
        
        Args:
            config: Redis configuration. If None, uses default config.
        """
        self.config = config or RedisCacheConfig()
        self._client = None
        self._lock = threading.RLock()
        self._connected = False
        
        # Try to connect
        self._connect()
    
    def _connect(self) -> bool:
        """
        Establish connection to Redis server.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            import redis
            
            self._client = redis.Redis(
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
            self._client.ping()
            self._connected = True
            logger.info(f"Redis connected: {self.config.host}:{self.config.port}/{self.config.db}")
            return True
            
        except ImportError:
            logger.error("Redis package not installed. Run: pip install redis")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._connected or not self._client:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False
    
    def _make_key(self, key: str) -> str:
        """Generate full Redis key with prefix."""
        return f"{self.config.key_prefix}{key}"
    
    # =========================================================================
    # Basic Key-Value Operations
    # =========================================================================
    
    def get(self, key: str) -> Optional[str]:
        """
        Get value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Value string or None if not found
        """
        if not self.is_connected():
            return None
        try:
            return self._client.get(self._make_key(key))
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
        """
        Set key-value pair.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (None uses default)
            nx: Only set if key does not exist
            xx: Only set if key exists
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            return False
        try:
            ttl = ttl if ttl is not None else self.config.default_ttl
            return bool(self._client.set(
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
        """
        Delete one or more keys.
        
        Args:
            keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        if not self.is_connected() or not keys:
            return 0
        try:
            full_keys = [self._make_key(k) for k in keys]
            return self._client.delete(*full_keys)
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.is_connected():
            return False
        try:
            return bool(self._client.exists(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for a key."""
        if not self.is_connected():
            return False
        try:
            return bool(self._client.expire(self._make_key(key), ttl))
        except Exception as e:
            logger.error(f"Redis EXPIRE error: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """Get remaining TTL for a key. Returns -1 if no TTL, -2 if key doesn't exist."""
        if not self.is_connected():
            return -2
        try:
            return self._client.ttl(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis TTL error: {e}")
            return -2
    
    # =========================================================================
    # JSON Operations (for complex objects)
    # =========================================================================
    
    def get_json(self, key: str) -> Optional[Any]:
        """
        Get JSON value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Deserialized Python object or None
        """
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
        """
        Set JSON value.
        
        Args:
            key: Cache key
            value: Python object to serialize and store
            ttl: Time-to-live in seconds
            nx: Only set if key does not exist
            xx: Only set if key exists
            
        Returns:
            True if successful
        """
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            return self.set(key, json_str, ttl=ttl, nx=nx, xx=xx)
        except Exception as e:
            logger.error(f"JSON encode error for key {key}: {e}")
            return False
    
    # =========================================================================
    # List Operations (for message buffers)
    # =========================================================================
    
    def lpush(self, key: str, *values: str) -> int:
        """
        Push values to the left of a list.
        
        Args:
            key: List key
            values: Values to push
            
        Returns:
            Length of list after push
        """
        if not self.is_connected() or not values:
            return 0
        try:
            return self._client.lpush(self._make_key(key), *values)
        except Exception as e:
            logger.error(f"Redis LPUSH error: {e}")
            return 0
    
    def rpush(self, key: str, *values: str) -> int:
        """
        Push values to the right of a list.
        
        Args:
            key: List key
            values: Values to push
            
        Returns:
            Length of list after push
        """
        if not self.is_connected() or not values:
            return 0
        try:
            return self._client.rpush(self._make_key(key), *values)
        except Exception as e:
            logger.error(f"Redis RPUSH error: {e}")
            return 0
    
    def lpop(self, key: str, count: int = 1) -> Optional[Union[str, List[str]]]:
        """
        Pop values from the left of a list.
        
        Args:
            key: List key
            count: Number of elements to pop
            
        Returns:
            Popped value(s) or None
        """
        if not self.is_connected():
            return None
        try:
            return self._client.lpop(self._make_key(key), count)
        except Exception as e:
            logger.error(f"Redis LPOP error: {e}")
            return None
    
    def rpop(self, key: str, count: int = 1) -> Optional[Union[str, List[str]]]:
        """
        Pop values from the right of a list.
        
        Args:
            key: List key
            count: Number of elements to pop
            
        Returns:
            Popped value(s) or None
        """
        if not self.is_connected():
            return None
        try:
            return self._client.rpop(self._make_key(key), count)
        except Exception as e:
            logger.error(f"Redis RPOP error: {e}")
            return None
    
    def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """
        Get range of elements from a list.
        
        Args:
            key: List key
            start: Start index
            end: End index (-1 for all)
            
        Returns:
            List of elements
        """
        if not self.is_connected():
            return []
        try:
            return self._client.lrange(self._make_key(key), start, end)
        except Exception as e:
            logger.error(f"Redis LRANGE error: {e}")
            return []
    
    def llen(self, key: str) -> int:
        """
        Get length of a list.
        
        Args:
            key: List key
            
        Returns:
            Length of list
        """
        if not self.is_connected():
            return 0
        try:
            return self._client.llen(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis LLEN error: {e}")
            return 0
    
    def ltrim(self, key: str, start: int, end: int) -> bool:
        """
        Trim a list to specified range.
        
        Args:
            key: List key
            start: Start index
            end: End index
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            return False
        try:
            return bool(self._client.ltrim(self._make_key(key), start, end))
        except Exception as e:
            logger.error(f"Redis LTRIM error: {e}")
            return False
    
    def lrange_json(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """
        Get range of JSON elements from a list.
        
        Args:
            key: List key
            start: Start index
            end: End index
            
        Returns:
            List of deserialized objects
        """
        items = self.lrange(key, start, end)
        result = []
        for item in items:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                result.append(item)
        return result
    
    def rpush_json(self, key: str, *values: Any) -> int:
        """
        Push JSON values to the right of a list.
        
        Args:
            key: List key
            values: Python objects to serialize and push
            
        Returns:
            Length of list after push
        """
        json_values = []
        for v in values:
            try:
                json_values.append(json.dumps(v, ensure_ascii=False, default=str))
            except Exception as e:
                logger.error(f"JSON encode error: {e}")
        if not json_values:
            return 0
        return self.rpush(key, *json_values)
    
    # =========================================================================
    # Hash Operations (for structured data)
    # =========================================================================
    
    def hget(self, key: str, field: str) -> Optional[str]:
        """Get a field from a hash."""
        if not self.is_connected():
            return None
        try:
            return self._client.hget(self._make_key(key), field)
        except Exception as e:
            logger.error(f"Redis HGET error: {e}")
            return None
    
    def hset(self, key: str, field: str, value: str) -> int:
        """Set a field in a hash."""
        if not self.is_connected():
            return 0
        try:
            return self._client.hset(self._make_key(key), field, value)
        except Exception as e:
            logger.error(f"Redis HSET error: {e}")
            return 0
    
    def hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        """Set multiple fields in a hash."""
        if not self.is_connected() or not mapping:
            return False
        try:
            return bool(self._client.hset(self._make_key(key), mapping=mapping))
        except Exception as e:
            logger.error(f"Redis HMSET error: {e}")
            return False
    
    def hgetall(self, key: str) -> Dict[str, str]:
        """Get all fields and values from a hash."""
        if not self.is_connected():
            return {}
        try:
            return self._client.hgetall(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis HGETALL error: {e}")
            return {}
    
    def hdel(self, key: str, *fields: str) -> int:
        """Delete fields from a hash."""
        if not self.is_connected() or not fields:
            return 0
        try:
            return self._client.hdel(self._make_key(key), *fields)
        except Exception as e:
            logger.error(f"Redis HDEL error: {e}")
            return 0
    
    def hlen(self, key: str) -> int:
        """Get number of fields in a hash."""
        if not self.is_connected():
            return 0
        try:
            return self._client.hlen(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis HLEN error: {e}")
            return 0
    
    def hexists(self, key: str, field: str) -> bool:
        """Check if a field exists in a hash."""
        if not self.is_connected():
            return False
        try:
            return bool(self._client.hexists(self._make_key(key), field))
        except Exception as e:
            logger.error(f"Redis HEXISTS error: {e}")
            return False
    
    # =========================================================================
    # Set Operations (for deduplication)
    # =========================================================================
    
    def sadd(self, key: str, *members: str) -> int:
        """Add members to a set."""
        if not self.is_connected() or not members:
            return 0
        try:
            return self._client.sadd(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis SADD error: {e}")
            return 0
    
    def srem(self, key: str, *members: str) -> int:
        """Remove members from a set."""
        if not self.is_connected() or not members:
            return 0
        try:
            return self._client.srem(self._make_key(key), *members)
        except Exception as e:
            logger.error(f"Redis SREM error: {e}")
            return 0
    
    def sismember(self, key: str, member: str) -> bool:
        """Check if a member is in a set."""
        if not self.is_connected():
            return False
        try:
            return bool(self._client.sismember(self._make_key(key), member))
        except Exception as e:
            logger.error(f"Redis SISMEMBER error: {e}")
            return False
    
    def smembers(self, key: str) -> Set[str]:
        """Get all members of a set."""
        if not self.is_connected():
            return set()
        try:
            return self._client.smembers(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis SMEMBERS error: {e}")
            return set()
    
    def scard(self, key: str) -> int:
        """Get number of members in a set."""
        if not self.is_connected():
            return 0
        try:
            return self._client.scard(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis SCARD error: {e}")
            return 0
    
    # =========================================================================
    # Atomic Operations
    # =========================================================================
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment a key's value atomically."""
        if not self.is_connected():
            return 0
        try:
            return self._client.incr(self._make_key(key), amount)
        except Exception as e:
            logger.error(f"Redis INCR error: {e}")
            return 0
    
    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement a key's value atomically."""
        if not self.is_connected():
            return 0
        try:
            return self._client.decr(self._make_key(key), amount)
        except Exception as e:
            logger.error(f"Redis DECR error: {e}")
            return 0
    
    def getset(self, key: str, value: str) -> Optional[str]:
        """Set a new value and return the old value atomically."""
        if not self.is_connected():
            return None
        try:
            return self._client.getset(self._make_key(key), value)
        except Exception as e:
            logger.error(f"Redis GETSET error: {e}")
            return None
    
    # =========================================================================
    # Lock Operations (for distributed locking)
    # =========================================================================
    
    def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: float = 5.0,
    ) -> Optional[str]:
        """
        Acquire a distributed lock.
        
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
            import uuid
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
                
        except Exception as e:
            logger.error(f"Redis acquire_lock error: {e}")
            return None
    
    def release_lock(self, lock_name: str, token: str) -> bool:
        """
        Release a distributed lock.
        
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
    # Utility Methods
    # =========================================================================
    
    def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "user:*")
            
        Returns:
            List of matching keys (without prefix)
        """
        if not self.is_connected():
            return []
        try:
            full_pattern = self._make_key(pattern)
            keys = self._client.keys(full_pattern)
            prefix_len = len(self.config.key_prefix)
            return [k[prefix_len:] if isinstance(k, str) else k.decode()[prefix_len:] for k in keys]
        except Exception as e:
            logger.error(f"Redis KEYS error: {e}")
            return []
    
    def flush_db(self) -> bool:
        """Flush the current database. Use with caution!"""
        if not self.is_connected():
            return False
        try:
            self._client.flushdb()
            logger.warning("Redis database flushed!")
            return True
        except Exception as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            return False
    
    def info(self) -> Dict[str, Any]:
        """Get Redis server info."""
        if not self.is_connected():
            return {}
        try:
            return self._client.info()
        except Exception as e:
            logger.error(f"Redis INFO error: {e}")
            return {}
    
    def close(self):
        """Close Redis connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._connected = False
            logger.info("Redis connection closed")


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
        logger.warning("Using in-memory cache fallback. Multi-instance sharing is NOT supported!")
    
    def is_connected(self) -> bool:
        return True
    
    def _check_expiry(self, key: str) -> bool:
        """Check if key is expired and remove if so."""
        if key in self._expiry:
            if datetime.now() > self._expiry[key]:
                self.delete(key)
                return True
        return False
    
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
    
    def rpush(self, key: str, *values: str) -> int:
        with self._lock:
            if key not in self._lists:
                self._lists[key] = []
            self._lists[key].extend(values)
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
    
    # Add other methods as needed...
    
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
    
    def close(self):
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
