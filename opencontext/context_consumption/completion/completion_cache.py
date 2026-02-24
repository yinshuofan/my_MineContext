#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Completion Cache Manager
Provides high-performance caching and optimization for completion results.
Supports both Redis (for multi-instance) and in-memory (fallback) caching.
"""

import hashlib
import json
import sys
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CacheStrategy(Enum):
    """Cache Strategy Enum"""

    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    HYBRID = "hybrid"  # Hybrid Strategy


@dataclass
class CacheEntry:
    """Cache Entry"""

    key: str
    suggestions: List[Any]  # Use Any to avoid circular imports
    created_at: str  # ISO format string for JSON serialization
    last_accessed: str  # ISO format string
    access_count: int
    confidence_score: float
    context_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "key": self.key,
            "suggestions": self.suggestions,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "confidence_score": self.confidence_score,
            "context_hash": self.context_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary"""
        return cls(
            key=data["key"],
            suggestions=data["suggestions"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            access_count=data["access_count"],
            confidence_score=data["confidence_score"],
            context_hash=data["context_hash"],
        )


class CompletionCache:
    """
    Intelligent Completion Cache Manager
    Supports multiple caching strategies and performance optimizations.
    
    Supports two storage backends:
    1. Redis (recommended): For multi-instance deployment with shared cache
    2. In-memory (fallback): When Redis is unavailable
    """

    # Redis key prefixes
    CACHE_KEY_PREFIX = "completion:cache:"
    ACCESS_ORDER_KEY = "completion:access_order"
    HOT_KEYS_KEY = "completion:hot_keys"
    STATS_KEY = "completion:stats"
    PRECOMPUTED_KEY_PREFIX = "completion:precomputed:"

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300,
        strategy: CacheStrategy = CacheStrategy.HYBRID,
        redis_config: Optional[Dict[str, Any]] = None,
    ):
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.ttl_seconds = ttl_seconds
        self.strategy = strategy

        # Local cache storage (fallback)
        self._local_cache: Dict[str, CacheEntry] = {}
        self._local_access_order: List[str] = []
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0,
            "average_response_time": 0.0,
        }

        # Performance optimizations
        self._precomputed_contexts: Dict[int, Dict[str, Any]] = {}
        self._hot_keys: set = set()

        # Redis cache
        self._redis_cache = None
        self._use_redis = False
        
        # Initialize Redis if configured
        if redis_config and redis_config.get("enabled", True):
            self._init_redis(redis_config)

        logger.info(
            f"CompletionCache initialized: max_size={max_size}, ttl={ttl_seconds}s, "
            f"strategy={strategy.value}, use_redis={self._use_redis}"
        )

    def _init_redis(self, redis_config: Dict[str, Any]):
        """Initialize Redis connection using global singleton"""
        try:
            from opencontext.storage.redis_cache import get_redis_cache

            self._redis_cache = get_redis_cache()
            self._use_redis = self._redis_cache.is_connected()
            
            if self._use_redis:
                logger.info("CompletionCache: Using Redis for multi-instance cache sharing")
            else:
                logger.warning("CompletionCache: Redis unavailable, using local memory cache")
        except Exception as e:
            logger.warning(f"CompletionCache: Failed to initialize Redis: {e}")
            self._use_redis = False

    def _make_cache_key(self, key: str) -> str:
        """Generate full cache key"""
        return f"{self.CACHE_KEY_PREFIX}{key}"

    def get(self, key: str, context_hash: str = None) -> Optional[List[Any]]:
        """Get cached completion suggestions"""
        import time
        start_time = time.time()

        if self._use_redis:
            result = self._get_redis(key, context_hash)
        else:
            result = self._get_local(key, context_hash)

        # Update response time stats
        response_time = time.time() - start_time
        self._update_average_response_time(response_time)

        return result

    def _get_redis(self, key: str, context_hash: str = None) -> Optional[List[Any]]:
        """Get from Redis cache"""
        try:
            self._increment_stat("total_requests")
            
            cache_key = self._make_cache_key(key)
            data = self._redis_cache.get_json(cache_key)
            
            if data is None:
                self._increment_stat("misses")
                return None
            
            entry = CacheEntry.from_dict(data)
            
            # Check TTL expiration
            created_at = datetime.fromisoformat(entry.created_at)
            if datetime.now() - created_at > self.ttl:
                self._redis_cache.delete(cache_key)
                self._increment_stat("misses")
                return None
            
            # Check context hash
            if context_hash and entry.context_hash != context_hash:
                self._increment_stat("misses")
                return None
            
            # Update access info
            entry.last_accessed = datetime.now().isoformat()
            entry.access_count += 1
            self._redis_cache.set_json(cache_key, entry.to_dict(), ttl=self.ttl_seconds)
            
            # Update access order (for LRU)
            self._redis_cache.lrem(self.ACCESS_ORDER_KEY, 0, key)
            self._redis_cache.rpush(self.ACCESS_ORDER_KEY, key)
            
            # Mark as hot key if frequently accessed
            if entry.access_count > 5:
                self._redis_cache.sadd(self.HOT_KEYS_KEY, key)
            
            self._increment_stat("hits")
            logger.debug(f"Cache hit (Redis): {key[:20]}...")
            return entry.suggestions
            
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            # Fallback to local cache
            return self._get_local(key, context_hash)

    def _get_local(self, key: str, context_hash: str = None) -> Optional[List[Any]]:
        """Get from local memory cache"""
        with self._lock:
            self._stats["total_requests"] += 1

            if key not in self._local_cache:
                self._stats["misses"] += 1
                return None

            entry = self._local_cache[key]

            # Check TTL expiration
            created_at = datetime.fromisoformat(entry.created_at)
            if datetime.now() - created_at > self.ttl:
                self._evict_local(key)
                self._stats["misses"] += 1
                return None

            # Check context hash
            if context_hash and entry.context_hash != context_hash:
                self._stats["misses"] += 1
                return None

            # Update access info
            entry.last_accessed = datetime.now().isoformat()
            entry.access_count += 1

            # Update LRU order
            if key in self._local_access_order:
                self._local_access_order.remove(key)
            self._local_access_order.append(key)

            # Mark as hot key
            if entry.access_count > 5:
                self._hot_keys.add(key)

            self._stats["hits"] += 1
            logger.debug(f"Cache hit (local): {key[:20]}...")
            return entry.suggestions

    def put(
        self,
        key: str,
        suggestions: List[Any],
        context_hash: str = None,
        confidence_score: float = 0.0,
    ):
        """Add completion suggestions to the cache"""
        if self._use_redis:
            self._put_redis(key, suggestions, context_hash, confidence_score)
        else:
            self._put_local(key, suggestions, context_hash, confidence_score)

    def _put_redis(
        self,
        key: str,
        suggestions: List[Any],
        context_hash: str = None,
        confidence_score: float = 0.0,
    ):
        """Add to Redis cache"""
        try:
            now = datetime.now()
            
            # Check cache size and evict if needed
            cache_size = len(self._redis_cache.keys(f"{self.CACHE_KEY_PREFIX}*"))
            if cache_size >= self.max_size:
                self._evict_entries_redis()
            
            # Create entry
            entry = CacheEntry(
                key=key,
                suggestions=suggestions,
                created_at=now.isoformat(),
                last_accessed=now.isoformat(),
                access_count=0,
                confidence_score=confidence_score,
                context_hash=context_hash or "",
            )
            
            # Store in Redis
            cache_key = self._make_cache_key(key)
            self._redis_cache.set_json(cache_key, entry.to_dict(), ttl=self.ttl_seconds)
            
            # Add to access order
            self._redis_cache.rpush(self.ACCESS_ORDER_KEY, key)
            
            logger.debug(f"Cache add (Redis): {key[:20]}... ({len(suggestions)} suggestions)")
            
        except Exception as e:
            logger.error(f"Redis cache put error: {e}")
            # Fallback to local cache
            self._put_local(key, suggestions, context_hash, confidence_score)

    def _put_local(
        self,
        key: str,
        suggestions: List[Any],
        context_hash: str = None,
        confidence_score: float = 0.0,
    ):
        """Add to local memory cache"""
        with self._lock:
            now = datetime.now()

            # Evict if cache is full
            if len(self._local_cache) >= self.max_size:
                self._evict_entries_local()

            # Create entry
            entry = CacheEntry(
                key=key,
                suggestions=suggestions,
                created_at=now.isoformat(),
                last_accessed=now.isoformat(),
                access_count=0,
                confidence_score=confidence_score,
                context_hash=context_hash or "",
            )

            self._local_cache[key] = entry
            self._local_access_order.append(key)

            logger.debug(f"Cache add (local): {key[:20]}... ({len(suggestions)} suggestions)")

    def invalidate(self, pattern: str = None):
        """Invalidate the cache"""
        if self._use_redis:
            self._invalidate_redis(pattern)
        self._invalidate_local(pattern)

    def _invalidate_redis(self, pattern: str = None):
        """Invalidate Redis cache"""
        try:
            if pattern is None:
                # Clear all cache
                keys = self._redis_cache.keys(f"{self.CACHE_KEY_PREFIX}*")
                if keys:
                    self._redis_cache.delete(*keys)
                self._redis_cache.delete(self.ACCESS_ORDER_KEY)
                self._redis_cache.delete(self.HOT_KEYS_KEY)
                logger.info("All Redis cache cleared")
            else:
                # Invalidate by pattern
                keys = self._redis_cache.keys(f"{self.CACHE_KEY_PREFIX}*{pattern}*")
                if keys:
                    self._redis_cache.delete(*keys)
                logger.info(f"Invalidated Redis cache by pattern: {pattern} ({len(keys)} items)")
        except Exception as e:
            logger.error(f"Redis invalidate error: {e}")

    def _invalidate_local(self, pattern: str = None):
        """Invalidate local cache"""
        with self._lock:
            if pattern is None:
                self._local_cache.clear()
                self._local_access_order.clear()
                self._hot_keys.clear()
                logger.info("All local cache cleared")
            else:
                keys_to_remove = [key for key in self._local_cache.keys() if pattern in key]
                for key in keys_to_remove:
                    self._evict_local(key)
                logger.info(f"Invalidated local cache by pattern: {pattern} ({len(keys_to_remove)} items)")

    def _evict_entries_redis(self):
        """Evict entries from Redis cache"""
        try:
            hot_keys = self._redis_cache.smembers(self.HOT_KEYS_KEY)
            
            # Get oldest keys from access order
            while True:
                oldest_key = self._redis_cache.lpop(self.ACCESS_ORDER_KEY)
                if not oldest_key:
                    break
                
                # Skip hot keys if possible
                cache_size = len(self._redis_cache.keys(f"{self.CACHE_KEY_PREFIX}*"))
                if oldest_key in hot_keys and cache_size < self.max_size * 1.2:
                    self._redis_cache.rpush(self.ACCESS_ORDER_KEY, oldest_key)
                    continue
                
                # Evict
                self._redis_cache.delete(self._make_cache_key(oldest_key))
                self._increment_stat("evictions")
                
                if cache_size < self.max_size:
                    break
                    
        except Exception as e:
            logger.error(f"Redis eviction error: {e}")

    def _evict_entries_local(self):
        """Evict entries from local cache"""
        while len(self._local_cache) >= self.max_size and self._local_access_order:
            oldest_key = self._local_access_order[0]

            # Protect hot keys
            if oldest_key in self._hot_keys and len(self._local_cache) < self.max_size * 1.2:
                self._local_access_order.remove(oldest_key)
                self._local_access_order.append(oldest_key)
                continue

            self._evict_local(oldest_key)

    def _evict_local(self, key: str):
        """Evict a single entry from local cache"""
        if key in self._local_cache:
            del self._local_cache[key]
            self._stats["evictions"] += 1

        if key in self._local_access_order:
            self._local_access_order.remove(key)

        if key in self._hot_keys:
            self._hot_keys.remove(key)

    def _increment_stat(self, stat_name: str, amount: int = 1):
        """Increment a statistic (thread-safe)"""
        if self._use_redis:
            try:
                self._redis_cache.incr(f"{self.STATS_KEY}:{stat_name}", amount)
            except Exception:
                pass
        with self._lock:
            self._stats[stat_name] = self._stats.get(stat_name, 0) + amount

    def _update_average_response_time(self, response_time: float):
        """Update the average response time"""
        with self._lock:
            if self._stats["average_response_time"] == 0:
                self._stats["average_response_time"] = response_time
            else:
                alpha = 0.1
                self._stats["average_response_time"] = (
                    alpha * response_time + (1 - alpha) * self._stats["average_response_time"]
                )

    def precompute_context(self, document_id: int, content: str):
        """Precompute document context"""
        try:
            lines = content.split("\n")
            patterns = {
                "headings": [line for line in lines if line.startswith("#")],
                "lists": [line for line in lines if line.strip().startswith(("-", "*", "+"))],
                "code_blocks": [line for line in lines if line.strip().startswith("```")],
                "links": [line for line in lines if "[" in line and "](" in line],
            }

            context_data = f"{document_id}:{len(content)}:{len(lines)}"
            context_hash = hashlib.md5(context_data.encode()).hexdigest()

            precomputed = {
                "hash": context_hash,
                "patterns": patterns,
                "computed_at": datetime.now().isoformat(),
            }

            if self._use_redis:
                try:
                    self._redis_cache.set_json(
                        f"{self.PRECOMPUTED_KEY_PREFIX}{document_id}",
                        precomputed,
                        ttl=3600  # 1 hour
                    )
                except Exception as e:
                    logger.error(f"Redis precompute error: {e}")

            self._precomputed_contexts[document_id] = precomputed
            logger.debug(f"Precomputed document context: {document_id}")

        except Exception as e:
            logger.error(f"Failed to precompute context: {e}")

    def get_precomputed_context(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get precomputed context"""
        # Try Redis first
        if self._use_redis:
            try:
                data = self._redis_cache.get_json(f"{self.PRECOMPUTED_KEY_PREFIX}{document_id}")
                if data:
                    return data
            except Exception:
                pass
        
        return self._precomputed_contexts.get(document_id)

    def optimize(self):
        """Cache optimization"""
        if self._use_redis:
            self._optimize_redis()
        self._optimize_local()

    def _optimize_redis(self):
        """Optimize Redis cache"""
        try:
            now = datetime.now()
            
            # Clean up expired entries
            keys = self._redis_cache.keys(f"{self.CACHE_KEY_PREFIX}*")
            for key in keys:
                data = self._redis_cache.get_json(key)
                if data:
                    created_at = datetime.fromisoformat(data["created_at"])
                    if now - created_at > self.ttl * 2:
                        self._redis_cache.delete(key)
            
            # Clean up old precomputed contexts
            precomputed_keys = self._redis_cache.keys(f"{self.PRECOMPUTED_KEY_PREFIX}*")
            for key in precomputed_keys:
                data = self._redis_cache.get_json(key)
                if data:
                    computed_at = datetime.fromisoformat(data["computed_at"])
                    if now - computed_at > timedelta(hours=1):
                        self._redis_cache.delete(key)
            
            logger.info("Redis cache optimization complete")
        except Exception as e:
            logger.error(f"Redis optimization error: {e}")

    def _optimize_local(self):
        """Optimize local cache"""
        with self._lock:
            now = datetime.now()
            
            # Clean up expired entries
            expired_keys = []
            for key, entry in self._local_cache.items():
                created_at = datetime.fromisoformat(entry.created_at)
                if now - created_at > self.ttl * 2:
                    expired_keys.append(key)

            for key in expired_keys:
                self._evict_local(key)

            # Update hot keys
            self._hot_keys = {
                key
                for key, entry in self._local_cache.items()
                if entry.access_count > 3 or key in self._hot_keys
            }

            # Compact access order list
            self._local_access_order = [key for key in self._local_access_order if key in self._local_cache]

            # Clean up old precomputed contexts
            old_contexts = [
                doc_id
                for doc_id, ctx in self._precomputed_contexts.items()
                if now - datetime.fromisoformat(ctx["computed_at"]) > timedelta(hours=1)
            ]

            for doc_id in old_contexts:
                del self._precomputed_contexts[doc_id]

            logger.info(
                f"Local cache optimization complete: cleaned {len(expired_keys)} expired entries, "
                f"{len(old_contexts)} old contexts"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["total_requests"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            stats = {
                "use_redis": self._use_redis,
                "max_size": self.max_size,
                "hit_rate": round(hit_rate, 2),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "total_requests": total_requests,
                "average_response_time_ms": round(self._stats["average_response_time"] * 1000, 2),
            }

            if self._use_redis:
                try:
                    stats["redis_cache_size"] = len(self._redis_cache.keys(f"{self.CACHE_KEY_PREFIX}*"))
                    stats["redis_hot_keys_count"] = self._redis_cache.scard(self.HOT_KEYS_KEY)
                except Exception:
                    pass

            stats["local_cache_size"] = len(self._local_cache)
            stats["local_hot_keys_count"] = len(self._hot_keys)
            stats["precomputed_contexts"] = len(self._precomputed_contexts)
            stats["memory_usage_estimate"] = self._estimate_memory_usage()

            return stats

    def _estimate_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage"""
        try:
            cache_size = sys.getsizeof(self._local_cache)
            for key, entry in self._local_cache.items():
                cache_size += sys.getsizeof(key)
                cache_size += sys.getsizeof(entry.suggestions)
                for suggestion in entry.suggestions:
                    if hasattr(suggestion, "text"):
                        cache_size += sys.getsizeof(suggestion.text)

            precomputed_size = sys.getsizeof(self._precomputed_contexts)
            for ctx in self._precomputed_contexts.values():
                precomputed_size += sys.getsizeof(ctx)

            return {
                "cache_bytes": cache_size,
                "precomputed_bytes": precomputed_size,
                "total_bytes": cache_size + precomputed_size,
                "cache_mb": round(cache_size / 1024 / 1024, 2),
                "total_mb": round((cache_size + precomputed_size) / 1024 / 1024, 2),
            }
        except Exception:
            return {"error": "Unable to estimate memory usage"}

    def export_hot_patterns(self) -> List[Dict[str, Any]]:
        """Export hot patterns for model training optimization"""
        hot_patterns = []

        # Get hot keys from Redis if available
        hot_keys = self._hot_keys.copy()
        if self._use_redis:
            try:
                redis_hot_keys = self._redis_cache.smembers(self.HOT_KEYS_KEY)
                hot_keys.update(redis_hot_keys)
            except Exception:
                pass

        with self._lock:
            for key in hot_keys:
                entry = None
                
                # Try local cache first
                if key in self._local_cache:
                    entry = self._local_cache[key]
                elif self._use_redis:
                    # Try Redis
                    try:
                        data = self._redis_cache.get_json(self._make_cache_key(key))
                        if data:
                            entry = CacheEntry.from_dict(data)
                    except Exception:
                        pass
                
                if entry:
                    hot_patterns.append(
                        {
                            "key_hash": hashlib.md5(key.encode()).hexdigest(),
                            "access_count": entry.access_count,
                            "confidence_score": entry.confidence_score,
                            "suggestion_types": [
                                (
                                    getattr(s, "completion_type", {}).get("value", "unknown")
                                    if hasattr(s, "completion_type")
                                    else "unknown"
                                )
                                for s in entry.suggestions
                            ],
                            "suggestion_count": len(entry.suggestions),
                        }
                    )

        return hot_patterns


# Global cache instance
_completion_cache_instance = None
_completion_cache_lock = threading.Lock()


def get_completion_cache(redis_config: Optional[Dict[str, Any]] = None) -> CompletionCache:
    """Get the global completion cache instance"""
    global _completion_cache_instance
    with _completion_cache_lock:
        if _completion_cache_instance is None:
            _completion_cache_instance = CompletionCache(redis_config=redis_config)
        return _completion_cache_instance


def init_completion_cache(
    max_size: int = 1000,
    ttl_seconds: int = 300,
    strategy: CacheStrategy = CacheStrategy.HYBRID,
    redis_config: Optional[Dict[str, Any]] = None,
) -> CompletionCache:
    """Initialize the global completion cache with specific configuration"""
    global _completion_cache_instance
    with _completion_cache_lock:
        _completion_cache_instance = CompletionCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            strategy=strategy,
            redis_config=redis_config,
        )
        return _completion_cache_instance


def clear_completion_cache():
    """Clear the global completion cache"""
    global _completion_cache_instance
    if _completion_cache_instance is not None:
        _completion_cache_instance.invalidate()


# Cache decorator
def cache_completion(ttl: int = 300):
    """
    Completion cache decorator

    Args:
        ttl: Cache time-to-live in seconds (not yet implemented, reserved parameter)
    """
    _ = ttl  # Mark parameter as used to avoid warnings

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cache = get_completion_cache()
            cached_result = cache.get(cache_key)

            if cached_result is not None:
                return cached_result

            # Execute original function
            result = func(*args, **kwargs)

            # Cache the result
            if result:
                cache.put(cache_key, result)

            return result

        return wrapper

    return decorator
