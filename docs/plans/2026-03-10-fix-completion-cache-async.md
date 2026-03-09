# Fix CompletionCache Async/Redis Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix CompletionCache's 100% broken Redis integration — every RedisCache method is `async def` but CompletionCache calls them all synchronously without `await`.

**Architecture:** Convert CompletionCache to fully async following the established MemoryCacheManager pattern. Three key improvements beyond just adding `await`:
1. **Resilience model** — Replace init-time `_use_redis` boolean (set via un-awaited `is_connected()` coroutine) with `_redis_configured` flag. Redis availability verified naturally by try/except on each operation, enabling auto-recovery when Redis reconnects.
2. **Data structure fix** — Replace list-based LRU tracking (uses `lrem` which doesn't exist on RedisCache) with sorted set (`ZADD`/`ZRANGEBYSCORE`/`ZREM`). Sorted sets give O(log N) access/eviction and natural LRU ordering by timestamp score.
3. **O(1) size checking** — Replace `keys("prefix*")` scan (O(N)) with `zcard()` (O(1)) for cache size checks in the write hot path.

**Tech Stack:** Python asyncio, Redis sorted sets, FastAPI async routes

**Context:** Completion routes are currently NOT registered in `api.py` (dead code per `server/MODULE.md`). Changes ensure correctness for when routes are activated.

---

## RedisCache Method Signatures (verified)

These are the exact signatures from `opencontext/storage/redis_cache.py` that CompletionCache uses:

```python
# Sorted Set (new — replacing list ops)
async def zadd(self, key: str, mapping: Dict[str, float]) -> int        # line 654
async def zrangebyscore(self, key: str, min_score: float, max_score: float, withscores: bool = False) -> List[str]  # line 664
async def zrem(self, key: str, *members: str) -> int                    # line 698
async def zcard(self, key: str) -> int                                  # line 718

# String/JSON
async def get_json(self, key: str) -> Optional[Any]                     # line 298
async def set_json(self, key: str, value: Any, ttl: int = None) -> bool # line 309
async def delete(self, *keys: str) -> int                               # line 253
async def keys(self, pattern: str = "*") -> List[str]                   # line 840
async def incr(self, key: str, amount: int = 1) -> int                  # line 732

# Set
async def sadd(self, key: str, *members: str) -> int                    # line 600
async def smembers(self, key: str) -> Set[str]                          # line 630
async def scard(self, key: str) -> int                                  # line 640
```

**Note:** `zrangebyscore` does NOT support `start`/`num` (LIMIT) parameters. Use Python slicing on the result instead. Max result size is bounded by `max_size` (default 1000).

**Note:** `lrem` does NOT exist on RedisCache. The current code's `lrem`/`rpush`/`lpop` list-based LRU is replaced by sorted set ops.

---

### Task 1: Rewrite CompletionCache initialization and resilience model

**Files:**
- Modify: `opencontext/context_consumption/completion/completion_cache.py:90-148`

**Step 1: Replace `_use_redis` with `_redis_configured` throughout the file**

In `__init__`, replace:
```python
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
```
with:
```python
        # Redis cache — flag indicates configuration, not connectivity.
        # Actual availability verified by try/except on each operation,
        # enabling auto-recovery when Redis reconnects after startup.
        self._redis_cache = None
        self._redis_configured = False

        # Initialize Redis if configured
        if redis_config and redis_config.get("enabled", True):
            self._init_redis(redis_config)

        logger.info(
            f"CompletionCache initialized: max_size={max_size}, ttl={ttl_seconds}s, "
            f"strategy={strategy.value}, redis_configured={self._redis_configured}"
        )
```

Replace `_init_redis`:
```python
    def _init_redis(self, redis_config: Dict[str, Any]):
        """Get Redis cache reference. Actual connectivity verified on first use via try/except."""
        try:
            from opencontext.storage.redis_cache import get_redis_cache

            self._redis_cache = get_redis_cache()
            self._redis_configured = True
            logger.info("CompletionCache: Redis configured for multi-instance cache sharing")
        except Exception as e:
            logger.warning(f"CompletionCache: Failed to get Redis cache: {e}")
            self._redis_configured = False
```

Then rename all remaining `self._use_redis` → `self._redis_configured` throughout the file (replace_all). Occurrences at approximately lines: 159, 266, 345, 439, 478, 497, 509, 589, 599, 646, 660.

**Step 2: Compile-check**

Run: `python -m py_compile opencontext/context_consumption/completion/completion_cache.py`
Expected: Success

**Step 3: Commit**

```bash
git add opencontext/context_consumption/completion/completion_cache.py
git commit -m "refactor(completion-cache): replace _use_redis init-time flag with _redis_configured

Remove is_connected() call from sync __init__ (was async, never awaited).
Redis availability now verified implicitly by try/except on each operation,
enabling auto-recovery if Redis reconnects after startup."
```

---

### Task 2: Convert CompletionCache read/write path to async with sorted set LRU

**Files:**
- Modify: `opencontext/context_consumption/completion/completion_cache.py`

**Step 1: Convert `get` and `_get_redis` to async, replace list ops with sorted set**

Replace `get` method:
```python
    async def get(self, key: str, context_hash: str = None) -> Optional[List[Any]]:
        """Get cached completion suggestions"""
        import time

        start_time = time.time()

        if self._redis_configured:
            result = await self._get_redis(key, context_hash)
        else:
            result = self._get_local(key, context_hash)

        # Update response time stats
        response_time = time.time() - start_time
        self._update_average_response_time(response_time)

        return result
```

Replace `_get_redis` method:
```python
    async def _get_redis(self, key: str, context_hash: str = None) -> Optional[List[Any]]:
        """Get from Redis cache"""
        try:
            await self._increment_stat("total_requests")

            cache_key = self._make_cache_key(key)
            data = await self._redis_cache.get_json(cache_key)

            if data is None:
                await self._increment_stat("misses")
                return None

            entry = CacheEntry.from_dict(data)

            # Check TTL expiration
            created_at = datetime.fromisoformat(entry.created_at)
            if datetime.now() - created_at > self.ttl:
                await self._redis_cache.delete(cache_key)
                await self._redis_cache.zrem(self.ACCESS_ORDER_KEY, key)
                await self._increment_stat("misses")
                return None

            # Check context hash
            if context_hash and entry.context_hash != context_hash:
                await self._increment_stat("misses")
                return None

            # Update access info
            now = datetime.now()
            entry.last_accessed = now.isoformat()
            entry.access_count += 1
            await self._redis_cache.set_json(cache_key, entry.to_dict(), ttl=self.ttl_seconds)

            # Update access order (sorted set with timestamp score for LRU)
            await self._redis_cache.zadd(self.ACCESS_ORDER_KEY, {key: now.timestamp()})

            # Mark as hot key if frequently accessed
            if entry.access_count > 5:
                await self._redis_cache.sadd(self.HOT_KEYS_KEY, key)

            await self._increment_stat("hits")
            logger.debug(f"Cache hit (Redis): {key[:20]}...")
            return entry.suggestions

        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            # Fallback to local cache
            return self._get_local(key, context_hash)
```

**Step 2: Convert `put` and `_put_redis` to async, use `zcard` for O(1) size check**

Replace `put` method:
```python
    async def put(
        self,
        key: str,
        suggestions: List[Any],
        context_hash: str = None,
        confidence_score: float = 0.0,
    ):
        """Add completion suggestions to the cache"""
        if self._redis_configured:
            await self._put_redis(key, suggestions, context_hash, confidence_score)
        else:
            self._put_local(key, suggestions, context_hash, confidence_score)
```

Replace `_put_redis` method:
```python
    async def _put_redis(
        self,
        key: str,
        suggestions: List[Any],
        context_hash: str = None,
        confidence_score: float = 0.0,
    ):
        """Add to Redis cache"""
        try:
            now = datetime.now()

            # O(1) size check via sorted set cardinality (replaces O(N) keys() scan)
            cache_size = await self._redis_cache.zcard(self.ACCESS_ORDER_KEY)
            if cache_size >= self.max_size:
                await self._evict_entries_redis()

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
            await self._redis_cache.set_json(cache_key, entry.to_dict(), ttl=self.ttl_seconds)

            # Add to access order sorted set (score = timestamp for LRU ordering)
            await self._redis_cache.zadd(self.ACCESS_ORDER_KEY, {key: now.timestamp()})

            logger.debug(f"Cache add (Redis): {key[:20]}... ({len(suggestions)} suggestions)")

        except Exception as e:
            logger.error(f"Redis cache put error: {e}")
            # Fallback to local cache
            self._put_local(key, suggestions, context_hash, confidence_score)
```

**Step 3: Compile-check**

Run: `python -m py_compile opencontext/context_consumption/completion/completion_cache.py`
Expected: Success

**Step 4: Commit**

```bash
git add opencontext/context_consumption/completion/completion_cache.py
git commit -m "fix(completion-cache): convert get/put to async, replace list LRU with sorted set

- Add await to all RedisCache calls in read/write path
- Replace list-based LRU (rpush/lrem/lpop — lrem doesn't exist on RedisCache)
  with sorted set (zadd with timestamp score) for correct O(log N) access tracking
- Replace O(N) keys() scan for size check with O(1) zcard()"
```

---

### Task 3: Convert eviction and invalidation to async with sorted set

**Files:**
- Modify: `opencontext/context_consumption/completion/completion_cache.py`

**Step 1: Rewrite `_evict_entries_redis` with sorted set batch eviction**

Replace:
```python
    async def _evict_entries_redis(self):
        """Evict entries from Redis cache using sorted set for efficient LRU"""
        try:
            hot_keys = await self._redis_cache.smembers(self.HOT_KEYS_KEY)

            # Amortized eviction: evict ~10% of max_size to avoid per-put eviction
            evict_target = max(1, self.max_size // 10)

            # Get oldest entries by access time (lowest scores in sorted set).
            # zrangebyscore returns all entries; slice in Python (bounded by max_size).
            oldest_entries = await self._redis_cache.zrangebyscore(
                self.ACCESS_ORDER_KEY, float("-inf"), float("inf")
            )
            candidates = oldest_entries[: evict_target * 2]

            evicted = 0
            for key in candidates:
                if evicted >= evict_target:
                    break

                # Protect hot keys from eviction
                if key in hot_keys:
                    continue

                # Evict entry and its access order record
                await self._redis_cache.delete(self._make_cache_key(key))
                await self._redis_cache.zrem(self.ACCESS_ORDER_KEY, key)
                self._stats["evictions"] += 1
                evicted += 1

        except Exception as e:
            logger.error(f"Redis eviction error: {e}")
```

**Step 2: Convert `invalidate` and `_invalidate_redis` to async**

Replace `invalidate`:
```python
    async def invalidate(self, pattern: str = None):
        """Invalidate the cache"""
        if self._redis_configured:
            await self._invalidate_redis(pattern)
        self._invalidate_local(pattern)
```

Replace `_invalidate_redis`:
```python
    async def _invalidate_redis(self, pattern: str = None):
        """Invalidate Redis cache"""
        try:
            if pattern is None:
                # Clear all cache
                keys = await self._redis_cache.keys(f"{self.CACHE_KEY_PREFIX}*")
                if keys:
                    await self._redis_cache.delete(*keys)
                await self._redis_cache.delete(self.ACCESS_ORDER_KEY)
                await self._redis_cache.delete(self.HOT_KEYS_KEY)
                logger.info("All Redis cache cleared")
            else:
                # Invalidate by pattern
                keys = await self._redis_cache.keys(f"{self.CACHE_KEY_PREFIX}*{pattern}*")
                if keys:
                    await self._redis_cache.delete(*keys)
                logger.info(f"Invalidated Redis cache by pattern: {pattern} ({len(keys)} items)")
        except Exception as e:
            logger.error(f"Redis invalidate error: {e}")
```

**Step 3: Convert `_increment_stat` to async**

Replace:
```python
    async def _increment_stat(self, stat_name: str, amount: int = 1):
        """Increment a statistic (thread-safe for local, async for Redis)"""
        if self._redis_configured:
            try:
                await self._redis_cache.incr(f"{self.STATS_KEY}:{stat_name}", amount)
            except Exception as e:
                logger.debug(f"Redis stat increment failed: {e}")
        with self._lock:
            self._stats[stat_name] = self._stats.get(stat_name, 0) + amount
```

**Step 4: Compile-check**

Run: `python -m py_compile opencontext/context_consumption/completion/completion_cache.py`
Expected: Success

**Step 5: Commit**

```bash
git add opencontext/context_consumption/completion/completion_cache.py
git commit -m "fix(completion-cache): convert eviction/invalidation/stats to async

- Rewrite eviction with sorted set batch eviction (amortized ~10% of max_size)
- Add await to all RedisCache calls in invalidate, increment_stat"
```

---

### Task 4: Convert remaining CompletionCache methods to async

**Files:**
- Modify: `opencontext/context_consumption/completion/completion_cache.py`

**Step 1: Convert `optimize` and `_optimize_redis` to async**

Replace `optimize`:
```python
    async def optimize(self):
        """Cache optimization"""
        if self._redis_configured:
            await self._optimize_redis()
        self._optimize_local()
```

Replace `_optimize_redis`:
```python
    async def _optimize_redis(self):
        """Optimize Redis cache"""
        try:
            now = datetime.now()

            # Clean up expired entries
            keys = await self._redis_cache.keys(f"{self.CACHE_KEY_PREFIX}*")
            for key in keys:
                data = await self._redis_cache.get_json(key)
                if data:
                    created_at = datetime.fromisoformat(data["created_at"])
                    if now - created_at > self.ttl * 2:
                        await self._redis_cache.delete(key)
                        # Also clean up sorted set entry
                        short_key = key.removeprefix(self.CACHE_KEY_PREFIX)
                        await self._redis_cache.zrem(self.ACCESS_ORDER_KEY, short_key)

            # Clean up old precomputed contexts
            precomputed_keys = await self._redis_cache.keys(f"{self.PRECOMPUTED_KEY_PREFIX}*")
            for key in precomputed_keys:
                data = await self._redis_cache.get_json(key)
                if data:
                    computed_at = datetime.fromisoformat(data["computed_at"])
                    if now - computed_at > timedelta(hours=1):
                        await self._redis_cache.delete(key)

            logger.info("Redis cache optimization complete")
        except Exception as e:
            logger.error(f"Redis optimization error: {e}")
```

**Step 2: Convert `get_stats` to async, use `zcard` for Redis size**

Replace:
```python
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["total_requests"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            stats = {
                "redis_configured": self._redis_configured,
                "max_size": self.max_size,
                "hit_rate": round(hit_rate, 2),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "total_requests": total_requests,
                "average_response_time_ms": round(self._stats["average_response_time"] * 1000, 2),
            }

        if self._redis_configured:
            try:
                stats["redis_cache_size"] = await self._redis_cache.zcard(self.ACCESS_ORDER_KEY)
                stats["redis_hot_keys_count"] = await self._redis_cache.scard(self.HOT_KEYS_KEY)
            except Exception as e:
                logger.debug(f"Failed to get Redis cache statistics: {e}")

        stats["local_cache_size"] = len(self._local_cache)
        stats["local_hot_keys_count"] = len(self._hot_keys)
        stats["precomputed_contexts"] = len(self._precomputed_contexts)
        stats["memory_usage_estimate"] = self._estimate_memory_usage()

        return stats
```

**Step 3: Convert `precompute_context` and `get_precomputed_context` to async**

Replace `precompute_context`:
```python
    async def precompute_context(self, document_id: int, content: str):
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

            if self._redis_configured:
                try:
                    await self._redis_cache.set_json(
                        f"{self.PRECOMPUTED_KEY_PREFIX}{document_id}",
                        precomputed,
                        ttl=3600,  # 1 hour
                    )
                except Exception as e:
                    logger.error(f"Redis precompute error: {e}")

            self._precomputed_contexts[document_id] = precomputed
            logger.debug(f"Precomputed document context: {document_id}")

        except Exception as e:
            logger.error(f"Failed to precompute context: {e}")
```

Replace `get_precomputed_context`:
```python
    async def get_precomputed_context(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get precomputed context"""
        # Try Redis first
        if self._redis_configured:
            try:
                data = await self._redis_cache.get_json(
                    f"{self.PRECOMPUTED_KEY_PREFIX}{document_id}"
                )
                if data:
                    return data
            except Exception as e:
                logger.debug(f"Redis get for precomputed context failed: {e}")

        return self._precomputed_contexts.get(document_id)
```

**Step 4: Convert `export_hot_patterns` to async**

Replace:
```python
    async def export_hot_patterns(self) -> List[Dict[str, Any]]:
        """Export hot patterns for model training optimization"""
        hot_patterns = []

        # Get hot keys from Redis if available
        hot_keys = self._hot_keys.copy()
        if self._redis_configured:
            try:
                redis_hot_keys = await self._redis_cache.smembers(self.HOT_KEYS_KEY)
                hot_keys.update(redis_hot_keys)
            except Exception as e:
                logger.debug(f"Redis hot key access failed: {e}")

        with self._lock:
            for key in hot_keys:
                entry = None

                # Try local cache first
                if key in self._local_cache:
                    entry = self._local_cache[key]
                elif self._redis_configured:
                    # Try Redis
                    try:
                        data = await self._redis_cache.get_json(self._make_cache_key(key))
                        if data:
                            entry = CacheEntry.from_dict(data)
                    except Exception as e:
                        logger.debug(f"Redis hot key access failed: {e}")

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
```

**Step 5: Update `clear_completion_cache` factory function to async**

Replace:
```python
async def clear_completion_cache():
    """Clear the global completion cache"""
    global _completion_cache_instance
    if _completion_cache_instance is not None:
        await _completion_cache_instance.invalidate()
```

**Step 6: Update `cache_completion` decorator to async**

Replace:
```python
def cache_completion(ttl: int = 300):
    """
    Completion cache decorator (async)

    Args:
        ttl: Cache time-to-live in seconds (not yet implemented, reserved parameter)
    """
    _ = ttl  # Mark parameter as used to avoid warnings

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cache = get_completion_cache()
            cached_result = await cache.get(cache_key)

            if cached_result is not None:
                return cached_result

            # Execute original function
            result = await func(*args, **kwargs)

            # Cache the result
            if result:
                await cache.put(cache_key, result)

            return result

        return wrapper

    return decorator
```

**Step 7: Compile-check**

Run: `python -m py_compile opencontext/context_consumption/completion/completion_cache.py`
Expected: Success

**Step 8: Commit**

```bash
git add opencontext/context_consumption/completion/completion_cache.py
git commit -m "fix(completion-cache): convert optimize/stats/precompute/export to async

Complete async conversion of all CompletionCache methods that interact with Redis.
Also update clear_completion_cache() and cache_completion decorator to async."
```

---

### Task 5: Update CompletionService to async

**Files:**
- Modify: `opencontext/context_consumption/completion/completion_service.py:88-425`

**Step 1: Add `await` to cache calls in `get_completions` (already async)**

In `get_completions()` (line 88), change line 119:
```python
            # Before:
            cached_result = self.cache.get(cache_key)
            # After:
            cached_result = await self.cache.get(cache_key)
```

And change line 146:
```python
            # Before:
            self.cache.put(cache_key, suggestions, context_hash, confidence_score)
            # After:
            await self.cache.put(cache_key, suggestions, context_hash, confidence_score)
```

**Step 2: Convert sync service methods to async**

Replace lines 410-425:
```python
    async def clear_cache(self):
        """Clear the cache"""
        await self.cache.invalidate()
        logger.info("Completion cache cleared")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return await self.cache.get_stats()

    async def precompute_document_context(self, document_id: int, content: str):
        """Precompute document context"""
        await self.cache.precompute_context(document_id, content)

    async def optimize_cache(self):
        """Optimize cache performance"""
        await self.cache.optimize()
```

**Step 3: Compile-check**

Run: `python -m py_compile opencontext/context_consumption/completion/completion_service.py`
Expected: Success

**Step 4: Commit**

```bash
git add opencontext/context_consumption/completion/completion_service.py
git commit -m "fix(completion-service): add await to async cache calls, convert methods to async"
```

---

### Task 6: Update completion routes (currently unregistered — for correctness)

**Files:**
- Modify: `opencontext/server/routes/completions.py:242-328`

**Context:** These routes are NOT registered in `api.py` (dead code per `server/MODULE.md`). Changes ensure correctness so the module works when routes are activated.

**Step 1: Add `await` to all service method calls that became async**

Line 249 — `get_completion_stats`:
```python
        # Before:
        cache_stats = completion_service.get_cache_stats()
        # After:
        cache_stats = await completion_service.get_cache_stats()
```

Line 271 — `get_cache_stats`:
```python
        # Before:
        stats = completion_service.get_cache_stats()
        # After:
        stats = await completion_service.get_cache_stats()
```

Line 287 — `optimize_cache`:
```python
        # Before:
        completion_service.optimize_cache()
        # After:
        await completion_service.optimize_cache()
```

Line 290 — `get_cache_stats` (after optimization):
```python
        # Before:
        stats = completion_service.get_cache_stats()
        # After:
        stats = await completion_service.get_cache_stats()
```

Line 306 — `precompute_document_context`:
```python
        # Before:
        completion_service.precompute_document_context(document_id, content)
        # After:
        await completion_service.precompute_document_context(document_id, content)
```

Line 322 — `clear_cache`:
```python
        # Before:
        completion_service.clear_cache()
        # After:
        await completion_service.clear_cache()
```

**Step 2: Compile-check**

Run: `python -m py_compile opencontext/server/routes/completions.py`
Expected: Success

**Step 3: Commit**

```bash
git add opencontext/server/routes/completions.py
git commit -m "fix(routes): add await to completion service async method calls"
```

---

### Task 7: Final cross-module verification

**Step 1: Compile-check all modified files**

```bash
python -m py_compile opencontext/context_consumption/completion/completion_cache.py && \
python -m py_compile opencontext/context_consumption/completion/completion_service.py && \
python -m py_compile opencontext/server/routes/completions.py
```

Expected: All pass

**Step 2: Verify no remaining sync calls to async cache methods**

Search for any `self.cache.get(`, `self.cache.put(`, `self.cache.invalidate(`, `self.cache.optimize(`, `self.cache.get_stats(`, `self.cache.precompute_context(` that are NOT preceded by `await`:

```bash
grep -rn "self\.cache\.\(get\|put\|invalidate\|optimize\|get_stats\|precompute_context\)" \
    opencontext/context_consumption/completion/completion_service.py | grep -v "await"
```

Expected: No output (all calls have `await`)

Also verify `self._redis_cache.` calls:
```bash
grep -rn "self\._redis_cache\." \
    opencontext/context_consumption/completion/completion_cache.py | grep -v "await"
```

Expected: No output (all Redis calls have `await`)

**Step 3: Verify no remaining `_use_redis` references**

```bash
grep -rn "_use_redis" opencontext/context_consumption/completion/completion_cache.py
```

Expected: No output

---

## Summary of Changes

| File | Changes |
|------|---------|
| `completion_cache.py` | Full async conversion. Replace `_use_redis` → `_redis_configured`. Replace list LRU → sorted set. Replace `keys()` size check → `zcard()`. |
| `completion_service.py` | Add `await` in `get_completions()`. Convert `clear_cache`, `get_cache_stats`, `precompute_document_context`, `optimize_cache` to async. |
| `completions.py` (routes) | Add `await` to 6 service method calls. |

## Key Design Decisions

1. **Why `_redis_configured` instead of `_use_redis`?** The old flag was set by un-awaited `is_connected()` (always truthy coroutine). The new flag tracks configuration intent, not runtime connectivity. Try/except on each operation naturally handles Redis going up/down — no stale flag.

2. **Why sorted set instead of list for LRU?** The original code used `lrem` which doesn't exist on RedisCache (`AttributeError` caught by except, silently broken). Sorted set with timestamp scores gives: natural LRU ordering, O(log N) add/update via `zadd`, O(1) size via `zcard`, no need for separate remove-then-add on access update.

3. **Why batch eviction (10%)?** The original code evicted one-by-one with an O(N) `keys()` scan per iteration. Batch eviction amortizes the cost and avoids pathological per-put eviction when cache is near capacity.

4. **Why keep local fallback?** Resilience. When Redis is down, completions still work with local in-memory cache. Every Redis operation's try/except falls back to local, matching the existing design intent.
