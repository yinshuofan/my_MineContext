# MineContext Performance, Scalability & High Availability Audit Report

> **Date:** 2026-03-09
> **Scope:** Full codebase review across 6 dimensions — concurrency, storage performance, API layer, caching, scheduler, architecture
> **Goal:** Identify issues blocking high-performance, high-availability, high-concurrency deployment at scale
> **Status:** Read-only audit — no code changes

---

## Summary

6 parallel review agents audited ~15,600 lines of Python code across 60+ files. After deduplication, **67 unique issues** were identified:

| Severity | Count | Description |
|----------|-------|-------------|
| **CRITICAL** | 10 | Runtime failures, data loss, broken features |
| **HIGH** | 17 | Performance degradation, scaling blockers |
| **MEDIUM** | 24 | Operational issues, moderate performance impact |
| **LOW** | 16 | Minor issues, code quality |

---

## Table of Contents

1. [CRITICAL Issues](#critical-issues)
2. [HIGH Issues](#high-issues)
3. [MEDIUM Issues](#medium-issues)
4. [LOW Issues](#low-issues)
5. [Positive Observations](#positive-observations)
6. [Prioritized Remediation Roadmap](#prioritized-remediation-roadmap)

---

## CRITICAL Issues

### C1. Missing `await` on async storage calls — multiple routes completely broken

**Files:**
- `opencontext/server/routes/conversation.py` — lines 102, 133, 154, 182, 210
- `opencontext/server/routes/messages.py` — lines 113, 146, 176, 207, 235, 260, 290
- `opencontext/server/routes/vaults.py` — lines 88, 128, 169, 221, 272
- `opencontext/server/routes/agent_chat.py` — lines 134, 146, 149, 158, 195, 208, 240, 250, 264, 282

**Description:** All `UnifiedStorage` methods are `async def`, but these routes call them without `await`. This returns coroutine objects instead of actual data. In `agent_chat.py`, `storage.append_message_content()` and `storage.mark_message_finished()` are never awaited, so streaming chat data is never persisted.

**Impact:** Conversation CRUD, message CRUD, vault CRUD, and streaming chat persistence are **completely non-functional** at runtime. Pydantic serialization will either fail or return garbage. This is a P0 bug.

**Fix approach:** Add `await` before every async storage method call. Mechanical fix — route handlers are already `async def`.

---

### C2. CompletionCache calls async Redis methods synchronously — entire Redis mode broken

**File:** `opencontext/context_consumption/completion/completion_cache.py` — lines 139, 176, 187, 199, 202-203, 207, 283, 300, 303, 354-358, 362-364, 388, 392, 397, 399, 403, 441, 480, 499, 519-534, 602-604, 648, 663

**Description:** Every `RedisCache` method is `async def`, but `CompletionCache` calls them all synchronously without `await`. Each call returns a coroutine object (truthy, not `None`), causing incorrect control flow. `put`, `invalidate`, `_evict_entries_redis`, and `optimize` silently produce unawaited coroutines — no data is ever written to Redis.

**Impact:** CompletionCache Redis integration is 100% non-functional. Falls back to local cache with RuntimeWarnings in Python 3.12+.

**Fix approach:** Convert all `CompletionCache` methods to `async def` and `await` Redis calls.

---

### C3. `KEYS` command used in production hot paths — O(N) scan of entire Redis keyspace

**Files:**
- `opencontext/context_consumption/completion/completion_cache.py` — lines 283, 354, 362, 397, 519, 528, 602
- `opencontext/context_capture/text_chat.py` — lines 205, 246
- `opencontext/storage/redis_cache.py` — line 846

**Description:** Redis `KEYS` command scans the entire keyspace and blocks the Redis server. Called on every `put` operation in CompletionCache and during eviction (in a loop — each iteration re-scans). Also used in `_flush_all_buffers` and `get_buffer_stats`.

**Impact:** On a production Redis with millions of keys, each `KEYS` call can take seconds, blocking **all** Redis operations across all service instances. The eviction loop compounds this catastrophically.

**Fix approach:** Replace `KEYS` with `SCAN` for enumeration. For size counting, use an atomic counter key (INCR/DECR on put/evict). Replace `RedisCache.keys()` with an async `SCAN`-based iterator.

---

### C4. Race condition in `schedule_user_task` — check-then-create is non-atomic

**File:** `opencontext/scheduler/redis_scheduler.py` — lines 199-266

**Description:** Multi-step read-check-write sequence across 3 separate Redis round-trips (check existing, check last_exec, check fail_count) then write (pipeline). Two instances processing concurrent pushes for the same user can both pass all checks and both create the task.

**Impact:** Duplicate task scheduling, inconsistent task state. The dedup check at line 200-201 has a TOCTOU window.

**Fix approach:** Consolidate into a single Lua script that atomically checks existence + creates task + enqueues. Or use a short-lived distributed lock per user+task_type key.

---

### C5. Profile merge race condition — read-modify-write without locking

**File:** `opencontext/context_processing/processor/profile_processor.py` — lines 52-71

**Description:** Reads existing profile → merges with new data via LLM → upserts merged result. Two concurrent profile updates for the same user will cause the second write to overwrite the first, silently losing data. The LLM call (seconds) makes the window large.

**Impact:** Silent data loss on user profiles under concurrent pushes from the same user across instances.

**Fix approach:** Distributed lock (Redis) per `user_id:device_id:agent_id` around the read-merge-write cycle. Or optimistic concurrency control with `updated_at` version check in MySQL.

---

### C6. No task execution timeout enforcement — scheduler can deadlock

**File:** `opencontext/scheduler/redis_scheduler.py` — lines 535-600

**Description:** `TaskConfig.timeout` is used for lock TTL but the actual `await handler(...)` has no `asyncio.wait_for` timeout. If a handler hangs (LLM call never returns, DB stall), the task holds its semaphore slot indefinitely. With the default `max_concurrent=5`, 5 hung tasks = complete scheduler deadlock.

**Impact:** A single LLM provider partial outage can cascade into total scheduler failure. Hierarchy summary tasks make 20+ sequential LLM calls, making this especially dangerous.

**Fix approach:** Wrap handler call in `asyncio.wait_for(handler(...), timeout=task_config.timeout)`. On timeout, mark task failed and release semaphore.

---

### C7. Chat buffer flush race — messages lost between `lrange` and `delete`

**Files:**
- `opencontext/context_capture/text_chat.py` — lines 337-362

**Description:** `_flush_buffer` acquires lock → `lrange_json` (read all) → process → `delete` (delete entire key). Between read and delete, new messages can arrive via `push_message` (which does NOT acquire the flush lock). The `delete` at line 356 removes these unprocessed messages.

**Impact:** Messages arriving during the flush window are silently lost. Under high message throughput, this is significant.

**Fix approach:** Replace `lrange + delete` with a Lua script: `LRANGE key 0 N-1` then `LTRIM key N -1`, operating only on the N messages that were read. New messages appended after the read are preserved.

---

### C8. Synchronous blocking file I/O on the async event loop

**Files:**
- `opencontext/server/routes/push.py` — lines 151, 244, 378, 558
- `opencontext/server/routes/media.py` — lines 99-102

**Description:** `_save_base64_to_temp_file`, `_save_media_base64`, `_upload_or_save_media`, and `upload_document_file` perform synchronous `open(file_path, "wb")` file writes inside async route handlers. For 50 MB video uploads, this blocks the entire event loop for seconds.

**Impact:** A single large file upload stalls all concurrent requests on that worker.

**Fix approach:** Use `asyncio.to_thread()` or `aiofiles` for all file write operations.

---

### C9. Qdrant `get_collection` RPC per collection on every search

**File:** `opencontext/storage/backends/qdrant_backend.py` — lines 465-469

**Description:** `search()` calls `await self._client.get_collection(collection_name)` to check `points_count` for every collection on every search. This is a full network round-trip per collection (3 per search).

**Impact:** Adds 15-60ms latency per search. Under concurrent load, creates unnecessary pressure on the Qdrant server.

**Fix approach:** Remove the pre-check entirely — Qdrant returns empty results for empty collections without error. Or cache collection point counts with a short TTL.

---

### C10. In-process Monitor state not shared across instances

**File:** `opencontext/monitoring/monitor.py` — lines 90-120

**Description:** `Monitor` maintains in-process `deque` structures as the primary data source for performance summaries. These are never persisted and exist only in the process that recorded them. Any instance serving the monitoring API shows only its own partial view.

**Impact:** With 10+ instances, monitoring dashboard shows misleading, fragmented metrics. Operators investigating issues see incomplete data.

**Fix approach:** Persist all metric categories to the relational DB. Or expose `/metrics` in Prometheus format and aggregate at the collection layer.

---

## HIGH Issues

### H1. No rate limiting on any endpoint

**Files:** All route files, `cli.py`

No rate limiting middleware, per-user throttling, or global concurrency limit. A single client can flood the service, exhausting event loop, thread pool, Redis/MySQL connections, and LLM quota.

**Fix:** Add Redis-backed token bucket/sliding window rate limiting (e.g., `slowapi`). Protect LLM-intensive endpoints first.

### H2. No request body size limit — unbounded base64 payloads

**File:** `opencontext/server/routes/push.py` — line 105

`PushDocumentRequest.base64_data` is `Optional[str]` with no `max_length`. Multi-GB payloads can cause OOM. The 10MB/50MB media guards only apply after full decoding in memory.

**Fix:** Add `max_length` to `base64_data`. Configure Uvicorn `--limit-max-request-size` or add a body-size middleware.

### H3. CORS configuration hardcoded for development

**File:** `opencontext/cli.py` — lines 129-135

`allow_origins=["http://localhost:5173", "http://localhost"]` is hardcoded. Production deployments on different domains will have all browser requests blocked.

**Fix:** Read allowed origins from `config.yaml` or environment variable.

### H4. Knowledge merger has non-atomic read-merge-delete cycle

**File:** `opencontext/context_processing/merger/context_merger.py` — lines 429-479

If global `periodic_memory_compression` and per-user `periodic_memory_compression_for_user` overlap for the same user, both read the same contexts, both merge them, both upsert new merged results, and both delete originals — creating duplicate knowledge entries.

**Fix:** Add a distributed lock per user around the merge operation.

### H5. Pipeline used without `transaction=True` — not atomic on Redis

**File:** `opencontext/scheduler/redis_scheduler.py` — lines 206, 262

Pipeline calls use default `transaction=False` — commands are batched but not atomic. A crash between `hset` and `zadd` creates orphaned task hashes never cleaned up.

**Fix:** Pass `transaction=True` or consolidate into a Lua script.

### H6. Qdrant search issues sequential queries per collection

**File:** `opencontext/storage/backends/qdrant_backend.py` — lines 465-496

Sequential `for` loop over target collections. Search latency = sum of all collection search times (150ms+) instead of max (50ms).

**Fix:** Use `asyncio.gather` to query all collections in parallel. Single-digit line change.

### H7. `BaseHTTPMiddleware` blocks event loop under concurrency

**File:** `opencontext/server/middleware/request_id.py` — lines 18-26

Starlette's `BaseHTTPMiddleware` creates additional task + intermediate queue per request. Well-documented performance issue.

**Fix:** Replace with pure ASGI middleware.

### H8. Qdrant `_check_connection` (full `get_collections` RPC) on every batch write

**File:** `opencontext/storage/backends/qdrant_backend.py` — lines 188-189

Health check RPC before every upsert adds 5-20ms per write.

**Fix:** Remove from hot write path. Use background periodic health check.

### H9. Embedding batch does not truly batch — N separate API calls

**File:** `opencontext/llm/global_embedding_client.py` — lines 218-250

Each item in `do_vectorize_batch` makes a separate HTTP request. 50 chunks = 50 API calls (with semaphore concurrency 15).

**Fix:** Use true batch API if available. Group text-only items into single requests.

### H10. No database migration strategy

**File:** `opencontext/storage/backends/mysql_backend.py` — line 116

`CREATE TABLE IF NOT EXISTS` only. No schema versioning, no migration tool, no ALTER TABLE support.

**Fix:** Adopt Alembic. Add schema versioning.

### H11. Single Redis instance is a SPOF

**Files:** `opencontext/storage/redis_cache.py`, `config/config.yaml`

No support for Redis Sentinel, Cluster, or replica sets. Redis is used for buffering, scheduling, caching, pub/sub. If it goes down, everything fails simultaneously.

**Fix:** Support Redis Sentinel config for automatic failover. Document HA requirement.

### H12. ThreadPoolExecutor size hardcoded at 10

**File:** `opencontext/cli.py` — line 48

10 threads shared by all `asyncio.to_thread()` operations. Document processing can saturate pool, blocking chat processing.

**Fix:** Make configurable. Consider separate pools for different workload types.

### H13. VikingDB `get_processed_context_count` fetches up to 100,000 records to count

**File:** `opencontext/storage/backends/vikingdb_backend.py` — lines 1451-1474

Count query uses `"limit": 100000` and reads response count. Sequential per-type calls compound the issue.

**Fix:** Use dedicated count API or cache counts in Redis. Parallelize per-type queries.

### H14. CompletionCache eviction loop can spin infinitely

**File:** `opencontext/context_consumption/completion/completion_cache.py` — lines 385-410

If all remaining keys are "hot keys" and cache is oversized, the `while True` loop cycles through them indefinitely, each iteration calling `KEYS`.

**Fix:** Add max iteration count. Force-evict oldest hot key after full cycle.

### H15. `get_cache()` creates throwaway `InMemoryCache` on every call when Redis is down

**File:** `opencontext/storage/redis_cache.py` — lines 1473-1489

Each call creates a fresh empty `InMemoryCache`. Data stored in one call is invisible to the next. Cache stampede prevention breaks.

**Fix:** Use a module-level singleton `InMemoryCache` as fallback.

### H16. `InMemoryCache` has no expiry cleanup — unbounded memory growth

**File:** `opencontext/storage/redis_cache.py` — lines 991-1470

Expiry checked lazily on read only. Keys never read again remain forever. No max size, no sweep.

**Fix:** Add periodic sweep or max capacity with LRU eviction.

### H17. Hierarchy summary unbounded cost per execution — 20+ sequential LLM calls

**File:** `opencontext/periodic_task/hierarchy_summary.py` — lines 179-268

Single `execute()` call: up to 7 daily + 1 weekly + 1 monthly summaries, each with LLM calls + batch overflow. Can run 5-10 minutes, monopolizing semaphore slots.

**Fix:** Rate-limit backfill days per execution. Add per-level timeouts. Parallelize independent queries.

---

## MEDIUM Issues

### M1. `close_redis_cache` mixes sync lock with async operation
**File:** `redis_cache.py:976-983` — Holds threading.Lock while awaiting Redis close. Can block threads.

### M2. ContextMerger._statistics not thread-safe
**File:** `context_merger.py:48,216,228,231` — `+= 1` without lock. Low impact (advisory counters).

### M3. `_context_lab_instance` global has no init protection
**File:** `cli.py:28-37` — No lock on lazy init. Low risk (lifespan called once per process).

### M4. InMemoryCache.release_lock is not atomic (check-then-delete)
**File:** `redis_cache.py:1424-1430` — Get + delete not under single lock. Breaks lock safety in fallback.

### M5. MySQL pool size hardcoded to maxsize=20
**File:** `mysql_backend.py:77-88` — 10 instances = 200 connections. MySQL default limit = 151. Not configurable.

### M6. Redis BlockingConnectionPool timeout hardcoded to 5 seconds
**File:** `redis_cache.py:144-156` — Under burst traffic, operations stall or fail silently.

### M7. Scheduler startup failure silently swallowed
**File:** `cli.py:70-76` — Scheduler fail → app runs without scheduling → memory compression, hierarchy, cleanup stop silently.

### M8. Heartbeat key collision across instances
**File:** `redis_scheduler.py:69` — Single `scheduler:heartbeat` key. Last writer wins. Masks per-instance health.

### M9. `_execute_task` silently skips if no handler after dequeue
**File:** `redis_scheduler.py:542-546` — Task dequeued, no handler → task lost. No `complete_task` called.

### M10. SQLite backend — single shared connection, no concurrency control
**File:** `sqlite_backend.py:36-52` — No pool, no mutex. Concurrent writes → "database is locked" errors.

### M11. Qdrant offset-based pagination — O(offset+limit) per page
**File:** `qdrant_backend.py:280-347` — Page 100 with 100/page fetches 10,000 records, discards 9,900.

### M12. VikingDB sequential per-type queries in single-collection setup
**File:** `vikingdb_backend.py:952-1027` — 4 sequential calls where 1 combined query would suffice.

### M13. `get_filtered_context_count` sequential per-type instead of parallel
**File:** `unified_storage.py:349-369` — Easy `asyncio.gather` optimization.

### M14. Qdrant/VikingDB serialization overhead — model_dump/validate + field sets recomputed per call
**Files:** `qdrant_backend.py:134-180,498-567`, `vikingdb_backend.py:1149-1151`

### M15. VikingDB `delete_by_source_file` unbounded search+delete, no pagination
**File:** `vikingdb_backend.py:1482-1573` — Searches with `limit:10000`, deletes all at once. Can timeout.

### M16. Memory cache snapshot has no size bound
**File:** `server/cache/memory_cache_manager.py:272-436` — Large profiles + events = huge Redis payloads.

### M17. No LLM response caching for idempotent embedding operations
**Files:** `llm_client.py`, `global_embedding_client.py` — Same text re-embedded on retry/re-index wastes API cost.

### M18. Config reload pub/sub has no debounce/dedup
**File:** `server/config_reload_manager.py:59-101` — Rapid reloads can interleave, causing partial config states.

### M19. `threading.Lock` in async settings routes blocks event loop
**File:** `server/routes/settings.py:26,172,377,541` — Holds sync lock across `await` LLM validation.

### M20. Storage not checked for `None` in conversation/messages/vaults routes
**Files:** `conversation.py`, `messages.py`, `vaults.py` — Returns 500 AttributeError instead of 503.

### M21. `convert_resp` double-serializes JSON on every response
**File:** `server/utils.py:28-41` — `json.dumps` + `json.loads` + `JSONResponse` = 3x serialization cost.

### M22. Vaults `get_document` fetches all documents to find one
**File:** `server/routes/vaults.py:168-176` — O(n) scan capped at 100. Documents beyond 100 unreachable.

### M23. Global agent instance without thread safety
**File:** `server/routes/agent_chat.py:33-42` — Lazy init with no lock. Potential state corruption.

### M24. Periodic task failure has no retry counter or backoff
**File:** `redis_scheduler.py:618-693` — Persistent failures cause log flooding with no circuit-breaking.

---

## LOW Issues

### L1. Fire-and-forget task in `search.py` not saved to reference set (GC risk)
### L2. Document processor `asyncio.run()` inside `asyncio.to_thread()` — creates separate event loops
### L3. Monitoring SQL avg_duration_ms — safe due to InnoDB row-level locking (non-issue)
### L4. DataCleanupTask duck-typing with `hasattr` — fragile, possible sync/async mismatch
### L5. Memory compression interval comment mismatch (172800 vs docstring "30 min")
### L6. `TaskResult.fail()` does not accept `data` parameter
### L7. User key parsing fragile when user_id contains ":"
### L8. Debug `print()` statements in static file setup
### L9. Hardcoded Chinese strings in media error messages
### L10. Completions streaming has artificial `asyncio.sleep(0.1)` delay
### L11. Unused import `from math import log` in auth middleware
### L12. API key fragment logged in warning message (security concern)
### L13. Internal error messages leaked to clients (stack traces, paths)
### L14. Scheduler runs on all instances without leader election option
### L15. InMemoryCache fallback silently breaks multi-instance semantics
### L16. Sync/async mixing in MetricsCollector decorators

---

## Positive Observations

The following patterns demonstrate strong engineering discipline:

1. **Atomic Lua scripts** — `_CONDITIONAL_ZPOPMIN_LUA`, `_RELEASE_LOCK_LUA`, `_RPUSH_EXPIRE_LLEN_LUA` — correct use of server-side atomicity
2. **Lazy storage via `@property`** — avoids caching `None` from uninitialized singletons, consistently applied
3. **`asyncio.shield()` in scheduler** — protects lock release from task cancellation during shutdown
4. **Pipeline batching** for task creation in scheduler
5. **MySQL `_get_connection` async context manager** with automatic rollback on exceptions
6. **Distributed flush lock** in text_chat (correct pattern, needs atomic read+trim fix)
7. **Fire-and-forget GC protection** in push.py (`_background_tasks` set + done_callback)
8. **Batch pre-vectorization** in both Qdrant and VikingDB backends
9. **Request ID middleware with ContextVar** for cross-stack request tracing
10. **Comprehensive pitfall documentation** in CLAUDE.md — captures real production bugs
11. **Readiness probe** (`/api/ready`) checking all core dependencies
12. **Stampede prevention** in memory cache manager with distributed lock + double-check

---

## Prioritized Remediation Roadmap

### Phase 1: Critical Bugs (P0) — Fix immediately

These are runtime correctness bugs causing broken features or data loss:

| # | Issue | Effort | Impact |
|---|-------|--------|--------|
| C1 | Add missing `await` in conversation/messages/vaults/agent_chat routes | Small | Fixes 4 completely broken API modules |
| C2 | Fix CompletionCache sync/async mismatch | Medium | Restores Redis cache mode |
| C7 | Fix chat buffer flush race (lrange+ltrim Lua script) | Medium | Prevents message loss |
| C5 | Add distributed lock for profile merge | Medium | Prevents profile data loss |

### Phase 2: Performance Critical (P1) — Before production

| # | Issue | Effort | Impact |
|---|-------|--------|--------|
| C3 | Replace KEYS with SCAN + atomic counters | Medium | Eliminates Redis blocking |
| C8 | Use asyncio.to_thread for file I/O | Small | Unblocks event loop |
| C9 | Remove Qdrant get_collection pre-check | Trivial | -15-60ms per search |
| H6 | Parallelize Qdrant search across collections | Trivial | -100ms per search |
| H8 | Remove Qdrant _check_connection from write path | Trivial | -5-20ms per write |
| H7 | Replace BaseHTTPMiddleware with pure ASGI | Small | Reduces per-request overhead |
| M21 | Fix double JSON serialization | Small | -66% serialization CPU |

### Phase 3: Scaling & Resilience (P2) — Before horizontal scale-out

| # | Issue | Effort | Impact |
|---|-------|--------|--------|
| C4+C6 | Fix scheduler: atomic scheduling + execution timeout | Large | Prevents deadlock & duplicate tasks |
| H1 | Add rate limiting | Medium | Prevents resource exhaustion |
| H2 | Add request body size limits | Small | Prevents OOM |
| H3 | Make CORS configurable | Small | Enables production deployment |
| H5 | Use transactional Redis pipelines | Small | Prevents orphaned state |
| H10 | Adopt Alembic for migrations | Medium | Enables schema evolution |
| M5 | Make MySQL pool size configurable | Small | Prevents connection exhaustion |
| H12 | Make ThreadPoolExecutor configurable | Small | Enables workload tuning |

### Phase 4: Architecture (P3) — For scale and HA

| # | Issue | Effort | Impact |
|---|-------|--------|--------|
| H11 | Redis Sentinel/Cluster support | Large | Eliminates Redis SPOF |
| C10 | Externalize monitoring metrics | Medium | Enables fleet-wide observability |
| H9 | True batch embedding API calls | Medium | Reduces API cost & latency |
| H17 | Bound hierarchy summary cost | Medium | Prevents scheduler starvation |
| M17 | Embedding result caching | Medium | Reduces API cost on retries |

---

*Report generated by 6 parallel review agents analyzing: concurrency/race conditions, storage performance, API layer, caching/memory, scheduler/background tasks, and architecture/horizontal scaling.*
