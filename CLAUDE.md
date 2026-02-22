# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MineContext is a **memory backend service** that captures, processes, stores, and retrieves context/memory from multiple sources (chat logs, documents, web links). It uses vector embeddings and LLM-powered analysis to organize information into 5 typed memory contexts with type-specific update strategies and storage routing.

## Common Commands

```bash
# Install dependencies (uv recommended)
uv sync

# Start the server (default port 1733)
uv run opencontext start
uv run opencontext start --config config/config.yaml
uv run opencontext start --port 1733 --host 0.0.0.0
uv run opencontext start --workers 4   # multi-process mode

# Code formatting (must use uv run — black/isort are dev deps, not globally installed)
uv run black opencontext --line-length 100
uv run isort opencontext
pre-commit run --all-files

# Docker
docker-compose up
```

There is no test suite in this project currently. To verify changes compile-check with:
```bash
python -m py_compile opencontext/path/to/file.py
```

## Architecture

### 5-Type Context System

All data flows through a unified `ProcessedContext` model, but each type has its own update strategy and storage destination. This is the central design decision — understand it before modifying any storage, processing, or retrieval code.

| Type | Update Strategy | Storage | Description |
|------|----------------|---------|-------------|
| `profile` | OVERWRITE | Relational DB | User's own preferences, habits, communication style. Key: `(user_id, device_id, agent_id)` |
| `entity` | OVERWRITE | Relational DB | Other people, projects, teams, organizations. Key: `(user_id, device_id, agent_id, entity_name)` |
| `document` | OVERWRITE | Vector DB | Uploaded files and web links. Overwrite = delete old chunks + insert new. Tracked by `source_file_key` |
| `event` | APPEND | Vector DB | Immutable activity records, meetings, status changes. Never modified after creation |
| `knowledge` | APPEND_MERGE | Vector DB | Reusable concepts, procedures, patterns. Similar entries merged to avoid duplication |

Defined in `opencontext/models/enums.py`:
- `ContextType` enum — the 5 types
- `UpdateStrategy` enum — OVERWRITE, APPEND, APPEND_MERGE
- `CONTEXT_UPDATE_STRATEGIES` dict — type → strategy mapping
- `CONTEXT_STORAGE_BACKENDS` dict — type → `"document_db"` or `"vector_db"`
- `ContextDescriptions` dict — rich descriptions with `key_indicators`, `examples`, `classification_priority` (used by LLM prompts for classification)

### Data Pipeline

```
Input → Processor → ProcessedContext → _handle_processed_context (routes by type) → Storage
```

1. **Process** (`opencontext/context_processing/`): `ProcessorFactory` routes by `ContextSource`:
   - `LOCAL_FILE`, `VAULT`, `WEB_LINK` → `DocumentProcessor`
   - `CHAT_LOG`, `INPUT` → `TextChatProcessor`
   - Entity extraction → `EntityProcessor`

2. **Route** (`opencontext/server/opencontext.py` → `_handle_processed_context()`): The central routing point. Reads `CONTEXT_STORAGE_BACKENDS` to decide per context:
   - profile → `storage.upsert_profile()` → relational DB
   - entity → `storage.upsert_entity()` → relational DB
   - document/event/knowledge → `storage.batch_upsert_processed_context()` → vector DB

3. **Store** (`opencontext/storage/`): `UnifiedStorage` wraps dual backends:
   - **Document DB** (MySQL or SQLite) — profiles table, entities table, plus context metadata
   - **Vector DB** (VikingDB, ChromaDB, or Qdrant) — document/event/knowledge embeddings

### Hierarchical Event Indexing

Events support a 4-level time-based hierarchy for efficient historical retrieval:
- **L0** — raw individual events
- **L1** — daily summaries (`time_bucket: "2026-02-21"`)
- **L2** — weekly summaries (`time_bucket: "2026-W08"`)
- **L3** — monthly summaries (`time_bucket: "2026-02"`)

Key fields on `ContextProperties`: `hierarchy_level`, `parent_id`, `children_ids`, `time_bucket`.

`HierarchicalEventTool` retrieval algorithm: search L1-L3 summaries top-down → drill through `children_ids` to L0 → merge with direct L0 semantic search as fallback → deduplicate by ID keeping higher score.

**Summary generation** (`hierarchy_summary.py`) uses `user_activity` trigger mode — summaries are scheduled per-user when data is pushed (via `_schedule_user_hierarchy_summary()` in `push.py`), not generated globally on a fixed schedule. This ensures:
- Only active users get summaries (no wasted LLM calls for inactive users)
- Tasks are processed sequentially by the scheduler (one per 10s cycle), providing natural rate limiting
- Each execution tries to generate the most recent completed daily/weekly/monthly summary, with idempotent dedup checks preventing regeneration
- Requires `HIERARCHY_SUMMARY_ENABLED=true` env var to activate

### Retrieval Tools

4 type-aligned tools registered in `opencontext/tools/tool_definitions.py`:

| Tool | Types Searched | Backend | Strategy |
|------|---------------|---------|----------|
| `ProfileEntityTool` | profile, entity | Relational DB | Exact name lookup + text search |
| `DocumentRetrievalTool` | document | Vector DB | Semantic similarity search |
| `KnowledgeRetrievalTool` | knowledge, event (L0) | Vector DB | Semantic similarity search |
| `HierarchicalEventTool` | event (all levels) | Vector DB | Summary drill-down + L0 fallback |

### Unified Search API (`POST /api/search`)

Two strategies available via `opencontext/server/search/`:

- **Fast** (`fast_strategy.py`): Zero LLM calls. Single embedding generation shared across all parallel storage queries via `asyncio.to_thread()`. Events filtered to L0 only, with parent summaries batch-attached.
- **Intelligent** (`intelligent_strategy.py`): LLM-driven agentic search. Uses `LLMContextStrategy` to select and execute retrieval tools. More accurate but slower.

Both return `TypedResults` with `profile`, `entities`, `documents`, `events`, `knowledge` fields.

Search results are automatically tracked as "recently accessed" via fire-and-forget `asyncio.create_task()` in the search route (see Memory Cache below).

### Memory Cache (`GET /api/memory-cache`)

Provides a single-call snapshot of a user's complete memory state, designed for downstream services that need quick access without running full searches. Located in `opencontext/server/cache/`.

**Three sections returned:**
1. **Profile + Entities** — from relational DB (cached in snapshot)
2. **Recently Accessed** — memories returned in past searches; stored in Redis Hash, always read real-time (not part of snapshot)
3. **Recent Memories** — hierarchical: today's L0 events + past N days' L1 daily summaries + recent documents/knowledge (cached in snapshot)

**Caching architecture (snapshot vs accessed are separate):**

| Data | Storage | Update Trigger | Read Mode |
|------|---------|---------------|-----------|
| Snapshot (profile + entities + recent memories) | Redis JSON String, 5-min TTL | Invalidated on push/write | Cache hit → return; miss → rebuild from DB |
| Recently Accessed | Redis Hash, 7-day TTL | Updated on every search | Always real-time from Redis |

**Key design decisions:**
- **Stampede prevention**: Distributed lock (`RedisCache.acquire_lock()`) + double-check pattern in `get_user_memory_cache()`. Lock timeout falls back to building without caching (duplicate build OK, better than failing).
- **Auto-invalidation**: `_invalidate_user_cache()` in `opencontext.py` fires after profile/entity/vector writes. Uses `asyncio.run_coroutine_threadsafe()` to bridge sync→async (see pitfall below).
- **Parallel snapshot build**: 6 concurrent `asyncio.to_thread()` queries via `asyncio.gather()`.
- **Singleton manager**: `get_memory_cache_manager()` in `memory_cache_manager.py`. All callers (route, search hook, opencontext.py invalidation) use the same instance.

Config section in `config/config.yaml`: `memory_cache` (snapshot_ttl, recent_days, max_recently_accessed, etc.).

### Storage Access

**Always use `get_storage()` from `opencontext/storage/global_storage.py`** to get the `UnifiedStorage` instance. This returns the object with all profile/entity/hierarchy methods.

Do NOT use `GlobalStorage.get_instance()` or `get_global_storage()` — these return the raw `GlobalStorage` wrapper which lacks `upsert_profile()`, `get_entity()`, `search_hierarchy()`, etc.

### Concurrency and Connection Management

The server is async-first (FastAPI + uvicorn) but many storage operations are synchronous. Key patterns:

- **ThreadPoolExecutor**: `lifespan()` in `cli.py` sets a 50-worker executor as the default for `asyncio.to_thread()`. The executor is stored and `shutdown(wait=False)` on lifespan exit.
- **MySQL**: SQLAlchemy `QueuePool` (pool_size=20, max_overflow=10). `_get_connection()` is a `@contextmanager` that auto-returns connections. Health-check ping on checkout via SQLAlchemy event listener.
- **SQLite**: `threading.local()` for per-thread connections. `_get_connection()` lazily creates a new connection per thread with WAL journal mode. **All method-level DB access must go through `self._get_connection()`** — only `initialize()` and `close()` may use `self.connection` directly.
- **Async timeout**: Push/search endpoints wrap processing in `asyncio.wait_for(..., timeout=60.0)`.

### Request Tracing

`RequestIDMiddleware` (`opencontext/server/middleware/request_id.py`) generates an 8-char UUID per request (or accepts `X-Request-ID` header). Stored in `contextvars.ContextVar`, injected into all loguru log records via a patcher. Use `from opencontext.server.middleware.request_id import request_id_var` to access in any code path.

### Server Layer

- **Entry point**: `opencontext/cli.py` → `opencontext start`
- **Framework**: FastAPI on Uvicorn, port 1733
- **Orchestrator**: `opencontext/server/opencontext.py` → `OpenContext`
- **Auth**: API key via `X-API-Key` header, disabled by default
- **Health check**: `check_components_health()` checks config, storage, llm, document_db, redis. MySQL uses context manager `with backend._get_connection()`, SQLite executes `SELECT 1`.

### Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with component status |
| `/api/push/chat/message` | POST | Push single chat message (buffered) |
| `/api/push/chat/messages` | POST | Push batch messages (buffered) |
| `/api/push/chat/process` | POST | Process messages directly (bypass buffer) |
| `/api/push/activity` | POST | Push activity/event record |
| `/api/push/context` | POST | Push generic context |
| `/api/push/document` | POST | Push document (file_path or base64) |
| `/api/search` | POST | Unified search (`strategy: "fast"` or `"intelligent"`) |
| `/api/memory-cache` | GET | User memory snapshot |

Request body uses OpenAI message format: `messages: [{role, content: [{type: "text", text: "..."}]}]`. The 3-key identifier `(user_id, device_id, agent_id)` is optional on all push endpoints, defaulting to `"default"`. Pydantic Field constraints (`min_length`, `max_length`) match DB column sizes (user_id: 255, device_id/agent_id: 100).

### Key Data Models (`opencontext/models/context.py`)

- `ProcessedContext` — the universal intermediate format all processors produce
- `ContextProperties` — includes `hierarchy_level`, `parent_id`, `children_ids`, `time_bucket`, `source_file_key`, `enable_merge`
- `ProfileData` — relational DB model for user profiles (PK: `user_id + device_id + agent_id`)
- `EntityData` — relational DB model for entities (unique key: `user_id + device_id + agent_id + entity_name`)
- `ProcessedContextModel` — API response model, must mirror all fields from `ContextProperties` that should be exposed

### LLM Integration (`opencontext/llm/`)

Three global singletons accessed via `get_instance()`:
- `GlobalEmbeddingClient` — text embeddings (Volcengine Doubao default)
- `GlobalVLMClient` — vision/image understanding
- `LLMClient` — text generation

All use OpenAI-compatible API format.

### Prompts (`config/prompts_en.yaml`, `config/prompts_zh.yaml`)

LLM prompts for extraction, classification, merging, and hierarchy summarization. Loaded by `PromptManager`. When modifying the type system, all prompts referencing context types must be updated in both language files.

### Scheduling (`opencontext/periodic_task/`, `opencontext/scheduler/`)

`RedisTaskScheduler` (async, stateless, Redis-backed) supports two trigger modes:

| Mode | Trigger | Use Case |
|------|---------|----------|
| `user_activity` | Scheduled per-user on data push, with configurable delay (`interval`) and dedup (`last_exec` check) | Tasks that should only run for active users |
| `periodic` | Global timer, runs at fixed intervals regardless of user activity | System-wide maintenance tasks |

Registered tasks:
- `MemoryCompressionTask` (`user_activity`) — deduplicates similar contexts (knowledge type only)
- `DataCleanupTask` (`periodic`) — retention-based cleanup
- `HierarchySummaryTask` (`user_activity`) — generates L1/L2/L3 event summaries, scheduled from push endpoints via `_schedule_user_hierarchy_summary()` in `push.py`

Push endpoints that schedule hierarchy summary: `push_chat_message`, `process_chat_messages`, `push_activity`, `push_context`.

## Configuration

- Main config: `config/config.yaml` — YAML with `${ENV_VAR:default}` syntax
- Environment variables: copy `.env.example` to `.env`
- Key env vars: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `MYSQL_HOST`, `MYSQL_PASSWORD`, `REDIS_HOST`, `HIERARCHY_SUMMARY_ENABLED`
- Service mode (`service_mode.enabled: true`): stateless deployment requiring external Redis + MySQL/vector DB

## Code Style

- Python 3.10+, Black (line length 100), isort (profile "black")
- Package name in imports: `opencontext`
- Logging: `loguru` via `from opencontext.utils.logging_utils import get_logger`
- Pre-commit hooks auto-format on commit

## Pitfalls and Lessons Learned

These are real bugs encountered during development. Check for them when modifying related code.

### All profile/entity operations require the 3-key identifier `(user_id, device_id, agent_id)`
Every storage method for profiles and entities (`upsert_profile`, `get_profile`, `get_entity`, `search_entities`, `list_entities`, etc.) requires all three identifiers. `device_id` and `agent_id` default to `"default"` for backward compatibility. When adding new code that calls these methods, always pass all three — omitting `device_id` or `agent_id` causes positional argument mismatches (e.g., `entity_name` gets interpreted as `device_id`). The same applies to `ProfileEntityTool`, `ProfileRetrievalTool`, memory cache manager, and search strategies. Redis cache keys also use all three: `memory_cache:snapshot:{user_id}:{device_id}:{agent_id}`.

### Storage singleton confusion
`get_storage()` returns `UnifiedStorage` (has profile/entity/hierarchy methods). `GlobalStorage.get_instance()` and `get_global_storage()` return `GlobalStorage` (lacks those methods). Always use `get_storage()` from `opencontext.storage.global_storage`.

### Qdrant Range filter only supports numeric/datetime
Qdrant's `models.Range(gte=..., lte=...)` does NOT work on string fields like `time_bucket`. Use in-code string comparison filtering instead (over-fetch with `top_k * 3`, then filter). See `qdrant_backend.py` `search_by_hierarchy()` for the pattern.

### ProcessedContextModel must declare all exposed fields
If you add a field to `ContextProperties` and want it in API responses, you must also declare it on `ProcessedContextModel` and update `from_processed_context()`. Missing declarations cause silent Pydantic field drops.

### MySQL lastrowid is 0 for VARCHAR primary keys
`cursor.lastrowid` only returns meaningful values for auto-increment integer PKs. For UUID/VARCHAR PKs (like the entities table), always SELECT back the persisted ID instead of relying on lastrowid.

### Use timezone-aware datetime everywhere
`datetime.utcfromtimestamp()` is deprecated in Python 3.12+. Use `datetime.fromtimestamp(ts, tz=datetime.timezone.utc)` instead. The same applies to `datetime.utcnow()` — use `datetime.now(tz=datetime.timezone.utc)`.

### Context merger only handles knowledge type
The merger (`context_merger.py`) now only processes `knowledge` contexts. Profile/entity use relational DB overwrite, document uses delete+insert, event is immutable append. Do not route non-knowledge types through the merger.

### Prompt files must stay in sync
`config/prompts_en.yaml` and `config/prompts_zh.yaml` must have identical keys. When adding/removing context types or changing classification logic, update both files and all `ContextDescriptions` in `enums.py`.

### EntityData.relationships stored in metadata JSON
The `relationships` field on `EntityData` is not a separate DB column — it's serialized inside the `metadata` JSON column. Keep this in mind when querying relationships.

### Database connections are not thread-safe — use `_get_connection()` everywhere
Both MySQL and SQLite backends use `threading.local()` for per-thread connections. `_get_connection()` is the only safe way to access connections from method-level code. In SQLite, `self.connection` is only for `initialize()` (setup) and `close()` (teardown) — all other methods must call `self._get_connection()` for cursor/commit/rollback. This was a major bug: 102 call sites originally used `self.connection` directly, causing thread-safety issues under `asyncio.to_thread()`.

### VikingDB hierarchy_level filter requires range format
VikingDB stores `hierarchy_level` as float32. Equality checks like `"hierarchy_level": 0` don't work. Use range format: `{"$gte": 0, "$lte": 0}`. This applies to all numeric filters in VikingDB.

### Context type routing must match CONTEXT_STORAGE_BACKENDS
`_handle_processed_context()` in `opencontext.py` routes profile/entity to MySQL and others to VikingDB. If you bypass this (e.g., calling `batch_upsert_processed_context()` for all types), profile/entity data will be stored in VikingDB instead of MySQL, making it unretrievable by the profile/entity search paths.

### Use `run_coroutine_threadsafe` not `create_task` from sync blocking code
When sync code runs inside the event loop thread (e.g., `_handle_processed_context` via `asyncio.to_thread`), `loop.create_task()` schedules the coroutine but never reliably wakes the event loop to execute it. Use `asyncio.run_coroutine_threadsafe(coro, loop)` instead — it calls `loop.call_soon_threadsafe()` internally, which writes to the event loop's self-pipe to wake it. The `_capture_loop()` pattern stores the loop reference from async entry points so sync callers can access it later.

### Scheduler `periodic` mode passes `user_id=None` — not for per-user tasks
`RedisTaskScheduler._process_periodic_tasks()` calls handlers with `(None, None, None)`. Any task whose `validate_context()` requires a non-null `user_id` will silently fail. Use `user_activity` trigger mode for per-user tasks; `periodic` is only for global system tasks (like `DataCleanupTask`). This was the root cause of `HierarchySummaryTask` never executing — it was configured as `periodic` but required `user_id`.

### Scheduler task type must be registered (enabled) or `schedule_user_task` silently fails
`schedule_user_task()` calls `get_task_config()` which looks up the task type in Redis. If the task's `enabled` flag is `false` in config, `_collect_task_types()` skips registration, and `get_task_config()` returns `None`, causing `schedule_user_task()` to log a warning and return `False`. The push endpoint continues normally — no error is raised. Always set `HIERARCHY_SUMMARY_ENABLED=true` in production to enable the task.

### MySQL InnoDB composite key length limit
InnoDB with utf8mb4 has a max key length of 3072 bytes. Each `VARCHAR(N)` uses `N*4` bytes in a key. Composite unique keys must keep total VARCHAR length under 768 chars. Current column sizes: `user_id VARCHAR(255)`, `device_id VARCHAR(100)`, `agent_id VARCHAR(100)`, `entity_name VARCHAR(255)` — total 710, just under the limit.

### Hierarchy summary weekly/monthly generation is idempotent, not day-gated
`execute()` always tries to generate summaries for the most recent completed week and month (not just on Mondays/1st of month). The existing dedup check (`search_hierarchy` for existing summaries) prevents regeneration. This design allows `user_activity` triggered tasks to generate weekly/monthly summaries whenever the user next becomes active, rather than requiring execution on a specific day.

## Extending the System

- **New context type**: Add to `ContextType` enum, `UpdateStrategy`, both mapping dicts, `ContextDescriptions`, `ContextSimpleDescriptions`, and update all prompt files
- **New processor**: Extend `BaseContextProcessor`, implement `can_process()` and `process()`, register in `ProcessorFactory`
- **New storage backend**: Implement `IVectorStorageBackend` or `IDocumentStorageBackend` from `base_storage.py`, register in factory
- **New retrieval tool**: Extend `BaseTool`, register in `tool_definitions.py` and `tools_executor.py`
