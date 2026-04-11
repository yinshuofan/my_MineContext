# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## First Principles Thinking

Think from first principles. Do not assume the user fully understands what they want or how best to achieve it. When receiving a request:

1. **Examine motivation and goals**: If the motivation or end goal is unclear, stop and discuss with the user before executing literally.
2. **Start from the root problem**: Go back to the original problem and think about what truly needs to be solved, rather than reflexively applying familiar patterns.
3. **Challenge the path**: If the goal is clear but the proposed path is not the shortest or most effective, proactively tell the user and suggest a better approach.
4. **Don't follow instructions blindly**: The user's instruction says WHAT to do, not necessarily the best HOW. Before executing, evaluate whether a simpler, more fundamental solution exists.

## Project Overview

MineContext is a **memory backend service** that captures, processes, stores, and retrieves context/memory from multiple sources (chat logs, documents, web links). It uses vector embeddings and LLM-powered analysis to organize information into 4 typed memory contexts with type-specific update strategies and storage routing.

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

### Testing

The project uses **pytest** with `pytest-asyncio` (`asyncio_mode = "auto"`) and `pytest-cov`. Tests live in `tests/`, mirroring the `opencontext/` package structure (e.g. `opencontext/models/enums.py` → `tests/models/test_enums.py`).

Two pytest markers are declared in `pyproject.toml` and enforced via `--strict-markers`:

- `@pytest.mark.unit` — no external dependencies (no Redis, no DB, no LLM)
- `@pytest.mark.integration` — requires real Redis / MySQL / SQLite

Common commands:

```bash
# Run the full suite
uv run pytest

# Unit tests only (fast, no infra needed)
uv run pytest -m unit

# Integration tests only
uv run pytest -m integration

# A single file / class / test
uv run pytest tests/models/test_enums.py
uv run pytest tests/models/test_enums.py::TestContextType
uv run pytest tests/models/test_enums.py::TestContextType::test_profile_uses_overwrite_strategy

# With coverage
uv run pytest --cov=opencontext --cov-report=term-missing
```

**Writing tests**:
- Place tests under `tests/<module>/test_<name>.py` matching the source layout.
- Every test class or function must carry a marker (`unit` or `integration`); strict-markers will fail otherwise.
- Prefer `unit` tests with fakes/stubs for Redis/storage where possible — they must run without any running services.
- For `async` code, just write `async def test_...` — `asyncio_mode = "auto"` handles the rest.

Any non-trivial code change should add or update tests. If a change genuinely cannot be unit-tested (e.g. pure wiring in `cli.py` lifespan), state that explicitly in the PR / commit message rather than silently skipping.

## Planning Workflow

**Mandatory rule**: Before planning any implementation, you MUST first use Explore subagents or agent teams to investigate the relevant codebase. Do NOT plan based solely on memory, CLAUDE.md descriptions, or assumptions — always explore the actual code first.

### Agent scale by task complexity

| Task Scope | Agent Strategy | Example |
|------------|---------------|---------|
| Simple (single-file change) | 1 Explore subagent — target file + direct dependencies | Fix a bug in one processor |
| Medium (cross-module change) | 2–3 Explore subagents in parallel — each covering a different module | Add a new API endpoint touching routes, storage, and models |
| Complex (architectural change) | Agent team with role-based exploration (architecture analysis, dependency tracing, pattern discovery, etc.) | Redesign the storage layer or add a new context type |

The specific number of agents, their query focus, and team composition should be determined by the actual task — the table above is guidance, not a rigid rule.

### Architecture-first design

When proposing code changes, **evaluate whether the solution fits the project's architecture** — not just whether it works. The simplest modification is not always the best one. Consider:

- **Layer boundaries**: Does the change respect the separation between generic infrastructure (e.g., `RedisCache` as thin wrapper) and domain logic (e.g., scheduler's task lifecycle)? Don't leak domain concepts into generic layers, and don't duplicate infrastructure in domain code.
- **Multi-instance deployment**: This service may run as multiple instances behind a load balancer. Any code touching shared state (Redis, MySQL) must be safe under concurrent access. Think about: race conditions, lock contention, atomic operations, and whether "read-then-write" sequences need to be made atomic (e.g., via Lua scripts or database transactions).
- **Existing patterns**: Before introducing a new approach, check how similar problems are already solved in the codebase. Follow established conventions unless there's a clear reason to deviate.

### What to explore

- Source files directly related to the task
- Existing patterns, conventions, and utilities that can be reused
- Dependency chains (callers and callees of the code being modified)
- Relevant `MODULE.md` files for implementation-level context

## Architecture

### 4-Type Context System

All data flows through a unified `ProcessedContext` model, but each type has its own update strategy and storage destination. This is the central design decision — understand it before modifying any storage, processing, or retrieval code.

| Type | Update Strategy | Storage | Description |
|------|----------------|---------|-------------|
| `profile` | OVERWRITE | Relational DB | User's own preferences, habits, communication style. Key: `(user_id, device_id, agent_id, context_type)` |
| `agent_profile` | OVERWRITE | Relational DB | Agent's perception and knowledge about a specific user. Same table as `profile`, distinguished by `context_type="agent_profile"` |
| `document` | OVERWRITE | Vector DB | Uploaded files and web links. Overwrite = delete old chunks + insert new. Tracked by `raw_type` + `raw_id` |
| `event` | APPEND | Vector DB | Immutable activity records, meetings, status changes. Never modified after creation |
| `knowledge` | APPEND_MERGE | Vector DB | Reusable concepts, procedures, patterns. Similar entries merged to avoid duplication |

Defined in `opencontext/models/enums.py`: `ContextType`, `UpdateStrategy`, `CONTEXT_UPDATE_STRATEGIES`, `CONTEXT_STORAGE_BACKENDS`, `ContextDescriptions`.

### Data Pipeline

```
Input → Processor → ProcessedContext → _handle_processed_context (routes by type) → Storage
```

1. **Process** (`opencontext/context_processing/`): `ProcessorFactory` routes by `ContextSource`:
   - `LOCAL_FILE`, `VAULT`, `WEB_LINK` → `DocumentProcessor`
   - `CHAT_LOG`, `INPUT` → `TextChatProcessor`

2. **Route** (`opencontext/server/opencontext.py` → `_handle_processed_context()`): The central routing point. Routes by context type:
   - profile / agent_profile / agent_base_profile → `_store_profile()` → `storage.upsert_profile()` → relational DB
   - document/event/knowledge → `storage.batch_upsert_processed_context()` → vector DB

3. **Store** (`opencontext/storage/`): `UnifiedStorage` wraps dual backends:
   - **Document DB** (MySQL or SQLite) — profiles table, plus context metadata
   - **Vector DB** (VikingDB, ChromaDB, or Qdrant) — document/event/knowledge embeddings

### Hierarchical Event Indexing

Events support a 4-level time-based hierarchy: **L0** (raw events) → **L1** (daily summaries) → **L2** (weekly) → **L3** (monthly). Summaries are generated per-user via `user_activity` scheduler trigger when data is pushed. See `periodic_task/MODULE.md` for generation details and `tools/MODULE.md` for retrieval algorithm.

### Key Components Overview

| Component | Location | Details |
|-----------|----------|---------|
| Retrieval Tools (3 type-aligned) | `opencontext/tools/` | `tools/MODULE.md` |
| Search API (event search with hierarchy drill-up) | `opencontext/server/search/` | `server/MODULE.md` |
| Memory Cache (user memory snapshot) | `opencontext/server/cache/` | `server/MODULE.md` |
| Scheduling (user_activity / periodic) | `opencontext/scheduler/` | `scheduler/MODULE.md` |
| Data Models | `opencontext/models/context.py` | `models/MODULE.md` |
| LLM Clients (embedding, VLM, text) | `opencontext/llm/` | `llm/MODULE.md` |
| Agent Registry | `opencontext/storage/` | `agent_registry` table with soft delete |
| Chat Batches | `opencontext/storage/`, `opencontext/server/routes/` | Persists raw chat messages; debug page traces batch → produced contexts |
| Agent Memory Processor | `opencontext/context_processing/` | Extracts agent-perspective memories from conversations |

### Storage Access

**Always use `get_storage()` from `opencontext/storage/global_storage.py`** to get the `UnifiedStorage` instance. Do NOT use `GlobalStorage.get_instance()` or `get_global_storage()` — these return the raw wrapper which lacks profile/hierarchy methods.

### Server Layer

- **Entry point**: `opencontext/cli.py` → `opencontext start` → FastAPI on Uvicorn, port 1733
- **Orchestrator**: `opencontext/server/opencontext.py` → `OpenContext`

### Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with component status |
| `/api/push/chat` | POST | Push chat messages with processor selection (`processors: ["user_memory", "agent_memory"]`) |
| `/api/push/document` | POST | Push document (file_path or base64) |
| `/api/agents` | POST/GET/PUT/DELETE | Agent CRUD (register, list, get, update, soft-delete) |
| `/api/agents/{id}/base/*` | POST/GET/DELETE | Agent base memory (profile + events) |
| `/api/search` | POST | Event search with semantic query, filters, hierarchy drill-up |
| `/api/memory-cache` | GET | User memory snapshot |
| `/api/chat-batches` | GET | List/get chat batches and their related vector contexts (debug) |

Request body uses OpenAI message format. The 3-key identifier `(user_id, device_id, agent_id)` is optional on all push endpoints, defaulting to `"default"`.

### Prompts (`config/prompts_en.yaml`, `config/prompts_zh.yaml`)

LLM prompts for extraction, classification, merging, and hierarchy summarization. Loaded by `PromptManager`. When modifying the type system, all prompts referencing context types must be updated in both language files.

> The above is a high-level architectural overview. For implementation details, method signatures, and extension patterns, see the corresponding `MODULE.md` in each module directory.

## Configuration

- Main config: `config/config.yaml` — YAML with `${ENV_VAR:default}` syntax
- Environment variables: copy `.env.example` to `.env`
- Key env vars: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `MYSQL_HOST`, `MYSQL_PASSWORD`, `REDIS_HOST`, `HIERARCHY_SUMMARY_ENABLED`
- `OPENCONTEXT_CONFIG_PATH`: Set by `cli.py` when `--config` is provided; used by `GlobalConfig._auto_initialize()` in multi-worker subprocess auto-init
- User settings: stored in `system_settings` table (MySQL or SQLite document backend), not in `user_setting.yaml`

## Code Style

- Python 3.10+, Black (line length 100), isort (profile "black")
- Package name in imports: `opencontext`
- Logging: `loguru` via `from opencontext.utils.logging_utils import get_logger`
- Pre-commit hooks auto-format on commit

## Pitfalls and Lessons Learned

These are real bugs encountered during development. Check for them when modifying related code.

### Multi-instance deployment — all shared-state operations must be concurrency-safe
This service runs as multiple instances sharing the same Redis and MySQL. Any "read-then-write" sequence on shared state is a potential race condition. Use atomic operations: Lua scripts for Redis (see `_CONDITIONAL_ZPOPMIN_LUA` in `redis_scheduler.py` for the pattern, executed via `RedisCache.eval_lua()`), database transactions for MySQL. When adding new Redis operations that check-then-modify, ask: "What happens if two instances run this at the same time?" If the answer is "data corruption or duplicate work", make it atomic.

### All profile operations require the 4-key identifier `(user_id, device_id, agent_id, context_type)`
Every storage method for profiles requires all four identifiers. `device_id` and `agent_id` default to `"default"`, `context_type` defaults to `"profile"`. Pass `context_type="agent_profile"` for agent profiles. Omitting them causes positional argument mismatches. The same applies to tools, cache manager, and search strategies. Redis cache keys also use the 3-key tuple: `memory_cache:snapshot:{user_id}:{device_id}:{agent_id}`.

### Qdrant Range filter only supports numeric/datetime
**Resolved**: The old workaround (in-code string comparison on `time_bucket`, over-fetching with `top_k * 3`) is no longer used. `search_by_hierarchy` now uses numeric `models.Range` filters on `event_time_start_ts`/`event_time_end_ts` float fields, which Qdrant handles natively.

### ProcessedContextModel must declare all exposed fields
If you add a field to `ContextProperties` and want it in API responses, you must also declare it on `ProcessedContextModel` and update `from_processed_context()`. Missing declarations cause silent Pydantic field drops.

### MySQL pitfalls
- **lastrowid is 0 for VARCHAR primary keys**: Always SELECT back the persisted ID for UUID/VARCHAR PKs.
- **InnoDB composite key length limit**: utf8mb4 max key length is 3072 bytes (`VARCHAR(N)` uses `N*4`). Current composite unique key total: 710 chars, just under the 768-char limit.
- **Connection pool stale MVCC snapshots**: With `autocommit=False` and `REPEATABLE READ`, pooled connections may carry uncommitted transactions from previous uses. Reads on such connections see an old MVCC snapshot, missing writes committed by other workers. `_get_connection()` commits on normal exit (else-branch) to ensure connections return to the pool with no lingering transaction. Do not remove this commit — it prevents stale reads in multi-instance deployments.

### Never cache `get_storage()` in `__init__` or sync init code
`GlobalStorage` requires async `ensure_initialized()` (runs in lifespan). Any `self.storage = get_storage()` in `__init__` or synchronous `initialize()` caches `None` permanently. Use a non-caching `@property` instead: `return get_storage()`. This is O(1) (singleton lookup) and matches the pattern in `BaseContextRetrievalTool`, `ContextMerger`, `OpenContext`, `ContextOperations`, etc. The same rule applies to all global singletons with async init.

### Use `time_utils.now()` instead of `datetime.now()`
All datetime generation must go through `opencontext.utils.time_utils`. Import as `from opencontext.utils.time_utils import now as tz_now`. This ensures all datetimes use the globally-configured timezone from `config.yaml` (`timezone` key, IANA name like `"Asia/Shanghai"`). Never use `datetime.now()`, `datetime.now(tz=timezone.utc)`, or `datetime.utcnow()` directly. For protocol-mandated UTC (e.g. AWS S3 auth), use `utc_now()`. **Important:** Do not call `now()` at module-import time — the timezone is initialized at startup, after imports.

### Context merger only handles knowledge type
The merger (`context_merger.py`) only processes `knowledge` contexts. Do not route non-knowledge types through the merger.

### Prompt files must stay in sync
`config/prompts_en.yaml` and `config/prompts_zh.yaml` must have identical keys. Update both files and `ContextDescriptions` in `enums.py` when changing classification logic.

### Database connections are not thread-safe — use `_get_connection()` everywhere
Both MySQL and SQLite backends use `threading.local()` for per-thread connections. `_get_connection()` is the only safe way to access connections from method-level code. In SQLite, `self.connection` is only for `initialize()` and `close()`.

### VikingDB hierarchy_level is int64, use must filter
VikingDB stores `hierarchy_level` as `int64`. Use `must` filter directly: `{"op": "must", "field": "hierarchy_level", "conds": [0]}`. Do NOT use `range` format — it's unnecessary for int64 fields. `must` also supports list filtering: `"conds": [0, 1, 2]`.

### Context type routing must match CONTEXT_STORAGE_BACKENDS
`_handle_processed_context()` routes `PROFILE`, `AGENT_PROFILE`, and `AGENT_BASE_PROFILE` to relational DB via `_store_profile()`, and others to vector DB. Bypassing this (e.g., calling `batch_upsert_processed_context()` for all types) makes profile data unretrievable.

### Use `run_coroutine_threadsafe` not `create_task` from sync blocking code
When sync code runs via `asyncio.to_thread`, `loop.create_task()` doesn't reliably wake the event loop. Use `asyncio.run_coroutine_threadsafe(coro, loop)` instead. The `_capture_loop()` pattern stores the loop reference from async entry points.

### Scheduler pitfalls
- **`periodic` mode passes `user_id=None`**: Any task requiring non-null `user_id` will silently fail. Use `user_activity` trigger mode for per-user tasks; `periodic` is only for global system tasks.
- **Disabled tasks cause silent failures**: `schedule_user_task()` silently returns `False` if the task type isn't registered (enabled). The push endpoint continues normally — no error is raised.

### Scheduler task disable requires Redis `enabled` flag, not just config skip
Disabling a task type in config only prevents NEW registration and handler creation. Existing Redis queue entries and other multi-instance workers remain unaffected. The Redis `scheduler:task_type:{type}.enabled` field is the **runtime authority** — `init_task_types()` marks disabled tasks as `enabled: "false"`, and both `_type_worker()` and `_process_periodic_tasks()` check this flag each cycle. If you add a new execution path for scheduled tasks, it must also check this flag.

### Hierarchy summary generation is idempotent, not day-gated
`execute()` always tries to generate summaries for the most recent completed period. The existing dedup check (`search_hierarchy`) prevents regeneration. This allows generation whenever the user next becomes active.

## Extending the System

- **New context type**: Add to `ContextType` enum, `UpdateStrategy`, both mapping dicts, `ContextDescriptions`, `ContextSimpleDescriptions`, and update all prompt files
- **New processor**: Extend `BaseContextProcessor`, implement `can_process()` and `process()`, register in `ProcessorFactory`
- **New storage backend**: Implement `IVectorStorageBackend` or `IDocumentStorageBackend` from `base_storage.py`, register in factory
- **New retrieval tool**: Extend `BaseTool`, register in `tool_definitions.py` and `tools_executor.py`

## Module Documentation

Each core module has a `MODULE.md` file providing implementation-level documentation for AI assistants. These files document class responsibilities, method signatures, data flow, and extension patterns.

**Modules with MODULE.md:**
- `opencontext/models/` -- Core data models and enums
- `opencontext/config/` -- Configuration loading and prompt management
- `opencontext/llm/` -- LLM client singletons
- `opencontext/storage/` -- Dual-backend storage layer
- `opencontext/context_capture/` -- Input source capture components
- `opencontext/context_processing/` -- Processing pipeline (processors, chunkers, merger)
- `opencontext/tools/` -- Retrieval tool framework and implementations
- `opencontext/server/` -- FastAPI server, routes, search strategies, cache
- `opencontext/scheduler/` -- Task scheduling (Redis-backed)
- `opencontext/periodic_task/` -- Periodic task implementations

**Maintenance rule**: When modifying a module's internal structure (adding/removing classes, changing method signatures, altering data flow), update the corresponding `MODULE.md` to keep it accurate.

## API Documentation

- `docs/api_reference.md` — Core API reference (push/chat, push/document, search, memory-cache) with request/response schemas and examples.
- `docs/curls.sh` — All API endpoints as cURL commands, organized by category. Used for Apifox import and as a quick API reference.
- **Maintenance rule**: When adding, modifying, or removing API endpoints, update `docs/curls.sh` and `docs/api_reference.md` to keep them in sync.
