# storage/ -- Dual-backend storage layer with vector and relational database support

## File Overview

| File | Responsibility |
|------|---------------|
| `base_storage.py` | Interface definitions: `IStorageBackend`, `IVectorStorageBackend`, `IDocumentStorageBackend`, plus shared data classes |
| `unified_storage.py` | `UnifiedStorage` facade and `StorageBackendFactory` -- routes operations to the correct backend |
| `global_storage.py` | `GlobalStorage` singleton wrapper; provides `get_storage()` accessor |
| `redis_cache.py` | `RedisCache` (async), `InMemoryCache` (fallback), distributed lock support |
| `__init__.py` | Re-exports Redis cache classes and functions |
| `backends/__init__.py` | Imports backend implementations: `ChromaDBBackend` and `SQLiteBackend` unconditionally; `MySQLBackend`, `QdrantBackend`, `VikingDBBackend` conditionally (try/except). `DashVectorBackend` is NOT imported. |
| `backends/chromadb_backend.py` | `ChromaDBBackend` -- ChromaDB vector storage (per-type collections) |
| `backends/qdrant_backend.py` | `QdrantBackend` -- Qdrant vector storage (per-type collections) |
| `backends/vikingdb_backend.py` | `VikingDBBackend` -- Volcengine VikingDB (single collection, field filtering) |
| `backends/dashvector_backend.py` | `DashVectorBackend` -- Aliyun DashVector (HTTP API, per-type collections) |
| `backends/sqlite_backend.py` | `SQLiteBackend` -- SQLite relational storage (profiles, entities, vaults, todos, monitoring) |
| `backends/mysql_backend.py` | `MySQLBackend` -- MySQL relational storage (same schema as SQLite, connection pooled) |
| `object_storage/` | Object storage sub-module for multimodal media files (see `object_storage/MODULE.md`) |

## Class Hierarchy

```
IStorageBackend (ABC)                        # base_storage.py
|
+-- IVectorStorageBackend (ABC)              # base_storage.py
|   +-- ChromaDBBackend                      # backends/chromadb_backend.py
|   +-- QdrantBackend                        # backends/qdrant_backend.py
|   +-- VikingDBBackend                      # backends/vikingdb_backend.py
|   +-- DashVectorBackend                    # backends/dashvector_backend.py
|
+-- IDocumentStorageBackend (ABC)            # base_storage.py
    +-- SQLiteBackend                        # backends/sqlite_backend.py
    +-- MySQLBackend                         # backends/mysql_backend.py
```

## Key Classes and Functions

### IStorageBackend (ABC) -- `base_storage.py`

Base interface for all storage backends.

| Abstract Method | Signature | Description |
|----------------|-----------|-------------|
| `initialize` | `(config: Dict[str, Any]) -> bool` | Initialize backend with config dict |
| `get_name` | `() -> str` | Return backend name string |
| `get_storage_type` | `() -> StorageType` | Return `StorageType.VECTOR_DB` or `StorageType.DOCUMENT_DB` |

### IVectorStorageBackend (ABC) -- `base_storage.py`

Extends `IStorageBackend`. All abstract methods that new vector backends must implement:

| Abstract Method | Signature | Returns |
|----------------|-----------|---------|
| `get_collection_names` | `() -> Optional[List[str]]` | All collection names |
| `delete_contexts` | `(ids: List[str], context_type: str) -> bool` | Success flag |
| `upsert_processed_context` | `(context: ProcessedContext) -> str` | Stored ID |
| `batch_upsert_processed_context` | `(contexts: List[ProcessedContext]) -> List[str]` | List of stored IDs |
| `get_all_processed_contexts` | `(context_types, limit, offset, filter, need_vector, user_id, device_id, agent_id, skip_slice) -> Dict[str, List[ProcessedContext]]` | Type-keyed dict. `skip_slice=True` skips per-type offset/limit slicing for correct cross-type pagination |
| `get_processed_context` | `(id: str, context_type: str) -> ProcessedContext` | Single context |
| `delete_processed_context` | `(id: str, context_type: str) -> bool` | Success flag |
| `search` | `(query: Vectorize, top_k, context_types, filters, user_id, device_id, agent_id, score_threshold) -> List[Tuple[ProcessedContext, float]]` | Scored results. `score_threshold` (0-1) filters out low-similarity results at the database level when supported |
| `get_processed_context_count` | `(context_type, filter, user_id, device_id, agent_id) -> int` | Count (all params except context_type are optional) |
| `get_filtered_context_count` | `(context_types, filter, user_id, device_id, agent_id) -> int` | Total count across multiple types with filters (UnifiedStorage only) |
| `get_all_processed_context_counts` | `() -> Dict[str, int]` | Type-keyed counts |
| `search_by_hierarchy` | `(context_type, hierarchy_level, time_start, time_end, user_id, device_id, agent_id, top_k) -> List[Tuple[ProcessedContext, float]]` | Scored results. `time_start`/`time_end` are UTC timestamps (floats); returns contexts whose `[event_time_start, event_time_end]` overlaps the query range |
| `get_by_ids` | `(ids: List[str], context_type: Optional[str], need_vector: bool) -> List[ProcessedContext]` | Contexts by ID |

Non-abstract methods with default implementation (backends may override):

| Method | Signature | Returns/Yields |
|--------|-----------|----------------|
| `scroll_processed_contexts` | `(context_types, batch_size=100, filter, need_vector, user_id, device_id, agent_id) -> Generator[ProcessedContext, None, None]` | `ProcessedContext` objects one at a time |
| `batch_update_refs` | `(context_ids: List[str], ref_key: str, ref_value: str, context_type: str) -> int` | Count of updated contexts |

**`scroll_processed_contexts`**: Generator that iterates all matching contexts. The default implementation uses offset-based `get_all_processed_contexts` calls (O(n^2) for backends like ChromaDB). `QdrantBackend` overrides this with native cursor-based scrolling via `_client.scroll(offset=next_page_offset)` for O(n) performance. Used by `ContextMerger._cleanup_contexts_by_type()` for data cleanup iteration.

**`batch_update_refs`** (replaces former `batch_set_parent_id`): Adds a `ref_key -> ref_value` entry to the `refs` dict of multiple contexts. For example, after storing a daily summary, calls `batch_update_refs(child_event_ids, ref_key="daily_summary", ref_value=summary_id, context_type="event")` to backfill upward refs on children. `QdrantBackend` overrides with native `set_payload` API (payload-only update, no vector traffic). `VikingDBBackend` overrides with `/api/vikingdb/data/update` API (batch limit 100). The default implementation raises `NotImplementedError`.

### IDocumentStorageBackend (ABC) -- `base_storage.py`

Extends `IStorageBackend`. All abstract methods that new document backends must implement:

| Abstract Method | Key Params | Returns |
|----------------|------------|---------|
| `insert_vaults` | `(title, summary, content, document_type, ...)` | `int` (vault ID) |
| `get_vaults` | `(limit, offset, is_deleted, document_type, created_after, ...)` | `List[Dict]` |
| `get_vault` | `(vault_id: int)` | `Optional[Dict]` |
| `update_vault` | `(vault_id: int, **kwargs)` | `bool` |
| `get_reports` | `(limit, offset, is_deleted)` | `List[Dict]` |
| `insert_todo` | `(content, start_time, end_time, status, urgency, assignee, reason)` | `int` |
| `get_todos` | `(status, limit, offset, start_time, end_time)` | `List[Dict]` |
| `update_todo_status` | `(todo_id: int, status: int, end_time)` | `bool` |
| `insert_tip` | `(content: str)` | `int` |
| `get_tips` | `(limit, offset)` | `List[Dict]` |
| `upsert_profile` | `(user_id, device_id, agent_id, factual_profile, behavioral_profile, entities, importance, metadata, refs, context_type="profile")` | `bool` |
| `get_profile` | `(user_id, device_id, agent_id, context_type="profile")` | `Optional[Dict]` |
| `delete_profile` | `(user_id, device_id, agent_id, context_type="profile")` | `bool` |
Note: Document backends also implement non-abstract methods for conversations, messages, message thinking, and monitoring (not listed in the interface but present in both SQLite and MySQL implementations).

### StorageBackendFactory -- `unified_storage.py`

Creates backend instances based on config. Backend selection is driven by `storage_type` + `backend` fields in config:

```
StorageType.VECTOR_DB:
    "chromadb"  -> ChromaDBBackend()
    "qdrant"    -> QdrantBackend()
    "vikingdb"  -> VikingDBBackend()
    # Note: DashVectorBackend exists in backends/ but is NOT registered in the factory

StorageType.DOCUMENT_DB:
    "mysql"     -> MySQLBackend()
    "sqlite"    -> SQLiteBackend()
```

If `backend` is `"default"`, selects the first registered backend for that storage type. After construction, calls `backend.initialize(config)` and returns `None` on failure.

### UnifiedStorage -- `unified_storage.py`

Facade that holds one `IVectorStorageBackend` and one `IDocumentStorageBackend`. Key fields:

- `_factory: StorageBackendFactory`
- `_vector_backend: IVectorStorageBackend`
- `_document_backend: IDocumentStorageBackend`
- `_initialized: bool`

`initialize()` reads config from `get_config("storage")`, iterates `backends` list, creates each via factory, and assigns to `_vector_backend` / `_document_backend` (preferring configs with `default: true`).

**Routing logic**: Most delegating methods use the `@_require_backend(backend_attr, default)` decorator (module-level) to check `_initialized` and backend availability, returning `default` on failure. Exceptions that keep manual guards: `scroll_processed_contexts` (async generator), `delete_conversation` (parameter-dependent default), `delete_document` (positive check pattern), `get_vector_collection_names` (no `_initialized` check).
- Vector operations (contexts, search, hierarchy) -> `_vector_backend`
- Document operations (vaults, todos, tips, profiles, entities, conversations, messages, monitoring) -> `_document_backend`

**Settings delegation**: Settings methods (`load_all_settings`, `save_setting`, `delete_all_settings`) follow the monitoring methods pattern — implemented on `MySQLBackend`, `SQLiteBackend`, and `UnifiedStorage`, not on `IDocumentStorageBackend`.

### GlobalStorage -- `global_storage.py`

Thread-safe singleton (double-checked locking) wrapping `UnifiedStorage`.

| Method | Returns | Description |
|--------|---------|-------------|
| `get_instance()` | `GlobalStorage` | Class method; auto-initializes on first call |
| `get_storage()` | `Optional[UnifiedStorage]` | Returns the wrapped UnifiedStorage |
| `is_initialized()` | `bool` | Whether storage is ready |
| `reset()` | `None` | Resets singleton (for testing) |

**Convenience methods** (delegate to UnifiedStorage, raise `RuntimeError` if not initialized):
- `upsert_processed_context(context) -> bool`
- `batch_upsert_processed_context(contexts) -> bool`
- `get_processed_context(doc_id, context_type) -> Optional[ProcessedContext]`
- `delete_processed_context(doc_id, context_type) -> bool`

Module-level convenience functions:
- `get_storage() -> Optional[UnifiedStorage]` -- **the recommended accessor** (returns UnifiedStorage directly)
- `get_global_storage() -> GlobalStorage` -- returns the wrapper (lacks profile/entity methods)

### RedisCache -- `redis_cache.py`

Async-only Redis client. Configured via `RedisCacheConfig` dataclass:

| Field | Type | Default |
|-------|------|---------|
| `host` | `str` | `"localhost"` |
| `port` | `int` | `6379` |
| `password` | `Optional[str]` | `None` |
| `db` | `int` | `0` |
| `key_prefix` | `str` | `"opencontext:"` |
| `default_ttl` | `int` | `3600` |
| `max_connections` | `int` | `50` |
| `socket_timeout` | `float` | `5.0` |
| `socket_connect_timeout` | `float` | `5.0` |
| `retry_on_timeout` | `bool` | `True` |
| `decode_responses` | `bool` | `True` |

**Connection pool**: Uses `redis.asyncio.BlockingConnectionPool` — when the pool is full, callers queue and wait (up to 5s timeout) instead of getting an immediate `ConnectionError`. This provides back-pressure under high concurrency. The `_ensure_async_client()` method uses `async with self._async_lock` with double-checked locking to prevent race conditions on first connection.

Operation groups: basic KV, JSON KV, lists (with JSON variants), hashes (with JSON variants), sets, sorted sets, atomic incr/decr, distributed locks, Lua scripts, pipelines.

Key lock methods:
- `acquire_lock(lock_name, timeout=10, blocking=True, blocking_timeout=5.0) -> Optional[str]` -- returns token
- `release_lock(lock_name, token) -> bool` -- only releases if token matches

Lua script methods:
- `rpush_expire_llen(key, value, ttl) -> int` -- atomic RPUSH + EXPIRE + LLEN in a single round-trip. Returns new list length. Used by `TextChatCapture.push_message()` to reduce 3 Redis calls to 1.

Pub/Sub methods:
- `publish(channel, message) -> int` -- publish to a Redis channel (auto-prefixed). Returns number of subscribers that received the message. Returns 0 if not connected.
- `create_pubsub() -> Optional[PubSub]` -- create a raw `redis.asyncio` PubSub instance. Caller manages lifecycle (subscribe, get_message loop, close). Channel names must be constructed with `config.key_prefix` manually. Returns None if not connected.

Pipeline support:
- `pipeline(transaction=False)` -- async context manager yielding a `_PrefixedPipeline` (auto-prefixes keys). Supports `hset()`, `expire()`, `zadd()`, `execute()`. Used by `RedisTaskScheduler.schedule_user_task()` to batch writes.

### InMemoryCache -- `redis_cache.py`

Fallback when Redis is unavailable. Same async interface as `RedisCache` but stores in local dicts. Does NOT support multi-instance sharing. Uses `_expiry: Dict[str, datetime]` for TTL tracking.

Module-level functions:
- `get_redis_cache(config=None) -> RedisCache` -- get/create global singleton
- `init_redis_cache(config) -> RedisCache` -- force-(re)create global singleton
- `close_redis_cache()` -- async, closes global instance
- `get_cache(config=None) -> Union[RedisCache, InMemoryCache]` -- async, returns Redis if connected, else InMemoryCache

### VikingDB V2 API and Multimodal Fields

**API version**: VikingDB backend uses V2 API format. Key differences from V1:

- **Control plane** (collection/index management): PascalCase parameters (e.g., `CollectionName`, `Fields`, `IndexName`, `ProjectName`)
- **Data plane** (upsert/search/delete/fetch): snake_case parameters, but with V2 naming:
  - `fields` → `data` (in upsert and update requests)
  - `primary_keys` → `ids` (in delete and fetch requests)
  - Response format: `id` is separated from `fields` (not duplicated inside `fields`)

**Multimodal collection fields** (stored as scalar fields in VikingDB):

| Field | Type | Description |
|-------|------|-------------|
| `content_modalities` | `string` | Comma-separated modality list, e.g. `"text"`, `"text,image"`, `"text,image,video"` |
| `media_refs` | `string` | JSON-serialized array of media references: `[{"type": "image", "url": "...", "local_path": "..."}, ...]` |

These fields are populated from `ProcessedContext.metadata["content_modalities"]` and `ProcessedContext.metadata["media_refs"]` during upsert, and parsed back during retrieval in `_doc_to_processed_context()`.

### Vector Backend Differences

| Feature | ChromaDB | Qdrant | VikingDB | DashVector |
|---------|----------|--------|----------|------------|
| Collection strategy | Per context_type | Per context_type | Single collection, field filtering | Per context_type |
| ID handling | String IDs directly | UUID5 from string ID | String IDs via HTTP API | String IDs via HTTP API |
| Connection | Local or HTTP client | `QdrantClient` | HTTP with V4 signature auth | HTTP with API key |
| `search_by_hierarchy` time range | In-code numeric range filter | Numeric `models.Range` filter on `event_time_start_ts`/`event_time_end_ts` | `must` filter with int64 `hierarchy_level` + numeric range | In-code numeric range filter |
| Write buffering | `_pending_writes` with flush | None | Batch HTTP requests | None |
| Graceful shutdown | `atexit` + signal handlers | None | None | None |

### Document Backend Differences

| Feature | SQLiteBackend | MySQLBackend |
|---------|--------------|--------------|
| Connection model | `threading.local()` per-thread connections | SQLAlchemy `QueuePool` (pool_size=20, max_overflow=10) |
| `_get_connection()` | Returns `sqlite3.Connection` (creates lazily per thread) | `@contextmanager` yielding pooled connection, auto-rollback on error, auto-commit on normal exit (prevents stale MVCC snapshots across pool reuses) |
| Journal mode | WAL | InnoDB (default) |
| Schema migration | `_create_tables()` | Same |
| Health check | `SELECT 1` on connection | `ping(reconnect=True)` on pool checkout |

### Settings Storage (MySQL and SQLite)

Both `MySQLBackend` and `SQLiteBackend` provide settings CRUD methods for DB-backed user settings management (not part of the `IDocumentStorageBackend` interface). Settings are stored in the `system_settings` table.

**Additional table:**

| Table | Purpose | Key |
|-------|---------|-----|
| `system_settings` | Multi-instance user settings (key-value with JSON values) | `setting_key VARCHAR(128) PK` |

**Settings methods (both backends):**

| Method | Description |
|--------|-------------|
| `load_all_settings()` | Load all settings rows as `{key: value}` dict (skips `_`-prefixed sentinel rows) |
| `save_setting(key, value)` | Atomic upsert. MySQL uses `JSON_MERGE_PATCH`; SQLite uses Python-side `deep_merge` |
| `delete_all_settings()` | Clear all settings (preserves sentinel rows) |

### Agent Registry (MySQL and SQLite)

Both backends implement agent CRUD for the `agent_registry` table. Methods are defined on `IDocumentStorageBackend` (with `raise NotImplementedError` defaults) and delegated through `UnifiedStorage`.

**Table DDL** (`agent_registry`):
| Column | Type | Notes |
|--------|------|-------|
| `agent_id` | `VARCHAR(100) PK` | Application-provided ID |
| `name` | `VARCHAR(255) NOT NULL` | Display name |
| `description` | `TEXT` | Optional description |
| `is_deleted` | `BOOLEAN DEFAULT FALSE` | Soft delete flag |
| `created_at` | `DATETIME` | Auto-set |
| `updated_at` | `DATETIME` | Auto-updated |

**Methods** (all async, on `UnifiedStorage` / `IDocumentStorageBackend`):

| Method | Signature | Returns |
|--------|-----------|---------|
| `create_agent` | `(agent_id: str, name: str, description: str = "")` | `bool` |
| `get_agent` | `(agent_id: str)` | `Optional[Dict]` (excludes soft-deleted) |
| `list_agents` | `()` | `List[Dict]` (active only, ordered by `created_at DESC`) |
| `update_agent` | `(agent_id: str, name: Optional[str], description: Optional[str])` | `bool` |
| `delete_agent` | `(agent_id: str)` | `bool` (soft delete: sets `is_deleted = TRUE`) |

### Chat Batches (MySQL and SQLite)

Persists raw chat messages before processing, enabling processors to reference the original input. Methods defined on `IDocumentStorageBackend` and delegated through `UnifiedStorage`.

**Table DDL** (`chat_batches`):
| Column | Type | Notes |
|--------|------|-------|
| `batch_id` | `VARCHAR(36) PK` | App-generated UUID |
| `messages` | `JSON NOT NULL` | Raw messages array |
| `user_id` | `VARCHAR(255)` | |
| `device_id` | `VARCHAR(100) DEFAULT 'default'` | |
| `agent_id` | `VARCHAR(100) DEFAULT 'default'` | |
| `message_count` | `INT NOT NULL` | |
| `created_at` | `DATETIME` | Indexed for cleanup |

**Methods** (all async):

| Method | Signature | Returns |
|--------|-----------|---------|
| `create_chat_batch` | `(batch_id: str, messages: List[Dict], user_id: Optional[str], device_id: str, agent_id: str)` | `bool` |
| `list_chat_batches` | `(user_id?, device_id?, agent_id?, start_date?, end_date?, limit=20, offset=0)` | `List[Dict]` — List batches without messages, with optional filters, ordered by created_at DESC |
| `count_chat_batches` | `(user_id?, device_id?, agent_id?, start_date?, end_date?)` | `int` — Count matching batches |
| `get_chat_batch` | `(batch_id)` | `Optional[Dict]` — Get single batch with messages (JSON parsed) |
| `cleanup_chat_batches` | `(retention_days: int = 90)` | `int` (rows deleted) |

## Internal Data Flow

```
Caller code
    |
    v
get_storage()                          # global_storage.py -> UnifiedStorage
    |
    +-- Profile operations ------> _document_backend.upsert_profile()
    |                                         |
    |                                         v
    |                                     SQLiteBackend or MySQLBackend
    |                                     (profiles table)
    |
    +-- Vector operations (contexts) ---> _vector_backend.batch_upsert_processed_context()
    |                                         |
    |                                         v
    |                                     ChromaDB / Qdrant / VikingDB / DashVector
    |                                     (per-type collections or single collection)
    |
    +-- Search --------------------------> _vector_backend.search(query: Vectorize, ...)
    |                                     Returns List[Tuple[ProcessedContext, float]]
    |
    +-- Hierarchy search ----------------> _vector_backend.search_by_hierarchy(...)
    |                                     Filter by hierarchy_level + timestamp range overlap
    |
    +-- Document overwrite --------------> _vector_backend.delete_by_source_file(...)
                                          Then batch_upsert new chunks
```

## Cross-Module Dependencies

**Imports from other modules:**
- `opencontext.models.context` -- `ProcessedContext`, `Vectorize`, `ContextProperties`, `ExtractedData`
- `opencontext.models.enums` -- `ContextType`, `ContentFormat`
- `opencontext.config.global_config` -- `get_config()`, `GlobalConfig`
- `opencontext.llm.global_embedding_client` -- `do_vectorize()` (called by vector backends when context lacks a vector)
- `opencontext.utils.logging_utils` -- `get_logger()`

**Depended on by:**
- `opencontext.server.opencontext` -- `OpenContext` orchestrator uses `get_storage()` for all write operations
- `opencontext.server.search/` -- Fast and intelligent strategies use `get_storage()` for search
- `opencontext.server.cache/` -- Memory cache manager uses `get_storage()` for snapshot building
- `opencontext.tools/` -- Retrieval tools use `get_storage()` for tool execution
- `opencontext.context_processing/` -- Processors may read existing contexts via storage
- `opencontext.periodic_task/` -- Compression, cleanup, hierarchy tasks read/write via storage
- `opencontext.server/push.py` -- Push endpoints use `get_storage()` and `get_redis_cache()`

## Extension Points

### Adding a new vector backend

1. Create `backends/new_backend.py` implementing `IVectorStorageBackend`
2. Implement all 16 abstract methods from the interface (see table above)
3. Register in `StorageBackendFactory.__init__()` under `StorageType.VECTOR_DB`:
   ```python
   "newbackend": self._create_newbackend,
   ```
4. Add lazy import factory method `_create_newbackend(self, config)` in the factory
5. Add optional import in `backends/__init__.py`
6. Configure in `config/config.yaml` under `storage.backends` with `backend: "newbackend"`

### Adding a new document backend

1. Create `backends/new_backend.py` implementing `IDocumentStorageBackend`
2. Implement all abstract methods (profile CRUD, vaults, todos, tips)
3. Also implement non-abstract methods used by `UnifiedStorage`: conversations, messages, message thinking, monitoring
4. Register in `StorageBackendFactory.__init__()` under `StorageType.DOCUMENT_DB`
5. Follow same pattern as vector backend for imports and config

## Conventions and Constraints

- **Always use `get_storage()`** from `global_storage.py` to access storage. Never use `GlobalStorage.get_instance()` or `get_global_storage()` directly -- they return the wrapper which lacks profile/hierarchy methods.
- **All profile calls require the 3-key tuple + context_type** `(user_id, device_id, agent_id, context_type)`. `context_type` defaults to `"profile"`; pass `"agent_profile"` for agent profiles. The `profiles` table PK is 4 columns: `(user_id, device_id, agent_id, context_type)`. A `refs JSON` column stores reference links (e.g., source context IDs). Migration from the old `owner_type` column happens at startup: rows with `owner_type = 'agent'` are updated to `context_type = 'agent_profile'`. Both MySQL and SQLite backends include `context_type` in all profile WHERE clauses.
- **SQLite `_get_connection()` must be used for all method-level DB access.** Only `initialize()` and `close()` may use `self.connection` directly. This prevents thread-safety issues under `asyncio.to_thread()`.
- **MySQL `_get_connection()` is a context manager** -- always use `with self._get_connection() as conn:`. It auto-returns to pool and auto-rolls-back on exceptions.
- **Qdrant `search_by_hierarchy`** now uses numeric `models.Range` filters on `event_time_start_ts`/`event_time_end_ts` for timestamp range overlap queries. The old in-code string comparison workaround on `time_bucket` is no longer used.
- **VikingDB `hierarchy_level`** is stored as int64. Use `must` filter directly (e.g., `{"op": "must", "field": "hierarchy_level", "conds": [0]}`). Supports list filtering: `"conds": [0, 1, 2]`.
- **MySQL `lastrowid` is 0 for VARCHAR PKs**. Always SELECT back the persisted ID.
- **Vector backends auto-vectorize** via `do_vectorize()` when `context.vectorize.vector` is None. This is a synchronous call to the embedding service.
- **Todo collection creation** is gated by `consumption.enabled` config flag. If disabled, todo-related methods on vector backends are no-ops.
