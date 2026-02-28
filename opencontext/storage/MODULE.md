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
| `get_all_processed_contexts` | `(context_types, limit, offset, filter, need_vector, user_id, device_id, agent_id) -> Dict[str, List[ProcessedContext]]` | Type-keyed dict |
| `get_processed_context` | `(id: str, context_type: str) -> ProcessedContext` | Single context |
| `delete_processed_context` | `(id: str, context_type: str) -> bool` | Success flag |
| `search` | `(query: Vectorize, top_k, context_types, filters, user_id, device_id, agent_id) -> List[Tuple[ProcessedContext, float]]` | Scored results |
| `get_processed_context_count` | `(context_type: str) -> int` | Count |
| `get_all_processed_context_counts` | `() -> Dict[str, int]` | Type-keyed counts |
| `delete_by_source_file` | `(source_file_key: str, user_id: Optional[str]) -> bool` | Success flag |
| `search_by_hierarchy` | `(context_type, hierarchy_level, time_bucket_start, time_bucket_end, user_id, top_k) -> List[Tuple[ProcessedContext, float]]` | Scored results |
| `get_by_ids` | `(ids: List[str], context_type: Optional[str]) -> List[ProcessedContext]` | Contexts by ID |
| `upsert_todo_embedding` | `(todo_id: int, content: str, embedding: List[float], metadata) -> bool` | Success flag |
| `search_similar_todos` | `(query_embedding: List[float], top_k, similarity_threshold) -> List[Tuple[int, str, float]]` | (todo_id, content, score) |
| `delete_todo_embedding` | `(todo_id: int) -> bool` | Success flag |

Non-abstract method with default implementation (backends may override):

| Method | Signature | Yields |
|--------|-----------|--------|
| `scroll_processed_contexts` | `(context_types, batch_size=100, filter, need_vector, user_id, device_id, agent_id) -> Generator[ProcessedContext, None, None]` | `ProcessedContext` objects one at a time |

**`scroll_processed_contexts`**: Generator that iterates all matching contexts. The default implementation uses offset-based `get_all_processed_contexts` calls (O(n^2) for backends like ChromaDB). `QdrantBackend` overrides this with native cursor-based scrolling via `_client.scroll(offset=next_page_offset)` for O(n) performance. Used by `ContextMerger._cleanup_contexts_by_type()` for data cleanup iteration.

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
| `upsert_profile` | `(user_id, device_id, agent_id, content, summary, keywords, entities, importance, metadata)` | `bool` |
| `get_profile` | `(user_id, device_id, agent_id)` | `Optional[Dict]` |
| `delete_profile` | `(user_id, device_id, agent_id)` | `bool` |
| `upsert_entity` | `(user_id, device_id, agent_id, entity_name, content, entity_type, summary, keywords, aliases, metadata)` | `str` (entity ID) |
| `get_entity` | `(user_id, device_id, agent_id, entity_name)` | `Optional[Dict]` |
| `list_entities` | `(user_id, device_id, agent_id, entity_type, limit, offset)` | `List[Dict]` |
| `search_entities` | `(user_id, device_id, agent_id, query_text, limit)` | `List[Dict]` |
| `delete_entity` | `(user_id, device_id, agent_id, entity_name)` | `bool` |

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

**Routing logic**: All methods check `_initialized` and the relevant backend, then delegate:
- Vector operations (contexts, search, hierarchy, todo embeddings) -> `_vector_backend`
- Document operations (vaults, todos, tips, profiles, entities, conversations, messages, monitoring) -> `_document_backend`

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

Pipeline support:
- `pipeline(transaction=False)` -- async context manager yielding a `_PrefixedPipeline` (auto-prefixes keys). Supports `hset()`, `expire()`, `zadd()`, `execute()`. Used by `RedisTaskScheduler.schedule_user_task()` to batch writes.

### InMemoryCache -- `redis_cache.py`

Fallback when Redis is unavailable. Same async interface as `RedisCache` but stores in local dicts. Does NOT support multi-instance sharing. Uses `_expiry: Dict[str, datetime]` for TTL tracking.

Module-level functions:
- `get_redis_cache(config=None) -> RedisCache` -- get/create global singleton
- `init_redis_cache(config) -> RedisCache` -- force-(re)create global singleton
- `close_redis_cache()` -- async, closes global instance
- `get_cache(config=None) -> Union[RedisCache, InMemoryCache]` -- async, returns Redis if connected, else InMemoryCache

### Vector Backend Differences

| Feature | ChromaDB | Qdrant | VikingDB | DashVector |
|---------|----------|--------|----------|------------|
| Collection strategy | Per context_type | Per context_type | Single collection, field filtering | Per context_type |
| ID handling | String IDs directly | UUID5 from string ID | String IDs via HTTP API | String IDs via HTTP API |
| Connection | Local or HTTP client | `QdrantClient` | HTTP with V4 signature auth | HTTP with API key |
| `search_by_hierarchy` time_bucket | In-code string filter | In-code string filter (Range doesn't work on strings) | Range filter with float `hierarchy_level` | In-code string filter |
| Write buffering | `_pending_writes` with flush | None | Batch HTTP requests | None |
| Graceful shutdown | `atexit` + signal handlers | None | None | None |

### Document Backend Differences

| Feature | SQLiteBackend | MySQLBackend |
|---------|--------------|--------------|
| Connection model | `threading.local()` per-thread connections | SQLAlchemy `QueuePool` (pool_size=20, max_overflow=10) |
| `_get_connection()` | Returns `sqlite3.Connection` (creates lazily per thread) | `@contextmanager` yielding pooled connection, auto-rollback on error |
| Journal mode | WAL | InnoDB (default) |
| Schema migration | `_create_tables()` + `_migrate_schema_v2()` | Same |
| Health check | `SELECT 1` on connection | `ping(reconnect=True)` on pool checkout |

## Internal Data Flow

```
Caller code
    |
    v
get_storage()                          # global_storage.py -> UnifiedStorage
    |
    +-- Profile/Entity operations ------> _document_backend.upsert_profile() / upsert_entity()
    |                                         |
    |                                         v
    |                                     SQLiteBackend or MySQLBackend
    |                                     (profiles / entities tables)
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
    |                                     Filter by hierarchy_level + time_bucket range
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
2. Implement all abstract methods (profile/entity CRUD, vaults, todos, tips)
3. Also implement non-abstract methods used by `UnifiedStorage`: conversations, messages, message thinking, monitoring
4. Register in `StorageBackendFactory.__init__()` under `StorageType.DOCUMENT_DB`
5. Follow same pattern as vector backend for imports and config

## Conventions and Constraints

- **Always use `get_storage()`** from `global_storage.py` to access storage. Never use `GlobalStorage.get_instance()` or `get_global_storage()` directly -- they return the wrapper which lacks profile/entity/hierarchy methods.
- **All profile/entity calls require the 3-key tuple** `(user_id, device_id, agent_id)`. Omitting `device_id` or `agent_id` causes positional argument mismatches.
- **SQLite `_get_connection()` must be used for all method-level DB access.** Only `initialize()` and `close()` may use `self.connection` directly. This prevents thread-safety issues under `asyncio.to_thread()`.
- **MySQL `_get_connection()` is a context manager** -- always use `with self._get_connection() as conn:`. It auto-returns to pool and auto-rolls-back on exceptions.
- **Qdrant `search_by_hierarchy`** uses in-code string comparison for `time_bucket` filtering because `models.Range` does not support string fields. Over-fetch with `top_k * 3`, then filter.
- **VikingDB `hierarchy_level`** is stored as float32. Use range format `{"$gte": N, "$lte": N}` instead of equality checks.
- **MySQL `lastrowid` is 0 for VARCHAR PKs** (entities table). Always SELECT back the persisted ID.
- **Vector backends auto-vectorize** via `do_vectorize()` when `context.vectorize.vector` is None. This is a synchronous call to the embedding service.
- **Todo collection creation** is gated by `consumption.enabled` config flag. If disabled, todo-related methods on vector backends are no-ops.
