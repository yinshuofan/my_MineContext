# opencontext/server/ -- HTTP Server, Routes, Search Strategies, Cache, and Middleware

FastAPI-based HTTP server layer: request routing, search strategy dispatch, per-user memory cache, and middleware (auth, request tracing).

## File Overview

| File | Responsibility |
|------|---------------|
| `opencontext.py` | `OpenContext` orchestrator -- integrates capture, processing, storage; routes processed contexts by type |
| `api.py` | Main `APIRouter` that aggregates all route sub-routers |
| `component_initializer.py` | `ComponentInitializer` -- initializes capture components, processors, merger, task scheduler |
| `context_operations.py` | `ContextOperations` -- CRUD and vector search operations delegated from `OpenContext` |
| `stream_interrupt.py` | `StreamInterruptManager` singleton -- cross-worker stream interrupt via Redis Pub/Sub, falls back to local dict |
| `utils.py` | `get_context_lab(request) -> OpenContext` dependency, `convert_resp()` standard JSON helper |
| `__init__.py` | Package marker (empty) |
| **routes/** | |
| `routes/push.py` | Push API endpoints (`/api/push/*`) -- unified chat, documents, contexts |
| `routes/search.py` | Unified search endpoint (`POST /api/search`) with strategy selection |
| `routes/memory_cache.py` | Memory cache endpoints (`GET/DELETE /api/memory-cache`) |
| `routes/health.py` | Health and readiness probes (`/health`, `/api/health`, `/api/ready`, `/api/auth/status`) |
| `routes/context.py` | Context CRUD and vector search (`/api/contexts/*`, `/api/context_types`, `/api/vector_search`) |
| `routes/documents.py` | Document/weblink upload (`/api/documents/upload`, `/api/weblinks/upload`) |
| `routes/agent_chat.py` | Agent chat interface (`/api/agent/chat`, `/api/agent/chat/stream`, workflow resume/state/cancel) |
| `routes/conversation.py` | Conversation CRUD (`/api/agent/chat/conversations/*`) |
| `routes/messages.py` | Message CRUD and streaming (`/api/agent/chat/message/*`) |
| `routes/monitoring.py` | Monitoring endpoints (`/api/monitoring/*`) -- overview, context-types, token-usage, processing, etc. |
| `routes/settings.py` | Model settings, general settings, prompts CRUD (`/api/model_settings/*`, `/api/settings/*`) |
| `routes/vaults.py` | Vault document management (`/api/vaults/*`) with background context processing |
| `routes/web.py` | HTML pages -- contexts list, vector search, chat, monitoring, settings, file serving |
| `routes/completions.py` | Intelligent completion suggestions (`/api/completions/*`) -- **NOT registered in `api.py`; routes are inactive/dead code** |
| **search/** | |
| `search/base_strategy.py` | `BaseSearchStrategy` ABC with `search()` abstract method |
| `search/fast_strategy.py` | `FastSearchStrategy` -- zero LLM calls, parallel storage queries |
| `search/intelligent_strategy.py` | `IntelligentSearchStrategy` -- LLM-driven agentic tool selection loop |
| `search/models.py` | Pydantic models: `UnifiedSearchRequest`, `TypedResults`, `VectorResult`, `ProfileResult`, `EntityResult` |
| **cache/** | |
| `cache/memory_cache_manager.py` | `UserMemoryCacheManager` singleton -- builds/caches per-user memory snapshots in Redis |
| `cache/models.py` | Response models: `UserMemoryCacheResponse`, `SimpleProfile`, `SimpleDailySummary`, `SimpleTodayEvent`, `RecentlyAccessedItem`; internal models: `RecentMemoryItem`, `DailySummaryItem` |
| **middleware/** | |
| `middleware/auth.py` | API key authentication via `X-API-Key` header or `api_key` query param |
| `middleware/request_id.py` | `RequestIDMiddleware` -- assigns 8-char UUID per request, stored in `ContextVar` |

## Key Classes and Functions

### OpenContext (opencontext.py)

Central orchestrator. Created in `cli.py` lifespan, stored on `app.state.context_lab_instance`.

```python
class OpenContext:
    def __init__(self)
    def initialize(self) -> None                    # Init all components in order
    def shutdown(self, graceful: bool = True) -> None

    # Context processing pipeline
    def _handle_captured_context(self, contexts: List[RawContextProperties]) -> bool
    def _handle_processed_context(self, contexts: List[ProcessedContext]) -> bool  # Routes by CONTEXT_STORAGE_BACKENDS
    def _store_profile(self, ctx: ProcessedContext) -> None     # -> storage.upsert_profile()
    def _store_entities(self, ctx: ProcessedContext) -> None    # -> storage.upsert_entity() per entity
    def _invalidate_user_cache(self, user_id, device_id, agent_id) -> None  # Fire-and-forget

    # Delegated operations
    def add_context(self, context_data: RawContextProperties) -> bool
    def add_document(self, file_path: str) -> Optional[str]     # Returns error msg or None
    def search(self, query, top_k, context_types, filters, user_id, device_id, agent_id) -> List[Dict]
    def get_all_contexts(self, limit, offset, filter_criteria) -> Dict[str, List[ProcessedContext]]
    def get_context(self, doc_id: str, context_type: str) -> Optional[ProcessedContext]
    def update_context(self, doc_id: str, context: ProcessedContext) -> bool
    def delete_context(self, doc_id: str, context_type: str) -> bool
    async def check_components_health(self) -> Dict[str, Any]  # Checks config, storage, llm, document_db, redis

    # Additional public methods
    def start_capture(self) -> None                  # Starts all capture components via capture_manager.start_all_components()
    def get_context_types(self) -> List[str]          # Delegates to context_operations.get_context_types()

    # Private helpers
    def _initialize_monitoring(self) -> None          # Initializes monitoring system; called from initialize()
    def _invalidate_cache_sync_fallback(self, user_id, device_id, agent_id) -> None  # Sync Redis DELETE fallback
```

Module-level: `main()` function -- entry point for `if __name__ == "__main__"`, parses args and runs uvicorn.

Key fields: `capture_manager` (ContextCaptureManager), `processor_manager` (ContextProcessorManager), `storage` (UnifiedStorage), `context_operations` (ContextOperations), `component_initializer` (ComponentInitializer).

### ComponentInitializer (component_initializer.py)

```python
class ComponentInitializer:
    def initialize_capture_components(self, capture_manager: ContextCaptureManager) -> None
    def initialize_processors(self, processor_manager: ContextProcessorManager, processed_context_callback) -> None
    def initialize_task_scheduler(self, processor_manager: Optional[ContextProcessorManager] = None) -> None
    async def start_task_scheduler(self) -> None   # Called after event loop is running
    def stop_task_scheduler(self) -> None

    # Private helpers
    def _to_camel_case(self, name: str) -> str             # Converts snake_case to CamelCase
    def _create_capture_component(self, name, config)      # Creates capture component from CAPTURE_COMPONENTS dict or dynamic import
```

### ContextOperations (context_operations.py)

Delegates to `UnifiedStorage` (obtained via `get_storage()`).

```python
class ContextOperations:
    def get_all_contexts(self, limit, offset, filter_criteria, need_vector) -> Dict[str, List[ProcessedContext]]
    def get_context(self, doc_id: str, context_type: str) -> Optional[ProcessedContext]
    def update_context(self, doc_id: str, context: ProcessedContext) -> bool
    def delete_context(self, doc_id: str, context_type: str) -> bool
    def add_document(self, file_path: str, context_processor_callback) -> Optional[str]
    def search(self, query, top_k, context_types, filters, user_id, device_id, agent_id) -> List[Dict]
    def get_context_types(self) -> List[str]
```

### FastSearchStrategy (search/fast_strategy.py)

Zero LLM calls. Single embedding generation shared across all parallel storage queries.

```python
class FastSearchStrategy(BaseSearchStrategy):
    async def search(self, query, context_types, top_k, time_range, user_id, device_id, agent_id) -> TypedResults

    # Internal
    def _build_time_filters(self, time_range: Optional[TimeRange]) -> Dict[str, Any]
    async def _attach_parent_summaries(self, event_results: List[Tuple[ProcessedContext, float]]) -> List[VectorResult]  # Batch-fetch parent summaries for L0 events via parent_id
    @staticmethod _to_vector_result(ctx: ProcessedContext, score: float) -> VectorResult
    @staticmethod _to_profile_result(data: Dict) -> ProfileResult
    @staticmethod _to_entity_result(data: Dict) -> EntityResult
```

Algorithm:
1. Generate embedding once via `do_vectorize()`
2. Build time filters from `TimeRange`
3. Dispatch parallel `asyncio.to_thread()` calls: profile lookup, entity search, document/event/knowledge vector search
4. Events filtered to L0 only (`hierarchy_level: {"$gte": 0, "$lte": 0}`), with parent summaries batch-attached via `_attach_parent_summaries()`: collects unique `parent_id` values from L0 events, batch-fetches parent contexts via `storage.get_contexts_by_ids()`, and populates the `parent_summary` field on each `VectorResult`. Requires `parent_id` to be backfilled by `HierarchySummaryTask._store_summary()`
5. Assemble `TypedResults`

### IntelligentSearchStrategy (search/intelligent_strategy.py)

LLM-driven agentic search. Reuses `LLMContextStrategy` for tool selection.

```python
class IntelligentSearchStrategy(BaseSearchStrategy):
    def __init__(self)  # Creates LLMContextStrategy instance
    async def search(self, query, context_types, top_k, time_range, user_id, device_id, agent_id) -> TypedResults

    # Internal
    async def _agentic_search_loop(self, query: str, user_id: Optional[str]) -> List[ContextItem]
    async def _direct_profile_entity_lookup(self, query, top_k, user_id, device_id, agent_id) -> tuple
    def _context_items_to_typed_results(self, items, context_types, top_k) -> TypedResults
    @staticmethod _item_to_vector_result(item: ContextItem, original: Dict) -> VectorResult
```

Algorithm:
1. Enhance query with time range info
2. Run agentic loop (MAX_ITERATIONS=1) in parallel with direct profile/entity lookup
3. Agentic loop: `analyze_and_plan_tools` -> `execute_tool_calls_parallel`, dedup by context ID (keeping higher score)
4. Convert `ContextItem` results to `TypedResults` using `_TOOL_TYPE_MAP`, with score threshold (≥0.3), per-type sort by score descending, and top_k truncation

### UserMemoryCacheManager (cache/memory_cache_manager.py)

Singleton via `get_memory_cache_manager()`. Manages per-user memory snapshots in Redis.

```python
class UserMemoryCacheManager:
    def __init__(self)
    async def track_accessed(self, user_id, items: List[Dict], device_id, agent_id) -> None
    async def invalidate_snapshot(self, user_id, device_id, agent_id) -> None
    async def get_user_memory_cache(self, user_id, device_id, agent_id, recent_days, max_recent_events_today, max_accessed, force_refresh) -> UserMemoryCacheResponse

    # Internal
    async def _get_recently_accessed(self, cache, user_id, max_items, device_id, agent_id) -> List[RecentlyAccessedItem]
    async def _build_snapshot(self, user_id, device_id, agent_id, recent_days, max_today_events) -> Dict[str, Any]
    def _merge_response(self, snapshot_data, accessed, cache_hit, ttl_remaining) -> UserMemoryCacheResponse
    async def _trim_accessed(self, cache, key: str, max_size: int) -> None
    @staticmethod _ctx_to_recent_item(ctx: ProcessedContext) -> Dict[str, Any]
```

Caching architecture:
- **Snapshot** (profile + today events + daily summaries): Redis JSON string, configurable TTL (default 300s). Key: `memory_cache:snapshot:{user_id}:{device_id}:{agent_id}`. Snapshot stores full internal data; response assembly in `_merge_response()` simplifies to `SimpleProfile` (content + keywords + metadata), `SimpleDailySummary` (time_bucket + summary), `SimpleTodayEvent` (title + summary + event_time).
- **Recently Accessed**: Redis Hash, 7-day TTL. Key: `memory_cache:accessed:{user_id}:{device_id}:{agent_id}`. Updated on every search (documents/events/knowledge only; profile/entity excluded), always read real-time.
- **Stampede prevention**: Distributed lock via `cache.acquire_lock()` + double-check pattern. If lock acquisition times out, tries cache once more then builds directly without caching.
- **Snapshot build**: 6 parallel `asyncio.to_thread()` queries (profile, entities, today events, daily summaries, recent docs, recent knowledge). Response only exposes profile, today_events, daily_summaries, recently_accessed.

### Auth Middleware (middleware/auth.py)

```python
def verify_api_key(request, api_key_header, api_key_query) -> str  # Returns key or raises 401
def is_auth_enabled() -> bool
def is_path_excluded(path: str) -> bool  # Wildcard matching via fnmatch
def reset_auth_cache() -> None           # Clears cached auth config (useful for testing)
def get_auth_config() -> dict            # Returns auth config dict from global config
def get_valid_api_keys() -> List[str]    # Returns list of valid API keys (filters out empty)
def get_excluded_paths() -> List[str]    # Returns list of excluded paths
auth_dependency = Depends(verify_api_key)  # Used as route dependency
```

Auth disabled by default. When enabled, checks `X-API-Key` header or `api_key` query param against `api_auth.api_keys` config list. Excluded paths bypass auth (default: `/health`, `/api/health`, `/`, `/static/*`).

### RequestIDMiddleware (middleware/request_id.py)

```python
request_id_var: ContextVar[str]  # Accessible from any code in the same request

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next) -> Response
```

Accepts `X-Request-ID` header or generates 8-char UUID. Sets `request_id_var` ContextVar and adds `X-Request-ID` to response.

### Utility Functions (utils.py)

```python
def get_context_lab(request: Request) -> OpenContext   # FastAPI dependency, reads app.state.context_lab_instance
def convert_resp(data=None, code=0, status=200, message="success") -> JSONResponse  # Standard response wrapper
```

## All Route Endpoints

### Push Routes (`/api/push/*`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/push/chat` | `push_chat` | Unified chat push (buffer or direct mode) |
| POST | `/api/push/document` | `push_document` | Push document (file_path or base64) |
| POST | `/api/push/document/upload` | `upload_document_file` | Upload document via multipart form |

Push endpoints that schedule hierarchy summary: `push_chat` (both modes).

### Search Routes (`/api/*`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/search` | `unified_search` | Unified search with fast/intelligent strategy |

### Memory Cache Routes (`/api/*`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/memory-cache` | `get_user_memory_cache` | Get user memory snapshot |
| DELETE | `/api/memory-cache` | `invalidate_user_memory_cache` | Invalidate user cache |

### Health Routes

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/health` | `health_check` | Simple health check |
| GET | `/api/health` | `api_health_check` | Detailed health with component status |
| GET | `/api/auth/status` | `auth_status` | Check if auth is enabled |
| GET | `/api/ready` | `readiness_check` | Readiness probe (all dependencies) |

### Context Routes

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/contexts/{context_id}` | `get_context_api` | Get single context by ID |
| GET | `/api/context_types` | `get_context_types` | List available context types |
| POST | `/api/vector_search` | `vector_search` | Direct vector DB search |
| POST | `/contexts/delete` | `delete_context` | Delete context |
| POST | `/contexts/detail` | `read_context_detail` | Context detail HTML page |

### Agent Chat Routes (`/api/agent/*`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/agent/chat` | `chat` | Non-streaming chat |
| POST | `/api/agent/chat/stream` | `chat_stream` | Streaming chat (SSE) |
| POST | `/api/agent/resume/{workflow_id}` | `resume_workflow` | Resume workflow |
| GET | `/api/agent/state/{workflow_id}` | `get_workflow_state` | Get workflow state |
| DELETE | `/api/agent/cancel/{workflow_id}` | `cancel_workflow` | Cancel workflow |
| GET | `/api/agent/test` | `test_agent` | Test agent health |

### Conversation Routes (`/api/agent/chat/*`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/agent/chat/conversations` | `create_conversation` | Create conversation |
| GET | `/api/agent/chat/conversations/list` | `get_conversation_list` | List conversations |
| GET | `/api/agent/chat/conversations/{cid}` | `get_conversation_detail` | Get conversation detail |
| PATCH | `/api/agent/chat/conversations/{cid}/update` | `update_conversation_title` | Update title |
| DELETE | `/api/agent/chat/conversations/{cid}/update` | `delete_conversation` | Soft delete |

### Message Routes (`/api/agent/chat/*`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/agent/chat/message/{mid}/create` | `create_message` | Create message |
| POST | `/api/agent/chat/message/stream/{mid}/create` | `create_streaming_message` | Create streaming placeholder |
| POST | `/api/agent/chat/message/{mid}/update` | `update_message` | Update message content |
| POST | `/api/agent/chat/message/{mid}/append` | `append_message` | Append to streaming message |
| POST | `/api/agent/chat/message/{mid}/finished` | `mark_message_finished_route` | Mark message complete |
| GET | `/api/agent/chat/conversations/{cid}/messages` | `get_conversation_messages` | List conversation messages |
| POST | `/api/agent/chat/messages/{mid}/interrupt` | `interrupt_message_generation` | Interrupt generation |

### Monitoring Routes (`/api/monitoring/*`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/monitoring/overview` | `get_system_overview` | System monitoring overview |
| GET | `/api/monitoring/context-types` | `get_context_type_stats` | Context type statistics |
| GET | `/api/monitoring/token-usage` | `get_token_usage_summary` | Token consumption |
| GET | `/api/monitoring/processing` | `get_processing_metrics` | Processor performance |
| GET | `/api/monitoring/stage-timing` | `get_stage_timing_metrics` | Stage timing metrics |
| GET | `/api/monitoring/data-stats` | `get_data_stats` | Data statistics |
| GET | `/api/monitoring/data-stats-trend` | `get_data_stats_trend` | Data stats time series |
| GET | `/api/monitoring/data-stats-range` | `get_data_stats_by_range` | Custom time range stats |
| POST | `/api/monitoring/refresh-context-stats` | `refresh_context_type_stats` | Refresh cache |
| GET | `/api/monitoring/health` | `monitoring_health` | Monitoring health |
| GET | `/api/monitoring/processing-errors` | `get_processing_errors` | Processing errors Top N |
| GET | `/api/monitoring/scheduler` | `get_scheduler_summary` | Scheduler execution summary (query param: `hours`, default 24) |
| GET | `/api/monitoring/scheduler/queues` | `get_scheduler_queue_depths` | Real-time queue depths for all task types from Redis `zcard` |
| GET | `/api/monitoring/scheduler/failures` | `get_scheduler_failures` | Scheduler failure rates and recent errors (query param: `hours`, default 1) for alerting |
| POST | `/api/monitoring/trigger-task` | `trigger_task` | Manually trigger periodic tasks. Params: `task_type` (required, e.g. `hierarchy_summary`), `user_id` (required), `device_id`/`agent_id` (default `"default"`), `level` (`auto`/`daily`/`weekly`/`monthly`, default `auto`), `target` (date/week/month string, required when level != auto) |

### Settings Routes (`/api/*`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/model_settings/get` | `get_model_settings` | Get all 3 model configs (llm, vlm_model, embedding_model) as flat objects |
| POST | `/api/model_settings/update` | `update_model_settings` | Partial update — validates then saves any non-null model section; reinitializes VLM/Embedding clients |
| POST | `/api/model_settings/validate` | `validate_llm_config` | Validate any combination of 3 models without saving |
| GET | `/api/settings/general` | `get_general_settings` | Get 7 config sections: capture, processing, logging, document_processing, scheduler, memory_cache, tools |
| POST | `/api/settings/general` | `update_general_settings` | Update any of the 7 general settings sections (partial) |
| GET | `/api/settings/prompts` | `get_prompts` | Get current prompts |
| POST | `/api/settings/prompts` | `update_prompts` | Update prompts |
| POST | `/api/settings/prompts/import` | `import_prompts` | Import prompts YAML |
| GET | `/api/settings/prompts/export` | `export_prompts` | Export prompts YAML |
| GET | `/api/settings/prompts/language` | `get_prompt_language` | Get prompt language |
| POST | `/api/settings/prompts/language` | `change_prompt_language` | Change language (zh/en) |
| POST | `/api/settings/reset` | `reset_settings` | Reset all to defaults |

### Vault Routes (`/api/vaults/*`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/vaults/list` | `get_documents_list` | List vault documents |
| POST | `/api/vaults/create` | `create_document` | Create vault document |
| GET | `/api/vaults/{id}` | `get_document` | Get document detail |
| POST | `/api/vaults/{id}` | `save_document` | Save/update document |
| DELETE | `/api/vaults/{id}` | `delete_document` | Soft delete document |
| GET | `/api/vaults/{id}/context` | `get_document_context_status` | Context processing status |

### Document/Weblink Routes

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/documents/upload` | `upload_document` | Upload document (local path) |
| POST | `/api/weblinks/upload` | `upload_weblink` | Submit web link for processing |

### Completion Routes (`/api/completions/*`) -- INACTIVE (not registered in `api.py`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/completions/suggest` | `get_completion_suggestions` | Get completion suggestions |
| POST | `/api/completions/suggest/stream` | `get_completion_suggestions_stream` | Stream suggestions (SSE) |
| POST | `/api/completions/feedback` | `submit_completion_feedback` | Submit feedback |
| GET | `/api/completions/stats` | `get_completion_stats` | Service statistics |
| GET | `/api/completions/cache/stats` | `get_cache_stats` | Cache statistics |
| POST | `/api/completions/cache/optimize` | `optimize_cache` | Optimize cache |
| POST | `/api/completions/precompute/{id}` | `precompute_document_context` | Precompute context |
| POST | `/api/completions/cache/clear` | `clear_completion_cache` | Clear cache |

### Web (HTML) Routes

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/` | `root` | Redirect to `/contexts` |
| GET | `/contexts` | `read_contexts` | Contexts list page |
| GET | `/vector_search` | `vector_search_page` | Vector search page |
| GET | `/memory_cache` | `memory_cache_page` | Memory cache page |
| GET | `/chat` | `chat_page` | Redirect to `/advanced_chat` |
| GET | `/advanced_chat` | `advanced_chat_page` | Redirect to `/vaults` |
| GET | `/vaults` | `vaults_workspace` | Agent chat page |
| GET | `/vaults/editor` | `note_editor_page` | Note editor page |
| GET | `/files/{path}` | `serve_file` | Static file serving (with security checks) |
| GET | `/monitoring` | `monitoring_page` | Monitoring dashboard |
| GET | `/assistant` | `assistant_page` | Intelligent assistant page |
| GET | `/settings` | `settings_page` | Settings page |

## Cross-Module Dependencies

### Imports FROM other modules

| Import | Source |
|--------|--------|
| `OpenContext`, routes, utils | `opencontext.config.global_config`, `opencontext.config.config_manager`, `opencontext.config.prompt_manager` |
| Processing pipeline | `opencontext.context_processing.processor.processor_factory`, `opencontext.context_processing.merger.context_merger` |
| Capture components | `opencontext.context_capture.vault_document_monitor`, `web_link_capture`, `text_chat` |
| Models | `opencontext.models.context` (ProcessedContext, RawContextProperties, Vectorize), `opencontext.models.enums` (ContextType, CONTEXT_STORAGE_BACKENDS, etc.) |
| Storage | `opencontext.storage.global_storage` (get_storage, GlobalStorage), `opencontext.storage.redis_cache` (get_redis_cache, get_cache) |
| LLM | `opencontext.llm.global_embedding_client` (do_vectorize, GlobalEmbeddingClient), `opencontext.llm.global_vlm_client`, `opencontext.llm.llm_client` |
| Scheduler | `opencontext.scheduler` (get_scheduler, init_scheduler) |
| Agent | `opencontext.context_consumption.context_agent` (ContextAgent), `context_agent.core.llm_context_strategy` (LLMContextStrategy) |
| Managers | `opencontext.managers.capture_manager`, `processor_manager` |
| Monitoring | `opencontext.monitoring` (get_monitor, initialize_monitor) |
| Completion | `opencontext.context_consumption.completion` (get_completion_service) |

### Modules that DEPEND on server

| Consumer | What it uses |
|----------|-------------|
| `opencontext/cli.py` | Imports `api.router`, creates FastAPI app, manages `OpenContext` lifecycle via lifespan |
| `opencontext/server/opencontext.py` (self) | Imports `cache.memory_cache_manager.get_memory_cache_manager` for cache invalidation |

## Internal Data Flow

### Request Lifecycle

```
HTTP Request
  -> RequestIDMiddleware (assigns X-Request-ID, sets ContextVar)
  -> CORSMiddleware (handled by FastAPI)
  -> auth_dependency (verify_api_key -- checks X-API-Key header)
  -> Route handler
     -> get_context_lab(request) dependency -> app.state.context_lab_instance -> OpenContext
     -> Business logic (search, push, cache, etc.)
  -> convert_resp() or direct JSONResponse
  -> Response with X-Request-ID header
```

### Push Flow (`POST /api/push/chat`)

Buffer mode (`process_mode="buffer"`):
```
push_chat()
  -> for msg in messages: text_chat.push_message()  # atomic Lua (rpush+expire+llen, 1 Redis call per msg)
  -> if flush_immediately: text_chat.flush_user_buffer()
  -> BackgroundTasks: _schedule_user_task("memory_compression")
  -> BackgroundTasks: _schedule_user_task("hierarchy_summary")
```

Direct mode (`process_mode="direct"`):
```
push_chat()
  -> BackgroundTasks: text_chat.process_messages_directly()
  -> BackgroundTasks: _schedule_user_task("memory_compression")
  -> BackgroundTasks: _schedule_user_task("hierarchy_summary")
```

Both modes use `BackgroundTasks` for scheduling — the scheduling runs after the response is sent, not inline.

When buffer flushes or direct processing runs:
```
TextChatCapture.process_messages_directly()
  -> processor_manager.process()
    -> TextChatProcessor.process()
      -> return List[ProcessedContext]
    -> Manager invokes callback with results:
      -> OpenContext._handle_processed_context()
         -> Routes by CONTEXT_STORAGE_BACKENDS:
            profile/entity -> _store_profile()/_store_entities() -> storage.upsert_profile()/upsert_entity()
            document/event/knowledge -> storage.batch_upsert_processed_context() -> vector DB
         -> _invalidate_user_cache() for affected users
```

Processors return `List[ProcessedContext]` to the manager, which centrally invokes the callback. Processors do not call storage or callbacks directly.

### Search Flow (`POST /api/search`)

```
unified_search()
  -> Validate context_types
  -> Select strategy (FastSearchStrategy or IntelligentSearchStrategy)
  -> strategy.search() with 30s timeout
  -> Fire-and-forget _track_accessed_safe() -> memory_cache_manager.track_accessed()
  -> Return UnifiedSearchResponse
```

Fast strategy internals:
```
FastSearchStrategy.search()
  -> do_vectorize(query)                           # 1 embedding call
  -> asyncio.gather(                               # All in parallel
       storage.get_profile(),                      # Relational DB
       storage.search_entities(),                  # Relational DB
       storage.search(vectorize, [document]),      # Vector DB
       storage.search(vectorize, [event], L0),     # Vector DB
       storage.search(vectorize, [knowledge]),     # Vector DB
     )
  -> _attach_parent_summaries() for events         # Batch fetch parents
  -> Assemble TypedResults
```

### Memory Cache Flow (`GET /api/memory-cache`)

```
get_user_memory_cache()
  -> _get_recently_accessed()          # Always real-time from Redis Hash
  -> Check cached snapshot (Redis JSON)
     HIT  -> _merge_response(snapshot, accessed)
     MISS -> acquire_lock()
          -> Double-check snapshot
          -> _build_snapshot()          # 6 parallel asyncio.to_thread() queries
             profile, entities, today_events, daily_summaries, recent_docs, recent_knowledge
          -> cache.set_json(snapshot, ttl=300s)
          -> release_lock()
          -> _merge_response(snapshot, accessed)
             # Simplifies snapshot to: SimpleProfile, SimpleTodayEvent (title+summary+event_time), SimpleDailySummary
             # Filters recently_accessed to exclude profile/entity types
```

### Cache Invalidation Flow

```
OpenContext._handle_processed_context()
  -> After successful storage writes
  -> _invalidate_user_cache(user_id, device_id, agent_id)
     -> get_memory_cache_manager().invalidate_snapshot()  # via run_coroutine_threadsafe
     -> Fallback: sync Redis DELETE of snapshot key
```

## Conventions and Constraints

1. **Always use `get_context_lab(request)`** as FastAPI dependency to get `OpenContext`. Do not access `app.state` directly in route handlers.

2. **Always use `get_storage()`** (from `opencontext.storage.global_storage`) for storage access. Do not use `GlobalStorage.get_instance()`.

3. **All push/search endpoints use 60s/30s timeout** via `asyncio.wait_for()`. Memory cache uses 15s. Do not remove these timeouts.

4. **3-key identifier required**: All profile/entity operations and cache keys use `(user_id, device_id, agent_id)`. Default to `"default"` for device_id and agent_id. Omitting them causes argument mismatches.

5. **Search strategies are lazy singletons** in `routes/search.py`. They are stateless and reusable across requests.

6. **Cache invalidation uses `run_coroutine_threadsafe`** because `_handle_processed_context` runs in a thread pool via `asyncio.to_thread()`. Using `loop.create_task()` from a thread would silently fail. The `_capture_loop()` pattern stores the event loop reference.

7. **Push endpoints schedule tasks via BackgroundTasks**: Both buffer and direct modes use `background_tasks.add_task()` with the unified `_schedule_user_task(task_type, ...)` helper. Scheduling runs post-response and must not fail the request.

8. **Stream interrupt uses `StreamInterruptManager`** (`stream_interrupt.py`). A singleton per worker with Redis Pub/Sub for cross-worker propagation. One persistent `PSUBSCRIBE stream:interrupt:*` pattern subscriber per worker handles all active streams. `register()` / `is_interrupted()` (sync) / `interrupt()` / `unregister()` are the public API. Falls back to local-only dict when Redis is unavailable. Access via `get_stream_interrupt_manager()`.

9. **Auth is a dependency, not middleware**: `auth_dependency = Depends(verify_api_key)` is added per-route, not as ASGI middleware. This allows per-route opt-in.

10. **Adding a new route module**: Create `routes/new_module.py` with `router = APIRouter(...)`, then add `from .routes import new_module` and `router.include_router(new_module.router)` in `api.py`.

11. **Adding a new search strategy**: Extend `BaseSearchStrategy`, implement `async search() -> TypedResults`, add lazy singleton getter in `routes/search.py`, add enum value to `SearchStrategy` in `search/models.py`.
