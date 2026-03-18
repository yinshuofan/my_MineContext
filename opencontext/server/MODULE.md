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
| `routes/search.py` | Event search endpoint (`POST /api/search`) with semantic query, filters, and hierarchy drill-up |
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
| `search/models.py` | Pydantic models: `EventSearchRequest` (multimodal content parts query, `memory_owner` field), `EventSearchResponse`, `EventNode` (with `refs` and `media_refs`), `SearchMetadata`, `TimeRange` |
| **cache/** | |
| `cache/memory_cache_manager.py` | `MemoryCacheManager` singleton -- builds/caches per-owner memory snapshots in Redis (parameterized by `memory_owner`: `"user"` or `"agent"`) |
| `cache/models.py` | Response models: `UserMemoryCacheResponse`, `SimpleProfile`, `SimpleDailySummary`, `SimpleTodayEvent`, `RecentlyAccessedItem` (with `media_refs`); internal models: `RecentMemoryItem`, `DailySummaryItem`. `UserMemoryCacheManager` is kept as a backward-compat alias for `MemoryCacheManager`. |
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
    def _invalidate_user_cache(self, user_id, device_id, agent_id) -> None  # Fire-and-forget

    # Delegated operations
    def add_context(self, context_data: RawContextProperties) -> bool
    def add_document(self, file_path: str) -> Optional[str]     # Returns error msg or None
    def search(self, query, top_k, context_types, filters, user_id, device_id, agent_id, score_threshold) -> List[Dict]
    def get_all_contexts(self, limit, offset, filter_criteria) -> Dict[str, List[ProcessedContext]]
    def get_context(self, doc_id: str, context_type: str) -> Optional[ProcessedContext]
    def update_context(self, doc_id: str, context: ProcessedContext) -> bool
    def delete_context(self, doc_id: str, context_type: str) -> bool
    async def check_components_health(self) -> Dict[str, Any]  # Checks config, storage, llm, document_db, redis, scheduler

    # Additional public methods
    def start_capture(self) -> None                  # Starts all capture components via capture_manager.start_all_components()
    def get_context_types(self) -> List[str]          # Delegates to context_operations.get_context_types()

    # Private helpers
    def _initialize_monitoring(self) -> None          # Initializes monitoring system; called from initialize()
    def _invalidate_cache_sync_fallback(self, user_id, device_id, agent_id) -> None  # Sync Redis DELETE fallback
```

Module-level: `main()` function -- entry point for `if __name__ == "__main__"`, parses args and runs uvicorn.

Key fields: `capture_manager` (ContextCaptureManager), `processor_manager` (ContextProcessorManager), `storage` (`@property` → `get_storage()`, lazy access), `context_operations` (ContextOperations), `component_initializer` (ComponentInitializer).

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

Stateless class — delegates to `UnifiedStorage` via a non-caching `@property storage` that calls `get_storage()` on each access (avoids init-order issues with async `GlobalStorage`).

```python
class ContextOperations:
    def get_all_contexts(self, limit, offset, filter_criteria, need_vector) -> Dict[str, List[ProcessedContext]]
    def get_context(self, doc_id: str, context_type: str) -> Optional[ProcessedContext]
    def update_context(self, doc_id: str, context: ProcessedContext) -> bool
    def delete_context(self, doc_id: str, context_type: str) -> bool
    def add_document(self, file_path: str, context_processor_callback) -> Optional[str]
    def search(self, query, top_k, context_types, filters, user_id, device_id, agent_id, score_threshold) -> List[Dict]
    def get_context_types(self) -> List[str]
```

### Event Search (routes/search.py)

Event-only search with three paths and optional upward hierarchy drill-up. Returns a **tree structure** where ancestors are parent nodes containing search hits as nested children. Parameterized by `memory_owner` (`"user"` or `"agent"`) via `EventSearchRequest.memory_owner` to dynamically resolve ContextTypes from `MEMORY_OWNER_TYPES`.

```python
# Route handler
async def search_events(request: EventSearchRequest, _auth) -> EventSearchResponse

# Internal functions
def _get_l0_type(memory_owner: str) -> str                         # Resolve L0 context type from MEMORY_OWNER_TYPES
def _get_context_types_for_levels(memory_owner: str, levels: Optional[List[int]]) -> List[str]  # Map levels to ContextType values
async def _execute_search(storage, request) -> Tuple[List[EventNode], List[EventNode]]
async def _filter_only_search(storage, request) -> List[Tuple[ProcessedContext, float]]
async def _collect_ancestors(storage, results, max_level, memory_owner="user") -> Dict[str, ProcessedContext]
def _build_filters(time_range, hierarchy_levels) -> Dict[str, Any]
def _time_range_to_buckets(start_ts, end_ts) -> Tuple[Optional[str], Optional[str]]
def _to_context_node(ctx: ProcessedContext) -> EventNode
def _to_search_hit_node(ctx: ProcessedContext, score: float) -> EventNode
async def _track_accessed_safe(user_id, results, device_id, agent_id) -> None
```

Algorithm:
1. **Search path selection** (priority: event_ids > query > filters-only):
   - `event_ids` → `storage.get_contexts_by_ids()`, score=1.0
   - `query` → `Vectorize` + `storage.search()` with time filter only, using context types from `_get_context_types_for_levels(memory_owner, None)`
   - filters-only → `storage.search_hierarchy()` per level, or `get_all_processed_contexts()` with time filter
2. **Collect ancestors** (if `drill_up=True`): Follows `refs` upward iteratively (max 3 rounds for L0→L3). Uses `seen` cache to avoid duplicate fetches when multiple events share the same parent ref. Strictly stops at `max_level`. Falls back to `metadata.parent_id` for old data written before the refs migration.
3. **Build node map**: Search hits become `EventNode` with is_search_hit=True (score/content/keywords populated), ancestors become lightweight `EventNode` with is_search_hit=False. Search hits are never overwritten by ancestors.
4. **Link tree**: Parent is extracted from `refs` via `_extract_parent_id_from_refs()` — finds the first ref ID that exists in the node map. Nodes without a valid parent become roots.
5. **Sort**: Roots and all children lists sorted by `time_bucket` ASC recursively.

### MemoryCacheManager (cache/memory_cache_manager.py)

Singleton via `get_memory_cache_manager()`. Manages per-owner memory snapshots in Redis. Renamed from `UserMemoryCacheManager` (kept as backward-compat alias). Parameterized by `memory_owner` (`"user"` or `"agent"`) to resolve which ContextTypes to query via `MEMORY_OWNER_TYPES`.

```python
class MemoryCacheManager:
    def __init__(self)
    async def track_accessed(self, user_id, items: List[Dict], device_id, agent_id) -> None
    async def invalidate_snapshot(self, user_id, device_id, agent_id, memory_owner="user") -> None
    async def refresh_snapshot(self, user_id, device_id, agent_id, memory_owner="user") -> bool
    async def get_user_memory_cache(self, user_id, device_id, agent_id, recent_days, max_recent_events_today, max_accessed, force_refresh, include_sections: Optional[Set[str]] = None, memory_owner="user") -> UserMemoryCacheResponse

    # Internal
    async def _get_recently_accessed(self, cache, user_id, max_items, device_id, agent_id) -> List[RecentlyAccessedItem]
    async def _build_snapshot(self, user_id, device_id, agent_id, recent_days, max_today_events, memory_owner="user") -> Dict[str, Any]
    def _merge_response(self, snapshot_data, accessed, cache_hit, ttl_remaining, include_sections: Optional[Set[str]] = None) -> UserMemoryCacheResponse
    async def _trim_accessed(self, cache, key: str, max_size: int) -> None
    @staticmethod _ctx_to_recent_item(ctx: ProcessedContext) -> Dict[str, Any]
    @staticmethod _snapshot_key(memory_owner, user_id, device_id, agent_id) -> str
```

**`memory_owner` parameter**: Controls which ContextTypes are used in snapshot queries. `_build_snapshot()` resolves types from `MEMORY_OWNER_TYPES[memory_owner]` (e.g., `"user"` → EVENT/DAILY_SUMMARY/..., `"agent"` → AGENT_EVENT/AGENT_DAILY_SUMMARY/...). Profile queries pass `owner_type` derived from `memory_owner`. Agent memory owner skips document/knowledge sections.

Caching architecture:
- **Snapshot** (profile + today events + daily summaries + recent docs + recent knowledge): Redis JSON string, configurable TTL (default 300s). Key: `memory_cache:snapshot:{memory_owner}:{user_id}:{device_id}:{agent_id}`. Snapshot always stores full internal data; response assembly in `_merge_response()` filters by `include_sections` and simplifies to `SimpleProfile`, `SimpleDailySummary`, `SimpleTodayEvent`.
- **Recently Accessed**: Redis Hash, 7-day TTL. Key: `memory_cache:accessed:{user_id}:{device_id}:{agent_id}`. Updated on every search (documents/events/knowledge only; profile excluded), always read real-time. Skipped entirely if `"accessed"` not in `include_sections`.
- **Stampede prevention**: Distributed lock via `cache.acquire_lock()` + double-check pattern. Lock key includes `memory_owner`: `memory_cache:build:{memory_owner}:{user_id}:{device_id}:{agent_id}`. If lock acquisition times out, tries cache once more then builds directly without caching.
- **Snapshot build**: 5 parallel queries (profile, today events, daily summaries, recent docs, recent knowledge). Snapshot is always built fully for caching efficiency; `include_sections` filtering is response-level only.
- **Section filtering**: `include_sections` controls which response fields are populated. Default: `{"profile", "events", "accessed"}`. Sections: `profile` → `profile`, `events` → `today_events` + `daily_summaries`, `accessed` → `recently_accessed`. Unrequested sections are `null` in response; requested but empty sections are `[]`.

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
| POST | `/api/search` | `search_events` | Event search with semantic query, filters, and hierarchy drill-up |

### Memory Cache Routes (`/api/*`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/memory-cache` | `get_user_memory_cache` | Get memory snapshot (query param: `memory_owner`, default `"user"`) |
| DELETE | `/api/memory-cache` | `invalidate_user_memory_cache` | Invalidate memory cache (query param: `memory_owner`, default `"user"`) |

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
| GET | `/api/monitoring/scheduler/queues` | `get_scheduler_queue_depths` | Real-time queue depths for all task types from Redis `zcard`. Falls back to remote heartbeat + `zcard` when no local scheduler |
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
| POST | `/api/settings/apply` | `apply_settings` | Broadcast config reload signal to all workers |

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
| GET | `/contexts` | `read_contexts` | Contexts list page (card layout). Query params: `page`, `limit`, `type`, `user_id`, `device_id`, `agent_id`, `hierarchy_level` (0-3), `start_date` (datetime-local or date), `end_date` (datetime-local or date). Type filter excludes profile (relational DB only) |
| GET | `/vector_search` | `vector_search_page` | Event search page |
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
| Scheduler | `opencontext.scheduler` (get_scheduler, init_scheduler, read_scheduler_heartbeat) |
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

Both text-only and multimodal messages (OpenAI format) are supported. Before processing, `_process_multimodal_messages()` (async) handles base64 media:
- Text-only messages (`content` is a string): pass through unchanged
- Multimodal messages (`content` is a list): base64 images/videos are uploaded to object storage (if configured, via `get_object_storage()`) and replaced with HTTPS URLs; without object storage, files are saved locally to `./uploads/media/{uuid}.{ext}`; HTTP URLs pass through unchanged; format and size constraints are validated (images < 10 MB, videos < 50 MB)

Buffer mode (`process_mode="buffer"`):
```
push_chat()
  -> await _process_multimodal_messages()  # upload media to object storage, validate
  -> for msg in messages: text_chat.push_message()  # content can be str or List[Dict]
  -> if flush_immediately: text_chat.flush_user_buffer()
  -> BackgroundTasks: _schedule_user_task("memory_compression")
  -> BackgroundTasks: _schedule_user_task("hierarchy_summary")
```

Direct mode (`process_mode="direct"`):
```
push_chat()
  -> await _process_multimodal_messages()  # upload media to object storage, validate
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
            profile -> _store_profile() -> storage.upsert_profile()
            document/event/knowledge -> storage.batch_upsert_processed_context() -> vector DB
         -> _invalidate_user_cache() for affected users
```

Processors return `List[ProcessedContext]` to the manager, which centrally invokes the callback. Processors do not call storage or callbacks directly.

### Search Flow (`POST /api/search`)

Query format uses OpenAI content parts (multimodal):
```json
{"query": [{"type": "text", "text": "找会议截图"}, {"type": "image_url", "image_url": {"url": "https://..."}}]}
```

```
search_events()
  -> Validate request (query/event_ids/time_range/hierarchy_levels at least one)
  -> Resolve context types from memory_owner via _get_context_types_for_levels()
  -> _execute_search() with 30s timeout
     Path A (event_ids): storage.get_contexts_by_ids()
     Path B (query):     Vectorize(input=request.query) -> do_vectorize() -> storage.search(owner_context_types, time_filter_only)
     Path C (filters):   storage.search_hierarchy() per level, or get_all_processed_contexts()
  -> _collect_ancestors() if drill_up=True     # Follow refs upward iteratively (max 3 rounds), with parent_id fallback
  -> Build tree: node map → refs-based parent-child linking → recursive sort by time_bucket ASC
  -> Fire-and-forget _track_accessed_safe()     # Includes media_refs in tracked items
  -> Return EventSearchResponse (tree roots + search hit count)
```

#### Response Format (`EventSearchResponse`)

Response is a **tree structure**: high-level summaries are root nodes, lower-level events are nested as `children`. All nodes use the single `EventNode` model. `is_search_hit` distinguishes search hits (with score/content/keywords/entities/metadata populated) from ancestor context nodes (lightweight, only id/level/time_bucket/title/summary).

```json
{
  "success": true,
  "events": [
    {
      "id": "f4b61534-...",
      "hierarchy_level": 1,
      "time_bucket": "2026-03-04",
      "refs": {},
      "title": "Daily Summary",
      "summary": "Daily summary text...",
      "event_time": null,
      "create_time": "2026-03-04T09:29:32.894067+00:00",
      "is_search_hit": false,
      "children": [
        {
          "id": "05278626-88c4-4f85-8eec-e69ac143914c",
          "hierarchy_level": 0,
          "time_bucket": "2026-03-04T09:17:26",
          "refs": {"daily_summary": ["f4b61534-..."]},
          "title": "Event title",
          "summary": "Event summary text",
          "event_time": "2026-03-04T09:17:26.626423+00:00",
          "create_time": "2026-03-04T09:17:26.626077",
          "is_search_hit": true,
          "media_refs": [{"type": "image", "url": "https://..."}],
          "children": [],
          "content": "id: ...\ntitle: ...\nsummary: ...\n...",
          "keywords": ["keyword1", "keyword2"],
          "entities": ["entity1", "entity2"],
          "score": 0.855,
          "metadata": {
            "content": "default",
            "created_at": "2026-03-04T09:18:38.598626",
            "is_happend": 0,
            "source": "default",
            "todo_id": "default"
          }
        }
      ]
    }
  ],
  "metadata": {
    "query": "search query text",
    "total_results": 1,
    "search_time_ms": 556.02
  }
}
```

**Field notes:**
- `events`: List of root nodes (tree roots). When `drill_up=true`, ancestors become parent nodes with search hits nested as children. When `drill_up=false`, all search hits are flat root nodes.
- `is_search_hit`: `true` for search results (content/keywords/entities/score/metadata/media_refs populated), `false` for ancestor context nodes (only id/level/time_bucket/title/summary)
- `media_refs`: List of media references for L0 events, e.g. `[{"type": "image", "url": "https://..."}, {"type": "video", "url": "https://..."}]`. Empty list for L1/L2/L3 summaries. Extracted from `ProcessedContext.metadata["media_refs"]`.
- `children`: Nested child nodes, sorted by `time_bucket` ASC. Root nodes are also sorted by `time_bucket` ASC.
- `hierarchy_level`: 0=raw event, 1=daily summary, 2=weekly summary, 3=monthly summary
- `time_bucket`: `"YYYY-MM-DDTHH:MM:SS"` for L0 events; `"YYYY-MM-DD"` for L1, `"YYYY-Www"` for L2, `"YYYY-MM"` for L3
- `refs`: Dict mapping ContextType values to lists of context IDs (replaces former `parent_id`). E.g. `{"daily_summary": ["sum-id"]}` on an L0 event points to its parent daily summary.
- `total_results`: Count of actual search hits (not total tree nodes)
- `score`: Semantic similarity score for query search; `1.0` for ID lookup and filter-only search

### Memory Cache Flow (`GET /api/memory-cache`)

```
get_user_memory_cache(include_sections, memory_owner)
  -> Parse include_sections (default: {profile, events, accessed})
  -> If "accessed" in sections:
       _get_recently_accessed()          # Real-time from Redis Hash
  -> If only "accessed" requested:
       return _merge_response(empty_snapshot, accessed, sections)
  -> Check cached snapshot (Redis JSON, key includes memory_owner)
     HIT  -> _merge_response(snapshot, accessed, sections)
     MISS -> acquire_lock()
          -> Double-check snapshot
          -> _build_snapshot(memory_owner=memory_owner)  # Resolves types from MEMORY_OWNER_TYPES
             profile (owner_type from memory_owner), today_events, daily_summaries,
             recent_docs (skipped for agent), recent_knowledge (skipped for agent)
          -> cache.set_json(snapshot, ttl=300s)
          -> release_lock()
          -> _merge_response(snapshot, accessed, sections)
             # Filters by include_sections: only populates requested sections
             # Deduplicates accessed against IDs in shown sections
```

### Cache Invalidation Flow (Proactive Refresh)

```
OpenContext._handle_processed_context()
  -> After successful storage writes
  -> _invalidate_user_cache(user_id, device_id, agent_id)
     -> get_memory_cache_manager().refresh_snapshot(memory_owner="user")
     -> get_memory_cache_manager().refresh_snapshot(memory_owner="agent")
        -> For each: acquire distributed lock (key includes memory_owner)
        -> delete old snapshot
        -> _build_snapshot(memory_owner=...) (5 parallel queries, types resolved from MEMORY_OWNER_TYPES)
        -> cache new snapshot with TTL
     -> Fallback (lock held by another worker): delete snapshot key only
```

## Conventions and Constraints

1. **Always use `get_context_lab(request)`** as FastAPI dependency to get `OpenContext`. Do not access `app.state` directly in route handlers.

2. **Always use `get_storage()`** (from `opencontext.storage.global_storage`) for storage access. Do not use `GlobalStorage.get_instance()`.

3. **All push/search endpoints use 60s/30s timeout** via `asyncio.wait_for()`. Memory cache uses 15s. Do not remove these timeouts.

4. **3-key identifier required**: All profile operations and cache keys use `(user_id, device_id, agent_id)`. Default to `"default"` for device_id and agent_id. Omitting them causes argument mismatches.

5. **Search logic lives directly in `routes/search.py`**. No strategy abstraction layer — the route handler calls storage directly.

6. **Cache invalidation uses `run_coroutine_threadsafe`** because `_handle_processed_context` runs in a thread pool via `asyncio.to_thread()`. Using `loop.create_task()` from a thread would silently fail. The `_capture_loop()` pattern stores the event loop reference.

7. **Push endpoints schedule tasks via BackgroundTasks**: Both buffer and direct modes use `background_tasks.add_task()` with the unified `_schedule_user_task(task_type, ...)` helper. Scheduling runs post-response and must not fail the request.

8. **Stream interrupt uses `StreamInterruptManager`** (`stream_interrupt.py`). A singleton per worker with Redis Pub/Sub for cross-worker propagation. One persistent `PSUBSCRIBE stream:interrupt:*` pattern subscriber per worker handles all active streams. `register()` / `is_interrupted()` (sync) / `interrupt()` / `unregister()` are the public API. Falls back to local-only dict when Redis is unavailable. Access via `get_stream_interrupt_manager()`.

9. **Auth is a dependency, not middleware**: `auth_dependency = Depends(verify_api_key)` is added per-route, not as ASGI middleware. This allows per-route opt-in.

10. **Adding a new route module**: Create `routes/new_module.py` with `router = APIRouter(...)`, then add `from .routes import new_module` and `router.include_router(new_module.router)` in `api.py`.

11. **Search uses memory_owner for type resolution**: The `/api/search` endpoint searches event-family contexts (EVENT/DAILY_SUMMARY/WEEKLY_SUMMARY/MONTHLY_SUMMARY for `memory_owner="user"`, or AGENT_EVENT/AGENT_DAILY_SUMMARY/... for `"agent"`). Context types are resolved dynamically from `MEMORY_OWNER_TYPES`, not hardcoded. To search other types, use `/api/vector_search` in `routes/context.py`.
