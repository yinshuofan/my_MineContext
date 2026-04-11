# periodic_task/ -- Task implementations for the scheduler

Defines the `IPeriodicTask` interface, base class `BasePeriodicTask`, and three concrete task implementations (memory compression, data cleanup, hierarchy summary).

## File Overview

| File | Responsibility |
|------|---------------|
| `base.py` | ABC (`IPeriodicTask`), base class `BasePeriodicTask`, dataclasses (`TaskContext`, `TaskResult`) |
| `memory_compression.py` | `MemoryCompressionTask` -- deduplicates similar knowledge contexts per user (`user_activity`) |
| `data_cleanup.py` | `DataCleanupTask` -- retention-based data cleanup (`periodic`, global) |
| `hierarchy_summary.py` | `HierarchySummaryTask` -- generates L1/L2/L3 event summaries (`user_activity`) |
| `agent_profile_update.py` | `AgentProfileUpdateTask` -- updates agent_profile from daily event commentary (`user_activity`) |
| `__init__.py` | Re-exports all public symbols including `create_*_handler` factory functions |

## Key Classes and Functions

### TaskResult (dataclass)

| Field | Type | Default |
|-------|------|---------|
| `success` | `bool` | -- |
| `message` | `str` | `""` |
| `data` | `Optional[Dict[str, Any]]` | `None` |
| `error` | `Optional[str]` | `None` |

Factory methods: `TaskResult.ok(message, data) -> TaskResult`, `TaskResult.fail(error, message) -> TaskResult`

### TaskContext (dataclass)

Execution context passed to every task. Created by handler factory functions.

| Field | Type | Default |
|-------|------|---------|
| `user_id` | `str` | -- |
| `device_id` | `Optional[str]` | `None` |
| `agent_id` | `Optional[str]` | `None` |
| `task_type` | `str` | `""` |
| `extra` | `Dict[str, Any]` | `{}` |

### IPeriodicTask (ABC)

| Member | Signature | Description |
|--------|-----------|-------------|
| `name` | `@property -> str` | Unique task type name |
| `description` | `@property -> str` | Human-readable description |
| `execute` | `async (context: TaskContext) -> TaskResult` | Async execution (main entry) |
| `validate_context` | `(context: TaskContext) -> bool` | Pre-execution validation (default: `True`) |

### BasePeriodicTask

Concrete base implementing `IPeriodicTask`. Subclasses override `async execute()`.

**Constructor**: `__init__(name, description="", interval=1800, timeout=300, task_ttl=7200, max_retries=3)`

- `async execute()` raises `NotImplementedError`

### MemoryCompressionTask

Deduplicates similar knowledge contexts for a specific user.

| Property | Value |
|----------|-------|
| `name` | `"memory_compression"` |
| `interval` | `172800` (48 hours) |
| `timeout` | `300` |

**Constructor**: `__init__(context_merger=None, interval=172800, timeout=300)`

- `set_context_merger(context_merger) -> None` -- late injection
- `async execute(context)` -- calls `context_merger.periodic_memory_compression_for_user(user_id, device_id, agent_id, interval_seconds)` or falls back to `periodic_memory_compression(interval_seconds)`

**Factory**: `create_compression_handler(context_merger) -> async TaskHandler`

### DataCleanupTask

Global retention-based cleanup. Delegates to `ContextMerger.intelligent_memory_cleanup()`.

| Property | Value |
|----------|-------|
| `name` | `"data_cleanup"` |
| `interval` | `86400` (24 hours) |
| `timeout` | `600` |

**Constructor**: `__init__(context_merger=None, storage=None, interval=86400, timeout=600, retention_days=30)`

- `set_context_merger(context_merger) -> None` / `set_storage(storage) -> None` -- late injection
- `async execute(context)` -- tries `context_merger.intelligent_memory_cleanup()`, then `cleanup_contexts_by_type()`, then falls back to storage `cleanup_expired_data()` or `delete_old_contexts()`
- Handler receives `(None, None, None)` from periodic scheduler -- uses `"global"` as user_id

**Factory**: `create_cleanup_handler(context_merger=None, storage=None, retention_days=30) -> async TaskHandler`

### HierarchySummaryTask

Generates hierarchical time-based summaries (L1 daily, L2 weekly, L3 monthly). Summaries are now stored with their own ContextType (DAILY_SUMMARY, WEEKLY_SUMMARY, MONTHLY_SUMMARY) instead of all being EVENT with hierarchy_level > 0.

| Property | Value |
|----------|-------|
| `name` | `"hierarchy_summary"` |
| `interval` | `86400` (24 hours) |
| `timeout` | `600` |
| `task_ttl` | `14400` |
| `max_retries` | `2` |

**Constructor**: `__init__(interval=86400, timeout=600)`

No external dependency injection -- uses `get_storage()` and LLM globals directly.

**Level-to-type mappings** (module-level constants):

```python
LEVEL_TO_CONTEXT_TYPE = {1: DAILY_SUMMARY, 2: WEEKLY_SUMMARY, 3: MONTHLY_SUMMARY}
LEVEL_TO_CHILD_TYPE = {1: EVENT, 2: DAILY_SUMMARY, 3: WEEKLY_SUMMARY}
```

**Core flow in `async execute(context)`**:
1. Backfill L1 daily summaries for recent N days via `_backfill_daily_summaries()` (N = `backfill_days` config, default 7). Batch-queries existing L1 summaries in the window (1 query) to skip already-generated dates, then generates missing ones most-recent-first.
2. Generate L2 weekly summary for the most recent completed ISO week
3. Generate L3 monthly summary for the most recent completed month

Each `async _generate_{level}_summary()` method:
1. **Dual-format dedup check**: Checks for existing summary in both old format (EVENT with hierarchy_level=N) and new format (DAILY_SUMMARY/WEEKLY_SUMMARY/MONTHLY_SUMMARY). This backward-compat check prevents regeneration of summaries created before the migration.
2. Queries child-level data from storage (all `await`ed)
3. Formats content, handles token overflow via batch splitting
4. Calls LLM (`await generate_with_messages()`), stores result as `ProcessedContext`

**Content formatting methods** (all `@staticmethod`):

| Method | Input | Output |
|--------|-------|--------|
| `_format_l0_events(contexts)` | L0 events | Numbered lines: `[i] Title | Time | Summary | Keywords` |
| `_format_weekly_hierarchical(l1_summaries)` | L1 | Day-grouped: `=== Daily Summary: date ===` |
| `_format_monthly_hierarchical(l2_summaries, l1_by_week)` | L2 + L1 | Week-grouped: `=== Weekly Summary: week ===` with nested daily summaries |

**Token overflow handling**:

Constants: `_MAX_INPUT_TOKENS = 60000`, `_BATCH_TOKEN_TARGET = 25000`, `_PROMPT_OVERHEAD_TOKENS = 800`

| Method | Level | Batching strategy |
|--------|-------|-------------------|
| `async _batch_summarize_and_merge(contexts, level, time_bucket, format_fn)` | L1 | Split by token budget per context |
| `async _batch_summarize_weekly(l1_contexts, time_bucket)` | L2 | Split by day-groups (3 days per batch) |
| `async _batch_summarize_monthly(l2_contexts, l1_by_week, time_bucket)` | L3 | Split by week-groups (2 weeks per batch) |

Each overflow handler: format -> check tokens -> if over limit, split into batches -> generate sub-summaries -> merge via `_call_llm_for_merge()`.

**LLM interaction**:

| Method | Purpose |
|--------|---------|
| `async _call_llm_for_summary(formatted_content, level, time_bucket, is_partial?, batch_info?)` | Generate summary from formatted content via `await generate_with_messages()` |
| `async _call_llm_for_merge(sub_summaries_text, level, time_bucket)` | Merge partial sub-summaries into one via `await generate_with_messages()` |

Prompt resolution: prompt group `"hierarchy_summary"` from YAML -> `{level}_summary` / `{level}_partial_summary` / `{level}_merge` keys -> fallback to `_FALLBACK_PROMPTS` dict.

**Storage**: `async _store_summary(user_id, summary_text, level, event_time_start, event_time_end, children_ids, device_id=None, agent_id=None) -> Optional[ProcessedContext]` -- parses the LLM JSON response (`{title, summary, keywords, entities, importance}`) to extract structured fields. Resolves `summary_context_type` from `LEVEL_TO_CONTEXT_TYPE[level]` and `child_type` from `LEVEL_TO_CHILD_TYPE[level]`. Builds `ProcessedContext` with:
- `extracted_data.context_type` = `summary_context_type` (e.g. `DAILY_SUMMARY`)
- `properties.refs` = `{child_type.value: children_ids}` (downward refs)

Generates embedding via `await do_vectorize()`, calls `await storage.upsert_processed_context()`. After successful upsert, backfills upward refs on all child contexts via `await storage.batch_update_refs(children_ids, ref_key=summary_context_type.value, ref_value=summary_id, context_type=child_type.value)` -- this enables upward traversal (child -> parent summary). The backfill is non-fatal: if it fails, the summary itself is still valid. Falls back to heuristic title extraction if the LLM response is not valid JSON. Also handles markdown code fence stripping (` ```json ... ``` `).

**Factory**: `create_hierarchy_handler() -> async TaskHandler`

### AgentProfileUpdateTask

Updates the `agent_profile` for a specific user by aggregating `agent_commentary` from today's events. Triggered via `user_activity` when `agent_memory` processor is used.

| Property | Value |
|----------|-------|
| `name` | `"agent_profile_update"` |
| `interval` | `86400` (24 hours) |
| `timeout` | `300` |
| `task_ttl` | `14400` |
| `max_retries` | `2` |

**Constructor**: `__init__(interval=86400, timeout=300)`

**Core flow in `async execute(context)`**:
1. Validates `agent_id` is non-default and agent is registered
2. Fetches today's EVENT-type contexts for the user via `storage.get_all_processed_contexts()`
3. Filters for events that have non-empty `agent_commentary` on `extracted_data`
4. Fetches existing `agent_profile` for context
5. Calls LLM with `agent_profile_update` prompt, passing existing profile and today's commentary
6. Calls `storage.upsert_profile()` directly with the LLM-generated profile — bypasses the LLM merge step in `profile_processor.refresh_profile()` since the LLM has already incorporated the existing profile in step 5

Returns early if:
- `agent_id` is missing or `"default"`
- Agent not found in registry
- No events with `agent_commentary` found today

**Factory**: `create_agent_profile_handler() -> async TaskHandler`

## Class Hierarchy

```
IPeriodicTask (ABC)
  └── BasePeriodicTask
        ├── MemoryCompressionTask
        ├── DataCleanupTask
        ├── HierarchySummaryTask
        └── AgentProfileUpdateTask
```

## Internal Data Flow

### Task registration and handler wiring (at startup)

```
cli.py / opencontext.py startup
  ├─ Create task instances (MemoryCompressionTask, DataCleanupTask, HierarchySummaryTask)
  ├─ create_*_handler() -> TaskHandler function (closure over task instance)
  └─ scheduler.register_handler(task_type, handler, trigger_mode=TriggerMode.XXX)
        -- wires handler into RedisTaskScheduler; trigger_mode is a code-declared contract
```

### Task execution (triggered by scheduler)

```
RedisTaskScheduler._execute_task() or _process_periodic_tasks()
  └─> await handler(user_id, device_id, agent_id)  -- the async closure from create_*_handler()
        ├─ Build TaskContext(user_id, device_id, agent_id, task_type)
        ├─ await task.execute(context) -> TaskResult
        └─> return result.success
```

### HierarchySummaryTask execution detail

```
execute(context)
  │  extracts user_id, device_id, agent_id from context
  │
  ├─> _backfill_daily_summaries(user_id, today, device_id, agent_id)
  │     ├─ Read backfill_days from config (default 7)
  │     ├─ search_hierarchy(level=1, context_type=EVENT or DAILY_SUMMARY, ...) -- batch dedup (dual-format)
  │     └─ For each missing date (most recent first):
  │           └─> _generate_daily_summary(user_id, date, device_id, agent_id)
  │                 ├─ Dual-format dedup: search_hierarchy(EVENT, level=1) + search_hierarchy(DAILY_SUMMARY)
  │                 ├─ get_all_processed_contexts(EVENT, filter=..., device_id, agent_id)
  │                 ├─ _batch_summarize_and_merge(l0_events, level=1, ..., _format_l0_events)
  │                 │     ├─ If fits: _call_llm_for_summary()
  │                 │     └─ If overflow: _split_into_batches() -> partial summaries -> _call_llm_for_merge()
  │                 └─ _store_summary(level=1, context_type=DAILY_SUMMARY, refs={EVENT: ids})
  │
  ├─> _generate_weekly_summary(user_id, prev_week, device_id, agent_id)
  │     ├─ Dual-format dedup: search_hierarchy(EVENT, level=2) + search_hierarchy(WEEKLY_SUMMARY)
  │     ├─ Fetch L1: search_hierarchy(EVENT, level=1) + search_hierarchy(DAILY_SUMMARY) -- merged
  │     ├─ _batch_summarize_weekly(l1, week_str)
  │     └─ _store_summary(level=2, context_type=WEEKLY_SUMMARY, refs={DAILY_SUMMARY: ids})
  │
  └─> _generate_monthly_summary(user_id, prev_month, device_id, agent_id)
        ├─ Dual-format dedup: search_hierarchy(EVENT, level=3) + search_hierarchy(MONTHLY_SUMMARY)
        ├─ For each ISO week in month:
        │     ├─ Fetch L2: search_hierarchy(EVENT, level=2) + search_hierarchy(WEEKLY_SUMMARY) -- merged
        │     └─ Fetch L1: search_hierarchy(EVENT, level=1) + search_hierarchy(DAILY_SUMMARY) -- merged
        ├─ _batch_summarize_monthly(l2, l1_by_week, month_str)
        └─ _store_summary(level=3, context_type=MONTHLY_SUMMARY, refs={WEEKLY_SUMMARY: ids})
```

Note: The dual-format dedup checks (querying both old EVENT-with-hierarchy-level and new typed ContextTypes) ensure backward compatibility with summaries generated before the migration. New summaries are always stored with their dedicated ContextType.

Note: All `search_hierarchy()` calls in the flow above are **storage-level calls** (`storage.search_hierarchy()` on `UnifiedStorage`) used by `HierarchySummaryTask` to fetch child contexts and check for existing summaries. They are unrelated to `EventSearchService` in `opencontext/server/search/`, which is the HTTP search API layer and does not call `search_hierarchy()` internally.

## Cross-Module Dependencies

**Imports from**:
- `loguru` -- `logger` imported directly in `memory_compression.py`, `data_cleanup.py`; `hierarchy_summary.py` uses `opencontext.utils.logging_utils.get_logger`
- `opencontext.storage.global_storage.get_storage` -- hierarchy_summary accesses storage directly
- `opencontext.llm.global_vlm_client.generate_with_messages` -- hierarchy_summary LLM calls
- `opencontext.llm.global_embedding_client.do_vectorize` -- hierarchy_summary embedding generation
- `opencontext.config.global_config.get_prompt_group` -- hierarchy_summary prompt resolution
- `opencontext.models.context` -- `ProcessedContext`, `ContextProperties`, `ExtractedData`, `Vectorize`
- `opencontext.models.enums` -- `ContextType`, `ContentFormat`

**Depended on by**:
- `opencontext.server.component_initializer` -- creates task instances, creates handlers, registers with scheduler
- `opencontext.server.push` -- calls `_schedule_user_hierarchy_summary()` which calls `scheduler.schedule_user_task("hierarchy_summary", ...)`

## Conventions and Constraints

- **Handlers are async**: The scheduler awaits handlers directly. The `create_*_handler()` factories return async closures that `await task.execute()`.
- **`validate_context` is called by `create_hierarchy_handler()` only**: `HierarchySummaryTask` overrides `validate_context()` to require non-empty `user_id`, and its handler calls it before `execute()`. Other handlers do not call `validate_context()`.
- **Idempotent summary generation**: `HierarchySummaryTask` checks for existing summaries before generating. Safe to re-trigger.
- **Late dependency injection**: `MemoryCompressionTask` and `DataCleanupTask` accept `context_merger`/`storage` via constructor or `set_*()` methods. `HierarchySummaryTask` uses globals (`get_storage()`, LLM singletons).
- **Handler factory pattern**: Each task has a `create_*_handler()` function that returns a `TaskHandler`-compatible closure. This is what gets registered with the scheduler via `scheduler.register_handler()`.

## Extension Points

### Adding a new periodic task

1. **Create task class** in a new file (e.g., `opencontext/periodic_task/my_task.py`):

```python
from opencontext.periodic_task.base import BasePeriodicTask, TaskContext, TaskResult

class MyTask(BasePeriodicTask):
    def __init__(self, interval: int = 3600, timeout: int = 300):
        super().__init__(
            name="my_task",                    # Must match config key in config.yaml
            description="What this task does",
            interval=interval,
            timeout=timeout,
        )

    async def execute(self, context: TaskContext) -> TaskResult:
        # Implementation here (async — can await storage/LLM calls)
        return TaskResult.ok("Done")

    def validate_context(self, context: TaskContext) -> bool:
        return bool(context.user_id)  # Required for USER_ACTIVITY tasks

def create_my_handler():
    task = MyTask()
    async def handler(user_id, device_id, agent_id):
        ctx = TaskContext(user_id=user_id, device_id=device_id,
                          agent_id=agent_id, task_type="my_task")
        if not task.validate_context(ctx):
            return False
        result = await task.execute(ctx)
        return result.success
    return handler
```

2. **Register in `__init__.py`**: Add imports and `__all__` entries.

3. **Add config** in `config/config.yaml` under `scheduler.tasks`. Do NOT add a `trigger_mode` field — it is declared in code at registration time (`_collect_task_types` strips stale YAML values with a deprecation warning):

```yaml
scheduler:
  tasks:
    my_task:
      enabled: true
      interval: 3600
      timeout: 300
```

4. **Wire up in `component_initializer.py`** startup: create handler, register with scheduler, and declare the trigger mode explicitly via the required keyword-only `trigger_mode` parameter:

```python
from opencontext.scheduler.base import TriggerMode

handler = create_my_handler()
scheduler.register_handler(
    "my_task",
    handler,
    trigger_mode=TriggerMode.USER_ACTIVITY,  # or TriggerMode.PERIODIC
)
```

5. **For `user_activity` tasks**: call `scheduler.schedule_user_task("my_task", user_id, ...)` from the appropriate push endpoint.
