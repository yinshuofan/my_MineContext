# periodic_task/ -- Task implementations for the scheduler

Defines the `IPeriodicTask` interface, base class `BasePeriodicTask`, and three concrete task implementations (memory compression, data cleanup, hierarchy summary).

## File Overview

| File | Responsibility |
|------|---------------|
| `base.py` | ABC (`IPeriodicTask`), base class `BasePeriodicTask`, dataclasses (`TaskContext`, `TaskResult`) |
| `memory_compression.py` | `MemoryCompressionTask` -- deduplicates similar knowledge contexts per user (`user_activity`) |
| `data_cleanup.py` | `DataCleanupTask` -- retention-based data cleanup (`periodic`, global) |
| `hierarchy_summary.py` | `HierarchySummaryTask` -- generates L1/L2/L3 event summaries (`user_activity`) |
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
| `default_config` | `@property -> TaskConfig` | Default `TaskConfig` for registration |
| `execute` | `async (context: TaskContext) -> TaskResult` | Async execution (main entry) |
| `validate_context` | `(context: TaskContext) -> bool` | Pre-execution validation (default: `True`) |

### BasePeriodicTask

Concrete base implementing `IPeriodicTask`. Subclasses override `async execute()`.

**Constructor**: `__init__(name, description="", trigger_mode=TriggerMode.USER_ACTIVITY, interval=1800, timeout=300, task_ttl=7200, max_retries=3)`

- `default_config` property builds a `TaskConfig` from constructor args
- `async execute()` raises `NotImplementedError`

### MemoryCompressionTask

Deduplicates similar knowledge contexts for a specific user.

| Property | Value |
|----------|-------|
| `name` | `"memory_compression"` |
| `trigger_mode` | `USER_ACTIVITY` |
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
| `trigger_mode` | `PERIODIC` |
| `interval` | `86400` (24 hours) |
| `timeout` | `600` |

**Constructor**: `__init__(context_merger=None, storage=None, interval=86400, timeout=600, retention_days=30)`

- `set_context_merger(context_merger) -> None` / `set_storage(storage) -> None` -- late injection
- `async execute(context)` -- tries `context_merger.intelligent_memory_cleanup()`, then `cleanup_contexts_by_type()`, then falls back to storage `cleanup_expired_data()` or `delete_old_contexts()`
- Handler receives `(None, None, None)` from periodic scheduler -- uses `"global"` as user_id

**Factory**: `create_cleanup_handler(context_merger=None, storage=None, retention_days=30) -> async TaskHandler`

### HierarchySummaryTask

Generates hierarchical time-based summaries (L1 daily, L2 weekly, L3 monthly) for EVENT contexts.

| Property | Value |
|----------|-------|
| `name` | `"hierarchy_summary"` |
| `trigger_mode` | `USER_ACTIVITY` |
| `interval` | `86400` (24 hours) |
| `timeout` | `600` |
| `task_ttl` | `14400` |
| `max_retries` | `2` |

**Constructor**: `__init__(interval=86400, timeout=600)`

No external dependency injection -- uses `get_storage()` and LLM globals directly.

**Core flow in `async execute(context)`**:
1. Generate L1 daily summary for yesterday
2. Generate L2 weekly summary for the most recent completed ISO week
3. Generate L3 monthly summary for the most recent completed month

Each `async _generate_{level}_summary()` method:
1. Checks for existing summary via `await storage.search_hierarchy()` (idempotent dedup)
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

**Storage**: `async _store_summary(user_id, summary_text, level, time_bucket, children_ids, device_id=None, agent_id=None) -> Optional[ProcessedContext]` -- parses the LLM JSON response (`{title, summary, keywords, entities, importance}`) to extract structured fields, builds `ProcessedContext` with hierarchy fields (including `device_id`/`agent_id` on `ContextProperties`), generates embedding via `await do_vectorize()`, calls `await storage.upsert_processed_context()`. After successful upsert, backfills `parent_id` on all child contexts via `await storage.batch_set_parent_id(children_ids, summary_id, "event")` — this enables upward traversal (child → parent summary). The backfill is non-fatal: if it fails, the summary itself is still valid. Falls back to heuristic title extraction if the LLM response is not valid JSON. Also handles markdown code fence stripping (` ```json ... ``` `).

**Factory**: `create_hierarchy_handler() -> async TaskHandler`

## Class Hierarchy

```
IPeriodicTask (ABC)
  └── BasePeriodicTask
        ├── MemoryCompressionTask
        ├── DataCleanupTask
        └── HierarchySummaryTask
```

## Internal Data Flow

### Task registration and handler wiring (at startup)

```
cli.py / opencontext.py startup
  ├─ Create task instances (MemoryCompressionTask, DataCleanupTask, HierarchySummaryTask)
  ├─ create_*_handler() -> TaskHandler function (closure over task instance)
  └─ scheduler.register_handler(task_type, handler)  -- wires handler into RedisTaskScheduler
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
  ├─> _generate_daily_summary(user_id, yesterday, device_id, agent_id)
  │     ├─ search_hierarchy(level=1, time_bucket=yesterday, device_id, agent_id) -- dedup check
  │     ├─ get_all_processed_contexts(EVENT, filter=..., device_id, agent_id)
  │     ├─ _batch_summarize_and_merge(l0_events, level=1, ..., _format_l0_events)
  │     │     ├─ If fits: _call_llm_for_summary()
  │     │     └─ If overflow: _split_into_batches() -> partial summaries -> _call_llm_for_merge()
  │     └─ _store_summary(level=1, ..., device_id, agent_id)
  │
  ├─> _generate_weekly_summary(user_id, prev_week, device_id, agent_id)
  │     ├─ search_hierarchy(level=2, time_bucket=week, device_id, agent_id) -- dedup check
  │     ├─ search_hierarchy(level=1, week date range, device_id, agent_id) -- fetch L1
  │     ├─ _batch_summarize_weekly(l1, week_str)
  │     └─ _store_summary(level=2, ..., device_id, agent_id)
  │
  └─> _generate_monthly_summary(user_id, prev_month, device_id, agent_id)
        ├─ search_hierarchy(level=3, time_bucket=month, device_id, agent_id) -- dedup check
        ├─ For each ISO week in month:
        │     ├─ search_hierarchy(level=2, week, device_id, agent_id) -- fetch L2
        │     └─ search_hierarchy(level=1, week date range, device_id, agent_id) -- fetch L1
        ├─ _batch_summarize_monthly(l2, l1_by_week, month_str)
        └─ _store_summary(level=3, ..., device_id, agent_id)
```

## Cross-Module Dependencies

**Imports from**:
- `loguru` -- `logger` imported directly in `memory_compression.py`, `data_cleanup.py`; `hierarchy_summary.py` uses `opencontext.utils.logging_utils.get_logger`
- `opencontext.scheduler.base` -- `TaskConfig`, `TriggerMode` (used by all task classes)
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
from opencontext.scheduler.base import TriggerMode

class MyTask(BasePeriodicTask):
    def __init__(self, interval: int = 3600, timeout: int = 300):
        super().__init__(
            name="my_task",                    # Must match config key in config.yaml
            description="What this task does",
            trigger_mode=TriggerMode.USER_ACTIVITY,  # or PERIODIC for global tasks
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

3. **Add config** in `config/config.yaml` under `scheduler.tasks`:

```yaml
scheduler:
  tasks:
    my_task:
      enabled: true
      trigger_mode: "user_activity"  # or "periodic"
      interval: 3600
      timeout: 300
```

4. **Wire up in `component_initializer.py`** startup: create handler, register with scheduler:

```python
handler = create_my_handler()
scheduler.register_handler("my_task", handler)
```

5. **For `user_activity` tasks**: call `scheduler.schedule_user_task("my_task", user_id, ...)` from the appropriate push endpoint.
