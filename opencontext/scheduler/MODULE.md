# scheduler/ -- Redis-backed task scheduling system

Stateless, async task scheduler that stores all state in Redis, supporting multi-instance deployment with distributed locking.

## File Overview

| File | Responsibility |
|------|---------------|
| `base.py` | ABCs (`ITaskScheduler`, `IUserKeyBuilder`), dataclasses (`TaskConfig`, `TaskInfo`, `UserKeyConfig`), enums (`TriggerMode`, `TaskStatus`), type alias `TaskHandler` |
| `redis_scheduler.py` | `RedisTaskScheduler` -- the only `ITaskScheduler` implementation; global singleton via `get_scheduler()` / `init_scheduler()` |
| `user_key_builder.py` | `UserKeyBuilder` -- builds composite user keys from `(user_id, device_id, agent_id)` dimensions |
| `__init__.py` | Re-exports all public symbols |

## Key Classes and Functions

### Enums

```python
class TriggerMode(str, Enum):
    USER_ACTIVITY = "user_activity"  # Triggered per-user on data push, delayed execution
    PERIODIC = "periodic"            # Fixed interval, global (not user-specific)

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
```

### TaskConfig (dataclass)

Configuration for a registered task type. Stored in Redis as a hash.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | -- | Unique task type name |
| `enabled` | `bool` | `True` | Whether the scheduler processes this type |
| `trigger_mode` | `TriggerMode` | `USER_ACTIVITY` | Trigger mode |
| `interval` | `int` | `1800` | Seconds between executions (delay for user_activity, period for periodic) |
| `timeout` | `int` | `300` | Distributed lock / execution timeout |
| `task_ttl` | `int` | `7200` | Redis key TTL for task state |
| `max_retries` | `int` | `3` | Max retry attempts |
| `description` | `str` | `""` | Human-readable description |

Methods: `to_dict() -> Dict[str, Any]`, `from_dict(data) -> TaskConfig`

### TaskInfo (dataclass)

Represents a single scheduled task instance for one user.

| Field | Type | Description |
|-------|------|-------------|
| `task_type` | `str` | Registered task type name |
| `user_key` | `str` | Composite key from UserKeyBuilder |
| `user_id` | `str` | Parsed user identifier |
| `device_id` | `Optional[str]` | Parsed device identifier |
| `agent_id` | `Optional[str]` | Parsed agent identifier |
| `status` | `TaskStatus` | Current status |
| `created_at` | `int` | Unix timestamp of creation |
| `scheduled_at` | `int` | Unix timestamp when task becomes eligible |
| `last_activity` | `int` | Last user activity timestamp |
| `retry_count` | `int` | Current retry attempt count (default `0`) |
| `lock_token` | `Optional[str]` | Distributed lock token (set when acquired) |

Methods: `to_dict() -> Dict[str, str]`, `from_dict(data) -> TaskInfo`

### UserKeyConfig (dataclass)

Controls which dimensions are included in composite user keys.

| Field | Type | Default |
|-------|------|---------|
| `use_user_id` | `bool` | `True` (always) |
| `use_device_id` | `bool` | `True` |
| `use_agent_id` | `bool` | `True` |
| `default_device_id` | `str` | `"default"` |
| `default_agent_id` | `str` | `"default"` |

Methods: `from_dict(data) -> UserKeyConfig`

### ITaskScheduler (ABC)

Async interface for task scheduling. All methods except `register_handler()` and `is_running()` are async.

| Method | Signature | Description |
|--------|-----------|-------------|
| `register_task_type` | `async (config: TaskConfig) -> bool` | Register a task type in Redis |
| `register_handler` | `(task_type: str, handler: TaskHandler) -> bool` | Register in-memory handler function |
| `schedule_user_task` | `async (task_type: str, user_id: str, device_id?: str, agent_id?: str) -> bool` | Schedule a user_activity task |
| `get_pending_task` | `async (task_type: str) -> Optional[TaskInfo]` | Get and lock a due task |
| `complete_task` | `async (task_type: str, user_key: str, lock_token: str, success: bool) -> None` | Mark done, release lock |
| `get_task_config` | `async (task_type: str) -> Optional[TaskConfig]` | Read task config from in-memory cache, falling back to Redis |
| `start` | `async () -> None` | Start background executor loop |
| `stop` | `async (timeout: float = 30.0) -> None` | Stop scheduler gracefully; waits up to `timeout` seconds, then force-cancels |
| `is_running` | `() -> bool` | Check running state |

### IUserKeyBuilder (ABC)

| Method | Signature |
|--------|-----------|
| `build_key` | `(user_id: str, device_id?: str, agent_id?: str) -> str` |
| `parse_key` | `(user_key: str) -> Dict[str, Optional[str]]` |
| `get_key_dimensions` | `() -> List[str]` |

### UserKeyBuilder

Implements `IUserKeyBuilder`. Separator: `":"`. Modes:
- 3-key (default): `user_id:device_id:agent_id`
- 2-key (`use_agent_id=False`): `user_id:device_id`
- 1-key (`use_device_id=False`): `user_id`

Additional methods: `from_dict(cls, config_dict: Dict) -> UserKeyBuilder`, `get_key_count() -> int`, properties `use_device_id`, `use_agent_id`, `default_device_id`, `default_agent_id`.

### RedisTaskScheduler

Implements `ITaskScheduler`. Stores all state in Redis via `RedisCache`.

**Constructor**: `__init__(redis_cache: RedisCache, config: Optional[Dict[str, Any]] = None)`

**Instance variables (concurrency)**:

| Variable | Type | Description |
|----------|------|-------------|
| `_check_interval` | `int` | Seconds between drain cycles. Read from `config["executor"]["check_interval"]`, default `10` |
| `_max_concurrent` | `int` | Max concurrent fire-and-forget tasks across all types. Read from `config["executor"]["max_concurrent"]`, default `5` |
| `_concurrency_sem` | `Optional[asyncio.Semaphore]` | Initialized in `_executor_loop` with value `_max_concurrent`. Acquired before claiming a task, released after execution completes (in `_run_and_release`) |
| `_in_flight` | `Set[asyncio.Task]` | Tracks all fire-and-forget `_run_and_release` tasks currently executing. Used during shutdown to await lock cleanup |
| `_task_config_cache` | `Dict[str, TaskConfig]` | In-memory cache of task configs, populated at `register_task_type()` time. `get_task_config()` reads from this first, falling back to Redis for configs registered by other instances |

**Redis key prefixes**:
| Prefix | Purpose |
|--------|---------|
| `scheduler:task_type:{name}` | Task type config hash |
| `scheduler:task:{type}:{user_key}` | Individual task state hash |
| `scheduler:queue:{type}` | Sorted set of pending tasks (score = `scheduled_at`) |
| `scheduler:last_exec:{type}:{user_key}` | Last execution timestamp (24h TTL) |
| `scheduler:lock:{type}:{user_key}` | Distributed lock |
| `scheduler:periodic:{type}` | Periodic task state hash (`last_run`, `next_run`, `status`) |
| `scheduler:fail_count:{type}:{user_key}` | Consecutive failure count (auto-expires via TTL) |

**Properties**: `user_key_builder -> UserKeyBuilder`

**Public methods (beyond ITaskScheduler)**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_queue_depths` | `async () -> Dict[str, int]` | Returns a mapping of `task_type` to pending task count. Iterates all registered task types and calls Redis `zcard` on each `scheduler:queue:{type}` sorted set. Useful for monitoring queue backlog in real time |

**Observability instrumentation**:

`_execute_task()` and `_process_periodic_tasks()` record execution metrics after each task completes (success or failure). They call `record_scheduler_execution()` from the monitoring module with the following fields:
- `task_type` -- registered task type name
- `user_key` -- composite user key (or `"global"` for periodic tasks)
- `success` -- boolean indicating whether the handler returned `True`
- `duration_ms` -- wall-clock execution time in milliseconds
- `trigger_mode` -- `"user_activity"` or `"periodic"`

Metrics are stored in an in-memory buffer and periodically persisted to the MySQL `monitoring_stage_timing` table using `stage_name="scheduler:{task_type}"`. This enables the `/api/monitoring/scheduler*` endpoints to query historical execution data.

**Key internal methods**:

| Method | Description |
|--------|-------------|
| `_collect_task_types()` | Reads `config["tasks"]` and queues enabled tasks for async registration |
| `init_task_types()` | `async` -- registers queued task configs in Redis (called from `start()`) |
| `_executor_loop()` | `async` -- coordinator that creates one `_type_worker` per registered handler + one `_periodic_worker`, then `asyncio.gather`s them. On `CancelledError`, cascades cancel to all workers and awaits in-flight tasks |
| `_type_worker(task_type)` | `async` -- independent per-type worker. Drain loop: acquires `_concurrency_sem` (1s timeout for shutdown responsiveness), calls `get_pending_task()`, fires off `_run_and_release` via `create_task` (fire-and-forget). Sleeps `check_interval` between drain cycles |
| `_run_and_release(task_type, task_info)` | `async` -- wraps `_execute_task` in `try/finally: _concurrency_sem.release()` to guarantee semaphore release |
| `_execute_task(task_type, task_info)` | `async` -- executes a single claimed task. Awaits async handler directly, calls `complete_task`. Uses `lock_released` flag + `finally` block with `asyncio.shield` to ensure lock is always released even on `CancelledError` |
| `_periodic_worker()` | `async` -- independent loop wrapping `_process_periodic_tasks()` with `check_interval` sleep. Handles `CancelledError` for clean shutdown |
| `_process_periodic_tasks()` | `async` -- iterates periodic task configs, checks `next_run`, acquires global lock, awaits async handler with `(None, None, None)` |

**Global singleton functions**:
- `get_scheduler() -> Optional[RedisTaskScheduler]`
- `set_scheduler(scheduler: RedisTaskScheduler) -> None`
- `init_scheduler(redis_cache: RedisCache, config?: Dict) -> RedisTaskScheduler` -- creates, sets, returns

### TaskHandler (type alias)

```python
TaskHandler = Callable[[str, Optional[str], Optional[str]], Awaitable[bool]]
# Signature: async (user_id, device_id, agent_id) -> success
```

## Class Hierarchy

```
ITaskScheduler (ABC)
  └── RedisTaskScheduler

IUserKeyBuilder (ABC)
  └── UserKeyBuilder
```

## Internal Data Flow

### user_activity task lifecycle

```
Push endpoint
  └─> scheduler.schedule_user_task(task_type, user_id, device_id, agent_id)
        ├─ get_task_config() -- reads in-memory cache first, falls back to Redis; returns None if not registered (silent skip)
        ├─ Check trigger_mode == USER_ACTIVITY
        ├─ build_key() -> user_key
        ├─ Check existing task in Redis (skip if PENDING/RUNNING; update via pipeline: HSET+EXPIRE in 1 round-trip)
        ├─ Check last_exec time (skip if within interval)
        ├─ Check fail_count >= max_retries (skip if exceeded; auto-resets via TTL)
        └─ Create TaskInfo, write via pipeline: HSET+EXPIRE+ZADD in 1 round-trip

_executor_loop (coordinator)
  └─> asyncio.gather(
        _type_worker("hierarchy_summary"),
        _type_worker("memory_compression"),
        _periodic_worker(),
      )

Each _type_worker (independent per-type, concurrent execution):
  while _running:
    drain loop:
      sem.acquire(timeout=1s)            # backpressure: max N total in-flight
      task = get_pending_task(type)       # Lua atomic pop (only if score <= now), orphan cleanup, acquire lock
      if no task: sem.release(); break
      _in_flight.add(create_task(         # fire-and-forget — runs concurrently
        _run_and_release(type, task)      # wraps _execute_task + sem.release in finally
      ))
    sleep(check_interval)

_execute_task(task_type, task_info):
  ├─ await handler(user_id, device_id, agent_id)  -- async handler called directly
  ├─ complete_task() -- update status, record last_exec (always), manage fail_count, release lock
  │     ├─ All Redis state updates in try block
  │     └─ release_lock() in finally -- lock always released even if state updates fail
  └─ lock_released flag + finally:
        ├─ CancelledError caught, logged, re-raised (complete_task called via finally with asyncio.shield)
        └─ On any other exception: complete_task called in finally with success=False
```

### graceful shutdown flow

```
stop(timeout=30.0)  [async]
  ├─ Sets self._running = False  (signals workers to exit after current drain cycle)
  ├─ asyncio.wait_for(asyncio.shield(executor_task), timeout=timeout)
  │     ├─ shield() prevents wait_for from cancelling the task on timeout
  │     ├─ On success: executor finished naturally, log "stopped gracefully"
  │     └─ On TimeoutError: force-cancel executor_task, await it (absorb CancelledError)
  ├─ Await remaining _in_flight tasks (ensures lock cleanup for fire-and-forget tasks)
  └─ self._executor_task = None

_executor_loop handles CancelledError
  ├─ Cancels all worker tasks
  ├─ Awaits workers with return_exceptions=True
  └─ Awaits all _in_flight tasks with return_exceptions=True

_type_worker handles CancelledError
  └─ Breaks out of drain and sleep loops cleanly

_execute_task handles CancelledError
  ├─ lock_released = False set before try block
  ├─ run_in_executor() interrupted by CancelledError -> caught, logged, re-raised
  └─ finally block: if not lock_released -> asyncio.shield(complete_task(success=False)) -> releases lock

_run_and_release handles cleanup
  └─ finally block: _concurrency_sem.release() -- semaphore always freed

complete_task lock safety
  ├─ All Redis state updates (status, last_exec, fail_count) in try block
  └─ release_lock() in finally -> lock always freed even if state updates raise

Atomic lock release (release_lock in redis_cache.py)
  └─ Lua script: checks token before DEL -> safe against releasing another instance's lock
```

Callers of `stop()`: `stop_task_scheduler()` in `component_initializer.py` is `async` and calls `await scheduler.stop()`.

### periodic task lifecycle

```
_periodic_worker (independent loop, runs alongside _type_workers)
  while _running:
    _process_periodic_tasks()
      ├─ Iterate config["tasks"] where trigger_mode == "periodic" and enabled
      ├─ Check next_run from Redis hash
      ├─ Acquire global lock (scheduler:lock:{type}:global)
      ├─ await handler(None, None, None)  -- async handler, no user context
      ├─ Release lock (asyncio.shield in finally), update next_run
      └─ On CancelledError: re-raise for clean shutdown
    sleep(check_interval)
```

## Cross-Module Dependencies

**Imports from**:
- `opencontext.storage.redis_cache.RedisCache` -- all Redis operations
- `loguru` (logger) -- used directly by `redis_scheduler.py` (not via `get_logger`)

**Depended on by**:
- `opencontext.periodic_task.base` -- imports `TaskConfig`, `TriggerMode`
- `opencontext.periodic_task.*` -- task implementations import `TriggerMode`
- `opencontext.server.push` -- calls `scheduler.schedule_user_task()` after push endpoints
- `opencontext.server.opencontext` -- initializes scheduler via `init_scheduler()`
- `opencontext.cli` -- starts/stops scheduler in lifespan

## Conventions and Constraints

- **Handlers are async**: The scheduler awaits them directly on the event loop. All `create_*_handler()` factories return async closures.
- **Periodic handlers receive `(None, None, None)`**: Tasks that need `user_id` must use `user_activity` trigger mode, not `periodic`.
- **Disabled tasks silently skip**: If `enabled: false` in config, `_collect_task_types()` skips registration, and `schedule_user_task()` returns `False` with a warning log. No error is raised.
- **Concurrent execution with global backpressure**: Each `_type_worker` drains its queue fully each cycle, firing off tasks concurrently via `create_task`. Total concurrency across all types is bounded by `_concurrency_sem` (default 5). Workers block on semaphore acquisition (1s timeout) before claiming tasks, providing natural backpressure.
- **Lock token is required for `complete_task`**: Always pass the `lock_token` from `TaskInfo` to `complete_task()`.
- **`check_interval` and `max_concurrent` read from `executor` sub-key**: Config path is `config["executor"]["check_interval"]` (default 10) and `config["executor"]["max_concurrent"]` (default 5). Not top-level scheduler config keys.
- **`last_exec` is recorded on both success and failure**: This enforces interval spacing after failures too, preventing rapid re-scheduling when a dependency (e.g., LLM) is down.
- **Consecutive failure tracking**: `fail_count` key tracks failures across task instances (since each push creates a new `TaskInfo`). When `fail_count >= max_retries`, new tasks are blocked. The key auto-expires via TTL (`max(max_retries * interval * 2, 86400)` seconds), allowing automatic recovery after transient outages. On success, `fail_count` is immediately deleted.
- **`stop()` is async and must be awaited**: The shutdown sequence sets `_running = False`, then waits up to `timeout` (default 30s) for the executor loop to finish its current work. Callers must `await stop()`. Use `timeout=0` for immediate cancellation.
- **Lock safety on cancellation**: `_execute_task()` uses a `lock_released` flag and a `finally` block with `asyncio.shield` to ensure `complete_task()` is always called even on `CancelledError`. `complete_task()` uses its own `try/finally` to guarantee `release_lock()` runs even if Redis state updates fail. `_run_and_release()` guarantees `_concurrency_sem.release()` in its own `finally` block.
