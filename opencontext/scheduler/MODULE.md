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
    CRON = "cron"                    # Defined but not implemented

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
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

Async interface for task scheduling. All methods except `register_handler()`, `stop()`, and `is_running()` are async.

| Method | Signature | Description |
|--------|-----------|-------------|
| `register_task_type` | `async (config: TaskConfig) -> bool` | Register a task type in Redis |
| `register_handler` | `(task_type: str, handler: TaskHandler) -> bool` | Register in-memory handler function |
| `schedule_user_task` | `async (task_type: str, user_id: str, device_id?: str, agent_id?: str) -> bool` | Schedule a user_activity task |
| `get_pending_task` | `async (task_type: str) -> Optional[TaskInfo]` | Get and lock a due task |
| `complete_task` | `async (task_type: str, user_key: str, lock_token: str, success: bool) -> None` | Mark done, release lock |
| `get_task_config` | `async (task_type: str) -> Optional[TaskConfig]` | Read task type config from Redis |
| `start` | `async () -> None` | Start background executor loop |
| `stop` | `() -> None` | Stop scheduler |
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

**Redis key prefixes**:
| Prefix | Purpose |
|--------|---------|
| `scheduler:task_type:{name}` | Task type config hash |
| `scheduler:task:{type}:{user_key}` | Individual task state hash |
| `scheduler:queue:{type}` | Sorted set of pending tasks (score = `scheduled_at`) |
| `scheduler:last_exec:{type}:{user_key}` | Last execution timestamp (24h TTL) |
| `scheduler:lock:{type}:{user_key}` | Distributed lock |
| `scheduler:periodic:{type}` | Periodic task state hash (`last_run`, `next_run`, `status`) |

**Properties**: `user_key_builder -> UserKeyBuilder`

**Key internal methods**:

| Method | Description |
|--------|-------------|
| `_collect_task_types()` | Reads `config["tasks"]` and queues enabled tasks for async registration |
| `init_task_types()` | `async` -- registers queued task configs in Redis (called from `start()`) |
| `_executor_loop()` | `async` -- main loop: processes user_activity tasks then periodic tasks, sleeps `check_interval` (default 10s) |
| `_process_task_type(task_type)` | `async` -- gets one pending task, runs handler via `run_in_executor`, calls `complete_task` |
| `_process_periodic_tasks()` | `async` -- iterates periodic task configs, checks `next_run`, acquires global lock, runs handler with `(None, None, None)` |

**Global singleton functions**:
- `get_scheduler() -> Optional[RedisTaskScheduler]`
- `set_scheduler(scheduler: RedisTaskScheduler) -> None`
- `init_scheduler(redis_cache: RedisCache, config?: Dict) -> RedisTaskScheduler` -- creates, sets, returns

### TaskHandler (type alias)

```python
TaskHandler = Callable[[str, Optional[str], Optional[str]], bool]
# Signature: (user_id, device_id, agent_id) -> success
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
        ├─ get_task_config() -- returns None if task type not registered/enabled (silent skip)
        ├─ Check trigger_mode == USER_ACTIVITY
        ├─ build_key() -> user_key
        ├─ Check existing task in Redis (skip if PENDING/RUNNING)
        ├─ Check last_exec time (skip if within interval)
        └─ Create TaskInfo, write to Redis hash + add to sorted set (score = now + interval)

_executor_loop (every check_interval seconds)
  └─> _process_task_type(task_type) for each registered handler
        ├─ get_pending_task() -- ZRANGEBYSCORE for due tasks, acquire distributed lock
        ├─ run_in_executor(handler, user_id, device_id, agent_id)
        └─ complete_task() -- update status, record last_exec, release lock
```

### periodic task lifecycle

```
_executor_loop
  └─> _process_periodic_tasks()
        ├─ Iterate config["tasks"] where trigger_mode == "periodic" and enabled
        ├─ Check next_run from Redis hash
        ├─ Acquire global lock (scheduler:lock:{type}:global)
        ├─ run_in_executor(handler, None, None, None)  -- no user context
        └─ Release lock, update next_run
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

- **Handlers are synchronous**: The scheduler wraps them with `run_in_executor()`. Do not make handlers async.
- **Periodic handlers receive `(None, None, None)`**: Tasks that need `user_id` must use `user_activity` trigger mode, not `periodic`.
- **Disabled tasks silently skip**: If `enabled: false` in config, `_collect_task_types()` skips registration, and `schedule_user_task()` returns `False` with a warning log. No error is raised.
- **One task per cycle per type**: `_process_task_type()` processes at most one pending task per executor loop iteration. This provides natural rate limiting.
- **Lock token is required for `complete_task`**: Always pass the `lock_token` from `TaskInfo` to `complete_task()`.
- **`check_interval` controls throughput**: Default 10s. All registered handler types are checked each cycle.
