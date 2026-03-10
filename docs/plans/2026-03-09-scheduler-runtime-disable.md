# Scheduler Runtime Disable Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When a scheduler task type is disabled via `/settings` → Apply, immediately stop all workers (including other multi-instance workers) from executing that task type, while preserving the Redis queue for future re-enablement. The monitoring page should display the paused status for each task type.

**Architecture:** Use the existing `scheduler:task_type:{type}` Redis hash `enabled` field as the authoritative runtime flag. On reload, mark disabled tasks as `enabled: "false"` in Redis. Type workers and periodic workers check this flag each cycle before consuming tasks. The `/api/monitoring/scheduler/queues` endpoint returns enabled status alongside queue depths, and the monitoring frontend displays a "已暂停" badge for disabled task types.

**Tech Stack:** Python 3.10+, Redis (async via `RedisCache`), `asyncio`, JavaScript (monitoring frontend)

---

### Task 1: Add `_sync_disabled_task_types()` and modify `init_task_types()`

When the scheduler starts (or restarts after reload), after registering enabled task types, scan the config for all defined task types and mark any non-enabled ones as `enabled: "false"` in their existing Redis hash. This ensures the Redis `enabled` field reflects the current config state.

**Files:**
- Modify: `opencontext/scheduler/redis_scheduler.py:130-134`

**Step 1: Add `_sync_disabled_task_types()` method**

Insert the new method after `init_task_types()` (after line 134):

```python
async def _sync_disabled_task_types(self, enabled_names: Set[str]) -> None:
    """Mark task types not in enabled_names as disabled in Redis.

    Reads all task type names from config (including disabled ones).
    For any that exist in Redis but are NOT in enabled_names,
    sets their 'enabled' field to 'false'. This ensures other
    workers' runtime guards see the updated state immediately.
    """
    all_task_names = set(self._config.get("tasks", {}).keys())
    disabled_names = all_task_names - enabled_names
    for name in disabled_names:
        key = f"{self.TASK_TYPE_PREFIX}{name}"
        if await self._redis.exists(key):
            await self._redis.hset(key, field="enabled", value="false")
            logger.info(f"Marked task type as disabled in Redis: {name}")
```

**Step 2: Modify `init_task_types()` to call the sync method**

Replace lines 130-134:

```python
async def init_task_types(self) -> None:
    """Initialize task types in Redis (async)"""
    enabled_names: Set[str] = set()
    for config in self._pending_task_configs:
        await self.register_task_type(config)
        enabled_names.add(config.name)
    self._pending_task_configs.clear()

    # Mark disabled task types in Redis so other workers' runtime guards take effect
    await self._sync_disabled_task_types(enabled_names)
```

**Step 3: Add `Set` import if not already present**

Check the imports at the top of `redis_scheduler.py`. The file already imports `Dict, Optional, Set, List` from `typing` (used by `_in_flight: Set[asyncio.Task]`). No new import needed.

**Step 4: Compile-check**

Run: `python -m py_compile opencontext/scheduler/redis_scheduler.py`
Expected: No output (success)

**Step 5: Commit**

```bash
git add opencontext/scheduler/redis_scheduler.py
git commit -m "feat(scheduler): mark disabled task types in Redis on init

When the scheduler starts or restarts after config reload, task types
that are defined in config but not enabled are now marked with
enabled='false' in their Redis hash. This is the first step toward
runtime disable — other workers can read this flag to stop execution.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Add runtime guard in `_type_worker()`

Add a Redis `enabled` check at the top of each drain cycle in `_type_worker()`. If the task type is disabled in Redis, skip the drain phase and sleep. This stops all workers (including un-reloaded multi-instance workers) from executing disabled user_activity tasks within one `check_interval` cycle.

**Files:**
- Modify: `opencontext/scheduler/redis_scheduler.py:482-526` (the `_type_worker` method)

**Step 1: Add the runtime guard**

Insert after `while self._running:` (line 491) and before the drain phase comment (line 492), shifting the entire drain phase into the else branch. The modified method should look like:

```python
async def _type_worker(self, task_type: str) -> None:
    """
    Independent worker for a single task type.

    Drain loop: claims tasks and fires them off concurrently via
    create_task, bounded by the global concurrency semaphore.
    Then sleeps for check_interval before the next drain cycle.
    """
    while self._running:
        # --- runtime guard: skip drain if task type disabled in Redis ---
        try:
            enabled = await self._redis.hget(
                f"{self.TASK_TYPE_PREFIX}{task_type}", "enabled"
            )
            if enabled != "true":
                if self._running:
                    try:
                        await asyncio.sleep(self._check_interval)
                    except asyncio.CancelledError:
                        break
                continue
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Failed to check enabled status for {task_type}: {e}")
            # On Redis error, fall through to normal drain (fail-open)

        # --- drain phase: claim tasks and fire them off concurrently ---
        try:
            while self._running:
                try:
                    await asyncio.wait_for(self._concurrency_sem.acquire(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue  # re-check _running

                # Wrap the region between acquire and ownership transfer to
                # create_task so the semaphore is released on any exception.
                try:
                    task_info = await self.get_pending_task(task_type)
                except Exception:
                    self._concurrency_sem.release()
                    raise

                if not task_info:
                    self._concurrency_sem.release()
                    break  # queue empty, go to sleep

                # Fire-and-forget: task runs concurrently.
                # Semaphore ownership transfers to _run_and_release.
                t = asyncio.create_task(self._run_and_release(task_type, task_info))
                self._in_flight.add(t)
                t.add_done_callback(self._in_flight.discard)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Error in type worker for {task_type}: {e}")

        # --- sleep phase ---
        if self._running:
            try:
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
```

Key design decisions:
- **Fail-open on Redis error**: If the `HGET` fails, we fall through to normal drain. This avoids a Redis outage from disabling ALL task execution.
- **Sleep before `continue`**: When disabled, the worker sleeps `check_interval` before re-checking, avoiding a tight loop.
- **`CancelledError` handled**: Ensures clean shutdown even during the enabled check.

**Step 2: Compile-check**

Run: `python -m py_compile opencontext/scheduler/redis_scheduler.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add opencontext/scheduler/redis_scheduler.py
git commit -m "feat(scheduler): add runtime enabled guard to type workers

Type workers now check the Redis task_type enabled flag at the start
of each drain cycle. When a task type is marked disabled, the worker
skips consumption and sleeps. This ensures multi-instance workers
stop executing disabled tasks within one check_interval, while
preserving the queue for future re-enablement.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Add runtime guard in `_process_periodic_tasks()`

Add the same Redis `enabled` check inside the periodic task loop. This prevents un-reloaded workers from executing disabled periodic tasks.

**Files:**
- Modify: `opencontext/scheduler/redis_scheduler.py:623-714` (the `_process_periodic_tasks` method)

**Step 1: Add the runtime guard**

Insert after the `trigger_mode` check (line 631) and before the handler check (line 633):

```python
            # Runtime guard: check if task type is still enabled in Redis
            try:
                enabled = await self._redis.hget(
                    f"{self.TASK_TYPE_PREFIX}{task_type}", "enabled"
                )
                if enabled != "true":
                    continue
            except Exception:
                pass  # Fail-open: on Redis error, allow execution
```

The surrounding context should look like:

```python
        for task_type, config in self._task_config_cache.items():
            if config.trigger_mode != TriggerMode.PERIODIC:
                continue

            # Runtime guard: check if task type is still enabled in Redis
            try:
                enabled = await self._redis.hget(
                    f"{self.TASK_TYPE_PREFIX}{task_type}", "enabled"
                )
                if enabled != "true":
                    continue
            except Exception:
                pass  # Fail-open: on Redis error, allow execution

            # Check if handler is registered
            handler = self._task_handlers.get(task_type)
```

**Step 2: Compile-check**

Run: `python -m py_compile opencontext/scheduler/redis_scheduler.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add opencontext/scheduler/redis_scheduler.py
git commit -m "feat(scheduler): add runtime enabled guard to periodic worker

Periodic tasks now check the Redis enabled flag before execution,
matching the guard added to type workers. Prevents un-reloaded
workers from executing disabled periodic tasks.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Update MODULE.md

Document the new runtime disable behavior in the scheduler module documentation.

**Files:**
- Modify: `opencontext/scheduler/MODULE.md`

**Step 1: Update the `init_task_types()` description**

In the "Key internal methods" table (line 166), replace the `init_task_types()` row:

Old:
```
| `init_task_types()` | `async` -- registers queued task configs in Redis (called from `start()`) |
```

New:
```
| `init_task_types()` | `async` -- registers queued task configs in Redis (called from `start()`), then calls `_sync_disabled_task_types()` to mark non-enabled task types as `enabled: "false"` in Redis |
```

**Step 2: Add `_sync_disabled_task_types()` to the internal methods table**

Add a new row after `init_task_types()`:

```
| `_sync_disabled_task_types(enabled_names)` | `async` -- reads all task names from config, marks any not in `enabled_names` as `enabled: "false"` in their Redis `scheduler:task_type:{name}` hash. Called by `init_task_types()` after registration |
```

**Step 3: Update `_type_worker` description**

Replace the `_type_worker` row (line 170):

Old:
```
| `_type_worker(task_type)` | `async` -- independent per-type worker. Drain loop: acquires `_concurrency_sem` (1s timeout for shutdown responsiveness), calls `get_pending_task()`, fires off `_run_and_release` via `create_task` (fire-and-forget). Sleeps `check_interval` between drain cycles |
```

New:
```
| `_type_worker(task_type)` | `async` -- independent per-type worker. Each cycle starts with a **runtime enabled guard**: reads `enabled` from `scheduler:task_type:{type}` in Redis; if not `"true"`, skips drain and sleeps (fail-open on Redis error). Drain loop: acquires `_concurrency_sem` (1s timeout for shutdown responsiveness), calls `get_pending_task()`, fires off `_run_and_release` via `create_task` (fire-and-forget). Sleeps `check_interval` between drain cycles |
```

**Step 4: Update `_process_periodic_tasks` description**

Replace the `_process_periodic_tasks()` row (line 174):

Old:
```
| `_process_periodic_tasks()` | `async` -- iterates periodic task configs, checks `next_run`, acquires global lock, awaits async handler with `(None, None, None)` |
```

New:
```
| `_process_periodic_tasks()` | `async` -- iterates periodic task configs. For each, checks **runtime enabled guard** from Redis (fail-open on error), then checks `next_run`, acquires global lock, awaits async handler with `(None, None, None)` |
```

**Step 5: Add a new convention entry**

Add to the "Conventions and Constraints" section (after line 339):

```
- **Runtime disable via Redis `enabled` flag**: When a task type is disabled via settings reload, `init_task_types()` marks it as `enabled: "false"` in Redis. Both `_type_worker()` and `_process_periodic_tasks()` check this flag each cycle before consuming tasks. This ensures all multi-instance workers stop executing disabled tasks within one `check_interval` (default 10s), without clearing the queue — pending tasks are preserved for re-enablement. The guard is **fail-open**: Redis errors do not block task execution.
```

**Step 6: Update the user_activity task lifecycle flow**

In the data flow section (around line 246), update the `_type_worker` flow:

Old:
```
Each _type_worker (independent per-type, concurrent execution):
  while _running:
    drain loop:
```

New:
```
Each _type_worker (independent per-type, concurrent execution):
  while _running:
    runtime guard: HGET scheduler:task_type:{type} enabled
      if != "true": sleep(check_interval), continue
    drain loop:
```

**Step 7: Update the periodic task lifecycle flow**

In the periodic task lifecycle section (around line 313), update:

Old:
```
    _process_periodic_tasks()
      ├─ Iterate config["tasks"] where trigger_mode == "periodic" and enabled
```

New:
```
    _process_periodic_tasks()
      ├─ Iterate config["tasks"] where trigger_mode == "periodic" and enabled
      ├─ Runtime guard: HGET scheduler:task_type:{type} enabled, skip if != "true"
```

**Step 8: Commit**

```bash
git add opencontext/scheduler/MODULE.md
git commit -m "docs(scheduler): document runtime disable via Redis enabled flag

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Add `get_queue_status()` to scheduler and update exports

Add a new public method that returns queue depth AND enabled status per task type for all configured tasks (not just handler-registered ones). Also add a module-level fallback function for workers without a local scheduler.

**Files:**
- Modify: `opencontext/scheduler/redis_scheduler.py:779-839`
- Modify: `opencontext/scheduler/__init__.py`

**Step 1: Add `get_queue_status()` method**

Insert after `get_queue_depths()` (after line 785):

```python
async def get_queue_status(self) -> Dict[str, Dict[str, Any]]:
    """Get queue depth and enabled status for all configured task types.

    Unlike get_queue_depths() which only covers handler-registered types,
    this method covers ALL task types defined in config — including disabled
    ones that may still have pending queue entries.

    Returns:
        Dict mapping task_type to {"depth": int, "enabled": bool}
    """
    status = {}
    all_task_names = set(self._config.get("tasks", {}).keys())
    all_task_names.update(self._task_handlers.keys())

    for task_type in all_task_names:
        queue_key = f"{self.QUEUE_PREFIX}{task_type}"
        type_key = f"{self.TASK_TYPE_PREFIX}{task_type}"

        depth = await self._redis.zcard(queue_key)
        enabled = await self._redis.hget(type_key, "enabled")

        status[task_type] = {
            "depth": depth,
            "enabled": enabled == "true",
        }
    return status
```

**Step 2: Add `read_scheduler_queue_status()` module-level function**

Insert after `read_scheduler_queue_depths()` (after line 839):

```python
async def read_scheduler_queue_status(
    redis_cache: RedisCache, task_types: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Read queue depths and enabled status from Redis for given task types.

    Fallback version of get_queue_status() for processes without a local scheduler.
    """
    status = {}
    for task_type in task_types:
        queue_key = f"{RedisTaskScheduler.QUEUE_PREFIX}{task_type}"
        type_key = f"{RedisTaskScheduler.TASK_TYPE_PREFIX}{task_type}"

        depth = await redis_cache.zcard(queue_key)
        enabled = await redis_cache.hget(type_key, "enabled")

        status[task_type] = {
            "depth": depth,
            "enabled": enabled == "true",
        }
    return status
```

**Step 3: Update `__init__.py` exports**

In `opencontext/scheduler/__init__.py`, add the new function to the imports and `__all__`:

Add `read_scheduler_queue_status` to the import from `redis_scheduler` (line 26, after `read_scheduler_queue_depths`):

```python
from opencontext.scheduler.redis_scheduler import (
    RedisTaskScheduler,
    get_scheduler,
    init_scheduler,
    read_scheduler_heartbeat,
    read_scheduler_queue_depths,
    read_scheduler_queue_status,
    set_scheduler,
)
```

Add `"read_scheduler_queue_status"` to `__all__` (after line 52, after `"read_scheduler_queue_depths"`).

**Step 4: Compile-check**

Run: `python -m py_compile opencontext/scheduler/redis_scheduler.py && python -m py_compile opencontext/scheduler/__init__.py`
Expected: No output (success)

**Step 5: Commit**

```bash
git add opencontext/scheduler/redis_scheduler.py opencontext/scheduler/__init__.py
git commit -m "feat(scheduler): add get_queue_status() with enabled flag

New method returns queue depth + enabled status for all configured
task types, including disabled ones. Used by monitoring to display
paused status. Also adds read_scheduler_queue_status() fallback for
workers without a local scheduler.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Update `/api/monitoring/scheduler/queues` endpoint

Modify the queues endpoint to return per-task-type enabled status alongside queue depths. The response format changes from `{"queues": {"type": 5}}` to `{"queues": {"type": {"depth": 5, "enabled": true}}}`.

**Files:**
- Modify: `opencontext/server/routes/monitoring.py:230-261`

**Step 1: Update the endpoint handler**

Replace the entire `get_scheduler_queue_depths` function (lines 230-261):

```python
@router.get("/scheduler/queues")
async def get_scheduler_queue_depths(
    _auth: str = auth_dependency,
):
    """Get current queue depths and enabled status for all scheduler task types (real-time from Redis)"""
    try:
        import json

        from opencontext.scheduler import get_scheduler

        scheduler = get_scheduler()
        if scheduler:
            queues = await scheduler.get_queue_status()
            return {"success": True, "data": {"queues": queues}}

        # Fallback: read from Redis heartbeat + config
        from opencontext.scheduler import read_scheduler_heartbeat, read_scheduler_queue_status
        from opencontext.storage.redis_cache import get_redis_cache

        redis_cache = get_redis_cache()
        if not redis_cache:
            return {"success": True, "data": {"queues": {}, "message": "Redis not available"}}

        # Get all task types from config
        from opencontext.config import GlobalConfig

        config = GlobalConfig.get_instance().config
        all_task_types = list(config.get("scheduler", {}).get("tasks", {}).keys())

        if not all_task_types:
            # Fallback to heartbeat registered handlers
            heartbeat = await read_scheduler_heartbeat(redis_cache)
            if not heartbeat:
                return {
                    "success": True,
                    "data": {"queues": {}, "message": "Scheduler not running"},
                }
            all_task_types = json.loads(heartbeat.get("registered_handlers", "[]"))

        queues = await read_scheduler_queue_status(redis_cache, all_task_types)
        return {"success": True, "data": {"queues": queues}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue depths: {str(e)}")
```

**Step 2: Compile-check**

Run: `python -m py_compile opencontext/server/routes/monitoring.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add opencontext/server/routes/monitoring.py
git commit -m "feat(monitoring): return enabled status in scheduler queues API

The /api/monitoring/scheduler/queues endpoint now returns per-task-type
enabled status alongside queue depth. Response format changes from
{type: count} to {type: {depth: int, enabled: bool}}. Also shows all
configured task types including disabled ones.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Update monitoring frontend to show paused status

Modify `loadQueueDepths()` and `updateQueueDepths()` in the monitoring page to handle the new API response format and display a "已暂停" badge for disabled task types.

**Files:**
- Modify: `opencontext/web/templates/monitoring.html:807-895`

**Step 1: Update `loadQueueDepths()` function**

Replace lines 807-822:

```javascript
async function loadQueueDepths() {
    try {
        const resp = await fetchData('/api/monitoring/scheduler/queues');
        const queues = resp.data.queues || {};
        updateQueueDepths(queues);

        // 更新概览卡片中的队列深度总数
        let totalDepth = 0;
        for (const key of Object.keys(queues)) {
            const val = queues[key];
            totalDepth += (typeof val === 'object') ? (val.depth || 0) : val;
        }
        document.getElementById('schedulerQueueDepth').textContent = totalDepth.toLocaleString();

        document.getElementById('queueLastUpdate').textContent =
            '上次更新: ' + new Date().toLocaleTimeString('zh-CN');
    } catch (error) {
        console.error('Failed to load queue depths:', error);
    }
}
```

**Step 2: Update `updateQueueDepths()` function**

Replace lines 870-895:

```javascript
function updateQueueDepths(queues) {
    const container = document.getElementById('queueDepthContainer');
    const taskTypes = Object.keys(queues);

    if (taskTypes.length === 0) {
        container.innerHTML = '<span class="text-muted">暂无队列数据</span>';
        return;
    }

    let html = '<div class="d-flex flex-wrap gap-3">';
    for (const taskType of taskTypes) {
        const val = queues[taskType];
        // Support both new format {depth, enabled} and legacy format (number)
        const count = (typeof val === 'object') ? (val.depth || 0) : val;
        const enabled = (typeof val === 'object') ? val.enabled : true;

        let badgeClass = 'bg-success';
        if (count > 5) {
            badgeClass = 'bg-danger';
        } else if (count > 0) {
            badgeClass = 'bg-warning text-dark';
        }

        const pausedBadge = !enabled
            ? ' <span class="badge bg-secondary">已暂停</span>'
            : '';

        html += `<div class="d-flex align-items-center gap-2">
            <span>${taskType}</span>
            <span class="badge ${badgeClass} queue-badge">${count}</span>${pausedBadge}
        </div>`;
    }
    html += '</div>';
    container.innerHTML = html;
}
```

**Step 3: Verify in browser**

Open the monitoring page in a browser and verify:
- Enabled tasks show queue depth badge only (green/warning/danger)
- Disabled tasks show queue depth badge + gray "已暂停" badge
- The overview card's total queue depth still calculates correctly

**Step 4: Commit**

```bash
git add opencontext/web/templates/monitoring.html
git commit -m "feat(monitoring): show paused status for disabled scheduler tasks

Queue depth display now shows a '已暂停' badge next to disabled task
types, so administrators can see which tasks have pending work but
are not being consumed. Backwards-compatible with the old API format.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Update documentation

Update CLAUDE.md pitfalls and MODULE.md with the monitoring changes.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `opencontext/scheduler/MODULE.md`

**Step 1: Add pitfall entry to CLAUDE.md**

Add to the "Pitfalls and Lessons Learned" section, after the "Scheduler pitfalls" entry:

```markdown
### Scheduler task disable requires Redis `enabled` flag, not just config skip
Disabling a task type in config only prevents NEW registration and handler creation. Existing Redis queue entries and other multi-instance workers remain unaffected. The Redis `scheduler:task_type:{type}.enabled` field is the **runtime authority** — `init_task_types()` marks disabled tasks as `enabled: "false"`, and both `_type_worker()` and `_process_periodic_tasks()` check this flag each cycle. If you add a new execution path for scheduled tasks, it must also check this flag.
```

**Step 2: Add `get_queue_status` to MODULE.md public methods table**

In the "Public methods (beyond ITaskScheduler)" table (line 148), add a new row after `get_queue_depths`:

```
| `get_queue_status` | `async () -> Dict[str, Dict[str, Any]]` | Returns a mapping of `task_type` to `{"depth": int, "enabled": bool}` for ALL configured task types (including disabled ones). Unlike `get_queue_depths()` which only covers handler-registered types, this includes disabled task types that may still have pending queue entries. Used by the monitoring API |
```

**Step 3: Add `read_scheduler_queue_status` to the remote observation functions**

In the "Remote observation functions" section (after line 183), add:

```
- `read_scheduler_queue_status(redis_cache: RedisCache, task_types: List[str]) -> Dict[str, Dict[str, Any]]` -- reads queue depth and `enabled` flag for each task type from Redis. Returns `{"depth": int, "enabled": bool}` per type
```

**Step 4: Commit**

```bash
git add CLAUDE.md opencontext/scheduler/MODULE.md
git commit -m "docs: document scheduler runtime disable and monitoring changes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
