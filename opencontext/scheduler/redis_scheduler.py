#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Redis-backed Task Scheduler Implementation (Async)

A stateless task scheduler that stores all state in Redis,
supporting multi-instance deployment. All Redis operations are async.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from opencontext.scheduler.base import (
    ITaskScheduler,
    TaskConfig,
    TaskHandler,
    TaskInfo,
    TaskStatus,
    TriggerMode,
    UserKeyConfig,
)
from opencontext.scheduler.user_key_builder import UserKeyBuilder
from opencontext.storage.redis_cache import RedisCache

# Lua script: atomically pop lowest-score member only if score <= max_score.
# Returns {member, score} if a due task exists, nil otherwise.
# Runs atomically on Redis server — no race window between check and remove.
_CONDITIONAL_ZPOPMIN_LUA = """
local result = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 1)
if #result == 0 then return nil end
local member = result[1]
local score = redis.call('ZSCORE', KEYS[1], member)
redis.call('ZREM', KEYS[1], member)
return {member, score}
"""


class RedisTaskScheduler(ITaskScheduler):
    """
    Redis-backed Task Scheduler (Async)

    A stateless task scheduler that stores all state in Redis,
    enabling multi-instance deployment with distributed task execution.

    Features:
    - Stateless: All state stored in Redis
    - Multi-instance: Distributed lock prevents duplicate execution
    - Multiple trigger modes: user_activity, periodic
    - Configurable user key dimensions: user_id, device_id, agent_id
    - All Redis operations are async (non-blocking)
    """

    # Redis Key prefixes
    TASK_TYPE_PREFIX = "scheduler:task_type:"
    TASK_PREFIX = "scheduler:task:"
    QUEUE_PREFIX = "scheduler:queue:"
    LAST_EXEC_PREFIX = "scheduler:last_exec:"
    LOCK_PREFIX = "scheduler:lock:"
    PERIODIC_PREFIX = "scheduler:periodic:"
    FAIL_COUNT_PREFIX = "scheduler:fail_count:"

    def __init__(self, redis_cache: RedisCache, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Redis Task Scheduler.

        Args:
            redis_cache: RedisCache instance for state storage
            config: Scheduler configuration dictionary
        """
        self._redis = redis_cache
        self._config = config or {}

        # Read from executor sub-key, matching the actual YAML structure
        executor_config = self._config.get("executor", {})
        self._check_interval = executor_config.get("check_interval", 10)
        self._max_concurrent = executor_config.get("max_concurrent", 5)

        # Initialize user key builder
        user_key_config = UserKeyConfig.from_dict(self._config.get("user_key_config", {}))
        self._user_key_builder = UserKeyBuilder(user_key_config)

        # Task handlers registry (in-memory, each instance has its own)
        self._task_handlers: Dict[str, TaskHandler] = {}

        # Running state
        self._running = False
        self._executor_task: Optional[asyncio.Task] = None

        # Concurrency control for fire-and-forget task execution
        self._concurrency_sem: Optional[asyncio.Semaphore] = None
        self._in_flight: Set[asyncio.Task] = set()

        # Store pending task type configs for async initialization
        self._pending_task_configs: list = []
        self._collect_task_types()

    def _collect_task_types(self) -> None:
        """Collect task types from configuration for async initialization"""
        tasks_config = self._config.get("tasks", {})
        for task_name, task_config in tasks_config.items():
            if task_config.get("enabled", False):
                config = TaskConfig(
                    name=task_name,
                    enabled=True,
                    trigger_mode=TriggerMode(task_config.get("trigger_mode", "user_activity")),
                    interval=task_config.get("interval", 1800),
                    timeout=task_config.get("timeout", 300),
                    task_ttl=task_config.get("task_ttl", 7200),
                    max_retries=task_config.get("max_retries", 3),
                    description=task_config.get("description", ""),
                )
                self._pending_task_configs.append(config)

    async def init_task_types(self) -> None:
        """Initialize task types in Redis (async)"""
        for config in self._pending_task_configs:
            await self.register_task_type(config)
        self._pending_task_configs.clear()

    @property
    def user_key_builder(self) -> UserKeyBuilder:
        """Get the user key builder instance"""
        return self._user_key_builder

    async def register_task_type(self, config: TaskConfig) -> bool:
        """Register a new task type in Redis (async)"""
        try:
            key = f"{self.TASK_TYPE_PREFIX}{config.name}"
            await self._redis.hmset(key, config.to_dict())
            logger.info(f"Registered task type: {config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register task type {config.name}: {e}")
            return False

    def register_handler(self, task_type: str, handler: TaskHandler) -> bool:
        """Register a handler function for a task type"""
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
        return True

    async def schedule_user_task(
        self,
        task_type: str,
        user_id: str,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Schedule a task for a specific user (async).

        For user_activity trigger mode, creates a delayed task.
        If a task already exists for the user, won't create duplicate.

        Returns:
            True if a new task was created, False if task already exists
        """
        # Get task type configuration
        task_config = await self.get_task_config(task_type)
        if not task_config:
            logger.warning(f"Task type {task_type} not found")
            return False

        if task_config.trigger_mode != TriggerMode.USER_ACTIVITY:
            logger.warning(
                f"Task type {task_type} is not user_activity mode, "
                f"cannot schedule via schedule_user_task()"
            )
            return False

        # Build user key
        user_key = self._user_key_builder.build_key(user_id, device_id, agent_id)

        # Redis keys
        task_key = f"{self.TASK_PREFIX}{task_type}:{user_key}"
        queue_key = f"{self.QUEUE_PREFIX}{task_type}"
        last_exec_key = f"{self.LAST_EXEC_PREFIX}{task_type}:{user_key}"

        interval = task_config.interval
        task_ttl = task_config.task_ttl

        # Check if task already exists
        existing = await self._redis.hgetall(task_key)
        if existing:
            status = existing.get("status", "")
            if status in (TaskStatus.PENDING.value, TaskStatus.RUNNING.value):
                # Task already exists, update activity time and refresh TTL
                now = int(time.time())
                await self._redis.hset(task_key, "last_activity", str(now))
                await self._redis.expire(task_key, task_ttl)
                logger.debug(f"Task {task_type} already exists for {user_key}")
                return False

        # Check last execution time
        last_exec = await self._redis.get(last_exec_key)
        if last_exec:
            last_exec_time = int(last_exec)
            if time.time() - last_exec_time < interval:
                logger.debug(
                    f"Skipping {task_type} for {user_key}, "
                    f"last executed {int(time.time() - last_exec_time)}s ago"
                )
                return False

        # Check consecutive failure count against max_retries
        fail_count_key = f"{self.FAIL_COUNT_PREFIX}{task_type}:{user_key}"
        fail_count_str = await self._redis.get(fail_count_key)
        if fail_count_str:
            fail_count = int(fail_count_str)
            # Defensive: ensure TTL exists (protect against orphaned keys)
            key_ttl = await self._redis.ttl(fail_count_key)
            if key_ttl == -1:  # No expiry set — re-apply default
                await self._redis.expire(fail_count_key, 86400)
            if fail_count >= task_config.max_retries:
                logger.debug(
                    f"Skipping {task_type} for {user_key}, "
                    f"consecutive failures ({fail_count}) >= "
                    f"max_retries ({task_config.max_retries})"
                )
                return False

        # Create new task
        now = int(time.time())
        scheduled_at = now + interval

        # Parse user_key to get dimensions
        key_parts = self._user_key_builder.parse_key(user_key)

        task_info = TaskInfo(
            task_type=task_type,
            user_key=user_key,
            user_id=key_parts.get("user_id") or user_id,
            device_id=key_parts.get("device_id") or device_id,
            agent_id=key_parts.get("agent_id") or agent_id,
            status=TaskStatus.PENDING,
            created_at=now,
            scheduled_at=scheduled_at,
            last_activity=now,
            retry_count=0,
        )

        # Write task state
        await self._redis.hmset(task_key, task_info.to_dict())
        await self._redis.expire(task_key, task_ttl)

        # Add to task queue (sorted set with scheduled_at as score)
        await self._redis.zadd(queue_key, {user_key: scheduled_at})

        logger.info(
            f"Scheduled {task_type} task for {user_key}, " f"will execute at {scheduled_at}"
        )
        return True

    async def get_pending_task(self, task_type: str) -> Optional[TaskInfo]:
        """
        Get a pending task ready for execution (async).

        Uses a Lua script for atomic conditional pop — only removes a task from
        the queue if its scheduled time has arrived. Each instance claims a unique
        task with no contention.

        Returns:
            TaskInfo if a task is available, None otherwise
        """
        task_config = await self.get_task_config(task_type)
        if not task_config:
            return None

        queue_key = f"{self.QUEUE_PREFIX}{task_type}"
        timeout = task_config.timeout
        now = int(time.time())

        max_attempts = 20  # Bound orphan cleanup per cycle

        for _ in range(max_attempts):
            # Atomically pop earliest task only if due (score <= now)
            result = await self._redis.eval_lua(
                _CONDITIONAL_ZPOPMIN_LUA, keys=[queue_key], args=[now]
            )
            if not result:
                return None  # Queue empty or nothing due

            user_key = result[0].decode("utf-8") if isinstance(result[0], bytes) else result[0]
            score = float(result[1].decode("utf-8") if isinstance(result[1], bytes) else result[1])

            # Orphan check: task hash expired but queue entry remained
            task_key = f"{self.TASK_PREFIX}{task_type}:{user_key}"
            task_exists = await self._redis.exists(task_key)
            if not task_exists:
                # Already removed by Lua, loop to next
                logger.debug(f"Cleaned orphaned queue entry for {task_type}:{user_key}")
                continue

            # Acquire lock (safety net — protects against crashed previous execution
            # whose lock hasn't expired. Near-impossible with atomic pop but defensive.)
            lock_key = f"{self.LOCK_PREFIX}{task_type}:{user_key}"
            lock_token = await self._redis.acquire_lock(lock_key, timeout=timeout, blocking=False)
            if not lock_token:
                # Previous execution still holds lock; put task back for later
                await self._redis.zadd(queue_key, {user_key: score})
                continue

            # Mark as running
            await self._redis.hset(task_key, "status", TaskStatus.RUNNING.value)

            # Parse user_key and return TaskInfo
            key_parts = self._user_key_builder.parse_key(user_key)

            return TaskInfo(
                task_type=task_type,
                user_key=user_key,
                user_id=key_parts.get("user_id") or "",
                device_id=key_parts.get("device_id"),
                agent_id=key_parts.get("agent_id"),
                status=TaskStatus.RUNNING,
                lock_token=lock_token,
            )

        return None

    async def complete_task(
        self, task_type: str, user_key: str, lock_token: str, success: bool = True
    ) -> None:
        """Mark a task as completed and release the lock (async)"""
        task_key = f"{self.TASK_PREFIX}{task_type}:{user_key}"
        lock_key = f"{self.LOCK_PREFIX}{task_type}:{user_key}"
        last_exec_key = f"{self.LAST_EXEC_PREFIX}{task_type}:{user_key}"
        fail_count_key = f"{self.FAIL_COUNT_PREFIX}{task_type}:{user_key}"

        try:
            # Update task status
            status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            await self._redis.hset(task_key, "status", status.value)

            # Always record last_exec to enforce interval spacing on both success and failure
            await self._redis.set(last_exec_key, str(int(time.time())))
            await self._redis.expire(last_exec_key, 86400)  # 24 hours

            if success:
                # Reset consecutive failure count on success
                await self._redis.delete(fail_count_key)
            else:
                # Increment consecutive failure count
                task_config = await self.get_task_config(task_type)
                max_retries = task_config.max_retries if task_config else 3
                interval = task_config.interval if task_config else 1800

                count = await self._redis.incr(fail_count_key)
                fail_count_ttl = max(max_retries * interval * 2, 86400)
                await self._redis.expire(fail_count_key, fail_count_ttl)

                logger.warning(
                    f"Task {task_type} for {user_key} failed "
                    f"(consecutive failures: {count}/{max_retries})"
                )
        except Exception as e:
            logger.exception(f"Error updating task state for {task_type}:{user_key}: {e}")
        finally:
            # Always release lock, even if state updates failed.
            # Use asyncio.shield() to protect against CancelledError propagation:
            # in Python 3.10, cancellation is "sticky" — after CancelledError is raised,
            # subsequent await calls in the same task also raise CancelledError.
            # shield() runs release_lock as an independent task unaffected by cancellation.
            try:
                await asyncio.shield(self._redis.release_lock(lock_key, lock_token))
            except (asyncio.CancelledError, Exception) as e:
                logger.error(f"Failed to release lock for {task_type}:{user_key}: {e}")

        logger.info(f"Task {task_type} for {user_key} {'completed' if success else 'failed'}")

    async def get_task_config(self, task_type: str) -> Optional[TaskConfig]:
        """Get configuration for a task type (async)"""
        key = f"{self.TASK_TYPE_PREFIX}{task_type}"
        data = await self._redis.hgetall(key)
        if not data:
            return None
        return TaskConfig.from_dict(data)

    async def start(self) -> None:
        """Start the scheduler background executor"""
        if self._running:
            logger.warning("Scheduler is already running")
            return

        # Initialize task types in Redis
        await self.init_task_types()

        self._running = True
        logger.info(f"Starting task scheduler, check interval: {self._check_interval}s")

        self._executor_task = asyncio.create_task(self._executor_loop())

    async def _executor_loop(self) -> None:
        """
        Coordinator that launches independent workers per task type.

        Each worker drains its queue concurrently (fire-and-forget) bounded by
        a global semaphore. A separate periodic worker handles global tasks.
        """
        self._concurrency_sem = asyncio.Semaphore(self._max_concurrent)

        workers: List[asyncio.Task] = []
        for task_type in self._task_handlers:
            workers.append(asyncio.create_task(self._type_worker(task_type)))
        workers.append(asyncio.create_task(self._periodic_worker()))

        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            for t in workers:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            # Also await all in-flight task executions for lock cleanup
            if self._in_flight:
                await asyncio.gather(*list(self._in_flight), return_exceptions=True)

    async def _type_worker(self, task_type: str) -> None:
        """
        Independent worker for a single task type.

        Drain loop: claims tasks and fires them off concurrently via
        create_task, bounded by the global concurrency semaphore.
        Then sleeps for check_interval before the next drain cycle.
        """
        while self._running:
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

    async def _run_and_release(self, task_type: str, task_info: TaskInfo) -> None:
        """Wrapper that executes a task and guarantees semaphore release."""
        try:
            await self._execute_task(task_type, task_info)
        finally:
            self._concurrency_sem.release()

    async def _execute_task(self, task_type: str, task_info: TaskInfo) -> None:
        """
        Execute a single claimed task.

        Preserves the exact lock lifecycle from the original _process_task_type:
        lock_released flag, finally block with asyncio.shield for cleanup.
        """
        handler = self._task_handlers.get(task_type)
        if not handler:
            logger.warning(f"No handler for task type: {task_type}")
            return

        lock_released = False
        exec_start = time.time()
        exec_success = False
        exec_error = ""
        try:
            logger.info(f"Executing {task_type} for {task_info.user_key}")

            # Execute in thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(
                None, handler, task_info.user_id, task_info.device_id, task_info.agent_id
            )

            await self.complete_task(
                task_type, task_info.user_key, task_info.lock_token or "", success
            )
            lock_released = True
            exec_success = success
        except asyncio.CancelledError:
            logger.warning(f"Task {task_type} for {task_info.user_key} cancelled during shutdown")
            exec_error = "cancelled_during_shutdown"
            raise  # Re-raise so _run_and_release's finally still fires
        except Exception as e:
            logger.exception(f"Task {task_type} failed for {task_info.user_key}: {e}")
            exec_error = str(e)
        finally:
            if not lock_released:
                # Use asyncio.shield() so complete_task runs as an independent task,
                # unaffected by the outer task's cancellation state. Without this,
                # CancelledError would propagate into every await inside complete_task.
                try:
                    await asyncio.shield(
                        self.complete_task(
                            task_type, task_info.user_key, task_info.lock_token or "", False
                        )
                    )
                except (asyncio.CancelledError, Exception) as cleanup_err:
                    logger.error(
                        f"Failed to release lock during cleanup for "
                        f"{task_type}:{task_info.user_key}: {cleanup_err}"
                    )

            # Record execution metrics
            duration_ms = int((time.time() - exec_start) * 1000)
            try:
                from opencontext.monitoring import record_scheduler_execution

                record_scheduler_execution(
                    task_type=task_type,
                    user_key=task_info.user_key,
                    success=exec_success,
                    duration_ms=duration_ms,
                    trigger_mode="user_activity",
                    error_message=exec_error,
                )
            except Exception:
                pass  # Never let monitoring break the scheduler

    async def _periodic_worker(self) -> None:
        """Independent worker loop for periodic (global) tasks."""
        while self._running:
            try:
                await self._process_periodic_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in periodic worker: {e}")

            if self._running:
                try:
                    await asyncio.sleep(self._check_interval)
                except asyncio.CancelledError:
                    break

    async def _process_periodic_tasks(self) -> None:
        """Process periodic tasks (global tasks, not user-specific) (async)"""
        tasks_config = self._config.get("tasks", {})

        for task_type, task_config in tasks_config.items():
            if not task_config.get("enabled", False):
                continue
            if task_config.get("trigger_mode") != TriggerMode.PERIODIC.value:
                continue

            # Check if handler is registered
            handler = self._task_handlers.get(task_type)
            if not handler:
                continue

            periodic_key = f"{self.PERIODIC_PREFIX}{task_type}"
            lock_key = f"{self.LOCK_PREFIX}{task_type}:global"

            # Check if it's time to execute
            periodic_state = await self._redis.hgetall(periodic_key)
            now = int(time.time())

            next_run = int(periodic_state.get("next_run", 0)) if periodic_state else 0
            if now < next_run:
                continue

            # Try to acquire lock
            timeout = task_config.get("timeout", 300)
            lock_token = await self._redis.acquire_lock(lock_key, timeout=timeout, blocking=False)
            if not lock_token:
                continue

            exec_start = time.time()
            exec_success = False
            exec_error = ""
            try:
                # Update state
                interval = task_config.get("interval", 3600)
                await self._redis.hmset(
                    periodic_key,
                    {"last_run": str(now), "next_run": str(now + interval), "status": "running"},
                )
                await self._redis.expire(periodic_key, interval * 3)

                # Execute task
                logger.info(f"Executing periodic task: {task_type}")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, handler, None, None, None)

                await self._redis.hset(periodic_key, "status", "idle")
                await self._redis.expire(periodic_key, interval * 3)
                exec_success = True
            except asyncio.CancelledError:
                logger.warning(f"Periodic task {task_type} cancelled during shutdown")
                raise
            except Exception as e:
                logger.exception(f"Periodic task {task_type} failed: {e}")
                exec_error = str(e)
                await self._redis.hset(periodic_key, "status", "failed")
                await self._redis.expire(periodic_key, interval * 3)
            finally:
                try:
                    await asyncio.shield(self._redis.release_lock(lock_key, lock_token))
                except (asyncio.CancelledError, Exception) as e:
                    logger.error(f"Failed to release periodic lock for {task_type}: {e}")

                # Record execution metrics
                duration_ms = int((time.time() - exec_start) * 1000)
                try:
                    from opencontext.monitoring import record_scheduler_execution

                    record_scheduler_execution(
                        task_type=task_type,
                        user_key="global",
                        success=exec_success,
                        duration_ms=duration_ms,
                        trigger_mode="periodic",
                        error_message=exec_error,
                    )
                except Exception:
                    pass

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the scheduler gracefully.

        Sets running flag to False, then waits up to `timeout` seconds
        for the executor loop to finish its current cycle. If the executor
        doesn't finish in time, it is cancelled.

        Args:
            timeout: Maximum seconds to wait for in-flight tasks to complete.
                     Default 30s. Set to 0 for immediate cancellation.
        """
        if not self._running and not self._executor_task:
            return

        self._running = False
        logger.info("Stopping task scheduler (graceful)...")

        if self._executor_task and not self._executor_task.done():
            if timeout > 0:
                try:
                    await asyncio.wait_for(asyncio.shield(self._executor_task), timeout=timeout)
                    logger.info("Task scheduler stopped gracefully")
                except asyncio.TimeoutError:
                    logger.warning(f"Scheduler did not stop within {timeout}s, cancelling")
                    self._executor_task.cancel()
                    try:
                        await self._executor_task
                    except (asyncio.CancelledError, Exception):
                        pass
                except asyncio.CancelledError:
                    self._executor_task.cancel()
                    try:
                        await self._executor_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    raise
            else:
                self._executor_task.cancel()
                try:
                    await self._executor_task
                except (asyncio.CancelledError, Exception):
                    pass

        # Await any remaining in-flight task executions to ensure lock cleanup
        if self._in_flight:
            logger.info(f"Waiting for {len(self._in_flight)} in-flight tasks to clean up...")
            await asyncio.gather(*list(self._in_flight), return_exceptions=True)
            self._in_flight.clear()

        self._executor_task = None
        logger.info("Task scheduler stopped")

    async def get_queue_depths(self) -> Dict[str, int]:
        """Get current queue depth for each registered task type."""
        depths = {}
        for task_type in self._task_handlers:
            queue_key = f"{self.QUEUE_PREFIX}{task_type}"
            depths[task_type] = await self._redis.zcard(queue_key)
        return depths

    def is_running(self) -> bool:
        """Check if the scheduler is running"""
        return self._running


# Global scheduler instance
_scheduler: Optional[RedisTaskScheduler] = None


def get_scheduler() -> Optional[RedisTaskScheduler]:
    """Get the global scheduler instance"""
    return _scheduler


def set_scheduler(scheduler: RedisTaskScheduler) -> None:
    """Set the global scheduler instance"""
    global _scheduler
    _scheduler = scheduler


def init_scheduler(
    redis_cache: RedisCache, config: Optional[Dict[str, Any]] = None
) -> RedisTaskScheduler:
    """
    Initialize and set the global scheduler instance.

    Args:
        redis_cache: RedisCache instance
        config: Scheduler configuration

    Returns:
        The initialized scheduler
    """
    scheduler = RedisTaskScheduler(redis_cache, config)
    set_scheduler(scheduler)
    return scheduler
