#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Redis-backed Task Scheduler Implementation (Async)

A stateless task scheduler that stores all state in Redis,
supporting multi-instance deployment. All Redis operations are async.
"""

import asyncio
import time
from typing import Any, Callable, Dict, Optional

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

    def __init__(
        self,
        redis_cache: RedisCache,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Redis Task Scheduler.

        Args:
            redis_cache: RedisCache instance for state storage
            config: Scheduler configuration dictionary
        """
        self._redis = redis_cache
        self._config = config or {}
        self._check_interval = self._config.get("check_interval", 10)

        # Initialize user key builder
        user_key_config = UserKeyConfig.from_dict(
            self._config.get("user_key_config", {})
        )
        self._user_key_builder = UserKeyBuilder(user_key_config)

        # Task handlers registry (in-memory, each instance has its own)
        self._task_handlers: Dict[str, TaskHandler] = {}

        # Running state
        self._running = False
        self._executor_task: Optional[asyncio.Task] = None

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
                    trigger_mode=TriggerMode(
                        task_config.get("trigger_mode", "user_activity")
                    ),
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

    def register_handler(
        self,
        task_type: str,
        handler: TaskHandler
    ) -> bool:
        """Register a handler function for a task type"""
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
        return True

    async def schedule_user_task(
        self,
        task_type: str,
        user_id: str,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None
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
            f"Scheduled {task_type} task for {user_key}, "
            f"will execute at {scheduled_at}"
        )
        return True

    async def get_pending_task(self, task_type: str) -> Optional[TaskInfo]:
        """
        Get a pending task ready for execution (async).

        Acquires a distributed lock to prevent duplicate execution.

        Returns:
            TaskInfo if a task is available, None otherwise
        """
        task_config = await self.get_task_config(task_type)
        if not task_config:
            return None

        queue_key = f"{self.QUEUE_PREFIX}{task_type}"
        timeout = task_config.timeout
        now = int(time.time())

        # Get all due tasks (score <= now)
        tasks = await self._redis.zrangebyscore(queue_key, 0, now)

        for user_key in tasks:
            # Try to acquire distributed lock
            lock_key = f"{self.LOCK_PREFIX}{task_type}:{user_key}"
            lock_token = await self._redis.acquire_lock(
                lock_key,
                timeout=timeout,
                blocking=False
            )

            if not lock_token:
                # Another instance is processing, skip
                continue

            # Update task status
            task_key = f"{self.TASK_PREFIX}{task_type}:{user_key}"
            await self._redis.hset(task_key, "status", TaskStatus.RUNNING.value)

            # Remove from queue
            await self._redis.zrem(queue_key, user_key)

            # Parse user_key
            key_parts = self._user_key_builder.parse_key(user_key)

            task_info = TaskInfo(
                task_type=task_type,
                user_key=user_key,
                user_id=key_parts.get("user_id") or "",
                device_id=key_parts.get("device_id"),
                agent_id=key_parts.get("agent_id"),
                status=TaskStatus.RUNNING,
                lock_token=lock_token,
            )

            return task_info

        return None

    async def complete_task(
        self,
        task_type: str,
        user_key: str,
        lock_token: str,
        success: bool = True
    ) -> None:
        """Mark a task as completed and release the lock (async)"""
        task_key = f"{self.TASK_PREFIX}{task_type}:{user_key}"
        lock_key = f"{self.LOCK_PREFIX}{task_type}:{user_key}"
        last_exec_key = f"{self.LAST_EXEC_PREFIX}{task_type}:{user_key}"

        # Update task status
        status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        await self._redis.hset(task_key, "status", status.value)

        # Record execution time
        if success:
            await self._redis.set(last_exec_key, str(int(time.time())))
            await self._redis.expire(last_exec_key, 86400)  # 24 hours

        # Release lock
        await self._redis.release_lock(lock_key, lock_token)

        logger.info(
            f"Task {task_type} for {user_key} "
            f"{'completed' if success else 'failed'}"
        )

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
        logger.info(
            f"Starting task scheduler, check interval: {self._check_interval}s"
        )

        self._executor_task = asyncio.create_task(self._executor_loop())

    async def _executor_loop(self) -> None:
        """Main executor loop"""
        while self._running:
            try:
                # Process user_activity tasks
                for task_type in self._task_handlers.keys():
                    await self._process_task_type(task_type)

                # Process periodic tasks
                await self._process_periodic_tasks()

            except Exception as e:
                logger.exception(f"Error in task scheduler: {e}")

            await asyncio.sleep(self._check_interval)

    async def _process_task_type(self, task_type: str) -> None:
        """Process tasks of a specific type (async)"""
        task_info = await self.get_pending_task(task_type)
        if not task_info:
            return

        handler = self._task_handlers.get(task_type)
        if not handler:
            logger.warning(f"No handler for task type: {task_type}")
            return

        try:
            logger.info(f"Executing {task_type} for {task_info.user_key}")

            # Execute in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                handler,
                task_info.user_id,
                task_info.device_id,
                task_info.agent_id
            )

            await self.complete_task(
                task_type,
                task_info.user_key,
                task_info.lock_token or "",
                success
            )
        except Exception as e:
            logger.exception(
                f"Task {task_type} failed for {task_info.user_key}: {e}"
            )
            await self.complete_task(
                task_type,
                task_info.user_key,
                task_info.lock_token or "",
                False
            )

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
            lock_token = await self._redis.acquire_lock(
                lock_key,
                timeout=timeout,
                blocking=False
            )
            if not lock_token:
                continue

            try:
                # Update state
                interval = task_config.get("interval", 3600)
                await self._redis.hmset(periodic_key, {
                    "last_run": str(now),
                    "next_run": str(now + interval),
                    "status": "running"
                })

                # Execute task
                logger.info(f"Executing periodic task: {task_type}")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, handler, None, None, None)

                await self._redis.hset(periodic_key, "status", "idle")
            except Exception as e:
                logger.exception(f"Periodic task {task_type} failed: {e}")
                await self._redis.hset(periodic_key, "status", "failed")
            finally:
                await self._redis.release_lock(lock_key, lock_token)

    def stop(self) -> None:
        """Stop the scheduler"""
        self._running = False
        if self._executor_task:
            self._executor_task.cancel()
            self._executor_task = None
        logger.info("Task scheduler stopped")

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
    redis_cache: RedisCache,
    config: Optional[Dict[str, Any]] = None
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
