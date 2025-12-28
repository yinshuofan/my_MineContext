#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Cleanup Periodic Task

Periodically cleans up expired or obsolete data from storage.
"""

import time
from typing import Any, Optional

from loguru import logger

from opencontext.periodic_task.base import (
    BasePeriodicTask,
    TaskContext,
    TaskResult,
)
from opencontext.scheduler.base import TaskConfig, TriggerMode


class DataCleanupTask(BasePeriodicTask):
    """
    Data Cleanup Task
    
    Cleans up expired data from storage periodically.
    This is a global task (not user-specific) that runs on a fixed schedule.
    """
    
    def __init__(
        self,
        storage: Any = None,
        interval: int = 86400,
        timeout: int = 600,
        retention_days: int = 30,
    ):
        """
        Initialize the data cleanup task.
        
        Args:
            storage: Storage backend instance
            interval: Interval in seconds between cleanups (default: 24 hours)
            timeout: Execution timeout in seconds (default: 10 min)
            retention_days: Number of days to retain data (default: 30)
        """
        super().__init__(
            name="data_cleanup",
            description="Periodically clean up expired data from storage",
            trigger_mode=TriggerMode.PERIODIC,
            interval=interval,
            timeout=timeout,
            task_ttl=86400,
            max_retries=3,
        )
        self._storage = storage
        self._retention_days = retention_days
    
    def set_storage(self, storage: Any) -> None:
        """Set the storage backend instance"""
        self._storage = storage
    
    def execute(self, context: TaskContext) -> TaskResult:
        """
        Execute data cleanup.
        
        This is a global task, so context.user_id may be None.
        
        Args:
            context: Task execution context
            
        Returns:
            TaskResult indicating success or failure
        """
        start_time = time.time()
        
        if not self._storage:
            return TaskResult.fail(
                error="Storage not configured",
                message="Data cleanup task requires storage backend"
            )
        
        logger.info(
            f"Starting data cleanup, retention_days={self._retention_days}"
        )
        
        try:
            deleted_count = 0
            
            # Clean up expired data
            if hasattr(self._storage, 'cleanup_expired_data'):
                deleted_count = self._storage.cleanup_expired_data(
                    retention_days=self._retention_days
                )
            elif hasattr(self._storage, 'delete_old_contexts'):
                # Alternative method name
                cutoff_time = time.time() - (self._retention_days * 86400)
                deleted_count = self._storage.delete_old_contexts(
                    before_timestamp=cutoff_time
                )
            else:
                logger.warning(
                    "Storage backend does not have cleanup method, skipping"
                )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Data cleanup completed, deleted {deleted_count} records "
                f"in {execution_time}ms"
            )
            
            return TaskResult.ok(
                message=f"Cleanup completed, deleted {deleted_count} records",
                data={
                    "deleted_count": deleted_count,
                    "retention_days": self._retention_days,
                    "execution_time_ms": execution_time,
                }
            )
            
        except Exception as e:
            logger.exception(f"Data cleanup failed: {e}")
            return TaskResult.fail(
                error=str(e),
                message="Data cleanup failed"
            )
    
    async def execute_async(self, context: TaskContext) -> TaskResult:
        """Async version of execute"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, context)


def create_cleanup_handler(storage: Any, retention_days: int = 30):
    """
    Create a cleanup handler function for the scheduler.
    
    Args:
        storage: Storage backend instance
        retention_days: Number of days to retain data
        
    Returns:
        Handler function compatible with TaskScheduler
    """
    task = DataCleanupTask(storage=storage, retention_days=retention_days)
    
    def handler(
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str]
    ) -> bool:
        # Global task, user info is not used
        context = TaskContext(
            user_id=user_id or "global",
            device_id=device_id,
            agent_id=agent_id,
            task_type="data_cleanup",
        )
        result = task.execute(context)
        return result.success
    
    return handler
