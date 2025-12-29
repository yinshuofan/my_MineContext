#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Cleanup Periodic Task

Periodically cleans up expired or obsolete data from storage using intelligent
cleanup strategies from context_merger.
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
    
    Cleans up expired data from storage periodically using intelligent
    cleanup strategies based on forgetting curves and importance scores.
    
    This task delegates to ContextMerger.intelligent_memory_cleanup() which
    uses type-specific strategies from merge_strategies.py to determine
    which contexts should be cleaned up.
    """
    
    def __init__(
        self,
        context_merger: Any = None,
        storage: Any = None,
        interval: int = 86400,
        timeout: int = 600,
        retention_days: int = 30,
    ):
        """
        Initialize the data cleanup task.
        
        Args:
            context_merger: ContextMerger instance for intelligent cleanup
            storage: Storage backend instance (fallback for simple cleanup)
            interval: Interval in seconds between cleanups (default: 24 hours)
            timeout: Execution timeout in seconds (default: 10 min)
            retention_days: Number of days to retain data (default: 30)
        """
        super().__init__(
            name="data_cleanup",
            description="Periodically clean up expired data using intelligent strategies",
            trigger_mode=TriggerMode.PERIODIC,
            interval=interval,
            timeout=timeout,
            task_ttl=86400,
            max_retries=3,
        )
        self._context_merger = context_merger
        self._storage = storage
        self._retention_days = retention_days
    
    def set_context_merger(self, context_merger: Any) -> None:
        """Set the context merger instance for intelligent cleanup"""
        self._context_merger = context_merger
    
    def set_storage(self, storage: Any) -> None:
        """Set the storage backend instance"""
        self._storage = storage
    
    def execute(self, context: TaskContext) -> TaskResult:
        """
        Execute data cleanup.
        
        This task uses ContextMerger's intelligent_memory_cleanup() method
        which applies type-specific cleanup strategies based on:
        - Forgetting curves
        - Importance scores
        - Access frequency
        - Retention periods per context type
        
        Args:
            context: Task execution context
            
        Returns:
            TaskResult indicating success or failure
        """
        start_time = time.time()
        
        logger.info(
            f"Starting data cleanup, retention_days={self._retention_days}"
        )
        
        try:
            cleanup_result = None
            
            # Primary method: Use ContextMerger's intelligent cleanup
            if self._context_merger:
                if hasattr(self._context_merger, 'intelligent_memory_cleanup'):
                    logger.info("Using intelligent memory cleanup from ContextMerger")
                    self._context_merger.intelligent_memory_cleanup()
                    cleanup_result = "intelligent_cleanup"
                elif hasattr(self._context_merger, 'cleanup_contexts_by_type'):
                    # Alternative method name
                    logger.info("Using cleanup_contexts_by_type from ContextMerger")
                    self._context_merger.cleanup_contexts_by_type()
                    cleanup_result = "type_cleanup"
            
            # Fallback: Use storage's simple cleanup methods
            if not cleanup_result and self._storage:
                deleted_count = 0
                
                if hasattr(self._storage, 'cleanup_expired_data'):
                    deleted_count = self._storage.cleanup_expired_data(
                        retention_days=self._retention_days
                    )
                    cleanup_result = "storage_cleanup"
                elif hasattr(self._storage, 'delete_old_contexts'):
                    cutoff_time = time.time() - (self._retention_days * 86400)
                    deleted_count = self._storage.delete_old_contexts(
                        before_timestamp=cutoff_time
                    )
                    cleanup_result = "storage_delete_old"
                
                if cleanup_result:
                    logger.info(f"Storage cleanup deleted {deleted_count} records")
            
            if not cleanup_result:
                logger.warning(
                    "No cleanup method available. "
                    "Ensure ContextMerger or Storage with cleanup methods is configured."
                )
                return TaskResult.fail(
                    error="No cleanup method available",
                    message="Neither ContextMerger nor Storage has cleanup methods"
                )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Data cleanup completed using {cleanup_result} "
                f"in {execution_time}ms"
            )
            
            return TaskResult.ok(
                message=f"Cleanup completed using {cleanup_result}",
                data={
                    "cleanup_method": cleanup_result,
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


def create_cleanup_handler(
    context_merger: Any = None,
    storage: Any = None,
    retention_days: int = 30
):
    """
    Create a cleanup handler function for the scheduler.
    
    Args:
        context_merger: ContextMerger instance for intelligent cleanup
        storage: Storage backend instance (fallback)
        retention_days: Number of days to retain data
        
    Returns:
        Handler function compatible with TaskScheduler
    """
    task = DataCleanupTask(
        context_merger=context_merger,
        storage=storage,
        retention_days=retention_days
    )
    
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
