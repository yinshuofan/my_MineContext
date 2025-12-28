#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory Compression Periodic Task

Periodically compresses user memory/context data to reduce storage
and improve retrieval performance.
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


class MemoryCompressionTask(BasePeriodicTask):
    """
    Memory Compression Task
    
    Compresses user context/memory data periodically.
    Triggered by user activity with delayed execution.
    """
    
    def __init__(
        self,
        context_merger: Any = None,
        # interval: int = 1800,
        interval: int = 172800,
        timeout: int = 300,
    ):
        """
        Initialize the memory compression task.
        
        Args:
            context_merger: ContextMerger instance for performing compression
            interval: Interval in seconds between compressions (default: 30 min)
            timeout: Execution timeout in seconds (default: 5 min)
        """
        super().__init__(
            name="memory_compression",
            description="Periodically compress user memory/context data",
            trigger_mode=TriggerMode.USER_ACTIVITY,
            interval=interval,
            timeout=timeout,
            task_ttl=7200,
            max_retries=3,
        )
        self._context_merger = context_merger
    
    def set_context_merger(self, context_merger: Any) -> None:
        """Set the context merger instance"""
        self._context_merger = context_merger
    
    def execute(self, context: TaskContext) -> TaskResult:
        """
        Execute memory compression for a user.
        
        Args:
            context: Task execution context with user info
            
        Returns:
            TaskResult indicating success or failure
        """
        start_time = time.time()
        
        if not self._context_merger:
            return TaskResult.fail(
                error="ContextMerger not configured",
                message="Memory compression task requires ContextMerger"
            )
        
        user_id = context.user_id
        device_id = context.device_id
        agent_id = context.agent_id
        
        logger.info(
            f"Starting memory compression for user={user_id}, "
            f"device={device_id}, agent={agent_id}"
        )
        
        try:
            # Call the context merger's compression method
            # This method should be implemented in ContextMerger
            if hasattr(self._context_merger, 'periodic_memory_compression_for_user'):
                self._context_merger.periodic_memory_compression_for_user(
                    user_id=user_id,
                    device_id=device_id,
                    agent_id=agent_id,
                    interval_seconds=self._interval
                )
            elif hasattr(self._context_merger, 'periodic_memory_compression'):
                # Fallback to old method signature
                self._context_merger.periodic_memory_compression(
                    interval_seconds=self._interval
                )
            else:
                return TaskResult.fail(
                    error="No compression method found",
                    message="ContextMerger does not have compression method"
                )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Memory compression completed for user={user_id} "
                f"in {execution_time}ms"
            )
            
            return TaskResult.ok(
                message=f"Compression completed for user {user_id}",
                data={
                    "user_id": user_id,
                    "device_id": device_id,
                    "agent_id": agent_id,
                    "execution_time_ms": execution_time,
                }
            )
            
        except Exception as e:
            logger.exception(f"Memory compression failed for user={user_id}: {e}")
            return TaskResult.fail(
                error=str(e),
                message=f"Compression failed for user {user_id}"
            )
    
    async def execute_async(self, context: TaskContext) -> TaskResult:
        """Async version of execute"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, context)
    
    def validate_context(self, context: TaskContext) -> bool:
        """Validate that user_id is provided"""
        return bool(context.user_id)


def create_compression_handler(context_merger: Any):
    """
    Create a compression handler function for the scheduler.
    
    Args:
        context_merger: ContextMerger instance
        
    Returns:
        Handler function compatible with TaskScheduler
    """
    task = MemoryCompressionTask(context_merger=context_merger)
    
    def handler(
        user_id: str,
        device_id: Optional[str],
        agent_id: Optional[str]
    ) -> bool:
        context = TaskContext(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            task_type="memory_compression",
        )
        result = task.execute(context)
        return result.success
    
    return handler
