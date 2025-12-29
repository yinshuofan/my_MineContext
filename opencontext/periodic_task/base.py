#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Periodic Task Base Module

Defines the abstract base classes and interfaces for periodic tasks.
"""

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from opencontext.scheduler.base import TaskConfig, TriggerMode

from loguru import logger

class TaskPriority(int, Enum):
    """Task execution priority"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TaskResult:
    """Result of a task execution"""
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    
    @classmethod
    def ok(cls, message: str = "Success", data: Dict[str, Any] = None) -> "TaskResult":
        """Create a successful result"""
        return cls(success=True, message=message, data=data)
    
    @classmethod
    def fail(cls, error: str, message: str = "Failed") -> "TaskResult":
        """Create a failed result"""
        return cls(success=False, message=message, error=error)


@dataclass
class TaskContext:
    """Context information passed to task execution"""
    user_id: str
    device_id: Optional[str] = None
    agent_id: Optional[str] = None
    task_type: str = ""
    retry_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def user_key(self) -> str:
        """Build user key from context"""
        parts = [self.user_id]
        if self.device_id:
            parts.append(self.device_id)
        if self.agent_id:
            parts.append(self.agent_id)
        return ":".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "device_id": self.device_id,
            "agent_id": self.agent_id,
            "task_type": self.task_type,
            "retry_count": self.retry_count,
            "extra": self.extra,
        }


class IPeriodicTask(abc.ABC):
    """
    Interface for periodic tasks.
    
    Each periodic task type should implement this interface.
    """
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Get the unique name of this task type.
        
        Returns:
            Task type name (e.g., "memory_compression", "data_cleanup")
        """
        pass
    
    @property
    @abc.abstractmethod
    def description(self) -> str:
        """
        Get a human-readable description of this task.
        
        Returns:
            Task description
        """
        pass
    
    @property
    @abc.abstractmethod
    def default_config(self) -> TaskConfig:
        """
        Get the default configuration for this task type.
        
        Returns:
            Default TaskConfig
        """
        pass
    
    @abc.abstractmethod
    def execute(self, context: TaskContext) -> TaskResult:
        """
        Execute the task.
        
        This is the main entry point for task execution.
        Should be implemented as a synchronous method.
        
        Args:
            context: Task execution context containing user info and metadata
            
        Returns:
            TaskResult indicating success or failure
        """
        pass
    
    @abc.abstractmethod
    async def execute_async(self, context: TaskContext) -> TaskResult:
        """
        Execute the task asynchronously.
        
        Async version of execute() for non-blocking execution.
        
        Args:
            context: Task execution context
            
        Returns:
            TaskResult indicating success or failure
        """
        pass
    
    def validate_context(self, context: TaskContext) -> bool:
        """
        Validate the task context before execution.
        
        Override this method to add custom validation logic.
        
        Args:
            context: Task execution context
            
        Returns:
            True if context is valid, False otherwise
        """
        return True
    
    def on_success(self, context: TaskContext, result: TaskResult) -> None:
        """
        Callback invoked after successful task execution.
        
        Override this method to add post-success logic.
        
        Args:
            context: Task execution context
            result: Execution result
        """
        pass
    
    def on_failure(self, context: TaskContext, result: TaskResult) -> None:
        """
        Callback invoked after failed task execution.
        
        Override this method to add post-failure logic.
        
        Args:
            context: Task execution context
            result: Execution result
        """
        pass
    
    def should_retry(self, context: TaskContext, result: TaskResult) -> bool:
        """
        Determine if the task should be retried after failure.
        
        Override this method to customize retry logic.
        
        Args:
            context: Task execution context
            result: Execution result
            
        Returns:
            True if task should be retried
        """
        config = self.default_config
        return not result.success and context.retry_count < config.max_retries


class IPeriodicTaskRegistry(abc.ABC):
    """Interface for periodic task registry"""
    
    @abc.abstractmethod
    def register(self, task: IPeriodicTask) -> bool:
        """
        Register a periodic task.
        
        Args:
            task: Task instance to register
            
        Returns:
            True if registration successful
        """
        pass
    
    @abc.abstractmethod
    def unregister(self, task_name: str) -> bool:
        """
        Unregister a periodic task.
        
        Args:
            task_name: Name of the task to unregister
            
        Returns:
            True if unregistration successful
        """
        pass
    
    @abc.abstractmethod
    def get(self, task_name: str) -> Optional[IPeriodicTask]:
        """
        Get a registered task by name.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task instance if found, None otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_all(self) -> Dict[str, IPeriodicTask]:
        """
        Get all registered tasks.
        
        Returns:
            Dictionary mapping task names to task instances
        """
        pass
    
    @abc.abstractmethod
    def get_enabled_tasks(self) -> Dict[str, IPeriodicTask]:
        """
        Get all enabled tasks.
        
        Returns:
            Dictionary mapping task names to enabled task instances
        """
        pass


class BasePeriodicTask(IPeriodicTask):
    """
    Base implementation of IPeriodicTask.
    
    Provides common functionality and sensible defaults.
    Subclasses should override execute() and execute_async().
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        trigger_mode: TriggerMode = TriggerMode.USER_ACTIVITY,
        interval: int = 1800,
        timeout: int = 300,
        task_ttl: int = 7200,
        max_retries: int = 3,
    ):
        self._name = name
        self._description = description
        self._trigger_mode = trigger_mode
        self._interval = interval
        self._timeout = timeout
        self._task_ttl = task_ttl
        self._max_retries = max_retries
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def default_config(self) -> TaskConfig:
        return TaskConfig(
            name=self._name,
            enabled=True,
            trigger_mode=self._trigger_mode,
            interval=self._interval,
            timeout=self._timeout,
            task_ttl=self._task_ttl,
            max_retries=self._max_retries,
            description=self._description,
        )
    
    def execute(self, context: TaskContext) -> TaskResult:
        """
        Default synchronous execution.
        
        Override this method in subclasses.
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    async def execute_async(self, context: TaskContext) -> TaskResult:
        """
        Default async execution that wraps synchronous execute().
        
        Override this method for true async implementation.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, context)
