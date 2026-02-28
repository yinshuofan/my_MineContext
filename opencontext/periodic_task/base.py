#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Periodic Task Base Module

Defines the abstract base classes and interfaces for periodic tasks.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from opencontext.scheduler.base import TaskConfig, TriggerMode


@dataclass
class TaskResult:
    """Result of a task execution"""

    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

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
    extra: Dict[str, Any] = field(default_factory=dict)


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


class BasePeriodicTask(IPeriodicTask):
    """
    Base implementation of IPeriodicTask.

    Provides common functionality and sensible defaults.
    Subclasses should override execute().
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
