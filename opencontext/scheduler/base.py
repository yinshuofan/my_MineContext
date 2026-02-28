#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task Scheduler Base Module

Defines the abstract base classes and interfaces for task scheduling.
"""

import abc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class TriggerMode(str, Enum):
    """Task trigger mode"""

    USER_ACTIVITY = "user_activity"  # Triggered by user activity, delayed execution
    PERIODIC = "periodic"  # Fixed interval execution


class TaskStatus(str, Enum):
    """Task execution status"""

    PENDING = "pending"  # Waiting to be executed
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Execution failed


@dataclass
class TaskConfig:
    """Configuration for a task type"""

    name: str
    enabled: bool = True
    trigger_mode: TriggerMode = TriggerMode.USER_ACTIVITY
    interval: int = 1800  # Execution interval in seconds
    timeout: int = 300  # Execution timeout in seconds
    task_ttl: int = 7200  # Task state TTL in seconds
    max_retries: int = 3  # Maximum retry attempts
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": str(self.enabled).lower(),
            "trigger_mode": self.trigger_mode.value,
            "interval": str(self.interval),
            "timeout": str(self.timeout),
            "task_ttl": str(self.task_ttl),
            "max_retries": str(self.max_retries),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        return cls(
            name=data.get("name", ""),
            enabled=data.get("enabled", "true").lower() == "true",
            trigger_mode=TriggerMode(data.get("trigger_mode", "user_activity")),
            interval=int(data.get("interval", 1800)),
            timeout=int(data.get("timeout", 300)),
            task_ttl=int(data.get("task_ttl", 7200)),
            max_retries=int(data.get("max_retries", 3)),
            description=data.get("description", ""),
        )


@dataclass
class TaskInfo:
    """Information about a scheduled task instance"""

    task_type: str
    user_key: str
    user_id: str
    device_id: Optional[str] = None
    agent_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: int = field(default_factory=lambda: int(time.time()))
    scheduled_at: int = 0
    last_activity: int = field(default_factory=lambda: int(time.time()))
    retry_count: int = 0
    lock_token: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        return {
            "task_type": self.task_type,
            "user_key": self.user_key,
            "user_id": self.user_id,
            "device_id": self.device_id or "",
            "agent_id": self.agent_id or "",
            "status": self.status.value,
            "created_at": str(self.created_at),
            "scheduled_at": str(self.scheduled_at),
            "last_activity": str(self.last_activity),
            "retry_count": str(self.retry_count),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "TaskInfo":
        return cls(
            task_type=data.get("task_type", ""),
            user_key=data.get("user_key", ""),
            user_id=data.get("user_id", ""),
            device_id=data.get("device_id") or None,
            agent_id=data.get("agent_id") or None,
            status=TaskStatus(data.get("status", "pending")),
            created_at=int(data.get("created_at", 0)),
            scheduled_at=int(data.get("scheduled_at", 0)),
            last_activity=int(data.get("last_activity", 0)),
            retry_count=int(data.get("retry_count", 0)),
        )


@dataclass
class UserKeyConfig:
    """Configuration for user key dimensions"""

    use_user_id: bool = True  # Must be True
    use_device_id: bool = True  # Whether to use device_id
    use_agent_id: bool = True  # Whether to use agent_id
    default_device_id: str = "default"
    default_agent_id: str = "default"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserKeyConfig":
        return cls(
            use_user_id=data.get("use_user_id", True),
            use_device_id=data.get("use_device_id", True),
            use_agent_id=data.get("use_agent_id", True),
            default_device_id=data.get("default_device_id", "default"),
            default_agent_id=data.get("default_agent_id", "default"),
        )


class IUserKeyBuilder(abc.ABC):
    """Interface for building user identification keys"""

    @abc.abstractmethod
    def build_key(
        self, user_id: str, device_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> str:
        """
        Build a user key from the given dimensions.

        Args:
            user_id: User identifier (required)
            device_id: Device identifier (optional)
            agent_id: Agent identifier (optional)

        Returns:
            A string key combining the specified dimensions
        """
        pass

    @abc.abstractmethod
    def parse_key(self, user_key: str) -> Dict[str, Optional[str]]:
        """
        Parse a user key back into its dimensions.

        Args:
            user_key: The combined user key string

        Returns:
            Dictionary with user_id, device_id, agent_id
        """
        pass

    @abc.abstractmethod
    def get_key_dimensions(self) -> List[str]:
        """
        Get the list of dimensions currently in use.

        Returns:
            List of dimension names (e.g., ["user_id", "device_id", "agent_id"])
        """
        pass


class ITaskScheduler(abc.ABC):
    """Interface for task scheduler (async)"""

    @abc.abstractmethod
    async def register_task_type(self, config: TaskConfig) -> bool:
        """
        Register a new task type (async).

        Args:
            config: Task type configuration

        Returns:
            True if registration successful
        """
        pass

    @abc.abstractmethod
    def register_handler(
        self, task_type: str, handler: Callable[[str, Optional[str], Optional[str]], bool]
    ) -> bool:
        """
        Register a handler function for a task type.

        Args:
            task_type: Name of the task type
            handler: Function that takes (user_id, device_id, agent_id) and returns success bool

        Returns:
            True if registration successful
        """
        pass

    @abc.abstractmethod
    async def schedule_user_task(
        self,
        task_type: str,
        user_id: str,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Schedule a task for a specific user (async).

        For user_activity trigger mode, this creates a delayed task.
        If a task already exists for the user, it won't create a duplicate.

        Args:
            task_type: Type of task to schedule
            user_id: User identifier
            device_id: Device identifier (optional)
            agent_id: Agent identifier (optional)

        Returns:
            True if a new task was created, False if task already exists
        """
        pass

    @abc.abstractmethod
    async def get_pending_task(self, task_type: str) -> Optional[TaskInfo]:
        """
        Get a pending task ready for execution (async).

        This should acquire a distributed lock to prevent duplicate execution.

        Args:
            task_type: Type of task to retrieve

        Returns:
            TaskInfo if a task is available, None otherwise
        """
        pass

    @abc.abstractmethod
    async def complete_task(
        self, task_type: str, user_key: str, lock_token: str, success: bool = True
    ) -> None:
        """
        Mark a task as completed and release the lock (async).

        Args:
            task_type: Type of the task
            user_key: User key identifying the task
            lock_token: Lock token to release
            success: Whether the task completed successfully
        """
        pass

    @abc.abstractmethod
    async def get_task_config(self, task_type: str) -> Optional[TaskConfig]:
        """
        Get configuration for a task type (async).

        Args:
            task_type: Name of the task type

        Returns:
            TaskConfig if found, None otherwise
        """
        pass

    @abc.abstractmethod
    async def start(self) -> None:
        """Start the scheduler background executor."""
        pass

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the scheduler."""
        pass

    @abc.abstractmethod
    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        pass


# Type alias for task handler function
TaskHandler = Callable[[str, Optional[str], Optional[str]], bool]
