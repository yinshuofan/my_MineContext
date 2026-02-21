#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Periodic Task Module

Provides base classes and implementations for periodic tasks
such as memory compression, data cleanup, etc.
"""

from opencontext.periodic_task.base import (
    BasePeriodicTask,
    IPeriodicTask,
    IPeriodicTaskRegistry,
    TaskContext,
    TaskPriority,
    TaskResult,
)
from opencontext.periodic_task.registry import (
    PeriodicTaskRegistry,
    get_registry,
    register_task,
    get_task,
)
from opencontext.periodic_task.memory_compression import (
    MemoryCompressionTask,
    create_compression_handler,
)
from opencontext.periodic_task.data_cleanup import (
    DataCleanupTask,
    create_cleanup_handler,
)
from opencontext.periodic_task.hierarchy_summary import (
    HierarchySummaryTask,
    create_hierarchy_handler,
)

__all__ = [
    # Interfaces
    "IPeriodicTask",
    "IPeriodicTaskRegistry",
    # Base classes
    "BasePeriodicTask",
    # Data classes
    "TaskContext",
    "TaskResult",
    # Enums
    "TaskPriority",
    # Registry
    "PeriodicTaskRegistry",
    "get_registry",
    "register_task",
    "get_task",
    # Task implementations
    "MemoryCompressionTask",
    "DataCleanupTask",
    "HierarchySummaryTask",
    # Handler factories
    "create_compression_handler",
    "create_cleanup_handler",
    "create_hierarchy_handler",
]
