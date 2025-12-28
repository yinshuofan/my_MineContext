#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task Scheduler Module

Provides a stateless, Redis-backed task scheduling system that supports
multi-instance deployment and various trigger modes.
"""

from opencontext.scheduler.base import (
    ITaskScheduler,
    IUserKeyBuilder,
    TaskConfig,
    TaskHandler,
    TaskInfo,
    TaskStatus,
    TriggerMode,
    UserKeyConfig,
)
from opencontext.scheduler.user_key_builder import UserKeyBuilder
from opencontext.scheduler.redis_scheduler import (
    RedisTaskScheduler,
    get_scheduler,
    set_scheduler,
    init_scheduler,
)

__all__ = [
    # Interfaces
    "ITaskScheduler",
    "IUserKeyBuilder",
    # Implementations
    "UserKeyBuilder",
    "RedisTaskScheduler",
    # Data classes
    "TaskConfig",
    "TaskInfo",
    "UserKeyConfig",
    # Enums
    "TaskStatus",
    "TriggerMode",
    # Type aliases
    "TaskHandler",
    # Global functions
    "get_scheduler",
    "set_scheduler",
    "init_scheduler",
]
