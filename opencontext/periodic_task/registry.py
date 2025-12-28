#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Periodic Task Registry

Manages registration and retrieval of periodic tasks.
"""

from typing import Dict, Optional

from loguru import logger

from opencontext.periodic_task.base import IPeriodicTask, IPeriodicTaskRegistry


class PeriodicTaskRegistry(IPeriodicTaskRegistry):
    """
    Periodic Task Registry
    
    A simple in-memory registry for managing periodic tasks.
    """
    
    def __init__(self):
        self._tasks: Dict[str, IPeriodicTask] = {}
        self._enabled: Dict[str, bool] = {}
    
    def register(self, task: IPeriodicTask) -> bool:
        """Register a periodic task"""
        if not task or not task.name:
            logger.error("Cannot register task without name")
            return False
        
        if task.name in self._tasks:
            logger.warning(f"Task {task.name} already registered, overwriting")
        
        self._tasks[task.name] = task
        self._enabled[task.name] = task.default_config.enabled
        
        logger.info(f"Registered periodic task: {task.name}")
        return True
    
    def unregister(self, task_name: str) -> bool:
        """Unregister a periodic task"""
        if task_name not in self._tasks:
            logger.warning(f"Task {task_name} not found in registry")
            return False
        
        del self._tasks[task_name]
        del self._enabled[task_name]
        
        logger.info(f"Unregistered periodic task: {task_name}")
        return True
    
    def get(self, task_name: str) -> Optional[IPeriodicTask]:
        """Get a registered task by name"""
        return self._tasks.get(task_name)
    
    def get_all(self) -> Dict[str, IPeriodicTask]:
        """Get all registered tasks"""
        return dict(self._tasks)
    
    def get_enabled_tasks(self) -> Dict[str, IPeriodicTask]:
        """Get all enabled tasks"""
        return {
            name: task
            for name, task in self._tasks.items()
            if self._enabled.get(name, False)
        }
    
    def enable(self, task_name: str) -> bool:
        """Enable a task"""
        if task_name not in self._tasks:
            return False
        self._enabled[task_name] = True
        return True
    
    def disable(self, task_name: str) -> bool:
        """Disable a task"""
        if task_name not in self._tasks:
            return False
        self._enabled[task_name] = False
        return True
    
    def is_enabled(self, task_name: str) -> bool:
        """Check if a task is enabled"""
        return self._enabled.get(task_name, False)


# Global registry instance
_registry: Optional[PeriodicTaskRegistry] = None


def get_registry() -> PeriodicTaskRegistry:
    """Get the global task registry instance"""
    global _registry
    if _registry is None:
        _registry = PeriodicTaskRegistry()
    return _registry


def register_task(task: IPeriodicTask) -> bool:
    """Register a task to the global registry"""
    return get_registry().register(task)


def get_task(task_name: str) -> Optional[IPeriodicTask]:
    """Get a task from the global registry"""
    return get_registry().get(task_name)
