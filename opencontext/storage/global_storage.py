#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Global storage manager singleton wrapper
Provides global access to UnifiedStorage instance
"""

import threading
from typing import List, Optional

from opencontext.models.context import ProcessedContext
from opencontext.models.enums import ContextType
from opencontext.storage.unified_storage import UnifiedStorage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GlobalStorage:
    """
    Global storage manager (singleton pattern)

    Provides global access to UnifiedStorage instance, avoiding passing Storage objects between components.
    All components can access UnifiedStorage through GlobalStorage.get_instance().
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize global storage manager"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._storage: Optional[UnifiedStorage] = None
                    self._auto_initialized = False
                    self._init_attempts = 0
                    GlobalStorage._initialized = True

    @classmethod
    def get_instance(cls) -> "GlobalStorage":
        """
        Get global storage manager instance

        Returns:
            GlobalStorage: Global storage manager singleton instance
        """
        instance = cls()
        return instance

    @classmethod
    def reset(cls):
        """Reset singleton instance (mainly for testing)"""
        with cls._lock:
            cls._instance = None
            cls._initialized = False

    async def _auto_initialize(self):
        """Auto-initialize storage manager"""
        if self._auto_initialized:
            return

        max_auto_init_attempts = 3
        if self._init_attempts >= max_auto_init_attempts:
            logger.error(
                f"GlobalStorage auto-init exceeded max attempts ({max_auto_init_attempts})"
            )
            return

        self._init_attempts += 1

        try:
            # Try to auto-initialize storage
            from opencontext.config.global_config import get_config

            storage_config = get_config("storage")
            if storage_config and storage_config.get("enabled", False):
                backend_configs = storage_config.get("backends", [])
                if backend_configs:
                    storage = UnifiedStorage()
                    if await storage.initialize():
                        self._storage = storage
                        logger.info("GlobalStorage auto-initialized successfully")
                    else:
                        logger.warning(
                            "GlobalStorage auto-initialization: storage initialization failed"
                        )
                else:
                    logger.warning("GlobalStorage auto-initialization: no backend configs found")
            else:
                logger.warning("GlobalStorage auto-initialization: storage not enabled in config")
            self._auto_initialized = True
        except Exception as e:
            logger.error(f"GlobalStorage auto-initialization failed: {e}")
            self._auto_initialized = False  # Allow retry on next call

    async def ensure_initialized(self):
        """Ensure storage is initialized (async). Call this before first use."""
        if not self._auto_initialized and self._storage is None:
            await self._auto_initialize()

    def get_storage(self) -> Optional[UnifiedStorage]:
        """
        Get storage instance

        Returns:
            UnifiedStorage: Storage instance, returns None if not initialized
        """
        return self._storage

    def is_initialized(self) -> bool:
        """
        Check if initialized

        Returns:
            bool: Whether initialized
        """
        return self._storage is not None

    # Convenience methods - directly call common UnifiedStorage methods

    async def upsert_processed_context(self, context: ProcessedContext) -> bool:
        """Store processed context"""
        if not self._storage:
            raise RuntimeError("Storage not initialized")
        return await self._storage.upsert_processed_context(context)

    async def batch_upsert_processed_context(self, contexts: List[ProcessedContext]) -> bool:
        """Batch store processed contexts"""
        if not self._storage:
            raise RuntimeError("Storage not initialized")
        return await self._storage.batch_upsert_processed_context(contexts)

    async def get_processed_context(
        self, doc_id: str, context_type: ContextType
    ) -> Optional[ProcessedContext]:
        """Get processed context"""
        if not self._storage:
            raise RuntimeError("Storage not initialized")
        return await self._storage.get_processed_context(doc_id, context_type)

    async def delete_processed_context(self, doc_id: str, context_type: ContextType) -> bool:
        """Delete processed context"""
        if not self._storage:
            raise RuntimeError("Storage not initialized")
        return await self._storage.delete_processed_context(doc_id, context_type)


# Convenience functions
def get_global_storage() -> GlobalStorage:
    """Convenience function to get global storage manager instance"""
    return GlobalStorage.get_instance()


def get_storage() -> Optional[UnifiedStorage]:
    """Convenience function to get storage instance"""
    return GlobalStorage.get_instance().get_storage()
