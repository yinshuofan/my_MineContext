#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
OpenContext main class - system entry point integrating all components.
Refactored for better separation of concerns and maintainability.
"""

import threading
from typing import Any, Dict, List, Optional

from opencontext.config.config_manager import ConfigManager
from opencontext.config.global_config import GlobalConfig
from opencontext.llm.global_embedding_client import GlobalEmbeddingClient
from opencontext.llm.global_vlm_client import GlobalVLMClient
from opencontext.managers.capture_manager import ContextCaptureManager
from opencontext.managers.processor_manager import ContextProcessorManager
from opencontext.models.context import ProcessedContext, RawContextProperties
from opencontext.models.enums import CONTEXT_STORAGE_BACKENDS, ContextType
from opencontext.server.component_initializer import ComponentInitializer
from opencontext.server.context_operations import ContextOperations
from opencontext.storage.global_storage import GlobalStorage, get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OpenContext:
    """OpenContext main class - integrates all components and provides unified API."""

    def __init__(self):
        """Initialize OpenContext."""
        # Initialize core components
        self.capture_manager = ContextCaptureManager()
        self.processor_manager = ContextProcessorManager()

        # Helper classes
        self.component_initializer = ComponentInitializer()
        self.context_operations: Optional[ContextOperations] = None

        # Web server state
        self.web_server: Optional[threading.Thread] = None
        self.web_server_running: bool = False

        self.storage = None

        logger.info("OpenContext initialization completed")

    def initialize(self) -> None:
        """Initialize all components in proper order."""
        logger.info("Starting initialization of all components...")

        try:
            GlobalConfig.get_instance()
            GlobalEmbeddingClient.get_instance()
            GlobalStorage.get_instance()
            GlobalVLMClient.get_instance()

            # Initialize Redis singleton from top-level config before any component
            redis_config = GlobalConfig.get_instance().get_config("redis") or {}
            if redis_config.get("enabled", True):
                from opencontext.storage.redis_cache import RedisCacheConfig, init_redis_cache

                redis_cfg = RedisCacheConfig(
                    host=redis_config.get("host", "localhost"),
                    port=int(redis_config.get("port", 6379)),
                    password=redis_config.get("password") or None,
                    db=int(redis_config.get("db", 0)),
                    key_prefix=redis_config.get("key_prefix", "opencontext:"),
                    max_connections=int(redis_config.get("max_connections", 10)),
                    socket_timeout=float(redis_config.get("socket_timeout", 5.0)),
                    socket_connect_timeout=float(redis_config.get("socket_connect_timeout", 5.0)),
                    retry_on_timeout=redis_config.get("retry_on_timeout", True),
                )
                init_redis_cache(redis_cfg)
                logger.info(
                    f"Redis singleton initialized: "
                    f"{redis_cfg.host}:{redis_cfg.port}/{redis_cfg.db}"
                )

            self.storage = GlobalStorage.get_instance().get_storage()

            self.context_operations = ContextOperations()
            self.capture_manager.set_callback(self._handle_captured_context)
            self.component_initializer.initialize_capture_components(self.capture_manager)
            logger.info("Capture modules initialization completed")
            self.component_initializer.initialize_processors(
                self.processor_manager, self._handle_processed_context
            )

            # Initialize task scheduler after processors (to reuse merger)
            self.component_initializer.initialize_task_scheduler(self.processor_manager)

            self._initialize_monitoring()
            logger.info("All components initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}", exc_info=True)
            self.shutdown(graceful=False)
            raise

    def _initialize_monitoring(self):
        """Initialize monitoring system"""
        try:
            from opencontext.monitoring import initialize_monitor

            initialize_monitor()
            logger.info("Monitoring system initialized with storage backend")
        except ImportError:
            logger.warning("Monitoring module not available, skipping monitoring initialization")
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {e}")

    async def _handle_captured_context(self, contexts: List[RawContextProperties]) -> bool:
        """
        Handle batch processing and storage of captured context data.
        """
        if not contexts:
            return False

        try:
            for context_data in contexts:
                await self.processor_manager.process(context_data)
            return True
        except Exception as e:
            logger.error(f"Error processing captured contexts: {e}")
            return False

    async def _handle_processed_context(self, contexts: List[ProcessedContext]) -> bool:
        """Store processed contexts, routing by type per CONTEXT_STORAGE_BACKENDS."""
        if not contexts:
            return False
        if not self.storage:
            logger.warning("Storage is not initialized.")
            return False

        try:
            vector_contexts = []
            db_contexts = []

            for ctx in contexts:
                ctx_type = ctx.extracted_data.context_type
                backend = CONTEXT_STORAGE_BACKENDS.get(ctx_type, "vector_db")
                if backend == "document_db":
                    db_contexts.append(ctx)
                else:
                    vector_contexts.append(ctx)

            # Store relational DB contexts (profile/entity) as a batch
            # If any fails, the exception propagates and we skip cache invalidation
            affected_users = set()
            if db_contexts:
                for ctx in db_contexts:
                    ctx_type = ctx.extracted_data.context_type
                    if ctx_type == ContextType.PROFILE:
                        await self._store_profile(ctx)
                    elif ctx_type == ContextType.ENTITY:
                        await self._store_entities(ctx)
                    uid = ctx.properties.user_id
                    if uid:
                        affected_users.add((uid, ctx.properties.device_id, ctx.properties.agent_id))

            # Store vector contexts
            success = True
            if vector_contexts:
                success = await self.storage.batch_upsert_processed_context(vector_contexts)
                for ctx in vector_contexts:
                    uid = ctx.properties.user_id
                    if uid:
                        affected_users.add((uid, ctx.properties.device_id, ctx.properties.agent_id))

            # Only invalidate cache after ALL operations succeed
            for uid, did, aid in affected_users:
                await self._invalidate_user_cache(uid, did, aid)

            return success
        except Exception as e:
            logger.error(f"Error storing processed contexts: {e}")
            return False

    async def _store_profile(self, ctx: ProcessedContext) -> None:
        """Store a profile context to relational DB with LLM-driven merge."""
        from opencontext.context_processing.processor.profile_processor import refresh_profile

        ed = ctx.extracted_data
        props = ctx.properties
        await refresh_profile(
            new_content=ed.summary or "",
            new_summary=None,
            new_keywords=ed.keywords,
            new_entities=ed.entities,
            new_importance=ed.importance,
            new_metadata=ctx.metadata,
            user_id=props.user_id or "default",
            device_id=props.device_id or "default",
            agent_id=props.agent_id or "default",
        )
        logger.info(
            f"Profile stored for user={props.user_id}, device={props.device_id}, agent={props.agent_id}"
        )

    async def _store_entities(self, ctx: ProcessedContext) -> None:
        """Store entity contexts to relational DB."""
        ed = ctx.extracted_data
        props = ctx.properties
        for entity_name in ed.entities:
            await self.storage.upsert_entity(
                user_id=props.user_id or "default",
                device_id=props.device_id or "default",
                agent_id=props.agent_id or "default",
                entity_name=entity_name,
                content=ed.summary or "",
                entity_type=None,
                summary=ed.summary,
                keywords=ed.keywords,
                metadata=ctx.metadata,
            )
            logger.info(f"Entity '{entity_name}' stored for user={props.user_id}")

    async def _invalidate_user_cache(
        self,
        user_id: Optional[str],
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """Invalidate user cache snapshot directly."""
        if not user_id:
            return
        try:
            from opencontext.server.cache.memory_cache_manager import get_memory_cache_manager

            manager = get_memory_cache_manager()
            did = device_id or "default"
            aid = agent_id or "default"
            await manager.invalidate_snapshot(user_id, did, aid)
            logger.debug(f"Cache invalidated for user={user_id}")
        except Exception as e:
            logger.warning(f"Cache invalidation failed for user={user_id}: {e}")

    def start_capture(self) -> None:
        """Start all capture components."""
        logger.info("Starting all capture components...")
        try:
            self.capture_manager.start_all_components()
            logger.info("All capture components started successfully")
        except Exception as e:
            logger.error(f"Failed to start capture components: {e}")
            raise

    def shutdown(self, graceful: bool = True) -> None:
        """Shutdown all components gracefully.

        Args:
            graceful: Whether to perform graceful shutdown
        """
        logger.info("Shutting down all components...")

        try:
            # Stop task scheduler (may be already stopped by lifespan teardown)
            try:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    asyncio.run_coroutine_threadsafe(
                        self.component_initializer.stop_task_scheduler(), loop
                    ).result(timeout=35)
                    logger.info("Task scheduler stopped")
                except RuntimeError:
                    logger.debug("No event loop — scheduler already stopped by lifespan")
            except Exception as e:
                logger.warning(f"Error stopping task scheduler: {e}")

            # Shutdown managers
            self.capture_manager.shutdown(graceful=graceful)
            self.processor_manager.shutdown(graceful=graceful)

            if self.web_server and self.web_server.is_alive():
                logger.info("Web server will close when main thread exits.")

            logger.info("All components shut down successfully.")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            if not graceful:
                raise

    async def add_context(self, context_data: RawContextProperties) -> bool:
        """
        Process a single context data item.
        """
        try:
            return await self.processor_manager.process(context_data)
        except Exception as e:
            logger.error(f"Error adding context: {e}")
            return False

    # Delegate context operations to ContextOperations helper
    def get_all_contexts(
        self, limit: int = 10, offset: int = 0, filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[ProcessedContext]]:
        """Get all processed contexts."""
        if not self.context_operations:
            logger.warning("Context operations not initialized.")
            return {}
        return self.context_operations.get_all_contexts(
            limit, offset, filter_criteria, need_vector=False
        )

    def get_context(self, doc_id: str, context_type: str) -> Optional[ProcessedContext]:
        """Get a single processed context."""
        if not self.context_operations:
            logger.warning("Context operations not initialized.")
            return None
        return self.context_operations.get_context(doc_id, context_type)

    def update_context(self, doc_id: str, context: ProcessedContext) -> bool:
        """Update a processed context."""
        if not self.context_operations:
            logger.warning("Context operations not initialized.")
            return False
        return self.context_operations.update_context(doc_id, context)

    def delete_context(self, doc_id: str, context_type: str) -> bool:
        """Delete a processed context."""
        if not self.context_operations:
            logger.warning("Context operations not initialized.")
            return False
        return self.context_operations.delete_context(doc_id, context_type)

    async def add_document(self, file_path: str) -> Optional[str]:
        """Add a document to the system."""
        if not self.context_operations:
            logger.warning("Context operations not initialized.")
            return "Context operations not initialized"
        return await self.context_operations.add_document(file_path, self.add_context)

    def search(
        self,
        query: str,
        top_k: int = 10,
        context_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector search without LLM processing.

        Args:
            query: Search query string
            top_k: Maximum number of results
            context_types: List of context types to search
            filters: Additional filter conditions
            user_id: User identifier for multi-user filtering
            device_id: Device identifier for multi-user filtering
            agent_id: Agent identifier for multi-user filtering
        """
        if not self.context_operations:
            raise RuntimeError("Context operations not initialized")
        return self.context_operations.search(
            query,
            top_k,
            context_types,
            filters,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
        )

    def get_context_types(self) -> List[str]:
        """Get all available context types."""
        if not self.context_operations:
            raise RuntimeError("Context operations not initialized")
        return self.context_operations.get_context_types()

    async def check_components_health(self) -> Dict[str, Any]:
        """Check health status of all components including actual connectivity."""
        health: Dict[str, Any] = {
            "config": GlobalConfig.get_instance().is_initialized(),
            "storage": GlobalStorage.get_instance().get_storage() is not None,
            "llm": GlobalEmbeddingClient.get_instance().is_initialized(),
        }

        # Check MySQL/SQLite connectivity
        try:
            storage = get_storage()
            if hasattr(storage, "document_storage") and storage.document_storage:
                backend = storage.document_storage
                if hasattr(backend, "_pool") and backend._pool:
                    # MySQL with pool: use context manager for safe checkout/return
                    with backend._get_connection():
                        pass
                    health["document_db"] = True
                elif hasattr(backend, "_get_connection"):
                    # SQLite: verify actual connectivity
                    conn = backend._get_connection()
                    conn.execute("SELECT 1")
                    health["document_db"] = True
        except Exception as e:
            logger.warning(f"Document DB health check failed: {e}")
            health["document_db"] = False

        # Check Redis connectivity
        try:
            from opencontext.storage.redis_cache import get_redis_cache

            cache = get_redis_cache()
            if cache:
                health["redis"] = await cache.is_connected()
            else:
                health["redis"] = False
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            health["redis"] = False

        # Check scheduler status
        try:
            from opencontext.scheduler import get_scheduler

            scheduler = get_scheduler()
            if scheduler:
                health["scheduler"] = {
                    "initialized": True,
                    "running": scheduler.is_running(),
                    "in_flight_tasks": len(scheduler._in_flight),
                    "registered_handlers": list(scheduler._task_handlers.keys()),
                }
            else:
                health["scheduler"] = {"initialized": False, "running": False}
        except Exception as e:
            logger.warning(f"Scheduler health check failed: {e}")
            health["scheduler"] = {"initialized": False, "error": str(e)}

        return health


def main():
    """Main entry point for running the server."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="OpenContext Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=1733, help="Port to bind to")
    parser.add_argument("--config", help="Configuration file path", default="./config/config.yaml")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    print(f"Starting OpenContext Server on {args.host}:{args.port}")
    if args.config:
        print(f"Using config file: {args.config}")

    uvicorn.run(
        "opencontext.cli:app", host=args.host, port=args.port, reload=args.reload, log_level="info"
    )


if __name__ == "__main__":
    main()
