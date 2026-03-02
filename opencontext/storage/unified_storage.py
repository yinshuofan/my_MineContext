#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Unified storage system - unified management supporting multiple storage backends
"""

import functools
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from opencontext.models.context import ProcessedContext, Vectorize
from opencontext.models.enums import ContextType
from opencontext.storage.base_storage import (
    DataType,
    DocumentData,
    IDocumentStorageBackend,
    IStorageBackend,
    IVectorStorageBackend,
    QueryResult,
    StorageType,
)
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _require_backend(backend_attr: str, default=None):
    """Decorator: check _initialized and backend availability before execution."""

    def decorator(method):
        @functools.wraps(method)
        async def wrapper(self, *args, **kwargs):
            if not self._initialized:
                logger.error("Storage not initialized")
                return default
            if not getattr(self, backend_attr, None):
                logger.error(f"Backend {backend_attr} not initialized")
                return default
            return await method(self, *args, **kwargs)

        return wrapper

    return decorator


class StorageBackendFactory:
    """Storage backend factory class"""

    def __init__(self):
        self._backends = {
            StorageType.VECTOR_DB: {
                "qdrant": self._create_qdrant_backend,
                "vikingdb": self._create_vikingdb_backend,
            },
            StorageType.DOCUMENT_DB: {
                "mysql": self._create_mysql_backend,
                "sqlite": self._create_sqlite_backend,
            },
        }

    async def create_backend(
        self, storage_type: StorageType, config: Dict[str, Any]
    ) -> Optional[IStorageBackend]:
        """Create storage backend"""
        backend_name = config.get("backend", "default")

        if storage_type not in self._backends:
            logger.error(f"Unsupported storage type: {storage_type}")
            return None

        # Get default backend
        type_backends = self._backends[storage_type]
        if backend_name == "default":
            backend_name = list(type_backends.keys())[0]

        if backend_name not in type_backends:
            logger.error(f"Unsupported {storage_type.value} backend: {backend_name}")
            return None

        try:
            backend = type_backends[backend_name](config)
            if await backend.initialize(config):
                return backend
            else:
                logger.error(f"Backend {backend_name} initialization failed")
                return None
        except Exception as e:
            logger.exception(f"Creating {backend_name} backend failed: {e}")
            return None

    def _create_qdrant_backend(self, config: Dict[str, Any]):
        from opencontext.storage.backends.qdrant_backend import QdrantBackend

        return QdrantBackend()

    def _create_sqlite_backend(self, config: Dict[str, Any]):
        from opencontext.storage.backends.sqlite_backend import SQLiteBackend

        return SQLiteBackend()

    def _create_mysql_backend(self, config: Dict[str, Any]):
        from opencontext.storage.backends.mysql_backend import MySQLBackend

        return MySQLBackend()

    def _create_vikingdb_backend(self, config: Dict[str, Any]):
        from opencontext.storage.backends.vikingdb_backend import VikingDBBackend

        return VikingDBBackend()


class UnifiedStorage:
    """
    Unified storage system - manages multiple storage backends, supports automatic routing based on data type and storage requirements
    """

    def __init__(self):
        self._factory = StorageBackendFactory()
        self._initialized = False
        self._vector_backend: IVectorStorageBackend = None
        self._document_backend: IDocumentStorageBackend = None

    async def get_vector_collection_names(self) -> Optional[List[str]]:
        """Get all collection names in vector database"""
        if not self._vector_backend:
            return None
        return await self._vector_backend.get_collection_names()

    async def initialize(self) -> bool:
        """
        Initialize unified storage system

        Args:
            storage_configs: Storage configuration list, each configuration contains:
                - name: Backend name
                - storage_type: Storage type (vector_db/document_db)
                - backend: Specific backend (chromadb/sqlite etc.)
                - config: Backend specific configuration
                - default: Whether it's the default backend for this type
                - data_types: List of supported data types
        """
        try:
            from opencontext.config.global_config import get_config

            storage_config = get_config("storage")
            backend_configs = storage_config.get("backends", [])
            if not backend_configs:
                logger.error("No storage backends configured")
                return False

            for config in backend_configs:
                storage_type = StorageType(config["storage_type"])
                backend = await self._factory.create_backend(storage_type, config)
                if backend:
                    # Set dedicated backend reference
                    if storage_type == StorageType.VECTOR_DB and isinstance(
                        backend, IVectorStorageBackend
                    ):
                        if self._vector_backend is None or config.get("default", False):
                            self._vector_backend = backend
                    elif storage_type == StorageType.DOCUMENT_DB and isinstance(
                        backend, IDocumentStorageBackend
                    ):
                        if self._document_backend is None or config.get("default", False):
                            self._document_backend = backend

                    logger.info(
                        f"Storage backend {config['name']} ({storage_type.value}) initialized successfully"
                    )
                else:
                    logger.error(f"Storage backend {config['name']} initialization failed")
                    return False

            self._initialized = True
            return True

        except Exception as e:
            logger.exception(f"Unified storage system initialization failed: {e}")
            return False

    def get_default_backend(self, storage_type: StorageType) -> Optional[IStorageBackend]:
        """Get default storage backend for specified type"""
        if storage_type == StorageType.VECTOR_DB:
            return self._vector_backend
        elif storage_type == StorageType.DOCUMENT_DB:
            return self._document_backend
        return None

    @_require_backend("_vector_backend")
    async def batch_upsert_processed_context(
        self, contexts: List[ProcessedContext]
    ) -> Optional[List[str]]:
        """Batch store processed contexts to vector database"""
        try:
            # Directly pass ProcessedContext to vector database
            doc_ids = await self._vector_backend.batch_upsert_processed_context(contexts)
            return doc_ids

        except Exception as e:
            logger.exception(f"Failed to store context: {e}")
            return None

    @_require_backend("_vector_backend")
    async def upsert_processed_context(self, context: ProcessedContext) -> Optional[str]:
        """Store processed context to vector database"""
        try:
            # Directly pass ProcessedContext to vector database
            doc_id = await self._vector_backend.upsert_processed_context(context)
            return doc_id

        except Exception as e:
            logger.exception(f"Failed to store context: {e}")
            return None

    async def get_processed_context(self, id: str, context_type: str):
        return await self._vector_backend.get_processed_context(id, context_type)

    async def delete_processed_context(self, id: str, context_type: str):
        return await self._vector_backend.delete_processed_context(id, context_type)

    async def delete_batch_processed_contexts(self, ids: List[str], context_type: str):
        return await self._vector_backend.delete_contexts(ids, context_type)

    @_require_backend("_vector_backend", default={})
    async def get_all_processed_contexts(
        self,
        context_types: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        filter: Optional[Dict[str, Any]] = None,
        need_vector: bool = False,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, List[ProcessedContext]]:
        """Get processed contexts, query only from vector database

        Args:
            context_types: List of context types to get
            limit: Maximum number of results per context type
            offset: Offset for pagination
            filter: Additional filter conditions
            need_vector: Whether to include vectors in results
            user_id: User identifier for multi-user filtering
            device_id: Device identifier for multi-user filtering
            agent_id: Agent identifier for multi-user filtering
        """
        if not context_types:
            context_types = [ct.value for ct in ContextType]
        try:
            return await self._vector_backend.get_all_processed_contexts(
                context_types=context_types,
                limit=limit,
                offset=offset,
                filter=filter,
                need_vector=need_vector,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            )
        except Exception as e:
            logger.exception(f"Failed to query ProcessedContext: {e}")
            return {}

    async def scroll_processed_contexts(
        self,
        context_types: Optional[List[str]] = None,
        batch_size: int = 100,
        filter: Optional[Dict[str, Any]] = None,
        need_vector: bool = False,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> AsyncGenerator[ProcessedContext, None]:
        """Iterate over all contexts matching the criteria, yielding one at a time.

        Delegates to vector backend's scroll_processed_contexts.
        """
        if not self._initialized:
            logger.error("Unified storage system not initialized")
            return

        if not self._vector_backend:
            logger.error("Vector database backend not initialized")
            return

        if not context_types:
            context_types = [ct.value for ct in ContextType]

        try:
            async for item in self._vector_backend.scroll_processed_contexts(
                context_types=context_types,
                batch_size=batch_size,
                filter=filter,
                need_vector=need_vector,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            ):
                yield item
        except Exception as e:
            logger.exception(f"Failed to scroll ProcessedContexts: {e}")

    @_require_backend("_vector_backend", default=0)
    async def get_processed_context_count(self, context_type: str) -> int:
        """Get record count for specified context_type"""
        try:
            return await self._vector_backend.get_processed_context_count(context_type)
        except Exception as e:
            logger.exception(f"Failed to get {context_type} record count: {e}")
            return 0

    @_require_backend("_vector_backend", default={})
    async def get_all_processed_context_counts(self) -> Dict[str, int]:
        """Get record count for all context_type"""
        try:
            return await self._vector_backend.get_all_processed_context_counts()
        except Exception as e:
            logger.exception(f"Failed to get all context_type record counts: {e}")
            return {}

    def get_available_context_types(self) -> List[str]:
        """Get all available context_type - all ProcessedContext use vector database"""
        # Return all ContextType enum values, as all ProcessedContext are stored in vector database
        from opencontext.models.enums import ContextType

        return [ct.value for ct in ContextType]

    @_require_backend("_vector_backend", default=[])
    async def search(
        self,
        query: Vectorize,
        top_k: int = 10,
        context_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Vector search, supports context_type filtering

        Args:
            query: Query vectorize object
            top_k: Maximum number of results to return
            context_types: List of context types to search
            filters: Additional filter conditions
            user_id: User identifier for multi-user filtering
            device_id: Device identifier for multi-user filtering
            agent_id: Agent identifier for multi-user filtering
        """
        try:
            # Execute vector search
            search_results = await self._vector_backend.search(
                query=query,
                top_k=top_k,
                context_types=context_types,
                filters=filters,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            )

            return search_results

        except Exception as e:
            logger.exception(f"Vector search failed: {e}")
            return []

    def _is_consumption_enabled(self) -> bool:
        """Check if consumption module is enabled in configuration."""
        try:
            from opencontext.config.global_config import GlobalConfig

            return (
                GlobalConfig.get_instance().get_config().get("consumption", {}).get("enabled", True)
            )
        except Exception as e:
            logger.debug(f"Config check for consumption failed, defaulting to enabled: {e}")
            return True  # Default to enabled if config not available

    async def upsert_todo_embedding(
        self,
        todo_id: int,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store todo embedding to vector database for deduplication"""
        if not self._is_consumption_enabled():
            logger.debug("Consumption disabled, skipping todo embedding upsert")
            return False
        if not self._initialized or not self._vector_backend:
            logger.warning("Storage not initialized, cannot store todo embedding")
            return False

        return await self._vector_backend.upsert_todo_embedding(
            todo_id, content, embedding, metadata
        )

    async def search_similar_todos(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[int, str, float]]:
        """Search for similar todos using vector similarity"""
        if not self._is_consumption_enabled():
            logger.debug("Consumption disabled, skipping todo search")
            return []
        if not self._initialized or not self._vector_backend:
            logger.warning("Storage not initialized, cannot search todos")
            return []

        return await self._vector_backend.search_similar_todos(
            query_embedding, top_k, similarity_threshold
        )

    async def delete_todo_embedding(self, todo_id: int) -> bool:
        """Delete todo embedding from vector database"""
        if not self._is_consumption_enabled():
            logger.debug("Consumption disabled, skipping todo embedding delete")
            return False
        if not self._initialized or not self._vector_backend:
            logger.warning("Storage not initialized, cannot delete todo embedding")
            return False

        return await self._vector_backend.delete_todo_embedding(todo_id)

    @_require_backend("_document_backend")
    async def get_document(self, doc_id: str) -> Optional[DocumentData]:
        """Get document"""
        return await self._document_backend.get(doc_id)

    @_require_backend("_document_backend")
    async def query_documents(
        self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> Optional[QueryResult]:
        """Query documents"""
        return await self._document_backend.query(query, limit, filters)

    async def delete_document(self, doc_id: str) -> bool:
        """Delete document"""
        if not self._initialized:
            logger.error("Unified storage system not initialized")
            return False

        # Try to delete from all backends
        if self._document_backend:
            await self._document_backend.delete(doc_id)
            return True
        return False

    @_require_backend("_document_backend")
    async def create_conversation(
        self,
        page_name: str,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a new conversation record."""
        return await self._document_backend.create_conversation(
            page_name=page_name,
            user_id=user_id,
            title=title,
            metadata=metadata,
        )

    @_require_backend("_document_backend")
    async def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Query single conversation details."""
        return await self._document_backend.get_conversation(conversation_id)

    @_require_backend("_document_backend", default={"items": [], "total": 0})
    async def get_conversation_list(
        self,
        limit: int = 20,
        offset: int = 0,
        page_name: Optional[str] = None,
        user_id: Optional[str] = None,
        status: str = "active",
    ) -> Dict[str, Any]:
        """List conversations with pagination/filtering."""
        return await self._document_backend.get_conversation_list(
            limit=limit,
            offset=offset,
            page_name=page_name,
            user_id=user_id,
            status=status,
        )

    @_require_backend("_document_backend")
    async def update_conversation(
        self,
        conversation_id: int,
        title: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update conversation metadata (title/status)."""
        return await self._document_backend.update_conversation(
            conversation_id=conversation_id,
            title=title,
            status=status,
        )

    async def delete_conversation(self, conversation_id: int) -> Dict[str, Any]:
        """Soft delete conversation."""
        if not self._initialized:
            logger.error("Unified storage system not initialized")
            return {"success": False, "id": conversation_id}

        if not self._document_backend:
            logger.error("Document database backend not initialized")
            return {"success": False, "id": conversation_id}

        return await self._document_backend.delete_conversation(conversation_id)

    @_require_backend("_document_backend")
    async def insert_vaults(
        self,
        title: str,
        summary: str,
        content: str,
        document_type: str,
        tags: str = None,
        parent_id: int = None,
        is_folder: bool = False,
    ) -> int:
        """Insert report"""
        return await self._document_backend.insert_vaults(
            title, summary, content, document_type, tags, parent_id, is_folder
        )

    @_require_backend("_document_backend", default=False)
    async def update_vault(
        self,
        vault_id: int,
        title: str = None,
        content: str = None,
        summary: str = None,
        tags: str = None,
        is_deleted: bool = None,
    ) -> bool:
        """Update report"""
        # Build kwargs, only include non-None values
        kwargs = {}
        if title is not None:
            kwargs["title"] = title
        if content is not None:
            kwargs["content"] = content
        if summary is not None:
            kwargs["summary"] = summary
        if tags is not None:
            kwargs["tags"] = tags
        if is_deleted is not None:
            kwargs["is_deleted"] = is_deleted

        return await self._document_backend.update_vault(vault_id, **kwargs)

    @_require_backend("_document_backend", default=[])
    async def get_reports(
        self, limit: int = 100, offset: int = 0, is_deleted: bool = False
    ) -> List[Dict]:
        """Get report"""
        return await self._document_backend.get_reports(limit, offset, is_deleted)

    @_require_backend("_document_backend", default=[])
    async def get_vaults(
        self,
        limit: int = 100,
        offset: int = 0,
        is_deleted: bool = False,
        document_type: str = None,
        created_after: datetime = None,
        created_before: datetime = None,
        updated_after: datetime = None,
        updated_before: datetime = None,
    ) -> List[Dict]:
        """Get vaults list, supports more filtering conditions"""
        return await self._document_backend.get_vaults(
            limit=limit,
            offset=offset,
            is_deleted=is_deleted,
            document_type=document_type,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
        )

    @_require_backend("_document_backend")
    async def get_vault(self, vault_id: int) -> Optional[Dict]:
        """Get vaults by ID"""
        return await self._document_backend.get_vault(vault_id)

    @_require_backend("_document_backend")
    async def insert_todo(
        self,
        content: str,
        start_time: datetime = None,
        end_time: datetime = None,
        status: int = 0,
        urgency: int = 0,
        assignee: str = None,
        reason: str = None,
    ) -> int:
        """Insert todo item"""
        return await self._document_backend.insert_todo(
            content, start_time, end_time, status, urgency, assignee, reason
        )

    @_require_backend("_document_backend", default=[])
    async def get_todos(
        self,
        status: int = None,
        limit: int = 100,
        offset: int = 0,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> List[Dict]:
        """Get todo items"""
        return await self._document_backend.get_todos(status, limit, offset, start_time, end_time)

    @_require_backend("_document_backend")
    async def insert_tip(self, content: str) -> int:
        """Insert tip"""
        return await self._document_backend.insert_tip(content)

    @_require_backend("_document_backend", default=[])
    async def get_tips(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """Get tips"""
        return await self._document_backend.get_tips(start_time, end_time, limit, offset)

    @_require_backend("_document_backend", default=False)
    async def update_todo_status(
        self, todo_id: int, status: int, end_time: datetime = None
    ) -> bool:
        """Update todo item status"""
        return await self._document_backend.update_todo_status(
            todo_id=todo_id, status=status, end_time=end_time
        )

    # Monitoring data operations - delegated to document backend
    async def save_monitoring_token_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ) -> bool:
        """Save token usage monitoring data"""
        return await self._document_backend.save_monitoring_token_usage(
            model, prompt_tokens, completion_tokens, total_tokens
        )

    async def save_monitoring_stage_timing(
        self,
        stage_name: str,
        duration_ms: int,
        status: str = "success",
        metadata: Optional[str] = None,
    ) -> bool:
        """Save stage timing monitoring data"""
        return await self._document_backend.save_monitoring_stage_timing(
            stage_name, duration_ms, status, metadata
        )

    async def save_monitoring_data_stats(
        self,
        data_type: str,
        count: int = 1,
        context_type: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> bool:
        """Save data statistics monitoring data"""
        return await self._document_backend.save_monitoring_data_stats(
            data_type, count, context_type, metadata
        )

    async def query_monitoring_token_usage(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Query token usage monitoring data"""
        return await self._document_backend.query_monitoring_token_usage(hours)

    async def query_monitoring_stage_timing(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Query stage timing monitoring data"""
        return await self._document_backend.query_monitoring_stage_timing(hours)

    async def query_monitoring_data_stats(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Query data statistics monitoring data"""
        return await self._document_backend.query_monitoring_data_stats(hours)

    async def query_monitoring_data_stats_by_range(
        self, start_time: Any, end_time: Any
    ) -> List[Dict[str, Any]]:
        """Query data statistics monitoring data by custom time range"""
        return await self._document_backend.query_monitoring_data_stats_by_range(
            start_time, end_time
        )

    async def query_monitoring_data_stats_trend(
        self, hours: int = 24, interval_hours: int = 1
    ) -> List[Dict[str, Any]]:
        """Query data statistics trend with time grouping"""
        return await self._document_backend.query_monitoring_data_stats_trend(hours, interval_hours)

    async def cleanup_old_monitoring_data(self, days: int = 7) -> bool:
        """Clean up monitoring data older than specified days"""
        return await self._document_backend.cleanup_old_monitoring_data(days)

    # Message management operations - delegated to document backend
    @_require_backend("_document_backend")
    async def create_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        is_complete: bool = True,
        token_count: int = 0,
        parent_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Create a new message, returns message ID"""
        result = await self._document_backend.create_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            is_complete=is_complete,
            token_count=token_count,
            parent_message_id=parent_message_id,
            metadata=metadata,
        )
        # create_message returns a dict, extract the ID
        return result.get("id") if result else None

    @_require_backend("_document_backend")
    async def create_streaming_message(
        self,
        conversation_id: int,
        role: str,
        parent_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Create a streaming message (initial content is empty), returns message ID"""
        result = await self._document_backend.create_streaming_message(
            conversation_id=conversation_id,
            role=role,
            parent_message_id=parent_message_id,
            metadata=metadata,
        )
        # create_streaming_message returns a dict, extract the ID
        return result.get("id") if result else None

    @_require_backend("_document_backend")
    async def update_message(
        self,
        message_id: int,
        new_content: str,
        is_complete: Optional[bool] = None,
        token_count: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update message content"""
        return await self._document_backend.update_message(
            message_id=message_id,
            new_content=new_content,
            is_complete=is_complete,
            token_count=token_count,
        )

    @_require_backend("_document_backend", default=False)
    async def append_message_content(
        self, message_id: int, content_chunk: str, token_count: int = 0
    ) -> bool:
        """Append content to a streaming message"""
        return await self._document_backend.append_message_content(
            message_id=message_id, content_chunk=content_chunk, token_count=token_count
        )

    @_require_backend("_document_backend", default=False)
    async def update_message_metadata(self, message_id: int, metadata: Dict[str, Any]) -> bool:
        """Update message metadata"""
        return await self._document_backend.update_message_metadata(
            message_id=message_id, metadata=metadata
        )

    @_require_backend("_document_backend", default=False)
    async def mark_message_finished(
        self, message_id: int, status: str = "completed", error_message: Optional[str] = None
    ) -> bool:
        """Mark a message as finished (completed, failed, or cancelled)"""
        return await self._document_backend.mark_message_finished(
            message_id=message_id, status=status, error_message=error_message
        )

    @_require_backend("_document_backend")
    async def get_message(self, message_id: int) -> Optional[Dict[str, Any]]:
        """Get a single message"""
        return await self._document_backend.get_message(message_id)

    @_require_backend("_document_backend", default=[])
    async def get_conversation_messages(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get all messages for a conversation"""
        return await self._document_backend.get_conversation_messages(conversation_id)

    @_require_backend("_document_backend", default=False)
    async def delete_message(self, message_id: int) -> bool:
        """Delete a message"""
        return await self._document_backend.delete_message(message_id)

    @_require_backend("_document_backend", default=False)
    async def interrupt_message(self, message_id: int) -> bool:
        """Interrupt message generation (mark as cancelled)"""
        return await self._document_backend.interrupt_message(message_id)

    # Message Thinking operations - delegated to document backend
    @_require_backend("_document_backend")
    async def add_message_thinking(
        self,
        message_id: int,
        content: str,
        stage: Optional[str] = None,
        progress: float = 0.0,
        sequence: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Add a thinking record to a message"""
        return await self._document_backend.add_message_thinking(
            message_id=message_id,
            content=content,
            stage=stage,
            progress=progress,
            sequence=sequence,
            metadata=metadata,
        )

    @_require_backend("_document_backend", default=[])
    async def get_message_thinking(self, message_id: int) -> List[Dict[str, Any]]:
        """Get all thinking records for a message"""
        return await self._document_backend.get_message_thinking(message_id)

    @_require_backend("_document_backend", default=False)
    async def clear_message_thinking(self, message_id: int) -> bool:
        """Clear all thinking records for a message"""
        return await self._document_backend.clear_message_thinking(message_id)

    # ── Profile routing (→ relational DB) ──

    @_require_backend("_document_backend", default=False)
    async def upsert_profile(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        content: str = "",
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        importance: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store/update user profile → relational DB"""
        return await self._document_backend.upsert_profile(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            content=content,
            summary=summary,
            keywords=keywords,
            entities=entities,
            importance=importance,
            metadata=metadata,
        )

    @_require_backend("_document_backend")
    async def get_profile(
        self, user_id: str, device_id: str = "default", agent_id: str = "default"
    ) -> Optional[Dict]:
        """Get user profile → relational DB"""
        return await self._document_backend.get_profile(user_id, device_id, agent_id)

    @_require_backend("_document_backend", default=False)
    async def delete_profile(
        self, user_id: str, device_id: str = "default", agent_id: str = "default"
    ) -> bool:
        """Delete user profile → relational DB"""
        return await self._document_backend.delete_profile(user_id, device_id, agent_id)

    # ── Entity routing (→ relational DB) ──

    @_require_backend("_document_backend", default="")
    async def upsert_entity(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        entity_name: str = "",
        content: str = "",
        entity_type: Optional[str] = None,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store/update entity → relational DB"""
        return await self._document_backend.upsert_entity(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            entity_name=entity_name,
            content=content,
            entity_type=entity_type,
            summary=summary,
            keywords=keywords,
            aliases=aliases,
            metadata=metadata,
        )

    @_require_backend("_document_backend")
    async def get_entity(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        entity_name: str = "",
    ) -> Optional[Dict]:
        """Get entity → relational DB"""
        return await self._document_backend.get_entity(user_id, device_id, agent_id, entity_name)

    @_require_backend("_document_backend", default=[])
    async def list_entities(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        entity_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """List entities for a user → relational DB"""
        return await self._document_backend.list_entities(
            user_id, device_id, agent_id, entity_type, limit, offset
        )

    @_require_backend("_document_backend", default=[])
    async def search_entities(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        query_text: str = "",
        limit: int = 20,
    ) -> List[Dict]:
        """Search entities by text → relational DB"""
        return await self._document_backend.search_entities(
            user_id, device_id, agent_id, query_text, limit
        )

    @_require_backend("_document_backend", default=False)
    async def delete_entity(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        entity_name: str = "",
    ) -> bool:
        """Delete entity → relational DB"""
        return await self._document_backend.delete_entity(user_id, device_id, agent_id, entity_name)

    # ── Document overwrite routing (→ vector DB) ──

    @_require_backend("_vector_backend", default=False)
    async def delete_document_chunks(
        self, source_file_key: str, user_id: Optional[str] = None
    ) -> bool:
        """Delete all chunks for a document (for overwrite) → vector DB"""
        return await self._vector_backend.delete_by_source_file(source_file_key, user_id)

    # ── Hierarchy routing (→ vector DB) ──

    @_require_backend("_vector_backend", default=[])
    async def search_hierarchy(
        self,
        context_type: str,
        hierarchy_level: int,
        time_bucket_start: Optional[str] = None,
        time_bucket_end: Optional[str] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Search by hierarchy level and time bucket → vector DB"""
        return await self._vector_backend.search_by_hierarchy(
            context_type=context_type,
            hierarchy_level=hierarchy_level,
            time_bucket_start=time_bucket_start,
            time_bucket_end=time_bucket_end,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            top_k=top_k,
        )

    @_require_backend("_vector_backend", default=[])
    async def get_contexts_by_ids(
        self,
        ids: List[str],
        context_type: Optional[str] = None,
        need_vector: bool = False,
    ) -> List[ProcessedContext]:
        """Get contexts by IDs → vector DB"""
        return await self._vector_backend.get_by_ids(ids, context_type, need_vector=need_vector)

    @_require_backend("_vector_backend", default=0)
    async def batch_set_parent_id(
        self,
        children_ids: List[str],
        parent_id: str,
        context_type: str,
    ) -> int:
        """Set parent_id on child contexts. Delegates to vector backend."""
        return await self._vector_backend.batch_set_parent_id(
            children_ids, parent_id, context_type
        )
