#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Unified storage base interfaces and abstract classes.
Defines contracts for different storage backend types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from opencontext.models.context import ProcessedContext, Vectorize


class StorageType(Enum):
    """Enumeration of supported storage types."""

    VECTOR_DB = "vector_db"  # Vector databases: Qdrant, VikingDB
    DOCUMENT_DB = "document_db"  # Document databases: SQLite, MySQL


class DataType(Enum):
    """Enumeration of supported data types."""

    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    MARKDOWN = "markdown"


@dataclass
class StorageConfig:
    """Base storage configuration class"""

    storage_type: StorageType
    name: str
    config: Dict[str, Any]


@dataclass
class DocumentData:
    """Document data structure"""

    id: str
    content: str
    metadata: Dict[str, Any]
    data_type: DataType = DataType.TEXT
    images: Optional[List[str]] = None  # Image paths or base64 data
    files: Optional[List[str]] = None  # Attachment file paths


@dataclass
class QueryResult:
    """Query result"""

    documents: List[DocumentData]
    total_count: int
    scores: Optional[List[float]] = None


class IStorageBackend(ABC):
    """Storage backend interface"""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize storage backend"""

    @abstractmethod
    def get_name(self) -> str:
        """Get storage backend name"""

    @abstractmethod
    def get_storage_type(self) -> StorageType:
        """Get storage type"""


class IVectorStorageBackend(IStorageBackend):
    """Vector storage backend interface"""

    @abstractmethod
    async def get_collection_names(self) -> Optional[List[str]]:
        """Get all collection names in vector database"""

    @abstractmethod
    async def delete_contexts(self, ids: List[str], context_type: str) -> bool:
        """Delete contexts of specified type"""

    @abstractmethod
    async def upsert_processed_context(self, context: ProcessedContext) -> str:
        """Store processed context"""

    @abstractmethod
    async def batch_upsert_processed_context(self, contexts: List[ProcessedContext]) -> List[str]:
        """Batch store processed contexts"""

    @abstractmethod
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
        skip_slice: bool = False,
    ) -> Dict[str, List[ProcessedContext]]:
        """Get processed contexts

        Args:
            context_types: List of context types to get
            limit: Maximum number of results per context type
            offset: Offset for pagination
            filter: Additional filter conditions
            need_vector: Whether to include vectors in results
            user_id: User identifier for multi-user filtering
            device_id: Device identifier for multi-user filtering
            agent_id: Agent identifier for multi-user filtering
            skip_slice: If True, skip per-type offset/limit slicing (caller handles global slice)
        """

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

        Uses backend-native cursors where available (e.g., Qdrant scroll) to
        avoid O(n^2) offset-based pagination. Backends that don't override this
        get a default implementation using get_all_processed_contexts with offset.

        Args:
            context_types: List of context types to iterate
            batch_size: Number of contexts per internal fetch
            filter: Additional filter conditions
            need_vector: Whether to include vectors in results
            user_id: User identifier for multi-user filtering
            device_id: Device identifier for multi-user filtering
            agent_id: Agent identifier for multi-user filtering

        Yields:
            ProcessedContext objects
        """
        if not context_types:
            context_types = list(await self.get_collection_names() or [])

        for context_type in context_types:
            offset = 0
            while True:
                result = await self.get_all_processed_contexts(
                    context_types=[context_type],
                    limit=batch_size,
                    offset=offset,
                    filter=filter,
                    need_vector=need_vector,
                    user_id=user_id,
                    device_id=device_id,
                    agent_id=agent_id,
                )
                batch = result.get(context_type, [])
                if not batch:
                    break
                for item in batch:
                    yield item
                if len(batch) < batch_size:
                    break
                offset += batch_size

    @abstractmethod
    async def get_processed_context(self, id: str, context_type: str) -> ProcessedContext:
        """Get specified context"""

    @abstractmethod
    async def delete_processed_context(self, id: str, context_type: str) -> bool:
        """Delete specified context"""

    @abstractmethod
    async def search(
        self,
        query: Vectorize,
        top_k: int = 10,
        context_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Vector similarity search

        Args:
            query: Query vectorize object
            top_k: Maximum number of results to return
            context_types: List of context types to search
            filters: Additional filter conditions
            user_id: User identifier for multi-user filtering
            device_id: Device identifier for multi-user filtering
            agent_id: Agent identifier for multi-user filtering
            score_threshold: Minimum similarity score (0-1). Results below this are excluded.
                Backends SHOULD implement at database query level; post-query filtering
                acceptable when native support is unavailable.
        """

    @abstractmethod
    async def get_processed_context_count(
        self,
        context_type: str,
        filter: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """Get record count for specified context_type"""

    @abstractmethod
    async def get_all_processed_context_counts(self) -> Dict[str, int]:
        """Get record counts for all context_types"""

    @abstractmethod
    async def search_by_hierarchy(
        self,
        context_type: str,
        hierarchy_level: int,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Search contexts by hierarchy level and numeric timestamp range overlap.

        Returns contexts whose [event_time_start, event_time_end] overlaps
        with the query range [time_start, time_end] (UTC timestamps as floats).

        Args:
            context_type: Context type to search
            hierarchy_level: Hierarchy level to search (0=original, 1=daily, 2=weekly, 3=monthly)
            time_start: Start of query time range as UTC timestamp (inclusive), or None for unbounded
            time_end: End of query time range as UTC timestamp (inclusive), or None for unbounded
            user_id: User identifier for multi-user filtering
            device_id: Device identifier for multi-device isolation
            agent_id: Agent identifier for multi-agent isolation
            top_k: Maximum number of results

        Returns:
            List of (context, score) tuples
        """

    @abstractmethod
    async def get_by_ids(
        self,
        ids: List[str],
        context_type: Optional[str] = None,
        need_vector: bool = False,
    ) -> List[ProcessedContext]:
        """Get contexts by their IDs

        Args:
            ids: List of context IDs to retrieve
            context_type: Optional context type for routing to correct collection
            need_vector: Whether to include vectors in results

        Returns:
            List of ProcessedContext objects
        """

    async def batch_update_refs(
        self,
        context_ids: List[str],
        ref_key: str,
        ref_value: str,
        context_type: str,
    ) -> int:
        """Add a ref entry (ref_key -> ref_value) to the refs dict of multiple contexts.

        Default implementation: no-op. Backends with refs support should override.
        """
        raise NotImplementedError


class IDocumentStorageBackend(IStorageBackend):
    """Document storage backend interface"""

    @abstractmethod
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
        """Insert vault"""

    @abstractmethod
    async def get_reports(
        self, limit: int = 100, offset: int = 0, is_deleted: bool = False
    ) -> List[Dict]:
        """Get reports"""

    @abstractmethod
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
        """Get vaults"""

    @abstractmethod
    async def get_vault(self, vault_id: int) -> Optional[Dict]:
        """Get vault by ID"""

    @abstractmethod
    async def update_vault(self, vault_id: int, **kwargs) -> bool:
        """Update vault"""

    @abstractmethod
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

    @abstractmethod
    async def get_todos(
        self,
        status: int = None,
        limit: int = 100,
        offset: int = 0,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> List[Dict]:
        """Get todo items"""

    @abstractmethod
    async def insert_tip(self, content: str) -> int:
        """Insert tip"""

    @abstractmethod
    async def get_tips(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get tips"""

    @abstractmethod
    async def update_todo_status(
        self, todo_id: int, status: int, end_time: datetime = None
    ) -> bool:
        """Update todo item status"""

    # ── Profile CRUD ──

    @abstractmethod
    async def upsert_profile(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        factual_profile: str = "",
        behavioral_profile: Optional[str] = None,
        entities: Optional[List[str]] = None,
        importance: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        refs: Optional[Dict] = None,
        context_type: str = "profile",
    ) -> bool:
        """Insert or update user profile (composite key: user_id + device_id + agent_id + context_type)

        Args:
            user_id: User identifier
            device_id: Device identifier
            agent_id: Agent identifier (same user can have different profiles per agent)
            factual_profile: Factual profile text (LLM-merged result)
            behavioral_profile: Behavioral profile text
            entities: Entities list
            importance: Importance score
            metadata: Additional metadata
            refs: Reference links (e.g. source context IDs)
            context_type: Context type ('profile', 'agent_profile', or 'agent_base_profile')

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    async def get_profile(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        context_type: str = "profile",
    ) -> Optional[Dict]:
        """Get user profile by composite key

        Args:
            user_id: User identifier
            device_id: Device identifier
            agent_id: Agent identifier
            context_type: Context type ('profile', 'agent_profile', or 'agent_base_profile')

        Returns:
            Profile dict or None if not found
        """

    @abstractmethod
    async def delete_profile(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        context_type: str = "profile",
    ) -> bool:
        """Delete user profile

        Args:
            user_id: User identifier
            device_id: Device identifier
            agent_id: Agent identifier
            context_type: Context type ('profile', 'agent_profile', or 'agent_base_profile')

        Returns:
            True if successful, False otherwise
        """

    # ── Agent Registry ──

    async def create_agent(self, agent_id: str, name: str, description: str = "") -> bool:
        """Register a new agent."""
        raise NotImplementedError

    async def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get agent by ID (excludes soft-deleted)."""
        raise NotImplementedError

    async def list_agents(self) -> List[Dict]:
        """List all active agents."""
        raise NotImplementedError

    async def update_agent(
        self, agent_id: str, name: Optional[str] = None, description: Optional[str] = None
    ) -> bool:
        """Update agent info."""
        raise NotImplementedError

    async def delete_agent(self, agent_id: str) -> bool:
        """Soft delete agent."""
        raise NotImplementedError

    # ── Chat Batches ──

    async def create_chat_batch(
        self,
        batch_id: str,
        messages: List[Dict],
        user_id: Optional[str],
        device_id: str = "default",
        agent_id: str = "default",
    ) -> bool:
        """Persist a chat batch. batch_id is app-generated UUID."""
        raise NotImplementedError

    async def cleanup_chat_batches(self, retention_days: int = 90) -> int:
        """Delete chat batches older than retention_days."""
        raise NotImplementedError

    async def list_chat_batches(
        self,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict]:
        """List chat batches (without messages) with optional filters."""
        raise NotImplementedError

    async def count_chat_batches(
        self,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """Count chat batches matching filters."""
        raise NotImplementedError

    async def get_chat_batch(self, batch_id: str) -> Optional[Dict]:
        """Get single chat batch with messages."""
        raise NotImplementedError
