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
from typing import Any, Dict, List, Optional, Tuple, Union

from opencontext.models.context import ProcessedContext, Vectorize


class StorageType(Enum):
    """Enumeration of supported storage types."""

    VECTOR_DB = "vector_db"  # Vector databases: ChromaDB, Qdrant
    DOCUMENT_DB = "document_db"  # Document databases: SQLite, MongoDB


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
    def initialize(self, config: Dict[str, Any]) -> bool:
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
    def get_collection_names(self) -> Optional[List[str]]:
        """Get all collection names in vector database"""

    @abstractmethod
    def delete_contexts(self, ids: List[str], context_type: str) -> bool:
        """Delete contexts of specified type"""

    @abstractmethod
    def upsert_processed_context(self, context: ProcessedContext) -> str:
        """Store processed context"""

    @abstractmethod
    def batch_upsert_processed_context(self, contexts: List[ProcessedContext]) -> List[str]:
        """Batch store processed contexts"""

    @abstractmethod
    def get_all_processed_contexts(
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
        """

    @abstractmethod
    def get_processed_context(self, id: str, context_type: str) -> ProcessedContext:
        """Get specified context"""

    @abstractmethod
    def delete_processed_context(self, id: str, context_type: str) -> bool:
        """Delete specified context"""

    @abstractmethod
    def search(
        self,
        query: Vectorize,
        top_k: int = 10,
        context_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
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
        """

    @abstractmethod
    def upsert_todo_embedding(
        self,
        todo_id: int,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store todo embedding to vector database for deduplication

        Args:
            todo_id: Todo ID
            content: Todo content text
            embedding: Text embedding vector
            metadata: Optional metadata (urgency, priority, etc.)

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def search_similar_todos(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[int, str, float]]:
        """Search for similar todos using vector similarity

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold (0-1)

        Returns:
            List of (todo_id, content, similarity_score) tuples
        """

    @abstractmethod
    def delete_todo_embedding(self, todo_id: int) -> bool:
        """Delete todo embedding from vector database

        Args:
            todo_id: Todo ID to delete

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def get_processed_context_count(self, context_type: str) -> int:
        """Get record count for specified context_type"""

    @abstractmethod
    def get_all_processed_context_counts(self) -> Dict[str, int]:
        """Get record counts for all context_types"""

    @abstractmethod
    def delete_by_source_file(self, source_file_key: str, user_id: Optional[str] = None) -> bool:
        """Delete all chunks belonging to a source file (for document overwrite)

        Args:
            source_file_key: Source file key (format: "user_id:file_path")
            user_id: Optional user identifier for additional filtering

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def search_by_hierarchy(
        self,
        context_type: str,
        hierarchy_level: int,
        time_bucket_start: Optional[str] = None,
        time_bucket_end: Optional[str] = None,
        user_id: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Search contexts by hierarchy level and time bucket range

        Args:
            context_type: Context type to search
            hierarchy_level: Hierarchy level to search (0=original, 1=daily, 2=weekly, 3=monthly)
            time_bucket_start: Start of time bucket range (inclusive)
            time_bucket_end: End of time bucket range (inclusive)
            user_id: User identifier for multi-user filtering
            top_k: Maximum number of results

        Returns:
            List of (context, score) tuples
        """

    @abstractmethod
    def get_by_ids(
        self, ids: List[str], context_type: Optional[str] = None
    ) -> List[ProcessedContext]:
        """Get contexts by their IDs

        Args:
            ids: List of context IDs to retrieve
            context_type: Optional context type for routing to correct collection

        Returns:
            List of ProcessedContext objects
        """


class IDocumentStorageBackend(IStorageBackend):
    """Document storage backend interface"""

    @abstractmethod
    def insert_vaults(
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
    def get_reports(
        self, limit: int = 100, offset: int = 0, is_deleted: bool = False
    ) -> List[Dict]:
        """Get reports"""

    @abstractmethod
    def get_vaults(
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
    def get_vault(self, vault_id: int) -> Optional[Dict]:
        """Get vault by ID"""
        pass

    @abstractmethod
    def update_vault(self, vault_id: int, **kwargs) -> bool:
        """Update vault"""
        pass

    @abstractmethod
    def insert_todo(
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
    def get_todos(
        self,
        status: int = None,
        limit: int = 100,
        offset: int = 0,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> List[Dict]:
        """Get todo items"""

    @abstractmethod
    def insert_activity(
        self,
        title: str,
        content: str,
        resources: str = None,
        metadata: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> int:
        """Insert activity

        Args:
            title: Activity title
            content: Activity content
            resources: Resource information (JSON string)
            metadata: Metadata information (JSON string), including categories, insights, etc.
            start_time: Start time
            end_time: End time

        Returns:
            int: Activity record ID
        """

    @abstractmethod
    def get_activities(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """Get activities"""

    @abstractmethod
    def insert_tip(self, content: str) -> int:
        """Insert tip"""

    @abstractmethod
    def get_tips(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get tips"""

    @abstractmethod
    def update_todo_status(self, todo_id: int, status: int, end_time: datetime = None) -> bool:
        """Update todo item status"""

    # ── Profile CRUD ──

    @abstractmethod
    def upsert_profile(
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
        """Insert or update user profile (composite key: user_id + device_id + agent_id)

        Args:
            user_id: User identifier
            device_id: Device identifier
            agent_id: Agent identifier (same user can have different profiles per agent)
            content: Full profile text (LLM-merged result)
            summary: Profile summary
            keywords: Keywords list
            entities: Entities list
            importance: Importance score
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def get_profile(
        self, user_id: str, device_id: str = "default", agent_id: str = "default"
    ) -> Optional[Dict]:
        """Get user profile by composite key

        Args:
            user_id: User identifier
            device_id: Device identifier
            agent_id: Agent identifier

        Returns:
            Profile dict or None if not found
        """

    @abstractmethod
    def delete_profile(
        self, user_id: str, device_id: str = "default", agent_id: str = "default"
    ) -> bool:
        """Delete user profile

        Args:
            user_id: User identifier
            device_id: Device identifier
            agent_id: Agent identifier

        Returns:
            True if successful, False otherwise
        """

    # ── Entity CRUD ──

    @abstractmethod
    def upsert_entity(
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
        """Insert or update entity (unique key: user_id + device_id + agent_id + entity_name)

        Args:
            user_id: Owner user identifier
            device_id: Device identifier
            agent_id: Agent identifier
            entity_name: Entity name (unique per user+device+agent)
            content: Entity description (LLM-merged result)
            entity_type: Entity type (person/project/team/org/other)
            summary: Entity summary
            keywords: Keywords list
            aliases: Alias names list
            metadata: Additional metadata

        Returns:
            Entity ID string
        """

    @abstractmethod
    def get_entity(
        self, user_id: str, device_id: str = "default", agent_id: str = "default",
        entity_name: str = "",
    ) -> Optional[Dict]:
        """Get entity by user_id + device_id + agent_id + entity_name

        Args:
            user_id: Owner user identifier
            device_id: Device identifier
            agent_id: Agent identifier
            entity_name: Entity name

        Returns:
            Entity dict or None if not found
        """

    @abstractmethod
    def list_entities(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        entity_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """List entities for a user

        Args:
            user_id: Owner user identifier
            device_id: Device identifier
            agent_id: Agent identifier
            entity_type: Optional filter by entity type
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of entity dicts
        """

    @abstractmethod
    def search_entities(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        query_text: str = "",
        limit: int = 20,
    ) -> List[Dict]:
        """Search entities by text (name, content, aliases)

        Args:
            user_id: Owner user identifier
            device_id: Device identifier
            agent_id: Agent identifier
            query_text: Search text
            limit: Maximum number of results

        Returns:
            List of matching entity dicts
        """

    @abstractmethod
    def delete_entity(
        self, user_id: str, device_id: str = "default", agent_id: str = "default",
        entity_name: str = "",
    ) -> bool:
        """Delete entity

        Args:
            user_id: Owner user identifier
            device_id: Device identifier
            agent_id: Agent identifier
            entity_name: Entity name

        Returns:
            True if successful, False otherwise
        """
