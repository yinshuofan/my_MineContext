#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
DashVector vector storage backend - Aliyun DashVector Service
https://help.aliyun.com/product/2510217.html
"""

import datetime
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import dashvector
    from dashvector import Doc
    DASHVECTOR_AVAILABLE = True
except ImportError:
    DASHVECTOR_AVAILABLE = False
    dashvector = None
    Doc = None

from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    Vectorize,
)
from opencontext.models.enums import ContentFormat, ContextType
from opencontext.storage.base_storage import IVectorStorageBackend, StorageType
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Collection and field constants
TODO_COLLECTION = "todo"
FIELD_DOCUMENT = "document"
FIELD_ORIGINAL_ID = "original_id"
FIELD_TODO_ID = "todo_id"
FIELD_CONTENT = "content"
FIELD_CREATED_AT = "created_at"
FIELD_USER_ID = "user_id"
FIELD_DEVICE_ID = "device_id"
FIELD_AGENT_ID = "agent_id"


class DashVectorBackend(IVectorStorageBackend):
    """
    DashVector vector storage backend.
    Aliyun DashVector is a fully managed vector search service.
    
    Features:
    - Cloud-native, no infrastructure management
    - High availability with SLA guarantee
    - Partition support for multi-tenant scenarios
    - SQL-style filter syntax
    """

    def __init__(self):
        self._client: Optional[Any] = None
        self._collections: Dict[str, Any] = {}  # context_type -> collection object
        self._initialized = False
        self._config = None
        self._vector_size = None
        self._max_retry_count = 3
        self._retry_delay = 1.0

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the DashVector backend.
        
        Args:
            config: Configuration dictionary containing:
                - config.api_key: DashVector API key
                - config.endpoint: DashVector cluster endpoint
                - config.vector_size: Vector dimension (default: 1536)
                - config.timeout: Request timeout in seconds (default: 30.0)
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not DASHVECTOR_AVAILABLE:
            logger.error("DashVector SDK not installed. Please install with: pip install dashvector")
            return False

        try:
            self._config = config
            dashvector_config = config.get("config", {})

            # Get configuration
            api_key = dashvector_config.get("api_key")
            endpoint = dashvector_config.get("endpoint")
            self._vector_size = dashvector_config.get("vector_size", 1536)
            timeout = dashvector_config.get("timeout", 30.0)

            if not api_key or not endpoint:
                raise ValueError("DashVector requires api_key and endpoint configuration")

            # Create client
            self._client = dashvector.Client(
                api_key=api_key,
                endpoint=endpoint,
                timeout=timeout
            )

            # Verify connection
            if not self._client:
                raise RuntimeError("Failed to create DashVector client")

            # Create collections for each context_type
            context_types = [ct.value for ct in ContextType]
            for context_type in context_types:
                collection_name = f"{context_type}"
                self._ensure_collection(collection_name)
                collection = self._client.get(collection_name)
                if collection:
                    self._collections[context_type] = collection
                else:
                    logger.warning(f"Failed to get collection: {collection_name}")

            # Create dedicated todo collection
            self._ensure_collection(TODO_COLLECTION)
            todo_collection = self._client.get(TODO_COLLECTION)
            if todo_collection:
                self._collections[TODO_COLLECTION] = todo_collection

            self._initialized = True
            logger.info(
                f"DashVector backend initialized successfully, "
                f"created {len(self._collections)} collections"
            )
            return True

        except Exception as e:
            logger.exception(f"DashVector backend initialization failed: {e}")
            return False

    def _ensure_collection(self, collection_name: str) -> None:
        """
        Ensure collection exists, create if not.
        
        Args:
            collection_name: Name of the collection to ensure
        """
        try:
            # Check if collection exists
            existing_collections = self._client.list()
            
            if collection_name not in existing_collections:
                # Create new collection with predefined schema
                ret = self._client.create(
                    name=collection_name,
                    dimension=self._vector_size,
                    metric='cosine',
                    dtype=float,
                    fields_schema={
                        FIELD_ORIGINAL_ID: str,
                        'context_type': str,
                        FIELD_DOCUMENT: str,
                        FIELD_CREATED_AT: str,
                        FIELD_USER_ID: str,
                        FIELD_DEVICE_ID: str,
                        FIELD_AGENT_ID: str,
                        # Additional fields for flexibility
                        'title': str,
                        'summary': str,
                        'source': str,
                        'confidence': float,
                    },
                    timeout=-1  # Async creation
                )
                
                if ret:
                    logger.info(f"Created DashVector collection: {collection_name}")
                    # Wait for collection to be ready
                    self._wait_for_collection_ready(collection_name)
                else:
                    logger.warning(
                        f"Failed to create collection {collection_name}: {ret.message}"
                    )
            else:
                logger.debug(f"DashVector collection already exists: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name}: {e}")
            raise

    def _wait_for_collection_ready(
        self, 
        collection_name: str, 
        max_wait: int = 60
    ) -> bool:
        """
        Wait for collection to be ready after async creation.
        
        Args:
            collection_name: Name of the collection
            max_wait: Maximum wait time in seconds
            
        Returns:
            True if collection is ready, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                collection = self._client.get(collection_name)
                if collection:
                    # Try to describe to verify it's ready
                    desc = collection.describe()
                    if desc and desc.output:
                        return True
            except Exception:
                pass
            time.sleep(1)
        
        logger.warning(f"Timeout waiting for collection {collection_name} to be ready")
        return False

    def _check_connection(self) -> bool:
        """Check if connection to DashVector is healthy."""
        if not self._client:
            return False

        try:
            # Try to list collections as health check
            self._client.list()
            return True
        except Exception as e:
            logger.warning(f"DashVector health check failed: {e}")
            return False

    def get_name(self) -> str:
        """Get storage backend name."""
        return "dashvector"

    def get_collection_names(self) -> Optional[List[str]]:
        """Get all collection names."""
        return list(self._collections.keys())

    def get_storage_type(self) -> StorageType:
        """Get storage type."""
        return StorageType.VECTOR_DB

    def _ensure_vectorized(self, context: ProcessedContext) -> List[float]:
        """
        Ensure context has a vector, generate if missing.
        
        Args:
            context: ProcessedContext to vectorize
            
        Returns:
            Vector as list of floats
        """
        if not context.vectorize:
            raise ValueError("Vectorize not set on context")
            
        if context.vectorize.vector:
            if not self._vector_size:
                self._vector_size = len(context.vectorize.vector)
            return context.vectorize.vector

        # Generate vector
        do_vectorize(context.vectorize)
        
        if not self._vector_size and context.vectorize.vector:
            self._vector_size = len(context.vectorize.vector)
            
        return context.vectorize.vector

    def _context_to_dashvector_format(
        self, 
        context: ProcessedContext
    ) -> Dict[str, Any]:
        """
        Convert ProcessedContext to DashVector fields format.
        
        Args:
            context: ProcessedContext to convert
            
        Returns:
            Dictionary of fields for DashVector Doc
        """
        fields = {}

        # Add basic context fields
        if context.extracted_data:
            extracted_dict = context.extracted_data.model_dump(exclude_none=True)
            for key, value in extracted_dict.items():
                fields[key] = self._serialize_value(value)

        # Add properties
        if context.properties:
            props_dict = context.properties.model_dump(exclude_none=True)
            for key, value in props_dict.items():
                fields[key] = self._serialize_value(value)

        # Add metadata
        if context.metadata:
            for key, value in context.metadata.items():
                fields[key] = self._serialize_value(value)

        # Add document text for retrieval
        if context.vectorize:
            if context.vectorize.content_format == ContentFormat.TEXT:
                fields[FIELD_DOCUMENT] = context.vectorize.text or ""

        # Add timestamp
        fields[FIELD_CREATED_AT] = datetime.datetime.now().isoformat()

        return fields

    def _serialize_value(self, value: Any) -> Any:
        """
        Serialize value for DashVector storage.
        DashVector fields support: str, int, float, bool, long, List types
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized value
        """
        if value is None:
            return ""
        elif isinstance(value, datetime.datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return str(value)
        elif isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)

    def upsert_processed_context(self, context: ProcessedContext) -> str:
        """
        Store a single ProcessedContext.
        
        Args:
            context: ProcessedContext to store
            
        Returns:
            ID of stored context
        """
        results = self.batch_upsert_processed_context([context])
        return results[0] if results else ""

    def batch_upsert_processed_context(
        self, 
        contexts: List[ProcessedContext]
    ) -> List[str]:
        """
        Batch store ProcessedContexts.
        
        Args:
            contexts: List of ProcessedContext to store
            
        Returns:
            List of stored context IDs
        """
        if not self._initialized:
            raise RuntimeError("DashVector backend not initialized")

        if not self._check_connection():
            raise RuntimeError("DashVector connection not available")

        # Group contexts by type
        contexts_by_type: Dict[str, List[ProcessedContext]] = {}
        for context in contexts:
            context_type = context.extracted_data.context_type.value
            if context_type not in contexts_by_type:
                contexts_by_type[context_type] = []
            contexts_by_type[context_type].append(context)

        stored_ids = []

        for context_type, type_contexts in contexts_by_type.items():
            collection = self._collections.get(context_type)
            if not collection:
                logger.warning(
                    f"No collection found for context_type '{context_type}', "
                    f"skipping storage"
                )
                continue

            docs = []
            doc_ids = []

            for context in type_contexts:
                try:
                    vector = self._ensure_vectorized(context)
                    fields = self._context_to_dashvector_format(context)
                    fields[FIELD_ORIGINAL_ID] = context.id

                    doc = Doc(
                        id=context.id,
                        vector=vector,
                        fields=fields
                    )
                    docs.append(doc)
                    doc_ids.append(context.id)

                except Exception as e:
                    logger.exception(f"Failed to process context {context.id}: {e}")
                    continue

            if not docs:
                continue

            # Upsert with retry
            for attempt in range(self._max_retry_count):
                try:
                    ret = collection.upsert(docs)
                    if ret:
                        stored_ids.extend(doc_ids)
                        logger.debug(
                            f"Stored {len(docs)} contexts to {context_type} collection"
                        )
                        break
                    else:
                        logger.error(
                            f"Batch upsert failed (attempt {attempt + 1}): {ret.message}"
                        )
                except Exception as e:
                    logger.error(
                        f"Batch storing to {context_type} failed "
                        f"(attempt {attempt + 1}): {e}"
                    )
                    if attempt < self._max_retry_count - 1:
                        time.sleep(self._retry_delay * (attempt + 1))
                    continue

        return stored_ids

    def get_processed_context(
        self, 
        id: str, 
        context_type: str,
        need_vector: bool = False
    ) -> Optional[ProcessedContext]:
        """
        Get a specific ProcessedContext by ID.
        
        Args:
            id: Context ID
            context_type: Type of context
            need_vector: Whether to include vector in result
            
        Returns:
            ProcessedContext if found, None otherwise
        """
        if not self._initialized:
            return None

        if context_type not in self._collections:
            return None

        collection = self._collections[context_type]
        
        try:
            ret = collection.fetch(ids=[id])
            
            if ret and ret.output:
                for doc in ret.output:
                    if doc:
                        return self._dashvector_result_to_context(doc, need_vector)
            
            return None

        except Exception as e:
            logger.debug(
                f"Failed to retrieve context {id} from {context_type}: {e}"
            )
            return None

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
        """
        Get all ProcessedContexts with optional filtering.
        
        Args:
            context_types: List of context types to retrieve
            limit: Maximum number of results per type
            offset: Offset for pagination
            filter: Additional filter conditions
            need_vector: Whether to include vectors
            user_id: Filter by user ID
            device_id: Filter by device ID
            agent_id: Filter by agent ID
            
        Returns:
            Dictionary mapping context_type to list of ProcessedContext
        """
        if not self._initialized:
            return {}

        result = {}
        
        if not context_types:
            context_types = [
                k for k in self._collections.keys() if k != TODO_COLLECTION
            ]

        for context_type in context_types:
            if context_type not in self._collections:
                continue

            collection = self._collections[context_type]
            
            try:
                # Build filter string
                filter_str = self._build_filter_string(
                    filter, user_id, device_id, agent_id
                )

                # DashVector doesn't have a direct scroll/list API like Qdrant
                # We use query with a dummy vector or filter-only query
                # For listing all, we can use query without vector (filter only)
                ret = collection.query(
                    topk=limit + offset,
                    filter=filter_str,
                    include_vector=need_vector,
                )

                if ret and ret.output:
                    contexts = []
                    # Apply offset
                    docs = list(ret.output)
                    if offset > 0:
                        docs = docs[offset:]
                    if len(docs) > limit:
                        docs = docs[:limit]

                    for doc in docs:
                        context = self._dashvector_result_to_context(doc, need_vector)
                        if context:
                            contexts.append(context)

                    if contexts:
                        result[context_type] = contexts

            except Exception as e:
                logger.exception(
                    f"Failed to get contexts from {context_type}: {e}"
                )
                continue

        return result

    def delete_processed_context(self, id: str, context_type: str) -> bool:
        """
        Delete a specific ProcessedContext.
        
        Args:
            id: Context ID to delete
            context_type: Type of context
            
        Returns:
            True if successful, False otherwise
        """
        return self.delete_contexts([id], context_type)

    def delete_contexts(self, ids: List[str], context_type: str) -> bool:
        """
        Delete multiple contexts.
        
        Args:
            ids: List of context IDs to delete
            context_type: Type of context
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False

        if context_type not in self._collections:
            return False

        collection = self._collections[context_type]
        
        try:
            ret = collection.delete(ids)
            if ret:
                logger.debug(f"Deleted {len(ids)} contexts from {context_type}")
                return True
            else:
                logger.error(f"Failed to delete contexts: {ret.message}")
                return False
        except Exception as e:
            logger.exception(f"Failed to delete contexts: {e}")
            return False

    def search(
        self,
        query: Vectorize,
        top_k: int = 10,
        context_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        need_vector: bool = False,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Tuple[ProcessedContext, float]]:
        """
        Vector similarity search.
        
        Args:
            query: Query vectorize object
            top_k: Maximum number of results
            context_types: List of context types to search
            filters: Additional filter conditions
            need_vector: Whether to include vectors in results
            user_id: Filter by user ID
            device_id: Filter by device ID
            agent_id: Filter by agent ID
            
        Returns:
            List of (ProcessedContext, score) tuples
        """
        if not self._initialized:
            return []

        # Determine target collections
        target_collections = {}
        if context_types:
            for context_type in context_types:
                if context_type in self._collections:
                    target_collections[context_type] = self._collections[context_type]
                else:
                    logger.warning(f"Collection not found: {context_type}")
        else:
            target_collections = {
                k: v for k, v in self._collections.items() if k != TODO_COLLECTION
            }

        # Get query vector
        query_vector = None
        if query.vector and len(query.vector) > 0:
            query_vector = query.vector
        else:
            do_vectorize(query)
            query_vector = query.vector

        if not query_vector:
            logger.warning("Unable to get query vector, search failed")
            return []

        # Build filter string
        filter_str = self._build_filter_string(filters, user_id, device_id, agent_id)

        all_results = []

        for context_type, collection in target_collections.items():
            try:
                # Check if collection has data
                stats = collection.stats()
                if stats and stats.output and stats.output.total_doc_count == 0:
                    continue

                # Execute query
                ret = collection.query(
                    vector=query_vector,
                    topk=top_k,
                    filter=filter_str,
                    include_vector=need_vector,
                )

                if ret and ret.output:
                    for doc in ret.output:
                        context = self._dashvector_result_to_context(doc, need_vector)
                        if context:
                            score = doc.score if hasattr(doc, 'score') else 0.0
                            all_results.append((context, score))

            except Exception as e:
                logger.exception(
                    f"Vector search failed in {context_type}: {e}"
                )
                continue

        # Sort by score descending
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def _dashvector_result_to_context(
        self, 
        doc: Any,
        need_vector: bool = False
    ) -> Optional[ProcessedContext]:
        """
        Convert DashVector Doc to ProcessedContext.
        
        Args:
            doc: DashVector Doc object
            need_vector: Whether to include vector
            
        Returns:
            ProcessedContext if conversion successful, None otherwise
        """
        try:
            if not doc or not doc.id:
                return None

            # Get fields from doc
            fields = doc.fields if hasattr(doc, 'fields') and doc.fields else {}
            
            # Separate fields into different categories
            extracted_data_field_names = set(ExtractedData.model_fields.keys())
            properties_field_names = set(ContextProperties.model_fields.keys())
            vectorize_field_names = set(Vectorize.model_fields.keys())

            extracted_data_dict = {}
            properties_dict = {}
            vectorize_dict = {}
            metadata_dict = {}

            # Get document text
            document = fields.pop(FIELD_DOCUMENT, None)
            if document:
                vectorize_dict["text"] = document

            # Get vector if needed
            if need_vector and hasattr(doc, 'vector') and doc.vector:
                vectorize_dict["vector"] = doc.vector

            # Get original ID
            original_id = fields.pop(FIELD_ORIGINAL_ID, doc.id)

            # Categorize fields
            for key, value in fields.items():
                if key.endswith("_ts"):
                    continue

                # Try to deserialize JSON strings
                val = value
                if isinstance(value, str) and value.startswith(("{", "[")):
                    try:
                        val = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if key in extracted_data_field_names:
                    extracted_data_dict[key] = val
                elif key in properties_field_names:
                    properties_dict[key] = val
                elif key in vectorize_field_names:
                    vectorize_dict[key] = val
                else:
                    metadata_dict[key] = val

            # Build context
            context_dict = {
                "id": original_id,
                "extracted_data": ExtractedData.model_validate(extracted_data_dict),
                "properties": ContextProperties.model_validate(properties_dict),
                "vectorize": Vectorize.model_validate(vectorize_dict),
            }

            if metadata_dict:
                context_dict["metadata"] = metadata_dict

            context = ProcessedContext.model_validate(context_dict)
            
            if not need_vector:
                context.vectorize.vector = None
                
            return context

        except Exception as e:
            logger.exception(
                f"Failed to convert DashVector result to ProcessedContext: {e}"
            )
            return None

    def _build_filter_string(
        self,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Build DashVector filter string.
        DashVector uses SQL WHERE clause style filter syntax.
        
        Args:
            filters: Additional filter conditions
            user_id: User ID filter
            device_id: Device ID filter
            agent_id: Agent ID filter
            
        Returns:
            Filter string or None if no filters
        """
        conditions = []

        # Add user identity filters
        if user_id:
            conditions.append(f"{FIELD_USER_ID} = '{user_id}'")
        if device_id:
            conditions.append(f"{FIELD_DEVICE_ID} = '{device_id}'")
        if agent_id:
            conditions.append(f"{FIELD_AGENT_ID} = '{agent_id}'")

        # Add custom filters
        if filters:
            for key, value in filters.items():
                # Skip special keys
                if key in ('context_type', 'entities'):
                    continue
                if value is None:
                    continue

                if isinstance(value, dict):
                    # Range queries
                    if '$gte' in value:
                        conditions.append(f"{key} >= {value['$gte']}")
                    if '$lte' in value:
                        conditions.append(f"{key} <= {value['$lte']}")
                    if '$gt' in value:
                        conditions.append(f"{key} > {value['$gt']}")
                    if '$lt' in value:
                        conditions.append(f"{key} < {value['$lt']}")
                elif isinstance(value, list):
                    # IN query
                    values_str = ', '.join([
                        f"'{v}'" if isinstance(v, str) else str(v) 
                        for v in value
                    ])
                    conditions.append(f"{key} IN ({values_str})")
                elif isinstance(value, str):
                    # Escape single quotes in string values
                    escaped_value = value.replace("'", "''")
                    conditions.append(f"{key} = '{escaped_value}'")
                elif isinstance(value, bool):
                    conditions.append(f"{key} = {str(value).lower()}")
                else:
                    conditions.append(f"{key} = {value}")

        if not conditions:
            return None

        return ' AND '.join(conditions)

    def get_processed_context_count(self, context_type: str) -> int:
        """
        Get count of records for a context type.
        
        Args:
            context_type: Type of context
            
        Returns:
            Number of records
        """
        if not self._initialized:
            return 0

        if context_type not in self._collections:
            return 0

        collection = self._collections[context_type]
        
        try:
            stats = collection.stats()
            if stats and stats.output:
                return stats.output.total_doc_count
            return 0
        except Exception as e:
            logger.error(f"Failed to get count for {context_type}: {e}")
            return 0

    def get_all_processed_context_counts(self) -> Dict[str, int]:
        """
        Get counts for all context types.
        
        Returns:
            Dictionary mapping context_type to count
        """
        if not self._initialized:
            return {}

        result = {}
        for context_type in self._collections.keys():
            if context_type != TODO_COLLECTION:
                result[context_type] = self.get_processed_context_count(context_type)

        return result

    # ==================== Todo Embedding Methods ====================

    def upsert_todo_embedding(
        self,
        todo_id: int,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Store todo embedding for deduplication.
        
        Args:
            todo_id: Todo ID
            content: Todo content text
            embedding: Text embedding vector
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            logger.warning("DashVector not initialized, cannot store todo embedding")
            return False

        try:
            collection = self._collections.get(TODO_COLLECTION)
            if not collection:
                logger.error("Todo collection not found")
                return False

            fields = {
                FIELD_TODO_ID: todo_id,
                FIELD_CONTENT: content,
                FIELD_CREATED_AT: datetime.datetime.now().isoformat(),
            }
            if metadata:
                for key, value in metadata.items():
                    fields[key] = self._serialize_value(value)

            doc = Doc(
                id=str(todo_id),
                vector=embedding,
                fields=fields
            )

            ret = collection.upsert([doc])
            if ret:
                logger.debug(f"Stored todo embedding: id={todo_id}")
                return True
            else:
                logger.error(f"Failed to store todo embedding: {ret.message}")
                return False

        except Exception as e:
            logger.error(f"Failed to store todo embedding (id={todo_id}): {e}")
            return False

    def search_similar_todos(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[int, str, float]]:
        """
        Search for similar todos using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of (todo_id, content, similarity_score) tuples
        """
        if not self._initialized:
            logger.warning("DashVector not initialized, cannot search todos")
            return []

        try:
            collection = self._collections.get(TODO_COLLECTION)
            if not collection:
                logger.error("Todo collection not found")
                return []

            # Check if collection has data
            stats = collection.stats()
            if stats and stats.output and stats.output.total_doc_count == 0:
                return []

            ret = collection.query(
                vector=query_embedding,
                topk=top_k,
                include_vector=False
            )

            similar_todos = []
            if ret and ret.output:
                for doc in ret.output:
                    score = doc.score if hasattr(doc, 'score') else 0.0
                    
                    if score >= similarity_threshold:
                        fields = doc.fields if hasattr(doc, 'fields') else {}
                        todo_id = fields.get(FIELD_TODO_ID, 0)
                        content = fields.get(FIELD_CONTENT, "")
                        
                        similar_todos.append((todo_id, content, score))

            return similar_todos

        except Exception as e:
            logger.error(f"Failed to search similar todos: {e}")
            return []

    def delete_todo_embedding(self, todo_id: int) -> bool:
        """
        Delete todo embedding.
        
        Args:
            todo_id: Todo ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            logger.warning("DashVector not initialized, cannot delete todo embedding")
            return False

        try:
            collection = self._collections.get(TODO_COLLECTION)
            if not collection:
                logger.error("Todo collection not found")
                return False

            ret = collection.delete([str(todo_id)])
            if ret:
                logger.debug(f"Deleted todo embedding: id={todo_id}")
                return True
            else:
                logger.error(f"Failed to delete todo embedding: {ret.message}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete todo embedding (id={todo_id}): {e}")
            return False
