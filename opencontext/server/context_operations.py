#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Context search operations for OpenContext.
Separated from main OpenContext class for better maintainability.
"""

import datetime
from typing import Any, Dict, List, Optional

from opencontext.models.context import ProcessedContext, RawContextProperties, Vectorize
from opencontext.models.enums import ContentFormat, ContextSource, ContextType
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ContextOperations:
    """Handles context CRUD and search operations."""

    def __init__(self):
        self.storage = get_storage()

    def get_all_contexts(
        self,
        limit: int = 10,
        offset: int = 0,
        filter_criteria: Optional[Dict[str, Any]] = None,
        need_vector: bool = False,
    ) -> Dict[str, List[ProcessedContext]]:
        """Get all processed contexts with pagination and filtering."""
        limit = min(limit, 1000)  # Prevent excessive memory usage
        if self.storage:
            return self.storage.get_all_processed_contexts(
                limit=limit, offset=offset, filter=filter_criteria or {}, need_vector=need_vector
            )
        logger.warning("Storage is not initialized.")
        return {}

    def get_context(self, doc_id: str, context_type: str) -> Optional[ProcessedContext]:
        """Get a single processed context by ID and type."""
        if self.storage:
            return self.storage.get_processed_context(doc_id, context_type)
        logger.warning("Storage is not initialized.")
        return None

    def update_context(self, doc_id: str, context: ProcessedContext) -> bool:
        """Update a processed context."""
        if self.storage:
            return self.storage.upsert_processed_context(context)
        logger.warning("Storage is not initialized.")
        return False

    def delete_context(self, doc_id: str, context_type: str) -> bool:
        """Delete a processed context."""
        if self.storage:
            return self.storage.delete_processed_context(doc_id, context_type)
        logger.warning("Storage is not initialized.")
        return False

    async def add_document(self, file_path: str, context_processor_callback) -> Optional[str]:
        """Add a document to the system."""
        import uuid
        from pathlib import Path

        # Validate inputs
        if not file_path:
            return "Document path cannot be empty"

        path = Path(file_path).expanduser()
        if not path.exists():
            return f"Document path {file_path} does not exist"

        if not path.is_file():
            return f"Path {file_path} is not a file"

        try:
            # Create RawContextProperties
            object_id = f"doc_{uuid.uuid4()}"

            raw_context = RawContextProperties(
                source=ContextSource.LOCAL_FILE,
                content_format=ContentFormat.FILE,
                create_time=datetime.datetime.now(),
                object_id=object_id,
                content_path=str(path),
                additional_info={
                    "filename": path.name,
                    "file_size": path.stat().st_size,
                    "file_extension": path.suffix,
                },
            )

            # Call processor
            if not await context_processor_callback(raw_context):
                return "Failed to add document"
            return None
        except Exception as e:
            error_msg = f"Failed to process document: {e}"
            logger.error(error_msg)
            return error_msg

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
        """
        Perform vector search without LLM processing.

        Args:
            query: Search query text
            top_k: Number of results to return
            context_types: Context type filter list
            filters: Additional filter conditions
            user_id: User identifier for multi-user filtering
            device_id: Device identifier for multi-user filtering
            agent_id: Agent identifier for multi-user filtering

        Returns:
            List of search results with context and scores
        """
        if not self.storage:
            raise RuntimeError("Storage not initialized")

        try:
            logger.debug(f"Searching with filters: {filters}")
            # Create query vector
            query_vectorize = Vectorize(text=query)

            # Execute vector search with multi-user filtering
            search_results = self.storage.search(
                query=query_vectorize,
                top_k=top_k,
                context_types=context_types,
                filters=filters,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            )

            # Format results
            results = []
            for context, score in search_results:
                results.append(
                    {
                        "context": {
                            "id": context.id,
                            "extracted_data": {
                                "title": context.extracted_data.title,
                                "summary": context.extracted_data.summary,
                                "context_type": context.extracted_data.context_type.value,
                                "keywords": context.extracted_data.keywords,
                            },
                            "properties": {
                                "create_time": context.properties.create_time,
                                "user_id": context.properties.user_id,
                                "device_id": context.properties.device_id,
                                "agent_id": context.properties.agent_id,
                            },
                        },
                        "score": score,
                    }
                )

            return results

        except Exception as e:
            logger.exception(f"Vector search failed: {e}")
            raise RuntimeError(f"Vector search failed: {str(e)}") from e

    def get_context_types(self) -> List[str]:
        """
        Get all available context types.

        Returns:
            List of context types (all 5 types are always available)
        """
        return [t.value for t in ContextType]
