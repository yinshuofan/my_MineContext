# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge context retrieval tool.

Retrieves KNOWLEDGE type contexts and also L0 EVENT contexts from the vector DB.
Extends BaseContextRetrievalTool but overrides execute to search both types,
providing comprehensive results that combine distilled knowledge with raw event records.
"""

from typing import Any, Dict, List, Optional, Tuple

from opencontext.models.context import ProcessedContext, Vectorize
from opencontext.models.enums import ContextType
from opencontext.tools.retrieval_tools.base_context_retrieval_tool import (
    BaseContextRetrievalTool,
    ContextRetrievalFilter,
    TimeRangeFilter,
)
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class KnowledgeRetrievalTool(BaseContextRetrievalTool):
    """
    Knowledge context retrieval tool.

    Retrieves knowledge concepts, technical principles, and operation workflows
    from the KNOWLEDGE context type.  Additionally searches L0 (raw individual)
    EVENT contexts to supplement knowledge results with concrete event records,
    ensuring comprehensive coverage when the user asks about topics that span
    both distilled knowledge and historical events.

    Results from both searches are merged, deduplicated by context ID (keeping
    the higher score for duplicates), and sorted by score descending.
    """

    CONTEXT_TYPE = ContextType.KNOWLEDGE

    @classmethod
    def get_name(cls) -> str:
        """Get tool name."""
        return "retrieve_knowledge_context"

    @classmethod
    def get_description(cls) -> str:
        """Get tool description."""
        return (
            "Retrieve knowledge concepts, technical principles, and operation workflows "
            "from the knowledge context store.  This tool also searches L0 (raw individual) "
            "event records to provide comprehensive results that combine distilled knowledge "
            "with concrete event details.\n"
            "\n"
            "**What this tool retrieves:**\n"
            "- Technical concept definitions and explanations\n"
            "- Operation workflows and procedural knowledge\n"
            "- Design patterns, best practices, and architectural principles\n"
            "- L0 event records that are semantically relevant to the query\n"
            "\n"
            "**When to use this tool:**\n"
            "- When you need to understand 'what is this' or 'how does it work'\n"
            "- When searching for technical knowledge combined with real-world examples\n"
            "- When looking for operation procedures or workflow documentation\n"
            "- When you want both conceptual knowledge and supporting event evidence\n"
            "\n"
            "**Two modes of operation:**\n"
            "1. **With query** (semantic search): Provide a natural language query to find "
            "semantically relevant knowledge and event records\n"
            "   - Example: 'deployment workflow', 'database migration steps'\n"
            "2. **Without query** (filter-only): Retrieve knowledge based on time range "
            "and/or entities (only knowledge type is retrieved in filter-only mode)\n"
            "\n"
            "**Filter options:**\n"
            "- Time range filtering (by event_time, create_time, or update_time)\n"
            "- Entity filtering (find knowledge mentioning specific technologies or concepts)\n"
            "- Configurable result count (top_k: 1-100, default 20)\n"
            "\n"
            "**Note:** Results are merged from both KNOWLEDGE and EVENT (L0) searches, "
            "deduplicated by ID, and sorted by relevance score descending."
        )

    @classmethod
    def get_parameters(cls) -> Dict[str, Any]:
        """Get tool parameters with knowledge-specific descriptions."""
        base_params = super().get_parameters()

        base_params["properties"]["query"]["description"] = (
            "Natural language query for semantic search of knowledge and event records. "
            "Examples: 'deployment workflow', 'authentication mechanism', "
            "'database migration steps'. "
            "Leave empty to perform filter-only retrieval (knowledge type only)."
        )

        return base_params

    def _search_l0_events(
        self,
        query: str,
        filters: ContextRetrievalFilter,
        top_k: int,
    ) -> List[Tuple[ProcessedContext, float]]:
        """
        Search L0 (raw individual) EVENT contexts via direct vector search.

        Args:
            query: Natural language query for semantic search.
            filters: Filter conditions including multi-user fields.
            top_k: Number of results to return.

        Returns:
            List of (context, score) tuples for matching L0 events.
        """
        try:
            built_filters = self._build_filters(filters)
            built_filters["hierarchy_level"] = 0

            vectorize = Vectorize(text=query)
            return self.storage.search(
                query=vectorize,
                context_types=[ContextType.EVENT.value],
                filters=built_filters,
                top_k=top_k,
                user_id=filters.user_id,
                device_id=filters.device_id,
                agent_id=filters.agent_id,
            )
        except Exception as e:
            logger.warning(f"L0 event search failed: {e}")
            return []

    def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute knowledge retrieval with supplementary L0 event search.

        Algorithm:
          1. Search KNOWLEDGE contexts using the standard _execute_search
             from the base class (supports both query and filter-only modes).
          2. If a query is provided, also search EVENT contexts filtered to
             hierarchy_level=0 (raw individual events) via direct storage
             search.
          3. Merge results from both searches, deduplicate by context ID,
             keeping the higher score when duplicates are found.
          4. Sort merged results by score descending, truncate to top_k.
          5. Format via _format_results().

        Args:
            query: Optional natural language search query.
            entities: Optional entity list for filtering.
            time_range: Optional time range filter dict.
            top_k: Number of results to return (default 20).
            user_id: User identifier for multi-user filtering.
            device_id: Device identifier for multi-user filtering.
            agent_id: Agent identifier for multi-user filtering.

        Returns:
            List of formatted context result dicts.
        """
        query = kwargs.get("query")
        entities = kwargs.get("entities", [])
        time_range = kwargs.get("time_range")
        top_k = kwargs.get("top_k", 20)
        user_id = kwargs.get("user_id")
        device_id = kwargs.get("device_id")
        agent_id = kwargs.get("agent_id")

        # Build filter conditions
        filters = ContextRetrievalFilter()
        filters.entities = entities
        filters.user_id = user_id
        filters.device_id = device_id
        filters.agent_id = agent_id

        if time_range:
            filters.time_range = TimeRangeFilter(**time_range)

        try:
            # Step 1: Search KNOWLEDGE contexts (standard base-class search)
            knowledge_results: List[Tuple[ProcessedContext, float]] = self._execute_search(
                query=query, filters=filters, top_k=top_k
            )

            # Step 2: Search L0 EVENT contexts (only when a query is provided)
            event_results: List[Tuple[ProcessedContext, float]] = []
            if query:
                event_results = self._search_l0_events(
                    query=query, filters=filters, top_k=top_k
                )

            # Step 3: Merge and deduplicate by context ID, keeping higher scores
            merged: Dict[str, Tuple[ProcessedContext, float]] = {}

            for ctx, score in knowledge_results:
                existing = merged.get(ctx.id)
                if existing is None or score > existing[1]:
                    merged[ctx.id] = (ctx, score)

            for ctx, score in event_results:
                existing = merged.get(ctx.id)
                if existing is None or score > existing[1]:
                    merged[ctx.id] = (ctx, score)

            # Step 4: Sort by score descending and truncate to top_k
            sorted_results: List[Tuple[ProcessedContext, float]] = sorted(
                merged.values(), key=lambda x: x[1], reverse=True
            )[:top_k]

            # Step 5: Format and return results
            return self._format_results(sorted_results)

        except Exception as e:
            logger.error(f"KnowledgeRetrievalTool execute exception: {e}")
            return [
                {
                    "error": (
                        f"Error occurred during knowledge retrieval: {str(e)}"
                    )
                }
            ]
