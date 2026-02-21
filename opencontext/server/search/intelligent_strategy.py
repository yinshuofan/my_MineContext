# -*- coding: utf-8 -*-

"""
Unified Search API - Intelligent Search Strategy

LLM-driven agentic search that reuses LLMContextStrategy for tool selection,
execution, and validation. Returns structured TypedResults instead of chat answers.
"""

import asyncio
from typing import Any, Dict, List, Optional

from opencontext.context_consumption.context_agent.core.llm_context_strategy import (
    LLMContextStrategy,
)
from opencontext.context_consumption.context_agent.models.enums import QueryType
from opencontext.context_consumption.context_agent.models.schemas import (
    ContextCollection,
    ContextItem,
    Intent,
)
from opencontext.server.search.base_strategy import BaseSearchStrategy
from opencontext.server.search.models import (
    EntityResult,
    ProfileResult,
    TimeRange,
    TypedResults,
    VectorResult,
)
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

MAX_ITERATIONS = 2


class IntelligentSearchStrategy(BaseSearchStrategy):
    """
    Intelligent search strategy — LLM decides which tools to call.

    Reuses LLMContextStrategy (analyze_and_plan_tools, execute_tool_calls_parallel,
    validate_and_filter_tool_results, evaluate_sufficiency) but only performs
    retrieval, not answer generation.
    """

    def __init__(self):
        self._strategy = LLMContextStrategy()

    async def search(
        self,
        query: str,
        context_types: List[str],
        top_k: int,
        time_range: Optional[TimeRange],
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str],
    ) -> TypedResults:
        # Run agentic loop and profile/entity lookup in parallel
        agentic_task = self._agentic_search_loop(query, user_id)
        profile_entity_task = self._direct_profile_entity_lookup(
            query, top_k, user_id, agent_id
        )

        agentic_items, (profile, entities) = await asyncio.gather(
            agentic_task, profile_entity_task, return_exceptions=False
        )

        # Convert ContextItems to TypedResults
        typed = self._context_items_to_typed_results(agentic_items)

        # Merge in profile/entity results (as fallback in case agent didn't call those tools)
        if profile and typed.profile is None:
            typed.profile = profile
        if entities and not typed.entities:
            typed.entities = entities

        return typed

    async def _agentic_search_loop(
        self, query: str, user_id: Optional[str]
    ) -> List[ContextItem]:
        """
        Run the LLM-driven tool selection loop (max MAX_ITERATIONS rounds).
        Returns collected ContextItems.
        """
        intent = Intent(
            original_query=query,
            query_type=QueryType.QA_ANALYSIS,
            enhanced_query=query,
        )
        collection = ContextCollection()
        all_items: List[ContextItem] = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            try:
                # LLM decides which tools to call
                tool_calls, _ = await self._strategy.analyze_and_plan_tools(
                    intent, collection, iteration
                )

                if not tool_calls:
                    logger.info(
                        f"Intelligent search iteration {iteration}: LLM returned no tool calls"
                    )
                    break

                # Execute tools in parallel
                tool_results = await self._strategy.execute_tool_calls_parallel(
                    tool_calls
                )

                if not tool_results:
                    logger.info(
                        f"Intelligent search iteration {iteration}: no results from tools"
                    )
                    break

                # LLM validates and filters results
                relevant_items, _ = await self._strategy.validate_and_filter_tool_results(
                    tool_calls, tool_results, intent, collection
                )

                # Add to collection
                for item in relevant_items:
                    collection.add_item(item)
                all_items.extend(relevant_items)

                # Check sufficiency
                sufficiency = await self._strategy.evaluate_sufficiency(
                    collection, intent
                )

                if sufficiency.value == "sufficient":
                    logger.info(
                        f"Intelligent search: sufficient after {iteration} iterations"
                    )
                    break

            except Exception as e:
                logger.error(
                    f"Intelligent search iteration {iteration} failed: {e}"
                )
                break

        return all_items

    async def _direct_profile_entity_lookup(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str],
        agent_id: Optional[str],
    ) -> tuple:
        """Direct profile/entity lookup as fallback (runs in parallel with agentic loop)."""
        profile = None
        entities = []

        if not user_id:
            return profile, entities

        storage = get_storage()

        try:
            profile_data = await asyncio.to_thread(
                storage.get_profile, user_id, agent_id or "default"
            )
            if profile_data:
                profile = ProfileResult(
                    user_id=profile_data.get("user_id", ""),
                    agent_id=profile_data.get("agent_id", "default"),
                    content=profile_data.get("content", ""),
                    summary=profile_data.get("summary"),
                    keywords=profile_data.get("keywords", []),
                    metadata=profile_data.get("metadata", {}),
                )
        except Exception as e:
            logger.error(f"Profile lookup failed: {e}")

        try:
            entity_list = await asyncio.to_thread(
                storage.search_entities, user_id, query, top_k
            )
            entities = [
                EntityResult(
                    id=e.get("id", ""),
                    entity_name=e.get("entity_name", ""),
                    entity_type=e.get("entity_type"),
                    content=e.get("content", ""),
                    summary=e.get("summary"),
                    aliases=e.get("aliases", []),
                    score=e.get("score", 1.0),
                )
                for e in entity_list
            ]
        except Exception as e:
            logger.error(f"Entity lookup failed: {e}")

        return profile, entities

    def _context_items_to_typed_results(
        self, items: List[ContextItem]
    ) -> TypedResults:
        """Convert ContextItems to TypedResults by routing based on tool_name."""
        typed = TypedResults()

        for item in items:
            tool_name = item.metadata.get("tool_name", "")
            original = item.metadata.get("original_data", {})

            vr = self._item_to_vector_result(item, original)

            if "document" in tool_name:
                typed.documents.append(vr)
            elif "event" in tool_name:
                typed.events.append(vr)
            elif "knowledge" in tool_name:
                # KnowledgeRetrievalTool may return both knowledge and L0 events
                ctx_type = original.get("context_type", "knowledge")
                if ctx_type == "event":
                    typed.events.append(vr)
                else:
                    typed.knowledge.append(vr)
            # Profile/entity from tools are handled by direct lookup fallback

        return typed

    @staticmethod
    def _item_to_vector_result(
        item: ContextItem, original: Dict[str, Any]
    ) -> VectorResult:
        """Convert a ContextItem + its original_data to VectorResult."""
        return VectorResult(
            id=item.id,
            title=item.title,
            summary=original.get("summary"),
            content=item.content,
            keywords=original.get("keywords", []),
            entities=original.get("entities", []),
            context_type=original.get("context_type", ""),
            score=item.relevance_score,
            create_time=original.get("create_time"),
            event_time=original.get("event_time"),
            hierarchy_level=original.get("hierarchy_level", 0),
            time_bucket=original.get("time_bucket"),
            parent_id=original.get("parent_id"),
            children_ids=original.get("children_ids", []),
            parent_summary=original.get("parent_summary"),
            source_file_key=original.get("source_file_key"),
            metadata=original.get("metadata", {}),
        )
