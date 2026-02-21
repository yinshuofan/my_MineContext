# -*- coding: utf-8 -*-

"""
Unified Search API - Fast Search Strategy

Zero LLM reasoning calls. Only 1 embedding generation + parallel storage searches.
For events, searches L0 directly and attaches parent summaries in one batch.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.models.context import ProcessedContext, Vectorize
from opencontext.models.enums import ContextType
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


class FastSearchStrategy(BaseSearchStrategy):
    """
    Fast search strategy — direct parallel search across all 5 types.

    - Zero LLM reasoning calls (only 1 embedding generation)
    - All storage calls run in parallel via asyncio.to_thread()
    - Events: search L0 directly, then batch-attach parent summaries
    """

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
        storage = get_storage()

        # Step 1: Pre-generate embedding once, share across all vector searches
        vectorize = Vectorize(text=query)
        await asyncio.to_thread(do_vectorize, vectorize)

        # Step 2: Build time filters
        time_filters = self._build_time_filters(time_range)

        # Step 3: Dispatch parallel searches
        tasks = {}

        if ContextType.PROFILE.value in context_types and user_id:
            tasks["profile"] = asyncio.to_thread(
                storage.get_profile, user_id, agent_id or "default"
            )

        if ContextType.ENTITY.value in context_types and user_id:
            tasks["entity"] = asyncio.to_thread(
                storage.search_entities, user_id, query, top_k
            )

        if ContextType.DOCUMENT.value in context_types:
            tasks["document"] = asyncio.to_thread(
                storage.search,
                vectorize,
                top_k,
                [ContextType.DOCUMENT.value],
                time_filters if time_filters else None,
                user_id,
                device_id,
                agent_id,
            )

        if ContextType.EVENT.value in context_types:
            event_filters = dict(time_filters) if time_filters else {}
            event_filters["hierarchy_level"] = 0  # Only L0 raw events
            tasks["event"] = asyncio.to_thread(
                storage.search,
                vectorize,
                top_k,
                [ContextType.EVENT.value],
                event_filters,
                user_id,
                device_id,
                agent_id,
            )

        if ContextType.KNOWLEDGE.value in context_types:
            tasks["knowledge"] = asyncio.to_thread(
                storage.search,
                vectorize,
                top_k,
                [ContextType.KNOWLEDGE.value],
                time_filters if time_filters else None,
                user_id,
                device_id,
                agent_id,
            )

        # Execute all in parallel
        task_names = list(tasks.keys())
        raw_results = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )
        results_map = dict(zip(task_names, raw_results))

        # Step 4: Assemble TypedResults
        typed = TypedResults()

        # Profile
        profile_data = results_map.get("profile")
        if profile_data and not isinstance(profile_data, Exception) and profile_data is not None:
            typed.profile = self._to_profile_result(profile_data)

        # Entities
        entity_data = results_map.get("entity")
        if entity_data and not isinstance(entity_data, Exception):
            typed.entities = [self._to_entity_result(e) for e in entity_data]

        # Documents
        doc_data = results_map.get("document")
        if doc_data and not isinstance(doc_data, Exception):
            typed.documents = [
                self._to_vector_result(ctx, score) for ctx, score in doc_data
            ]

        # Knowledge
        knowledge_data = results_map.get("knowledge")
        if knowledge_data and not isinstance(knowledge_data, Exception):
            typed.knowledge = [
                self._to_vector_result(ctx, score) for ctx, score in knowledge_data
            ]

        # Events — attach parent summaries
        event_data = results_map.get("event")
        if event_data and not isinstance(event_data, Exception):
            typed.events = await self._attach_parent_summaries(event_data)

        # Log exceptions for debugging
        for name, result in results_map.items():
            if isinstance(result, Exception):
                logger.error(f"Fast search failed for {name}: {result}")

        return typed

    def _build_time_filters(self, time_range: Optional[TimeRange]) -> Dict[str, Any]:
        """Build time filter dict for storage backends"""
        if not time_range:
            return {}

        filters = {}
        if time_range.start is not None or time_range.end is not None:
            time_filter = {}
            if time_range.start is not None:
                time_filter["$gte"] = time_range.start
            if time_range.end is not None:
                time_filter["$lte"] = time_range.end
            filters["event_time_ts"] = time_filter
        return filters

    async def _attach_parent_summaries(
        self, event_results: List[Tuple[ProcessedContext, float]]
    ) -> List[VectorResult]:
        """For L0 events, batch-fetch parent summaries and attach them."""
        # Collect unique parent_ids
        parent_ids = set()
        for ctx, _ in event_results:
            pid = ctx.properties.parent_id
            if pid:
                parent_ids.add(pid)

        # Batch fetch parents in one call
        parent_map: Dict[str, ProcessedContext] = {}
        if parent_ids:
            storage = get_storage()
            parents = await asyncio.to_thread(
                storage.get_contexts_by_ids,
                list(parent_ids),
                ContextType.EVENT.value,
            )
            parent_map = {p.id: p for p in parents}

        # Build results with parent_summary attached
        results = []
        for ctx, score in event_results:
            vr = self._to_vector_result(ctx, score)
            parent = parent_map.get(ctx.properties.parent_id)
            if parent and parent.extracted_data:
                vr.parent_summary = parent.extracted_data.summary
            results.append(vr)
        return results

    @staticmethod
    def _to_vector_result(ctx: ProcessedContext, score: float) -> VectorResult:
        """Convert ProcessedContext + score to VectorResult"""
        ed = ctx.extracted_data
        props = ctx.properties

        create_time = None
        if props.create_time:
            create_time = props.create_time.isoformat() if hasattr(props.create_time, "isoformat") else str(props.create_time)

        event_time = None
        if props.event_time:
            event_time = props.event_time.isoformat() if hasattr(props.event_time, "isoformat") else str(props.event_time)

        return VectorResult(
            id=ctx.id,
            title=ed.title if ed else None,
            summary=ed.summary if ed else None,
            content=ctx.get_llm_context_string(),
            keywords=ed.keywords if ed else [],
            entities=ed.entities if ed else [],
            context_type=ed.context_type.value if ed and ed.context_type else "",
            score=score,
            create_time=create_time,
            event_time=event_time,
            hierarchy_level=props.hierarchy_level,
            time_bucket=props.time_bucket,
            parent_id=props.parent_id,
            children_ids=props.children_ids,
            source_file_key=props.source_file_key,
            metadata=ctx.metadata or {},
        )

    @staticmethod
    def _to_profile_result(data: Dict[str, Any]) -> ProfileResult:
        """Convert profile dict from storage to ProfileResult"""
        return ProfileResult(
            user_id=data.get("user_id", ""),
            agent_id=data.get("agent_id", "default"),
            content=data.get("content", ""),
            summary=data.get("summary"),
            keywords=data.get("keywords", []),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def _to_entity_result(data: Dict[str, Any]) -> EntityResult:
        """Convert entity dict from storage to EntityResult"""
        return EntityResult(
            id=data.get("id", ""),
            entity_name=data.get("entity_name", ""),
            entity_type=data.get("entity_type"),
            content=data.get("content", ""),
            summary=data.get("summary"),
            aliases=data.get("aliases", []),
            score=data.get("score", 1.0),
        )
