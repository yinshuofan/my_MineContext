# -*- coding: utf-8 -*-

"""
Event Search Service — reusable search logic for routes and processors.

Unified search: builds filters from parameters (time_range, hierarchy_levels),
resolves context types, and delegates to storage.search() (vector) or
storage.get_all_processed_contexts() (filter-only) in a single call.

Storage backends automatically skip user_id/device_id filtering for agent_base_* types.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.models.context import ProcessedContext, Vectorize
from opencontext.models.enums import ContentFormat, ContextType
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Return type for search."""

    hits: List[Tuple[ProcessedContext, float]] = field(default_factory=list)
    ancestors: Dict[str, ProcessedContext] = field(default_factory=dict)


class EventSearchService:
    """Stateless search service. Access storage via property (never cache in __init__)."""

    @property
    def storage(self):
        return get_storage()

    # ── Public API ──

    async def search(
        self,
        query: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        hierarchy_levels: Optional[List[int]] = None,
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        time_range: Optional[Any] = None,
        drill_up: bool = False,
    ) -> SearchResult:
        """Unified search — vector or filter-only, with optional drill-up.

        If query is provided, performs vector similarity search.
        If query is None, performs filter-only retrieval.
        Both paths use the same filter construction from time_range and hierarchy_levels.
        Storage backends skip user_id/device_id for agent_base_* types automatically.
        """
        query_text = ""
        if query:
            query_text = " ".join(
                item.get("text", "") for item in query if item.get("type") == "text"
            )
        logger.debug(
            f"[EventSearchService] search: query='{query_text[:100]}', "
            f"user_id={user_id}, agent_id={agent_id}, "
            f"hierarchy_levels={hierarchy_levels}, top_k={top_k}, drill_up={drill_up}"
        )

        # Build unified filters and context types
        filters = self._build_filters(time_range, hierarchy_levels)
        context_types = self._get_context_types_for_levels(hierarchy_levels)

        # Execute search
        if query:
            raw_results = await self._vector_search(
                query,
                context_types,
                filters,
                user_id,
                device_id,
                agent_id,
                top_k,
                score_threshold,
            )
        else:
            raw_results = await self._filter_search(
                context_types,
                filters,
                user_id,
                device_id,
                agent_id,
                top_k,
            )

        # Drill-up
        ancestors: Dict[str, ProcessedContext] = {}
        if drill_up and raw_results:
            ancestors = await self.collect_ancestors(raw_results, max_level=3)

        logger.debug(
            f"[EventSearchService] search results: "
            f"{len(raw_results)} hits, {len(ancestors)} ancestors"
        )
        for ctx, score in raw_results:
            ed = ctx.extracted_data
            logger.debug(
                f"[EventSearchService]   hit: score={score:.4f}, "
                f"title='{ed.title if ed else ''}', "
                f"type={ed.context_type.value if ed else ''}, "
                f"event_time_start={ctx.properties.event_time_start if ctx.properties else ''}"
            )

        return SearchResult(hits=raw_results, ancestors=ancestors)

    # ── Internal search paths ──

    async def _vector_search(
        self,
        query: List[Dict],
        context_types: List[str],
        filters: Dict[str, Any],
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str],
        top_k: int,
        score_threshold: Optional[float],
    ) -> List[Tuple[ProcessedContext, float]]:
        """Vector similarity search — single storage.search() call."""
        query_types = {item.get("type") for item in query}
        has_multimodal = bool(query_types & {"image_url", "video_url"})
        vectorize = Vectorize(
            input=query,
            content_format=(ContentFormat.MULTIMODAL if has_multimodal else ContentFormat.TEXT),
        )
        await do_vectorize(vectorize, role="query")

        return await self.storage.search(
            query=vectorize,
            top_k=top_k,
            context_types=context_types,
            filters=filters,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            score_threshold=score_threshold,
        )

    async def _filter_search(
        self,
        context_types: List[str],
        filters: Dict[str, Any],
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str],
        top_k: int,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Filter-only retrieval — single storage.get_all_processed_contexts() call."""
        result = await self.storage.get_all_processed_contexts(
            context_types=context_types,
            limit=top_k,
            filter=filters,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
        )

        contexts: List[ProcessedContext] = []
        for ct in context_types:
            contexts.extend(result.get(ct, []))
        return [(ctx, 1.0) for ctx in contexts[:top_k]]

    # ── Helpers ──

    @staticmethod
    def get_l0_type() -> str:
        """Get the L0 event ContextType value."""
        return ContextType.EVENT.value

    @staticmethod
    def _get_context_types_for_levels(levels: Optional[List[int]] = None) -> List[str]:
        """Map hierarchy_levels to combined user + agent_base ContextType values."""
        user_types = [
            ContextType.EVENT,
            ContextType.DAILY_SUMMARY,
            ContextType.WEEKLY_SUMMARY,
            ContextType.MONTHLY_SUMMARY,
        ]
        base_types = [
            ContextType.AGENT_BASE_EVENT,
            ContextType.AGENT_BASE_L1_SUMMARY,
            ContextType.AGENT_BASE_L2_SUMMARY,
            ContextType.AGENT_BASE_L3_SUMMARY,
        ]
        if levels:
            result = []
            for l in levels:
                if l < len(user_types):
                    result.append(user_types[l].value)
                if l < len(base_types):
                    result.append(base_types[l].value)
            return result
        return [t.value for t in user_types + base_types]

    @staticmethod
    def _build_filters(
        time_range: Optional[Any],
        hierarchy_levels: Optional[List[int]],
    ) -> Dict[str, Any]:
        """Build storage filter dict from request parameters.

        Uses range-overlap pattern on two fields (event_time_start_ts, event_time_end_ts)
        so that events whose [start, end] interval overlaps the query [start, end] are matched.
        """
        filters: Dict[str, Any] = {}
        if time_range:
            if time_range.end is not None:
                filters["event_time_start_ts"] = {"$lte": time_range.end}
            if time_range.start is not None:
                filters["event_time_end_ts"] = {"$gte": time_range.start}

        if hierarchy_levels is not None and len(hierarchy_levels) == 1:
            filters["hierarchy_level"] = hierarchy_levels[0]
        elif hierarchy_levels is not None and len(hierarchy_levels) > 1:
            filters["hierarchy_level"] = hierarchy_levels

        return filters

    async def collect_ancestors(
        self,
        results: List[Tuple[ProcessedContext, float]],
        max_level: int,
    ) -> Dict[str, ProcessedContext]:
        """Collect ancestors by following refs upward (to summary types)."""
        user_summary_types = {
            ContextType.DAILY_SUMMARY.value,
            ContextType.WEEKLY_SUMMARY.value,
            ContextType.MONTHLY_SUMMARY.value,
        }
        base_summary_types = {
            ContextType.AGENT_BASE_L1_SUMMARY.value,
            ContextType.AGENT_BASE_L2_SUMMARY.value,
            ContextType.AGENT_BASE_L3_SUMMARY.value,
        }
        summary_type_values = user_summary_types | base_summary_types

        all_ancestors = {}
        seen = set()
        current_batch = []

        for ctx, score in results:
            seen.add(ctx.id)
            if not ctx.properties or not ctx.properties.refs:
                continue
            for ref_key, ref_ids in ctx.properties.refs.items():
                if ref_key in summary_type_values:
                    for pid in ref_ids:
                        if pid not in seen:
                            seen.add(pid)
                            current_batch.append(pid)

        rounds = 0
        while current_batch and rounds < max_level:
            parents = await self.storage.get_contexts_by_ids(current_batch)
            next_batch = []
            for parent in parents:
                all_ancestors[parent.id] = parent
                if not parent.properties or not parent.properties.refs:
                    continue
                for ref_key, ref_ids in parent.properties.refs.items():
                    if ref_key in summary_type_values:
                        for pid in ref_ids:
                            if pid not in seen:
                                seen.add(pid)
                                next_batch.append(pid)
            current_batch = next_batch
            rounds += 1

        return all_ancestors
