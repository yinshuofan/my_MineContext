# -*- coding: utf-8 -*-

"""
Event Search Service — reusable search logic for routes and processors.

The storage backends automatically skip user_id filtering for agent_base_* types,
so all search methods can pass the caller's user_id directly with combined type lists.
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
    """Return type for semantic_search."""

    hits: List[Tuple[ProcessedContext, float]] = field(default_factory=list)
    ancestors: Dict[str, ProcessedContext] = field(default_factory=dict)


class EventSearchService:
    """Stateless search service. Access storage via property (never cache in __init__)."""

    @property
    def storage(self):
        return get_storage()

    # ── Public API ──

    async def semantic_search(
        self,
        query: List[Dict],
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        time_range: Optional[Any] = None,
        drill_up: bool = False,
    ) -> SearchResult:
        """Semantic search with optional drill-up.

        Handles vectorization internally — caller provides raw query content
        in OpenAI content parts format. Automatically includes agent base memories
        (backend skips user_id filter for agent_base_* types).
        """
        query_text = " ".join(item.get("text", "") for item in query if item.get("type") == "text")
        logger.debug(
            f"[EventSearchService] semantic_search: query='{query_text[:100]}', "
            f"user_id={user_id}, agent_id={agent_id}, "
            f"top_k={top_k}, drill_up={drill_up}"
        )

        # Vectorize query
        query_types = {item.get("type") for item in query}
        has_multimodal = bool(query_types & {"image_url", "video_url"})
        vectorize = Vectorize(
            input=query,
            content_format=(ContentFormat.MULTIMODAL if has_multimodal else ContentFormat.TEXT),
        )
        await do_vectorize(vectorize, role="query")

        # Single search across all types (backend handles user_id skipping for base types)
        filters = self._build_filters(time_range, None)
        context_types = self._get_context_types_for_levels(None)
        raw_results = await self.storage.search(
            query=vectorize,
            top_k=top_k,
            context_types=context_types,
            filters=filters,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            score_threshold=score_threshold,
        )

        # Drill-up
        ancestors: Dict[str, ProcessedContext] = {}
        if drill_up and raw_results:
            ancestors = await self.collect_ancestors(raw_results, max_level=3)

        logger.debug(
            f"[EventSearchService] semantic_search results: "
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

    async def filter_search(
        self,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        hierarchy_levels: Optional[List[int]] = None,
        time_range: Optional[Any] = None,
        top_k: int = 20,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Filter-only search (no semantic vector).

        Backend handles user_id skipping for agent_base_* types automatically.
        """
        if hierarchy_levels:
            time_start = time_range.start if time_range else None
            time_end = time_range.end if time_range else None

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
            tasks = []
            for level in hierarchy_levels:
                for types in (user_types, base_types):
                    if level < len(types):
                        tasks.append(
                            self.storage.search_hierarchy(
                                context_type=types[level].value,
                                hierarchy_level=level,
                                time_start=time_start,
                                time_end=time_end,
                                user_id=user_id,
                                device_id=device_id,
                                agent_id=agent_id,
                                top_k=top_k,
                            )
                        )

            import asyncio

            level_results = await asyncio.gather(*tasks)
            merged: List[Tuple[ProcessedContext, float]] = []
            for results in level_results:
                merged.extend(results)
            seen_ids = set()
            deduped = []
            for item in merged:
                if item[0].id not in seen_ids:
                    seen_ids.add(item[0].id)
                    deduped.append(item)
            return deduped[:top_k]

        # Only time_range — single query across all types
        filters: Dict[str, Any] = {}
        if time_range:
            if time_range.start is not None:
                filters["event_time_end_ts"] = {"$gte": time_range.start}
            if time_range.end is not None:
                filters["event_time_start_ts"] = {"$lte": time_range.end}

        all_types = self._get_context_types_for_levels(None)
        result = await self.storage.get_all_processed_contexts(
            context_types=all_types,
            limit=top_k,
            filter=filters,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
        )

        contexts: List[ProcessedContext] = []
        for ct in all_types:
            contexts.extend(result.get(ct, []))
        return [(ctx, 1.0) for ctx in contexts[:top_k]]

    # ── Helpers (public — used by route layer) ──

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
        while current_batch and rounds < 3:
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
