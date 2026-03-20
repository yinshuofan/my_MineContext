# -*- coding: utf-8 -*-

"""
Event Search Service — reusable search logic for routes and processors.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.models.context import ProcessedContext, Vectorize
from opencontext.models.enums import MEMORY_OWNER_TYPES, ContentFormat
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
        memory_owner: str = "user",
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        time_range: Optional[Any] = None,
        drill_up: bool = False,
    ) -> SearchResult:
        """Semantic search with optional drill-up.

        Handles vectorization internally — caller provides raw query content
        in OpenAI content parts format.
        """
        query_text = " ".join(
            item.get("text", "") for item in query if item.get("type") == "text"
        )
        logger.debug(
            f"[EventSearchService] semantic_search: query='{query_text[:100]}', "
            f"user_id={user_id}, agent_id={agent_id}, memory_owner={memory_owner}, "
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

        # Search
        filters = self._build_filters(time_range, None)
        context_types = self._get_context_types_for_levels(memory_owner, None)
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
            ancestors = await self.collect_ancestors(
                raw_results, max_level=3, memory_owner=memory_owner
            )

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
                f"time_bucket={ctx.properties.time_bucket if ctx.properties else ''}"
            )

        return SearchResult(hits=raw_results, ancestors=ancestors)

    async def filter_search(
        self,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory_owner: str = "user",
        hierarchy_levels: Optional[List[int]] = None,
        time_range: Optional[Any] = None,
        top_k: int = 20,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Filter-only search (no semantic vector)."""
        if hierarchy_levels:
            time_bucket_start = None
            time_bucket_end = None
            if time_range:
                time_bucket_start, time_bucket_end = self._time_range_to_buckets(
                    time_range.start, time_range.end
                )

            owner_types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
            tasks = []
            for level in hierarchy_levels:
                if level < len(owner_types):
                    tasks.append(
                        self.storage.search_hierarchy(
                            context_type=owner_types[level].value,
                            hierarchy_level=level,
                            time_bucket_start=time_bucket_start,
                            time_bucket_end=time_bucket_end,
                            user_id=user_id,
                            device_id=device_id,
                            agent_id=agent_id,
                            top_k=top_k,
                        )
                    )

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

        # Only time_range — fetch all events
        filters: Dict[str, Any] = {}
        if time_range:
            ts_filter: Dict[str, Any] = {}
            if time_range.start is not None:
                ts_filter["$gte"] = time_range.start
            if time_range.end is not None:
                ts_filter["$lte"] = time_range.end
            if ts_filter:
                filters["event_time_ts"] = ts_filter

        all_types = self._get_context_types_for_levels(memory_owner, None)
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
        return [(ctx, 1.0) for ctx in contexts]

    # ── Helpers (public — used by route layer) ──

    @staticmethod
    def get_l0_type(memory_owner: str) -> str:
        """Get the L0 event ContextType value for a memory owner."""
        types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
        return types[0].value

    @staticmethod
    def _get_context_types_for_levels(
        memory_owner: str, levels: Optional[List[int]]
    ) -> List[str]:
        """Map hierarchy_levels + memory_owner to ContextType values."""
        types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
        if levels:
            return [types[l].value for l in levels if l < len(types)]
        return [t.value for t in types]

    @staticmethod
    def _build_filters(
        time_range: Optional[Any],
        hierarchy_levels: Optional[List[int]],
    ) -> Dict[str, Any]:
        """Build storage filter dict from request parameters."""
        filters: Dict[str, Any] = {}
        if time_range:
            ts_filter: Dict[str, Any] = {}
            if time_range.start is not None:
                ts_filter["$gte"] = time_range.start
            if time_range.end is not None:
                ts_filter["$lte"] = time_range.end
            if ts_filter:
                filters["event_time_ts"] = ts_filter

        if hierarchy_levels is not None and len(hierarchy_levels) == 1:
            filters["hierarchy_level"] = hierarchy_levels[0]
        elif hierarchy_levels is not None and len(hierarchy_levels) > 1:
            filters["hierarchy_level"] = hierarchy_levels

        return filters

    @staticmethod
    def _time_range_to_buckets(
        start_ts: Optional[int],
        end_ts: Optional[int],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Convert Unix timestamps to date bucket strings for search_hierarchy."""
        bucket_start = None
        bucket_end = None
        if start_ts is not None:
            dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
            bucket_start = dt.strftime("%Y-%m-%d")
        if end_ts is not None:
            dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
            bucket_end = dt.strftime("%Y-%m-%d")
        return bucket_start, bucket_end

    async def collect_ancestors(
        self,
        results: List[Tuple[ProcessedContext, float]],
        max_level: int,
        memory_owner: str = "user",
    ) -> Dict[str, ProcessedContext]:
        """Collect ancestors by following refs upward (to summary types)."""
        owner_types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
        summary_type_values = {t.value for t in owner_types[1:]}

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
