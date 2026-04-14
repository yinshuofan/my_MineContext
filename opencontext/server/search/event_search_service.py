"""
Event Search Service — reusable search logic for routes and processors.

Unified search: builds filters from parameters (time_range, hierarchy_levels),
resolves context types, and delegates to storage.search() (vector) or
storage.get_all_processed_contexts() (filter-only) in a single call.

Storage backends automatically skip user_id/device_id filtering for agent_base_* types.
"""

from dataclasses import dataclass, field
from typing import Any

from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.models.context import ProcessedContext, Vectorize
from opencontext.models.enums import ContentFormat, ContextType
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Return type for search."""

    hits: list[tuple[ProcessedContext, float]] = field(default_factory=list)
    ancestors: dict[str, ProcessedContext] = field(default_factory=dict)
    descendants: dict[str, ProcessedContext] = field(default_factory=dict)


class EventSearchService:
    """Stateless search service. Access storage via property (never cache in __init__)."""

    @property
    def storage(self):
        return get_storage()

    # ── Public API ──

    async def search(
        self,
        query: list[dict] | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
        hierarchy_levels: list[int] | None = None,
        top_k: int = 20,
        score_threshold: float | None = None,
        time_range: Any | None = None,
        drill: str = "none",
    ) -> SearchResult:
        """Unified search — vector or filter-only, with optional drill traversal.

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
            f"hierarchy_levels={hierarchy_levels}, top_k={top_k}, drill={drill}"
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

        # Drill traversal
        ancestors: dict[str, ProcessedContext] = {}
        descendants: dict[str, ProcessedContext] = {}
        if raw_results:
            if drill in ("up", "both"):
                max_level = max(hierarchy_levels) if hierarchy_levels else 3
                ancestors = await self.collect_ancestors(raw_results, max_level=max_level)
            if drill in ("down", "both"):
                min_level = min(hierarchy_levels) if hierarchy_levels else 0
                descendants = await self.collect_descendants(raw_results, min_level=min_level)

        logger.debug(
            f"[EventSearchService] search results: "
            f"{len(raw_results)} hits, {len(ancestors)} ancestors, "
            f"{len(descendants)} descendants"
        )
        for ctx, score in raw_results:
            ed = ctx.extracted_data
            logger.debug(
                f"[EventSearchService]   hit: score={score:.4f}, "
                f"title='{ed.title if ed else ''}', "
                f"type={ed.context_type.value if ed else ''}, "
                f"event_time_start={ctx.properties.event_time_start if ctx.properties else ''}"
            )

        return SearchResult(hits=raw_results, ancestors=ancestors, descendants=descendants)

    # ── Internal search paths ──

    async def _vector_search(
        self,
        query: list[dict],
        context_types: list[str],
        filters: dict[str, Any],
        user_id: str | None,
        device_id: str | None,
        agent_id: str | None,
        top_k: int,
        score_threshold: float | None,
    ) -> list[tuple[ProcessedContext, float]]:
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
        context_types: list[str],
        filters: dict[str, Any],
        user_id: str | None,
        device_id: str | None,
        agent_id: str | None,
        top_k: int,
    ) -> list[tuple[ProcessedContext, float]]:
        """Filter-only retrieval — single storage.get_all_processed_contexts() call."""
        result = await self.storage.get_all_processed_contexts(
            context_types=context_types,
            limit=top_k,
            filter=filters,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
        )

        contexts: list[ProcessedContext] = []
        for ct in context_types:
            contexts.extend(result.get(ct, []))
        return [(ctx, 1.0) for ctx in contexts[:top_k]]

    # ── Helpers ──

    @staticmethod
    def get_l0_type() -> str:
        """Get the L0 event ContextType value."""
        return ContextType.EVENT.value

    @staticmethod
    def _get_context_types_for_levels(levels: list[int] | None = None) -> list[str]:
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
            for level in levels:
                if level < len(user_types):
                    result.append(user_types[level].value)
                if level < len(base_types):
                    result.append(base_types[level].value)
            return result
        return [t.value for t in user_types + base_types]

    @staticmethod
    def _build_filters(
        time_range: Any | None,
        hierarchy_levels: list[int] | None,
    ) -> dict[str, Any]:
        """Build storage filter dict from request parameters.

        Uses range-overlap pattern on two fields (event_time_start_ts, event_time_end_ts)
        so that events whose [start, end] interval overlaps the query [start, end] are matched.
        """
        filters: dict[str, Any] = {}
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
        results: list[tuple[ProcessedContext, float]],
        max_level: int,
    ) -> dict[str, ProcessedContext]:
        """Collect ancestors by following refs upward (to higher hierarchy levels)."""
        all_ancestors: dict[str, ProcessedContext] = {}
        seen: set[str] = set()
        # Batch: list of (ref_id, child_hierarchy_level)
        current_batch: list[tuple[str, int]] = []

        for ctx, _score in results:
            seen.add(ctx.id)
            level = ctx.properties.hierarchy_level if ctx.properties else 0
            if not ctx.properties or not ctx.properties.refs:
                continue
            for ref_ids in ctx.properties.refs.values():
                for pid in ref_ids:
                    if pid not in seen:
                        seen.add(pid)
                        current_batch.append((pid, level))

        rounds = 0
        while current_batch and rounds < max_level:
            batch_ids = [pid for pid, _ in current_batch]
            child_levels = {pid: clvl for pid, clvl in current_batch}
            parents = await self.storage.get_contexts_by_ids(batch_ids)
            next_batch: list[tuple[str, int]] = []
            for parent in parents:
                parent_level = parent.properties.hierarchy_level if parent.properties else 0
                clvl = child_levels.get(parent.id, 0)
                if parent_level <= clvl:
                    continue  # not an ancestor (same level or downward ref)
                all_ancestors[parent.id] = parent
                if parent.properties and parent.properties.refs:
                    for ref_ids in parent.properties.refs.values():
                        for pid in ref_ids:
                            if pid not in seen:
                                seen.add(pid)
                                next_batch.append((pid, parent_level))
            current_batch = next_batch
            rounds += 1

        return all_ancestors

    async def collect_descendants(
        self,
        results: list[tuple[ProcessedContext, float]],
        min_level: int = 0,
    ) -> dict[str, ProcessedContext]:
        """Collect descendants by following refs downward (to lower hierarchy levels)."""
        all_descendants: dict[str, ProcessedContext] = {}
        seen: set[str] = set()
        # Batch: list of (ref_id, parent_hierarchy_level)
        current_batch: list[tuple[str, int]] = []

        for ctx, _score in results:
            seen.add(ctx.id)
            level = ctx.properties.hierarchy_level if ctx.properties else 0
            if level <= min_level or not ctx.properties or not ctx.properties.refs:
                continue
            for ref_ids in ctx.properties.refs.values():
                for cid in ref_ids:
                    if cid not in seen:
                        seen.add(cid)
                        current_batch.append((cid, level))

        max_rounds = 4  # L3 → L0 is at most 3 hops; generous limit
        rounds = 0
        while current_batch and rounds < max_rounds:
            batch_ids = [cid for cid, _ in current_batch]
            parent_levels = {cid: plvl for cid, plvl in current_batch}
            children = await self.storage.get_contexts_by_ids(batch_ids)
            next_batch: list[tuple[str, int]] = []
            for child in children:
                child_level = child.properties.hierarchy_level if child.properties else 0
                plvl = parent_levels.get(child.id, 0)
                if child_level >= plvl:
                    continue  # not a descendant (same level or upward ref)
                all_descendants[child.id] = child
                if child_level > min_level and child.properties and child.properties.refs:
                    for ref_ids in child.properties.refs.values():
                        for cid in ref_ids:
                            if cid not in seen:
                                seen.add(cid)
                                next_batch.append((cid, child_level))
            current_batch = next_batch
            rounds += 1

        return all_descendants
