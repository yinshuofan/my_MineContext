# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Hierarchical event context retrieval tool.

Retrieves EVENT type contexts using a hierarchical time-based indexing strategy.
Events are organized in a 4-level hierarchy:
  L0 — raw individual events (original records)
  L1 — daily summaries (time_bucket: "YYYY-MM-DD")
  L2 — weekly summaries (time_bucket: "YYYY-Www")
  L3 — monthly summaries (time_bucket: "YYYY-MM")

The retrieval algorithm first searches higher-level summaries to identify
relevant time periods, then drills down through refs to gather
fine-grained L0 details.  A parallel direct L0 semantic search acts as a
fallback so that recent events not yet rolled up are still discoverable.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from opencontext.llm.global_embedding_client import do_vectorize as do_vectorize_fn
from opencontext.models.context import ProcessedContext, Vectorize
from opencontext.models.enums import MEMORY_OWNER_TYPES, ContentFormat, ContextType
from opencontext.storage.global_storage import get_storage
from opencontext.tools.base import BaseTool
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Derive all summary type values (upward refs — these should NOT be treated as children)
_ALL_SUMMARY_TYPES = {t.value for types in MEMORY_OWNER_TYPES.values() for t in types[1:]}


class HierarchicalEventTool(BaseTool):
    """
    Retrieval tool for EVENT contexts with hierarchical time-based indexing.

    Instead of relying solely on flat vector search, this tool exploits the
    hierarchy of event summaries (L1-L3) to locate relevant time windows
    first, then drills down to the underlying L0 events.  Results from the
    top-down traversal are merged with a direct L0 semantic search to ensure
    both recall and precision.
    """

    CONTEXT_TYPE = ContextType.EVENT

    # ── Tool metadata ────────────────────────────────────────────────

    @classmethod
    def get_name(cls) -> str:
        return "retrieve_event_context"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Retrieve event contexts using hierarchical time-based indexing.\n"
            "\n"
            "Events are stored in a 4-level hierarchy:\n"
            "  L0 — raw individual events (meetings, tasks, incidents, etc.)\n"
            "  L1 — daily summaries aggregating L0 events for a single day\n"
            "  L2 — weekly summaries aggregating L1 daily summaries\n"
            "  L3 — monthly summaries aggregating L2 weekly summaries\n"
            "\n"
            "The retrieval algorithm works in two parallel paths:\n"
            "  1. **Top-down**: Search L1/L2/L3 summaries to find relevant time\n"
            "     periods, then drill down through refs to collect the\n"
            "     underlying L0 events.  Each child's final score is a weighted\n"
            "     blend of the child's own relevance and the parent summary's\n"
            "     match score (0.5 * child_score + 0.5 * parent_score).\n"
            "  2. **Direct L0 fallback**: A standard semantic search over L0\n"
            "     events catches recent records that haven't been rolled up yet.\n"
            "\n"
            "Results from both paths are merged, deduplicated by context ID,\n"
            "and returned sorted by score descending.\n"
            "\n"
            "**When to use this tool:**\n"
            "- When you need to find events within a specific date/week/month\n"
            "- When you want a high-level overview of what happened over a period\n"
            "- When searching for specific meetings, tasks, or incidents by topic\n"
            "- When you need both summary-level and detail-level event information\n"
            "\n"
            "**Supports:**\n"
            "- Optional time range filtering via start/end timestamps\n"
            "- Multi-user isolation via user_id, device_id, agent_id\n"
            "- Configurable result count (top_k, default 5)"
        )

    @classmethod
    def get_parameters(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language query for semantic search of event records. "
                        "Examples: 'team standup meetings last week', "
                        "'production incidents in January', 'project milestone reviews'. "
                        "Required for meaningful hierarchical retrieval."
                    ),
                },
                "image_url": {
                    "type": "string",
                    "description": (
                        "Optional image URL for multimodal search. "
                        "Can be an HTTP URL or data:image/...;base64,... string."
                    ),
                },
                "video_url": {
                    "type": "string",
                    "description": (
                        "Optional video URL for multimodal search. "
                        "Can be an HTTP URL or data:video/...;base64,... string."
                    ),
                },
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {
                            "type": "integer",
                            "description": (
                                "Start timestamp in seconds (Unix epoch). "
                                "MUST be a pre-calculated integer, not a string or expression."
                            ),
                        },
                        "end": {
                            "type": "integer",
                            "description": (
                                "End timestamp in seconds (Unix epoch). "
                                "MUST be a pre-calculated integer, not a string or expression."
                            ),
                        },
                    },
                    "description": (
                        "Optional time range to narrow the search window. "
                        "When provided, only summaries and events whose time_bucket "
                        "overlaps the given range are considered."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of results to return (default 5).",
                },
                "user_id": {
                    "type": "string",
                    "description": (
                        "User identifier for multi-user filtering. "
                        "Filter results to this specific user."
                    ),
                },
                "device_id": {
                    "type": "string",
                    "description": (
                        "Device identifier for multi-user filtering. "
                        "Filter results to this specific device."
                    ),
                },
                "agent_id": {
                    "type": "string",
                    "description": (
                        "Agent identifier for multi-user filtering. "
                        "Filter results to this specific agent."
                    ),
                },
            },
            "required": ["query"],
        }

    # ── Helpers ──────────────────────────────────────────────────────

    @property
    def storage(self):
        """Lazy access to the global storage singleton."""
        return get_storage()

    async def _search_summaries(
        self,
        level: int,
        time_start: Optional[float],
        time_end: Optional[float],
        user_id: Optional[str],
        top_k: int,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Search summary contexts at a given hierarchy level."""
        try:
            return await self.storage.search_hierarchy(
                context_type=self.CONTEXT_TYPE.value,
                hierarchy_level=level,
                time_start=time_start,
                time_end=time_end,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning(f"search_hierarchy L{level} failed: {e}")
            return []

    async def _drill_down_children(
        self,
        parent_contexts: List[Tuple[ProcessedContext, float]],
        user_id: Optional[str] = None,
    ) -> List[Tuple[ProcessedContext, float, int]]:
        """
        Recursively drill down through refs of matched summaries
        until L0 events are reached.

        Returns a list of (context, blended_score, hierarchy_level) triples.
        The blended score is computed as:
            score = 0.5 * child_score + 0.5 * parent_score
        where child_score defaults to 0.3 for direct children lookups (lower
        baseline so semantic search results rank higher) and parent_score is
        the match score of the summary that led us here.
        """
        results: List[Tuple[ProcessedContext, float, int]] = []

        # Also keep summaries themselves for transparency
        for ctx, score in parent_contexts:
            results.append((ctx, score, ctx.properties.hierarchy_level))

        # BFS through children (deque for O(1) popleft)
        queue: deque = deque(parent_contexts)
        visited_ids: set = {ctx.id for ctx, _ in parent_contexts}

        while queue:
            parent_ctx, parent_score = queue.popleft()
            # Get child IDs from refs (exclude upward/parent refs)
            children_ids = []
            if parent_ctx.properties and parent_ctx.properties.refs:
                for key, ids in parent_ctx.properties.refs.items():
                    if key not in _ALL_SUMMARY_TYPES:
                        children_ids.extend(ids)
            if not children_ids:
                continue

            # Fetch children in batch
            try:
                children = await self.storage.get_contexts_by_ids(
                    ids=children_ids,
                    context_type=self.CONTEXT_TYPE.value,
                )
            except Exception as e:
                logger.warning(f"get_contexts_by_ids failed for {len(children_ids)} IDs: {e}")
                continue

            for child in children:
                if child.id in visited_ids:
                    continue
                visited_ids.add(child.id)

                # Filter by user_id if provided (H4)
                child_user_id = getattr(child.properties, "user_id", None)
                if user_id and child_user_id and child_user_id != user_id:
                    continue

                # Lower baseline score (0.3) so semantic search results rank higher (H3)
                child_intrinsic_score = 0.3
                blended_score = 0.5 * child_intrinsic_score + 0.5 * parent_score
                child_level = child.properties.hierarchy_level

                results.append((child, blended_score, child_level))

                # If this child is itself a summary (level > 0), keep drilling
                if child_level > 0:
                    queue.append((child, blended_score))

        return results

    async def _direct_l0_search(
        self,
        query: str,
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str],
        filters: Dict[str, Any],
        top_k: int,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
    ) -> List[Tuple[ProcessedContext, float, int]]:
        """
        Perform a direct semantic search over L0 events as a fallback.
        This catches recent events that haven't been aggregated into
        higher-level summaries yet.
        """
        try:
            search_filters = dict(filters)
            search_filters["hierarchy_level"] = 0

            has_multimodal = bool(image_url or video_url)
            ark_input = [{"type": "text", "text": query}]
            if image_url:
                ark_input.append({"type": "image_url", "image_url": {"url": image_url}})
            if video_url:
                ark_input.append({"type": "video_url", "video_url": {"url": video_url}})
            vectorize = Vectorize(
                input=ark_input,
                content_format=(
                    ContentFormat.MULTIMODAL if has_multimodal else ContentFormat.TEXT
                ),
            )
            await do_vectorize_fn(vectorize, role="query")
            raw_results = await self.storage.search(
                query=vectorize,
                context_types=[self.CONTEXT_TYPE.value],
                filters=search_filters,
                top_k=top_k,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            )
            return [(ctx, score, 0) for ctx, score in raw_results]
        except Exception as e:
            logger.warning(f"Direct L0 search failed: {e}")
            return []

    @staticmethod
    def _format_result(
        context: ProcessedContext,
        score: float,
        hierarchy_level: int,
    ) -> Dict[str, Any]:
        """Format a single context into the output dict."""
        ed = context.extracted_data
        props = context.properties
        return {
            "id": context.id,
            "context": context.get_llm_context_string(),
            "similarity_score": round(score, 4),
            "context_type": "event",
            "title": ed.title or "",
            "summary": ed.summary or "",
            "keywords": ed.keywords or [],
            "entities": ed.entities or [],
            "create_time": props.create_time.isoformat() if props.create_time else None,
            "event_time_start": props.event_time_start.isoformat() if props.event_time_start else None,
            "event_time_end": props.event_time_end.isoformat() if props.event_time_end else None,
            "hierarchy_level": hierarchy_level,
            "refs": props.refs if props else {},
            "metadata": context.metadata or {},
        }

    # ── Main execute ─────────────────────────────────────────────────

    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute hierarchical event retrieval.

        Algorithm:
          1. Search L1/L2/L3 summaries via storage.search_hierarchy() to
             identify relevant time periods.
          2. For matched summaries, drill down through refs to
             collect L0 detail events. Each child's score is blended with
             its parent's match score (0.5 * child + 0.5 * parent).
          3. Also perform a direct L0 semantic search as fallback for
             events not yet rolled into summaries.
          4. Merge results from both paths, deduplicate by context ID,
             keeping the higher score when duplicates exist.
          5. Sort by score descending, return top_k results.

        Returns:
            List of dicts with keys: context, similarity_score,
            context_type, hierarchy_level.
        """
        query: str = kwargs.get("query", "")
        image_url: Optional[str] = kwargs.get("image_url")
        video_url: Optional[str] = kwargs.get("video_url")
        time_range: Optional[Dict[str, Any]] = kwargs.get("time_range")
        top_k: int = kwargs.get("top_k", 5)
        user_id: Optional[str] = kwargs.get("user_id")
        device_id: Optional[str] = kwargs.get("device_id")
        agent_id: Optional[str] = kwargs.get("agent_id")

        if not query:
            return [{"error": "A query parameter is required for hierarchical event retrieval."}]

        try:
            # ── Derive time bounds from time_range ──────────────────
            ts_start: Optional[float] = None
            ts_end: Optional[float] = None
            time_filters: Dict[str, Any] = {}

            if time_range:
                ts_start = time_range.get("start")
                ts_end = time_range.get("end")

                if ts_start is not None:
                    time_filters["event_time_end_ts"] = {"$gte": ts_start}
                if ts_end is not None:
                    time_filters["event_time_start_ts"] = {"$lte": ts_end}

            # ── Path 1: Top-down hierarchical search ─────────────────
            # Search each summary level and collect matched summaries.
            all_summary_hits: List[Tuple[ProcessedContext, float]] = []

            for level in [1, 2, 3]:
                hits = await self._search_summaries(
                    level=level,
                    time_start=ts_start,
                    time_end=ts_end,
                    user_id=user_id,
                    top_k=top_k,
                    device_id=device_id,
                    agent_id=agent_id,
                )
                all_summary_hits.extend(hits)

            # Drill down from summaries to L0 events
            drilldown_results: List[Tuple[ProcessedContext, float, int]] = []
            if all_summary_hits:
                drilldown_results = await self._drill_down_children(
                    all_summary_hits, user_id=user_id
                )

            # ── Path 2: Direct L0 semantic search (fallback) ─────────
            direct_l0_results = await self._direct_l0_search(
                query=query,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
                filters=time_filters,
                top_k=top_k,
                image_url=image_url,
                video_url=video_url,
            )

            # ── Merge & deduplicate ──────────────────────────────────
            # Use a dict keyed by context ID; keep the higher score.
            merged: Dict[str, Tuple[ProcessedContext, float, int]] = {}

            for ctx, score, level in drilldown_results:
                existing = merged.get(ctx.id)
                if existing is None or score > existing[1]:
                    merged[ctx.id] = (ctx, score, level)

            for ctx, score, level in direct_l0_results:
                existing = merged.get(ctx.id)
                if existing is None or score > existing[1]:
                    merged[ctx.id] = (ctx, score, level)

            # ── Sort & truncate ──────────────────────────────────────
            sorted_results = sorted(merged.values(), key=lambda x: x[1], reverse=True)
            top_results = sorted_results[:top_k]

            # ── Format output ────────────────────────────────────────
            return [self._format_result(ctx, score, level) for ctx, score, level in top_results]

        except Exception as e:
            logger.error(f"HierarchicalEventTool execute exception: {e}")
            return [{"error": f"Error during hierarchical event retrieval: {str(e)}"}]
