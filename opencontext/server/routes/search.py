# -*- coding: utf-8 -*-

"""
Event Search API Route

POST /api/search — Search events with optional semantic query, filters, and upward hierarchy
drill-up. Returns a tree structure where ancestors are parent nodes containing search hits as
children.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.models.context import ProcessedContext, Vectorize
from opencontext.models.enums import ContentFormat, ContextType
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.search.models import (
    EventNode,
    EventSearchRequest,
    EventSearchResponse,
    SearchMetadata,
)
from opencontext.storage.global_storage import get_storage
from opencontext.utils.media_refs import normalize_media_refs
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["search"])

EVENT_TYPE = ContextType.EVENT.value


@router.post("/search")
async def search_events(
    request: EventSearchRequest,
    _auth: str = auth_dependency,
):
    """
    Search events with optional semantic query and filters.

    Supports three search paths (priority order):
    1. **event_ids** — Exact ID lookup
    2. **query** — Semantic search with optional time_range / hierarchy_levels filters
    3. **filters-only** — Browse by time_range and/or hierarchy_levels

    When `drill_up=True`, results are returned as a hierarchy tree where ancestor
    summaries are parent nodes containing search hits as nested children.
    """
    start_time = time.monotonic()
    query_preview = str(request.query)[:50] if request.query else ""

    query_text_for_meta = None
    if request.query:
        text_parts = [item["text"] for item in request.query if item.get("type") == "text"]
        non_text = [item.get("type") for item in request.query if item.get("type") != "text"]
        preview = " ".join(text_parts) if text_parts else ""
        if non_text:
            modalities = []
            if "image_url" in non_text:
                modalities.append("[图片]")
            if "video_url" in non_text:
                modalities.append("[视频]")
            preview = (preview + " " + " ".join(modalities)).strip()
        query_text_for_meta = preview or None

    logger.info(
        f"Event search: query='{query_preview}', "
        f"event_ids={bool(request.event_ids)}, "
        f"drill_up={request.drill_up}, user_id={request.user_id}"
    )

    storage = get_storage()
    if storage is None:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Storage not available"},
        )

    try:
        search_hits, tree_roots = await asyncio.wait_for(
            _execute_search(storage, request),
            timeout=30.0,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Fire-and-forget: track accessed items for memory cache
        if request.user_id and search_hits:
            asyncio.create_task(
                _track_accessed_safe(
                    request.user_id,
                    search_hits,
                    device_id=request.device_id or "default",
                    agent_id=request.agent_id or "default",
                )
            )

        return EventSearchResponse(
            success=True,
            events=tree_roots,
            metadata=SearchMetadata(
                query=query_text_for_meta,
                total_results=len(search_hits),
                search_time_ms=round(elapsed_ms, 2),
            ),
        )

    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.warning(f"Event search timed out after {elapsed_ms:.0f}ms")
        return EventSearchResponse(
            success=False,
            events=[],
            metadata=SearchMetadata(
                query=query_text_for_meta,
                total_results=0,
                search_time_ms=round(elapsed_ms, 2),
            ),
        )

    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.exception(f"Event search failed: {e}")
        return EventSearchResponse(
            success=False,
            events=[],
            metadata=SearchMetadata(
                query=query_text_for_meta,
                total_results=0,
                search_time_ms=round(elapsed_ms, 2),
            ),
        )


async def _execute_search(
    storage,
    request: EventSearchRequest,
) -> Tuple[List[EventNode], List[EventNode]]:
    """Execute the search and return (search_hits_for_tracking, tree_roots)."""

    # ── Step 1: Get raw results ──
    raw_results: List[Tuple[ProcessedContext, float]] = []

    if request.event_ids:
        # Path A: Exact ID lookup
        contexts = await storage.get_contexts_by_ids(request.event_ids, EVENT_TYPE)
        raw_results = [(ctx, 1.0) for ctx in contexts]

    elif request.query:
        # Path B: Semantic search with optional filters (supports multimodal query)
        query_types = {item.get("type") for item in request.query}
        has_multimodal = bool(query_types & {"image_url", "video_url"})
        vectorize = Vectorize(
            input=request.query,
            content_format=(
                ContentFormat.MULTIMODAL if has_multimodal else ContentFormat.TEXT
            ),
        )
        await do_vectorize(vectorize, role="query")

        filters = _build_filters(request.time_range, None)
        raw_results = await storage.search(
            query=vectorize,
            top_k=request.top_k,
            context_types=[EVENT_TYPE],
            filters=filters,
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
            score_threshold=request.score_threshold,
        )

    else:
        # Path C: Filters-only (time_range and/or hierarchy_levels)
        raw_results = await _filter_only_search(storage, request)

    # ── Step 2: Collect ancestors ──
    all_ancestors: Dict[str, ProcessedContext] = {}
    if request.drill_up and raw_results:
        max_level = max(request.hierarchy_levels) if request.hierarchy_levels else 3
        all_ancestors = await _collect_ancestors(storage, raw_results, max_level)

    # ── Step 3: Build node map ──
    nodes: Dict[str, EventNode] = {}

    # Search results -> EventNode with is_search_hit=True, score, content, etc.
    search_hits: List[EventNode] = []
    for ctx, score in raw_results:
        node = _to_search_hit_node(ctx, score)
        nodes[ctx.id] = node
        search_hits.append(node)

    # Ancestors -> EventNode (lightweight, is_search_hit=False)
    for aid, actx in all_ancestors.items():
        if aid not in nodes:  # Don't overwrite search hits
            nodes[aid] = _to_context_node(actx)

    # ── Step 4: Link children to parents, build tree ──
    roots: List[EventNode] = []
    linked: set = set()
    for node_id, node in nodes.items():
        pid = node.parent_id
        if pid and pid in nodes and node_id not in linked:
            nodes[pid].children.append(node)
            linked.add(node_id)
        else:
            roots.append(node)

    # ── Step 5: Sort ──
    def sort_tree(node_list: List[EventNode]):
        node_list.sort(key=lambda n: (n.time_bucket or ""))
        for n in node_list:
            if n.children:
                sort_tree(n.children)

    sort_tree(roots)

    return search_hits, roots


async def _filter_only_search(
    storage,
    request: EventSearchRequest,
) -> List[Tuple[ProcessedContext, float]]:
    """Search by filters only (no semantic query)."""

    # If specific hierarchy levels are requested, query each level
    if request.hierarchy_levels:
        time_bucket_start = None
        time_bucket_end = None
        if request.time_range:
            time_bucket_start, time_bucket_end = _time_range_to_buckets(
                request.time_range.start, request.time_range.end
            )

        tasks = []
        for level in request.hierarchy_levels:
            tasks.append(
                storage.search_hierarchy(
                    context_type=EVENT_TYPE,
                    hierarchy_level=level,
                    time_bucket_start=time_bucket_start,
                    time_bucket_end=time_bucket_end,
                    user_id=request.user_id,
                    device_id=request.device_id,
                    agent_id=request.agent_id,
                    top_k=request.top_k,
                )
            )

        level_results = await asyncio.gather(*tasks)
        merged: List[Tuple[ProcessedContext, float]] = []
        for results in level_results:
            merged.extend(results)
        # Deduplicate by ID, keep first occurrence
        seen_ids = set()
        deduped = []
        for item in merged:
            if item[0].id not in seen_ids:
                seen_ids.add(item[0].id)
                deduped.append(item)
        return deduped[: request.top_k]

    # Only time_range provided — fetch all events with time filter
    filters: Dict[str, Any] = {}
    if request.time_range:
        ts_filter: Dict[str, Any] = {}
        if request.time_range.start is not None:
            ts_filter["$gte"] = request.time_range.start
        if request.time_range.end is not None:
            ts_filter["$lte"] = request.time_range.end
        if ts_filter:
            filters["event_time_ts"] = ts_filter

    result = await storage.get_all_processed_contexts(
        context_types=[EVENT_TYPE],
        limit=request.top_k,
        filter=filters,
        user_id=request.user_id,
        device_id=request.device_id,
        agent_id=request.agent_id,
    )

    # get_all_processed_contexts returns Dict[str, List[ProcessedContext]]
    contexts = result.get(EVENT_TYPE, [])
    return [(ctx, 1.0) for ctx in contexts]


async def _collect_ancestors(
    storage,
    results: List[Tuple[ProcessedContext, float]],
    max_level: int,
) -> Dict[str, ProcessedContext]:
    """
    Batch collect ancestors: for each result, follow parent_id chain upward.

    Returns a mapping of ancestor_id -> ProcessedContext (excluding search results themselves).
    Uses a shared cache to avoid duplicate fetches (multiple L0 events may share the same L1 parent).
    """
    # Map of context_id -> ProcessedContext (cache)
    seen: Dict[str, ProcessedContext] = {}

    # Seed the cache with the search results themselves
    for ctx, _ in results:
        seen[ctx.id] = ctx

    # Collect initial parent_ids to fetch
    current_round: Dict[str, List[str]] = {}  # parent_id -> list of child context IDs needing it
    for ctx, _ in results:
        if ctx.properties and ctx.properties.parent_id:
            pid = ctx.properties.parent_id
            if pid not in seen:
                current_round.setdefault(pid, []).append(ctx.id)

    # Iterative batch fetch (max 3 rounds: L0->L1, L1->L2, L2->L3)
    for _ in range(3):
        if not current_round:
            break

        # Batch fetch all needed parent IDs
        parent_ids = list(current_round.keys())
        parents = await storage.get_contexts_by_ids(parent_ids, EVENT_TYPE)
        parent_by_id = {p.id: p for p in parents}

        next_round: Dict[str, List[str]] = {}

        for pid, child_ids in current_round.items():
            parent = parent_by_id.get(pid)
            if parent is None:
                continue

            # Check if this parent exceeds max_level
            parent_level = parent.properties.hierarchy_level if parent.properties else 0
            if parent_level > max_level:
                continue

            seen[pid] = parent

            # Queue next level parent
            if parent.properties and parent.properties.parent_id:
                next_pid = parent.properties.parent_id
                if next_pid not in seen:
                    for cid in child_ids:
                        next_round.setdefault(next_pid, []).append(cid)

        current_round = next_round

    # Return only ancestors (exclude search results themselves)
    result_ids = {ctx.id for ctx, _ in results}
    return {k: v for k, v in seen.items() if k not in result_ids}


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
        # Single level: exact match (MatchValue for Qdrant compatibility)
        filters["hierarchy_level"] = hierarchy_levels[0]
    elif hierarchy_levels is not None and len(hierarchy_levels) > 1:
        # Multiple levels: list match (MatchAny for Qdrant compatibility)
        filters["hierarchy_level"] = hierarchy_levels

    return filters


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


def _format_timestamp(value) -> Optional[str]:
    """Format a timestamp value to ISO string."""
    if not value:
        return None
    try:
        return value.isoformat()
    except (AttributeError, ValueError):
        return str(value)


def _normalize_parent_id(props) -> Optional[str]:
    """Return parent_id or None."""
    if props and props.parent_id:
        return props.parent_id
    return None


def _extract_media_refs(ctx: ProcessedContext) -> List[Dict[str, Any]]:
    """Extract normalized media_refs from context metadata."""
    if not ctx.metadata:
        return []
    return normalize_media_refs(ctx.metadata.get("media_refs"))


def _to_context_node(ctx: ProcessedContext) -> EventNode:
    """Convert a ProcessedContext to a lightweight EventNode (for ancestors)."""
    props = ctx.properties
    extracted = ctx.extracted_data

    return EventNode(
        id=ctx.id,
        hierarchy_level=props.hierarchy_level if props else 0,
        time_bucket=props.time_bucket if props else None,
        parent_id=_normalize_parent_id(props),
        title=extracted.title if extracted else None,
        summary=extracted.summary if extracted else None,
        event_time=_format_timestamp(props.event_time if props else None),
        create_time=_format_timestamp(props.create_time if props else None),
        is_search_hit=False,
        media_refs=_extract_media_refs(ctx),
    )


def _to_search_hit_node(ctx: ProcessedContext, score: float) -> EventNode:
    """Convert a ProcessedContext to a search-hit EventNode with full data."""
    props = ctx.properties
    extracted = ctx.extracted_data

    return EventNode(
        id=ctx.id,
        title=extracted.title if extracted else None,
        summary=extracted.summary if extracted else None,
        keywords=extracted.keywords if extracted and extracted.keywords else [],
        entities=extracted.entities if extracted and extracted.entities else [],
        score=score,
        hierarchy_level=props.hierarchy_level if props else 0,
        time_bucket=props.time_bucket if props else None,
        parent_id=_normalize_parent_id(props),
        event_time=_format_timestamp(props.event_time if props else None),
        create_time=_format_timestamp(props.create_time if props else None),
        is_search_hit=True,
        media_refs=_extract_media_refs(ctx),
    )


async def _track_accessed_safe(
    user_id: str,
    results: List[EventNode],
    device_id: str = "default",
    agent_id: str = "default",
) -> None:
    """Fire-and-forget: record accessed event IDs in Redis for memory cache."""
    try:
        items: List[dict] = []
        for er in results:
            items.append(
                {
                    "id": er.id,
                    "context_type": EVENT_TYPE,
                    "title": er.title,
                    "summary": er.summary,
                    "keywords": er.keywords,
                    "score": er.score,
                    "event_time": er.event_time,
                    "create_time": er.create_time,
                    "media_refs": er.media_refs,
                }
            )
        if items:
            from opencontext.server.cache.memory_cache_manager import get_memory_cache_manager

            await get_memory_cache_manager().track_accessed(
                user_id, items, device_id=device_id, agent_id=agent_id
            )
    except Exception as e:
        logger.debug(f"Access tracking failed (non-critical): {e}")
