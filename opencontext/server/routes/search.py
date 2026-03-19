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
from opencontext.models.enums import MEMORY_OWNER_TYPES, ContentFormat, ContextType
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

def _get_l0_type(memory_owner: str) -> str:
    """Get the L0 event ContextType value for a memory owner."""
    types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
    return types[0].value  # index 0 = L0


def _get_context_types_for_levels(memory_owner: str, levels: Optional[List[int]]) -> List[str]:
    """Map hierarchy_levels + memory_owner to ContextType values."""
    types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
    if levels:
        return [types[l].value for l in levels if l < len(types)]
    return [t.value for t in types]


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
                    memory_owner=request.memory_owner,
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

    l0_type = _get_l0_type(request.memory_owner)

    if request.event_ids:
        # Path A: Exact ID lookup
        contexts = await storage.get_contexts_by_ids(request.event_ids, l0_type)
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
            context_types=_get_context_types_for_levels(request.memory_owner, None),
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
        all_ancestors = await _collect_ancestors(
            storage, raw_results, max_level, memory_owner=request.memory_owner
        )

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
    owner_types = MEMORY_OWNER_TYPES.get(request.memory_owner, MEMORY_OWNER_TYPES["user"])
    summary_type_values = {t.value for t in owner_types[1:]}

    roots: List[EventNode] = []
    linked: set = set()
    for node_id, node in nodes.items():
        pid = _extract_parent_id_from_refs(node.refs, nodes, summary_type_values)
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

        owner_types = MEMORY_OWNER_TYPES.get(
            request.memory_owner, MEMORY_OWNER_TYPES["user"]
        )
        tasks = []
        for level in request.hierarchy_levels:
            if level < len(owner_types):
                tasks.append(
                    storage.search_hierarchy(
                        context_type=owner_types[level].value,
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

    all_types = _get_context_types_for_levels(request.memory_owner, None)
    result = await storage.get_all_processed_contexts(
        context_types=all_types,
        limit=request.top_k,
        filter=filters,
        user_id=request.user_id,
        device_id=request.device_id,
        agent_id=request.agent_id,
    )

    # get_all_processed_contexts returns Dict[str, List[ProcessedContext]]
    contexts: List[ProcessedContext] = []
    for ct in all_types:
        contexts.extend(result.get(ct, []))
    return [(ctx, 1.0) for ctx in contexts]


async def _collect_ancestors(
    storage,
    results: List[Tuple[ProcessedContext, float]],
    max_level: int,
    memory_owner: str = "user",
) -> Dict[str, ProcessedContext]:
    """
    Collect ancestors by following refs upward (to summary types).

    Uses refs-based traversal with parent_id fallback for old data.
    Returns a mapping of ancestor_id -> ProcessedContext (excluding search results themselves).
    """
    owner_types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
    summary_type_values = {t.value for t in owner_types[1:]}

    ancestor_map = {}
    all_ancestors = {}
    seen = set()
    current_batch = []

    for ctx, score in results:
        seen.add(ctx.id)
        if not ctx.properties or not ctx.properties.refs:
            # Fallback to parent_id for old data (field removed from model, may exist in metadata)
            old_pid = (ctx.metadata or {}).get("parent_id") if ctx.properties else None
            if old_pid:
                if old_pid not in seen:
                    seen.add(old_pid)
                    current_batch.append(old_pid)
                    ancestor_map.setdefault(ctx.id, []).append(old_pid)
            continue
        for ref_key, ref_ids in ctx.properties.refs.items():
            if ref_key in summary_type_values:
                for pid in ref_ids:
                    if pid not in seen:
                        seen.add(pid)
                        current_batch.append(pid)
                        ancestor_map.setdefault(ctx.id, []).append(pid)

    rounds = 0
    while current_batch and rounds < 3:
        parents = await storage.get_contexts_by_ids(current_batch)
        next_batch = []
        for parent in parents:
            all_ancestors[parent.id] = parent
            if not parent.properties or not parent.properties.refs:
                # Fallback to parent_id for old data (field removed from model, may exist in metadata)
                old_pid = (parent.metadata or {}).get("parent_id") if parent.properties else None
                if old_pid:
                    if old_pid not in seen:
                        seen.add(old_pid)
                        next_batch.append(old_pid)
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


def _extract_parent_id_from_refs(
    refs: Dict[str, List[str]], nodes: Dict[str, Any], summary_type_values: set = None
) -> Optional[str]:
    """Find the first upward ref target that exists in the nodes map (i.e., a known ancestor).

    Only considers refs whose keys are summary types (upward/parent direction).
    Skips child refs to avoid incorrect parent-child linkage.
    """
    for ref_key, ref_ids in refs.items():
        if summary_type_values and ref_key not in summary_type_values:
            continue  # skip downward (child) refs
        for pid in ref_ids:
            if pid in nodes:
                return pid
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
        refs=props.refs if props else {},
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
        refs=props.refs if props else {},
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
    memory_owner: str = "user",
) -> None:
    """Fire-and-forget: record accessed event IDs in Redis for memory cache."""
    try:
        l0_type = _get_l0_type(memory_owner)
        items: List[dict] = []
        for er in results:
            items.append(
                {
                    "id": er.id,
                    "context_type": l0_type,
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
