# -*- coding: utf-8 -*-

"""
Event Search API Route

POST /api/search — Search events with optional semantic query, filters, and upward hierarchy
drill-up. Returns a tree structure where ancestors are parent nodes containing search hits as
children.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from opencontext.models.context import ProcessedContext
from opencontext.models.enums import ContextType
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.search.event_search_service import EventSearchService
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

_search_service = EventSearchService()


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

    # ── Step 1: Get raw results + ancestors via service ──
    raw_results: List[Tuple[ProcessedContext, float]] = []
    all_ancestors: Dict[str, ProcessedContext] = {}

    l0_type = _search_service.get_l0_type()

    if request.event_ids:
        # Path A: Exact ID lookup (stays in route — not a search)
        contexts = await storage.get_contexts_by_ids(request.event_ids, l0_type)
        raw_results = [(ctx, 1.0) for ctx in contexts]

        if request.drill_up and raw_results:
            max_level = max(request.hierarchy_levels) if request.hierarchy_levels else 3
            all_ancestors = await _search_service.collect_ancestors(
                raw_results, max_level
            )

    elif request.query:
        # Path B: Semantic search → delegate to service
        result = await _search_service.semantic_search(
            query=request.query,
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            time_range=request.time_range,
            drill_up=request.drill_up,
        )
        raw_results = result.hits
        all_ancestors = result.ancestors

    else:
        # Path C: Filters-only → delegate to service
        raw_results = await _search_service.filter_search(
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
            hierarchy_levels=request.hierarchy_levels,
            time_range=request.time_range,
            top_k=request.top_k,
        )

        if request.drill_up and raw_results:
            max_level = max(request.hierarchy_levels) if request.hierarchy_levels else 3
            all_ancestors = await _search_service.collect_ancestors(
                raw_results, max_level
            )

    # ── Step 2: Build node map (stays in route — EventNode is a response model) ──
    nodes: Dict[str, EventNode] = {}
    search_hits: List[EventNode] = []
    for ctx, score in raw_results:
        node = _to_search_hit_node(ctx, score)
        nodes[ctx.id] = node
        search_hits.append(node)

    for aid, actx in all_ancestors.items():
        if aid not in nodes:
            nodes[aid] = _to_context_node(actx)

    # ── Step 3: Link children to parents, build tree ──
    user_summary_types = {ContextType.DAILY_SUMMARY.value, ContextType.WEEKLY_SUMMARY.value, ContextType.MONTHLY_SUMMARY.value}
    base_summary_types = {ContextType.AGENT_BASE_L1_SUMMARY.value, ContextType.AGENT_BASE_L2_SUMMARY.value, ContextType.AGENT_BASE_L3_SUMMARY.value}
    summary_type_values = user_summary_types | base_summary_types

    roots: List[EventNode] = []
    linked: set = set()
    for node_id, node in nodes.items():
        pid = _extract_parent_id_from_refs(node.refs, nodes, summary_type_values)
        if pid and pid in nodes and node_id not in linked:
            nodes[pid].children.append(node)
            linked.add(node_id)
        else:
            roots.append(node)

    # ── Step 4: Sort ──
    def sort_tree(node_list: List[EventNode]):
        node_list.sort(key=lambda n: (n.event_time_start or ""))
        for n in node_list:
            if n.children:
                sort_tree(n.children)

    sort_tree(roots)
    return search_hits, roots


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
        refs=props.refs if props else {},
        title=extracted.title if extracted else None,
        summary=extracted.summary if extracted else None,
        event_time_start=_format_timestamp(props.event_time_start if props else None),
        event_time_end=_format_timestamp(props.event_time_end if props else None),
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
        refs=props.refs if props else {},
        event_time_start=_format_timestamp(props.event_time_start if props else None),
        event_time_end=_format_timestamp(props.event_time_end if props else None),
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
        l0_type = _search_service.get_l0_type()
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
                    "event_time_start": er.event_time_start,
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
