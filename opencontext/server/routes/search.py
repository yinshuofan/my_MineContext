"""
Event Search API Route

POST /api/search — Search events with optional semantic query, filters, and upward hierarchy
drill-up. Returns a tree structure where ancestors are parent nodes containing search hits as
children.
"""

import asyncio
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from opencontext.models.context import ProcessedContext
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.search.event_search_service import EventSearchService
from opencontext.server.search.models import (
    EventNode,
    EventSearchRequest,
    EventSearchResponse,
    SearchMetadata,
)
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.media_refs import normalize_media_refs

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

    When `drill` is 'up', 'down', or 'both', results include hierarchy context
    (ancestors, descendants, or both) and are returned as a tree structure.
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
        f"drill={request.drill}, user_id={request.user_id}"
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

    except TimeoutError:
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
) -> tuple[list[EventNode], list[EventNode]]:
    """Execute the search and return (search_hits_for_tracking, tree_roots)."""

    # ── Step 1: Get raw results + ancestors via service ──
    raw_results: list[tuple[ProcessedContext, float]] = []
    all_ancestors: dict[str, ProcessedContext] = {}
    all_descendants: dict[str, ProcessedContext] = {}

    if request.event_ids:
        # Path A: Exact ID lookup — search all collections (IDs may be any type)
        contexts = await storage.get_contexts_by_ids(request.event_ids)
        raw_results = [(ctx, 1.0) for ctx in contexts]

        if raw_results:
            if request.drill in ("up", "both"):
                max_level = max(request.hierarchy_levels) if request.hierarchy_levels else 3
                all_ancestors = await _search_service.collect_ancestors(raw_results, max_level)
            if request.drill in ("down", "both"):
                min_level = min(request.hierarchy_levels) if request.hierarchy_levels else 0
                all_descendants = await _search_service.collect_descendants(
                    raw_results, min_level=min_level
                )

    else:
        # Path B: Unified search (vector if query provided, filter-only otherwise)
        result = await _search_service.search(
            query=request.query,
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
            hierarchy_levels=request.hierarchy_levels,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            time_range=request.time_range,
            drill=request.drill,
        )
        raw_results = result.hits
        all_ancestors = result.ancestors
        all_descendants = result.descendants

    # ── Step 2: Build node map (stays in route — EventNode is a response model) ──
    nodes: dict[str, EventNode] = {}
    search_hits: list[EventNode] = []
    for ctx, score in raw_results:
        node = _to_search_hit_node(ctx, score)
        nodes[ctx.id] = node
        search_hits.append(node)

    for aid, actx in all_ancestors.items():
        if aid not in nodes:
            nodes[aid] = _to_context_node(actx)

    for did, dctx in all_descendants.items():
        if did not in nodes:
            nodes[did] = _to_context_node(dctx)

    # ── Step 3: Link children to parents, build tree ──
    roots: list[EventNode] = []
    linked: set = set()
    for node_id, node in nodes.items():
        pid = _extract_parent_id_from_refs(node, nodes)
        if pid and pid in nodes and node_id not in linked:
            nodes[pid].children.append(node)
            linked.add(node_id)
        else:
            roots.append(node)

    # ── Step 4: Sort ──
    def sort_tree(node_list: list[EventNode]):
        node_list.sort(key=lambda n: n.event_time_start or "")
        for n in node_list:
            if n.children:
                sort_tree(n.children)

    sort_tree(roots)
    return search_hits, roots


def _format_timestamp(value) -> str | None:
    """Format a timestamp value to ISO string."""
    if not value:
        return None
    try:
        return value.isoformat()
    except (AttributeError, ValueError):
        return str(value)


def _extract_parent_id_from_refs(node: EventNode, nodes: dict[str, Any]) -> str | None:
    """Find the first upward ref target that exists in the nodes map.

    Uses hierarchy_level comparison: a ref target with higher level than the
    current node is a parent. Skips same-level or lower-level refs.
    """
    node_level = node.hierarchy_level
    for ref_ids in node.refs.values():
        for pid in ref_ids:
            if pid in nodes and nodes[pid].hierarchy_level > node_level:
                return pid
    return None


def _extract_media_refs(ctx: ProcessedContext) -> list[dict[str, Any]]:
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
        agent_commentary=extracted.agent_commentary if extracted else None,
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
        agent_commentary=extracted.agent_commentary if extracted else None,
        is_search_hit=True,
        media_refs=_extract_media_refs(ctx),
    )


async def _track_accessed_safe(
    user_id: str,
    results: list[EventNode],
    device_id: str = "default",
    agent_id: str = "default",
) -> None:
    """Fire-and-forget: record accessed event IDs in Redis for memory cache."""
    try:
        l0_type = _search_service.get_l0_type()
        items: list[dict] = []
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
