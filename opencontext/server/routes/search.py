# -*- coding: utf-8 -*-

"""
Unified Search API Route

POST /api/search — Single endpoint with fast and intelligent search strategies.
"""

import asyncio
import time
from typing import List

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from opencontext.models.enums import ContextType
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.search.fast_strategy import FastSearchStrategy
from opencontext.server.search.intelligent_strategy import IntelligentSearchStrategy
from opencontext.server.search.models import (
    SearchMetadata,
    SearchStrategy,
    TypedResults,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["search"])

# Lazy-initialized strategy singletons (stateless, reusable)
_fast_strategy = None
_intelligent_strategy = None

ALL_CONTEXT_TYPES = [ct.value for ct in ContextType]


def _get_fast_strategy() -> FastSearchStrategy:
    global _fast_strategy
    if _fast_strategy is None:
        _fast_strategy = FastSearchStrategy()
    return _fast_strategy


def _get_intelligent_strategy() -> IntelligentSearchStrategy:
    global _intelligent_strategy
    if _intelligent_strategy is None:
        _intelligent_strategy = IntelligentSearchStrategy()
    return _intelligent_strategy


@router.post("/search")
async def unified_search(
    request: UnifiedSearchRequest,
    _auth: str = auth_dependency,
):
    """
    Unified search endpoint with strategy selection.

    - **fast**: Direct parallel search across all types. Zero LLM reasoning calls.
    - **intelligent**: LLM-driven agentic search with tool selection and result validation.

    Both strategies return the same response shape with results grouped by type.
    """
    start_time = time.monotonic()

    logger.info(
        f"Search request: strategy={request.strategy.value}, "
        f"query='{request.query[:50]}', user_id={request.user_id}"
    )

    # Validate and resolve context_types
    context_types = request.context_types or ALL_CONTEXT_TYPES
    context_types = [ct for ct in context_types if ct in ALL_CONTEXT_TYPES]

    if not context_types:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"No valid context types specified. Valid types: {ALL_CONTEXT_TYPES}",
            },
        )

    # Select strategy
    if request.strategy == SearchStrategy.FAST:
        strategy = _get_fast_strategy()
    else:
        strategy = _get_intelligent_strategy()

    try:
        results = await asyncio.wait_for(
            strategy.search(
                query=request.query,
                context_types=context_types,
                top_k=request.top_k,
                time_range=request.time_range,
                user_id=request.user_id,
                device_id=request.device_id,
                agent_id=request.agent_id,
            ),
            timeout=30.0,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        total = (
            (1 if results.profile else 0)
            + len(results.entities)
            + len(results.documents)
            + len(results.events)
            + len(results.knowledge)
        )

        # Track actually searched types (profile/entity need user_id)
        actually_searched = context_types
        if not request.user_id:
            actually_searched = [
                ct
                for ct in context_types
                if ct not in (ContextType.PROFILE.value, ContextType.ENTITY.value)
            ]

        # Fire-and-forget: track accessed items for memory cache
        if request.user_id and total > 0:
            asyncio.create_task(
                _track_accessed_safe(
                    request.user_id,
                    results,
                    device_id=request.device_id or "default",
                    agent_id=request.agent_id or "default",
                )
            )

        return UnifiedSearchResponse(
            success=True,
            results=results,
            metadata=SearchMetadata(
                strategy=request.strategy.value,
                query=request.query,
                total_results=total,
                search_time_ms=round(elapsed_ms, 2),
                types_searched=actually_searched,
            ),
        )

    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.warning(
            f"Search timed out after {elapsed_ms:.0f}ms for query='{request.query[:50]}'"
        )
        return UnifiedSearchResponse(
            success=False,
            results=TypedResults(),
            metadata=SearchMetadata(
                strategy=request.strategy.value,
                query=request.query,
                total_results=0,
                search_time_ms=round(elapsed_ms, 2),
                types_searched=context_types,
            ),
        )

    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.exception(f"Unified search failed: {e}")

        return UnifiedSearchResponse(
            success=False,
            results=TypedResults(),
            metadata=SearchMetadata(
                strategy=request.strategy.value,
                query=request.query,
                total_results=0,
                search_time_ms=round(elapsed_ms, 2),
                types_searched=context_types,
            ),
        )


async def _track_accessed_safe(
    user_id: str,
    results: TypedResults,
    device_id: str = "default",
    agent_id: str = "default",
) -> None:
    """Fire-and-forget: record accessed context IDs in Redis for memory cache."""
    try:
        items: List[dict] = []
        for vr in results.documents + results.events + results.knowledge:
            items.append(
                {
                    "id": vr.id,
                    "context_type": vr.context_type,
                    "title": vr.title,
                    "summary": vr.summary,
                    "keywords": vr.keywords,
                    "score": vr.score,
                    "event_time": vr.event_time,
                    "create_time": vr.create_time,
                }
            )
        if items:
            from opencontext.server.cache.memory_cache_manager import get_memory_cache_manager

            await get_memory_cache_manager().track_accessed(
                user_id, items, device_id=device_id, agent_id=agent_id
            )
    except Exception as e:
        logger.debug(f"Access tracking failed (non-critical): {e}")
