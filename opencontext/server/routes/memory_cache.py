# -*- coding: utf-8 -*-

"""
User Memory Cache API Route

GET /api/memory-cache — Returns a user's current memory state snapshot.
DELETE /api/memory-cache — Invalidates the cache for a user.
"""

import asyncio
import time
from typing import Optional, Set

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from opencontext.server.cache.memory_cache_manager import get_memory_cache_manager
from opencontext.server.middleware.auth import auth_dependency
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["memory-cache"])

VALID_SECTIONS = {"profile", "events", "accessed"}
DEFAULT_SECTIONS = {"profile", "events", "accessed"}


def _parse_include(include: Optional[str]) -> Set[str]:
    """Parse comma-separated include parameter into a set of valid section names."""
    if include is None:
        return DEFAULT_SECTIONS.copy()
    sections = {s.strip().lower() for s in include.split(",") if s.strip()}
    if "all" in sections:
        return VALID_SECTIONS.copy()
    valid = sections & VALID_SECTIONS
    return valid if valid else DEFAULT_SECTIONS.copy()


@router.get("/memory-cache")
async def get_user_memory_cache(
    user_id: str = Query(..., description="User identifier (required)"),
    device_id: str = Query(default="default", description="Device identifier"),
    agent_id: str = Query(default="default", description="Agent identifier"),
    memory_owner: str = Query(
        default="user",
        description="Memory owner type: 'user' or 'agent'. Controls which context types are queried.",
    ),
    include: Optional[str] = Query(
        default=None,
        description="Comma-separated response sections: profile,events,accessed,all. Default: profile,events,accessed",
    ),
    recent_days: int = Query(default=None, description="Recent memory window in days"),
    max_recent_events_today: int = Query(default=None, description="Max today L0 events"),
    max_accessed: int = Query(default=5, ge=1, le=100, description="Max recently accessed items"),
    force_refresh: bool = Query(default=False, description="Force rebuild cache"),
    _auth: str = auth_dependency,
):
    """
    Get a user's current memory state in a single call.

    Returns:
    - profile: User profile data
    - entities: Known entities for this user
    - recently_accessed: Memories recently returned in search results (real-time)
    - recent_memories: Hierarchical recent memories (today L0 events + daily summaries)
    """
    if memory_owner not in ("user", "agent"):
        raise HTTPException(400, "memory_owner must be 'user' or 'agent'")

    t0 = time.monotonic()
    manager = get_memory_cache_manager()
    include_sections = _parse_include(include)

    try:
        response = await asyncio.wait_for(
            manager.get_user_memory_cache(
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
                recent_days=recent_days,
                max_recent_events_today=max_recent_events_today,
                max_accessed=max_accessed,
                force_refresh=force_refresh,
                include_sections=include_sections,
                memory_owner=memory_owner,
            ),
            timeout=15.0,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(f"Memory cache response time: {elapsed_ms:.2f}ms")
        return response
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"success": False, "error": "Memory cache request timed out"},
        )
    except Exception as e:
        logger.exception(f"Memory cache request failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@router.delete("/memory-cache")
async def invalidate_user_memory_cache(
    user_id: str = Query(..., description="User identifier"),
    device_id: str = Query(default="default", description="Device identifier"),
    agent_id: str = Query(default="default", description="Agent identifier"),
    memory_owner: str = Query(
        default="user",
        description="Memory owner type: 'user' or 'agent'.",
    ),
    _auth: str = auth_dependency,
):
    """Manually invalidate a user's (or agent's) memory cache snapshot."""
    manager = get_memory_cache_manager()
    await manager.invalidate_snapshot(user_id, device_id, agent_id, memory_owner=memory_owner)
    return {
        "success": True,
        "message": (
            f"Cache invalidated for owner={memory_owner}, user={user_id}, "
            f"device={device_id}, agent={agent_id}"
        ),
    }
