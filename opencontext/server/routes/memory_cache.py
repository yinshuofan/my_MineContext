"""
User Memory Cache API Route

GET /api/memory-cache — Returns a user's current memory state snapshot.
DELETE /api/memory-cache — Invalidates the cache for a user.
"""

import asyncio
import time

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from opencontext.server.cache.memory_cache_manager import get_memory_cache_manager
from opencontext.server.middleware.auth import auth_dependency
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["memory-cache"])

VALID_SECTIONS = {"profile", "agent_prompt", "events", "accessed"}
DEFAULT_SECTIONS = {"profile", "agent_prompt", "events", "accessed"}


def _parse_include(include: str | None) -> set[str]:
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
    user_id: str = Query(..., description="User identifier (required)"),  # noqa: B008
    device_id: str = Query(default="default", description="Device identifier"),  # noqa: B008
    agent_id: str = Query(default="default", description="Agent identifier"),  # noqa: B008
    include: str | None = Query(  # noqa: B008
        default=None,
        description=(
            "Comma-separated response sections: "
            "profile,agent_prompt,events,accessed,all. "
            "Default: profile,agent_prompt,events,accessed"
        ),
    ),
    recent_days: int = Query(default=None, description="Recent memory window in days"),  # noqa: B008
    max_recent_events_today: int = Query(default=None, description="Max today L0 events"),  # noqa: B008
    max_accessed: int = Query(default=5, ge=1, le=100, description="Max recently accessed items"),  # noqa: B008
    force_refresh: bool = Query(default=False, description="Force rebuild cache"),  # noqa: B008
    _auth: str = auth_dependency,
):
    """
    Get a user's current memory state in a single call.

    Returns:
    - profile: User profile data
    - recently_accessed: Memories recently returned in search results (real-time)
    - recent_memories: Hierarchical recent memories (today L0 events + daily summaries)
    """
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
            ),
            timeout=15.0,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(f"Memory cache response time: {elapsed_ms:.2f}ms")
        return response
    except TimeoutError:
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
    user_id: str = Query(..., description="User identifier"),  # noqa: B008
    device_id: str = Query(default="default", description="Device identifier"),  # noqa: B008
    agent_id: str = Query(default="default", description="Agent identifier"),  # noqa: B008
    _auth: str = auth_dependency,
):
    """Manually invalidate a user's memory cache snapshot."""
    manager = get_memory_cache_manager()
    await manager.invalidate_snapshot(user_id, device_id, agent_id)
    return {
        "success": True,
        "message": (f"Cache invalidated for user={user_id}, device={device_id}, agent={agent_id}"),
    }
