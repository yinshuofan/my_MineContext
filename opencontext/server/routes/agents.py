#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Agent registry CRUD routes and agent base memory endpoints."""

import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    ProcessedContextModel,
    Vectorize,
)
from opencontext.models.enums import ContentFormat, ContextType
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.utils import convert_resp
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.time_utils import get_timezone
from opencontext.utils.time_utils import now as tz_now

logger = get_logger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])


# ============================================================================
# Request Models
# ============================================================================


class CreateAgentRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""


class UpdateAgentRequest(BaseModel):
    name: str | None = Field(None, max_length=255)
    description: str | None = None


class BaseProfileRequest(BaseModel):
    factual_profile: str
    behavioral_profile: str | None = None
    entities: list[str] = Field(default_factory=list)
    importance: int = 0


# Mapping from hierarchy_level to context type for base events
_BASE_HIERARCHY_LEVEL_TO_TYPE = {
    0: ContextType.AGENT_BASE_EVENT,
    1: ContextType.AGENT_BASE_L1_SUMMARY,
    2: ContextType.AGENT_BASE_L2_SUMMARY,
    3: ContextType.AGENT_BASE_L3_SUMMARY,
}

# All AGENT_BASE_* context types (for queries)
_ALL_AGENT_BASE_TYPES = [ct.value for ct in _BASE_HIERARCHY_LEVEL_TO_TYPE.values()]


class BaseEventItem(BaseModel):
    title: str
    summary: str
    event_time_start: str | None = None  # ISO 8601, defaults to current time
    event_time_end: str | None = None  # Required for hierarchy_level > 0
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    importance: int = 5
    hierarchy_level: int = 0  # 0/1/2/3, pure hierarchy depth
    children: list["BaseEventItem"] | None = None  # Nested child events


class BaseEventsRequest(BaseModel):
    events: list[BaseEventItem] = Field(..., min_length=1)


def _validate_base_event_tree(
    events: list[BaseEventItem],
    path: str = "events",
) -> int:
    """Validate a list of base events (potentially with nested children).

    Returns the total count of events across all levels.
    Raises HTTPException(400) on validation failure with path-based error message.
    """
    total_count = 0

    for i, event in enumerate(events):
        node_path = f"{path}[{i}]"
        level = event.hierarchy_level

        # Validate hierarchy_level range
        if level not in _BASE_HIERARCHY_LEVEL_TO_TYPE:
            raise HTTPException(
                status_code=400,
                detail=f"{node_path}: hierarchy_level must be 0-3, got {level}",
            )

        # Parse event times for this node
        ets = _parse_event_time(event.event_time_start, node_path, "event_time_start")

        if level > 0:
            # Summary node validations
            if not event.event_time_end:
                raise HTTPException(
                    status_code=400,
                    detail=f"{node_path}: event_time_end is required for hierarchy_level > 0",
                )
            if not event.children:
                raise HTTPException(
                    status_code=400,
                    detail=f"{node_path}: children is required for hierarchy_level > 0",
                )

            ete = _parse_event_time(event.event_time_end, node_path, "event_time_end")

            if ets and ete and ets > ete:
                raise HTTPException(
                    status_code=400,
                    detail=f"{node_path}: event_time_start must be <= event_time_end",
                )

            # Validate children hierarchy_level
            for j, child in enumerate(event.children):
                if child.hierarchy_level != level - 1:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"{node_path}.children[{j}]: hierarchy_level must be "
                            f"{level - 1} (parent is {level}), got {child.hierarchy_level}"
                        ),
                    )

            # Time range coverage: parent must cover all direct children
            child_starts = []
            child_ends = []
            for j, child in enumerate(event.children):
                child_path = f"{node_path}.children[{j}]"
                cs = _parse_event_time(child.event_time_start, child_path, "event_time_start")
                if cs:
                    child_starts.append(cs)
                if child.event_time_end:
                    ce = _parse_event_time(child.event_time_end, child_path, "event_time_end")
                    if ce:
                        child_ends.append(ce)
                elif cs:
                    child_ends.append(cs)  # L0: event_time_end defaults to event_time_start

            if ets and child_starts and ets > min(child_starts):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{node_path}: event_time_start ({ets.isoformat()}) must be <= "
                        f"min child event_time_start ({min(child_starts).isoformat()})"
                    ),
                )
            if ete and child_ends and ete < max(child_ends):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{node_path}: event_time_end ({ete.isoformat()}) must be >= "
                        f"max child event_time_end ({max(child_ends).isoformat()})"
                    ),
                )

            # Recurse into children
            total_count += _validate_base_event_tree(event.children, f"{node_path}.children")
        else:
            # L0 node validations
            if event.children:
                raise HTTPException(
                    status_code=400,
                    detail=f"{node_path}: hierarchy_level 0 cannot have children",
                )

        total_count += 1

    return total_count


def _parse_event_time(
    value: str | None, node_path: str, field_name: str
) -> datetime.datetime | None:
    """Parse an ISO 8601 string to datetime. Returns None if value is None."""
    if value is None:
        return None
    try:
        dt = datetime.datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=get_timezone())
        return dt
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"{node_path}: invalid ISO 8601 format for {field_name}: '{value}'",
        ) from e


def _flatten_base_event_tree(
    events: list[BaseEventItem],
    agent_id: str,
    parent_id: str | None = None,
    parent_context_type: ContextType | None = None,
) -> list[ProcessedContext]:
    """Flatten a nested event tree into a list of ProcessedContext objects.

    Builds bidirectional refs in-memory:
    - Downward: parent.refs[child_type] = [child_ids]
    - Upward: child.refs[parent_type] = [parent_id]
    """
    result: list[ProcessedContext] = []

    for event in events:
        now = tz_now()
        level = event.hierarchy_level
        context_type = _BASE_HIERARCHY_LEVEL_TO_TYPE[level]

        # Parse times
        event_time_start = _parse_event_time(event.event_time_start, "", "event_time_start") or now
        event_time_end = (
            _parse_event_time(event.event_time_end, "", "event_time_end") or event_time_start
        )

        # Build text for embedding
        text_for_embedding = f"{event.title}\n{event.summary}\n{', '.join(event.keywords)}"
        vectorize = Vectorize(
            input=[{"type": "text", "text": text_for_embedding}],
            content_format=ContentFormat.TEXT,
        )

        # Build refs (upward ref to parent, if any)
        refs: dict[str, list[str]] = {}
        if parent_id and parent_context_type:
            refs[parent_context_type.value] = [parent_id]

        ctx = ProcessedContext(
            properties=ContextProperties(
                raw_properties=[],
                create_time=now,
                update_time=now,
                event_time_start=event_time_start,
                event_time_end=event_time_end,
                is_processed=True,
                user_id="__base__",
                agent_id=agent_id,
                hierarchy_level=level,
                refs=refs,
            ),
            extracted_data=ExtractedData(
                title=event.title,
                summary=event.summary,
                keywords=event.keywords,
                entities=event.entities,
                context_type=context_type,
                importance=event.importance,
                confidence=10,
            ),
            vectorize=vectorize,
        )
        result.append(ctx)

        # Recurse into children and build downward refs
        if event.children:
            child_contexts = _flatten_base_event_tree(
                event.children,
                agent_id,
                parent_id=ctx.id,
                parent_context_type=context_type,
            )

            # Build downward refs: only direct children (not grandchildren)
            child_type_value = _BASE_HIERARCHY_LEVEL_TO_TYPE[level - 1].value
            direct_child_ids = [
                c.id for c in child_contexts if c.properties.hierarchy_level == level - 1
            ]
            ctx.properties.refs[child_type_value] = direct_child_ids

            result.extend(child_contexts)

    return result


# ============================================================================
# Agent CRUD Endpoints
# ============================================================================


@router.post("")
async def create_agent(request: CreateAgentRequest, _auth: str = auth_dependency):
    """Register a new agent."""
    if request.agent_id == "__base__":
        raise HTTPException(
            status_code=400, detail="agent_id '__base__' is reserved for system use"
        )

    storage = get_storage()
    success = await storage.create_agent(request.agent_id, request.name, request.description)  # type: ignore[union-attr]
    if not success:
        raise HTTPException(status_code=400, detail="Agent creation failed (ID may already exist)")
    return convert_resp(data={"agent_id": request.agent_id}, message="Agent created")


@router.get("")
async def list_agents(_auth: str = auth_dependency):
    """List all active agents."""
    storage = get_storage()
    agents = await storage.list_agents()  # type: ignore[union-attr]
    return convert_resp(data={"agents": agents})


@router.get("/{agent_id}")
async def get_agent(agent_id: str, _auth: str = auth_dependency):
    """Get a single agent by ID."""
    storage = get_storage()
    agent = await storage.get_agent(agent_id)  # type: ignore[union-attr]
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return convert_resp(data={"agent": agent})


@router.put("/{agent_id}")
async def update_agent(agent_id: str, request: UpdateAgentRequest, _auth: str = auth_dependency):
    """Update agent name and/or description."""
    storage = get_storage()
    success = await storage.update_agent(  # type: ignore[union-attr]
        agent_id, name=request.name, description=request.description
    )
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found or update failed")
    return convert_resp(message="Agent updated")


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, _auth: str = auth_dependency):
    """Soft-delete an agent."""
    storage = get_storage()
    success = await storage.delete_agent(agent_id)  # type: ignore[union-attr]
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return convert_resp(message="Agent deleted")


# ============================================================================
# Agent Base Memory — Profile
# ============================================================================


@router.post("/{agent_id}/base/profile")
async def set_base_profile(
    agent_id: str, request: BaseProfileRequest, _auth: str = auth_dependency
):
    """Set or overwrite the agent's base profile.

    The base profile is stored with ``context_type="agent_base_profile"``
    to distinguish it from per-user profiles.
    """
    storage = get_storage()
    agent = await storage.get_agent(agent_id)  # type: ignore[union-attr]
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    success = await storage.upsert_profile(  # type: ignore[union-attr]
        user_id="__base__",
        device_id="default",
        agent_id=agent_id,
        context_type="agent_base_profile",
        factual_profile=request.factual_profile,
        behavioral_profile=request.behavioral_profile,
        entities=request.entities,
        importance=request.importance,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save base profile")
    return convert_resp(message="Base profile saved")


@router.get("/{agent_id}/base/profile")
async def get_base_profile(agent_id: str, _auth: str = auth_dependency):
    """Retrieve the agent's base profile."""
    storage = get_storage()
    profile = await storage.get_profile(  # type: ignore[union-attr]
        user_id="__base__",
        device_id="default",
        agent_id=agent_id,
        context_type="agent_base_profile",
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Base profile not found")
    return convert_resp(data={"profile": profile})


# ============================================================================
# Agent Base Memory — Events
# ============================================================================


@router.post("/{agent_id}/base/events")
async def push_base_events(agent_id: str, request: BaseEventsRequest, _auth: str = auth_dependency):
    """Push structured base events for an agent (no LLM extraction).

    Supports flat L0 events and nested hierarchy trees (L0-L3).
    Generates embeddings from ``title + summary + keywords``, builds
    bidirectional refs, and batch-writes to vector DB.
    """
    storage = get_storage()
    agent = await storage.get_agent(agent_id)  # type: ignore[union-attr]
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Validate tree structure, time ranges, hierarchy consistency
    total_count = _validate_base_event_tree(request.events)
    if total_count > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Total event count {total_count} exceeds maximum of 500",
        )

    # Flatten tree into ProcessedContext list with bidirectional refs
    contexts = _flatten_base_event_tree(request.events, agent_id)

    # Batch write (vectorization happens inside batch_upsert via do_vectorize_batch)
    result = await storage.batch_upsert_processed_context(contexts)  # type: ignore[union-attr]
    success = result is not None
    return convert_resp(
        data={"count": len(contexts) if success else 0},
        message="Base events saved" if success else "Failed to save base events",
    )


@router.get("/{agent_id}/base/events")
async def list_base_events(
    agent_id: str,
    limit: int = 50,
    offset: int = 0,
    hierarchy_level: int | None = None,
    _auth: str = auth_dependency,
):
    """List base events for an agent.

    Returns all hierarchy levels by default. Use ``hierarchy_level`` query param
    to filter (0=events, 1/2/3=summaries).
    """
    storage = get_storage()

    # Determine which context types to query
    if hierarchy_level is not None:
        if hierarchy_level not in _BASE_HIERARCHY_LEVEL_TO_TYPE:
            raise HTTPException(
                status_code=400,
                detail=f"hierarchy_level must be 0-3, got {hierarchy_level}",
            )
        query_types = [_BASE_HIERARCHY_LEVEL_TO_TYPE[hierarchy_level].value]
    else:
        query_types = _ALL_AGENT_BASE_TYPES

    result = await storage.get_all_processed_contexts(  # type: ignore[union-attr]
        context_types=query_types,
        user_id="__base__",
        agent_id=agent_id,
        limit=limit + offset,
    )

    # Merge results from all queried types
    all_contexts = []
    for ct_value in query_types:
        all_contexts.extend(result.get(ct_value, []))

    # Sort by event_time_start descending (most recent first)
    all_contexts.sort(key=lambda c: c.properties.event_time_start, reverse=True)
    page = all_contexts[offset : offset + limit]

    return convert_resp(
        data={
            "events": [
                ProcessedContextModel.from_processed_context(c, Path(".")).model_dump()
                for c in page
            ]
        }
    )


@router.delete("/{agent_id}/base/events/{event_id}")
async def delete_base_event(agent_id: str, event_id: str, _auth: str = auth_dependency):
    """Delete a single base event or summary by ID."""
    storage = get_storage()
    # Try deleting from each AGENT_BASE_* type (we don't know which type it is)
    for ct_value in _ALL_AGENT_BASE_TYPES:
        success = await storage.delete_processed_context(event_id, ct_value)  # type: ignore[union-attr]
        if success:
            return convert_resp(message="Event deleted")
    raise HTTPException(status_code=404, detail="Event not found")
