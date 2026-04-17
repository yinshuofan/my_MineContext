#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Agent base-event routes — replace-tree semantics.

POST replaces the agent's entire base-event tree (diff existing vs new, delete missing, upsert new).
DELETE prunes a subtree and invokes the same replace logic. Both operations serialize per-agent
through a Redis lock.
"""

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
router = APIRouter(prefix="/api/agents", tags=["agent-base-events"])


# ============================================================================
# Constants
# ============================================================================

_BASE_HIERARCHY_LEVEL_TO_TYPE = {
    0: ContextType.AGENT_BASE_EVENT,
    1: ContextType.AGENT_BASE_L1_SUMMARY,
    2: ContextType.AGENT_BASE_L2_SUMMARY,
    3: ContextType.AGENT_BASE_L3_SUMMARY,
}

_ALL_AGENT_BASE_TYPES = [ct.value for ct in _BASE_HIERARCHY_LEVEL_TO_TYPE.values()]

_MAX_TOTAL_EVENTS = 500
_LOCK_TIMEOUT_SECONDS = 60
_LOCK_BLOCKING_TIMEOUT_SECONDS = 10


# ============================================================================
# Request Models
# ============================================================================


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


# ============================================================================
# Validation / parsing helpers (moved from agents.py)
# ============================================================================


def _parse_event_time(
    value: str | None, node_path: str, field_name: str
) -> datetime.datetime | None:
    """Parse an ISO 8601 string into tz-aware datetime. Returns None if value is None."""
    if value is None:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=get_timezone())
        return parsed
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"{node_path}: invalid ISO 8601 format for {field_name}: '{value}'",
        ) from e


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

        if level not in _BASE_HIERARCHY_LEVEL_TO_TYPE:
            raise HTTPException(
                status_code=400,
                detail=f"{node_path}: hierarchy_level must be 0/1/2/3, got {level}",
            )

        if level > 0 and not event.event_time_end:
            raise HTTPException(
                status_code=400,
                detail=f"{node_path}: event_time_end is required when hierarchy_level > 0",
            )

        if event.children:
            for j, child in enumerate(event.children):
                child_path = f"{node_path}.children[{j}]"
                if child.hierarchy_level >= level:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"{child_path}: child hierarchy_level ({child.hierarchy_level}) "
                            f"must be less than parent ({level})"
                        ),
                    )
            total_count += _validate_base_event_tree(event.children, path=f"{node_path}.children")

        total_count += 1

    return total_count


def _flatten_base_event_tree(
    events: list[BaseEventItem],
    agent_id: str,
    parent_id: str | None = None,
    parent_context_type: ContextType | None = None,
) -> list[ProcessedContext]:
    """Flatten a nested event tree into ProcessedContext list with bidirectional refs.

    - Downward: parent.refs[child_type] = [direct_child_ids]
    - Upward:   child.refs[parent_type] = [parent_id]
    """
    result: list[ProcessedContext] = []

    for event in events:
        now = tz_now()
        level = event.hierarchy_level
        context_type = _BASE_HIERARCHY_LEVEL_TO_TYPE[level]

        event_time_start = _parse_event_time(event.event_time_start, "", "event_time_start") or now
        event_time_end = (
            _parse_event_time(event.event_time_end, "", "event_time_end") or event_time_start
        )

        text_for_embedding = f"{event.title}\n{event.summary}\n{', '.join(event.keywords)}"
        vectorize = Vectorize(
            input=[{"type": "text", "text": text_for_embedding}],
            content_format=ContentFormat.TEXT,
        )

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

        if event.children:
            child_contexts = _flatten_base_event_tree(
                event.children,
                agent_id,
                parent_id=ctx.id,
                parent_context_type=context_type,
            )

            child_type_value = _BASE_HIERARCHY_LEVEL_TO_TYPE[level - 1].value
            direct_child_ids = [
                c.id for c in child_contexts if c.properties.hierarchy_level == level - 1
            ]
            ctx.properties.refs[child_type_value] = direct_child_ids

            result.extend(child_contexts)

    return result


# ============================================================================
# Endpoints (migrated as-is; rewritten in later tasks)
# ============================================================================


@router.post("/{agent_id}/base/events")
async def push_base_events(agent_id: str, request: BaseEventsRequest, _auth: str = auth_dependency):
    """Push structured base events for an agent (no LLM extraction)."""
    storage = get_storage()
    agent = await storage.get_agent(agent_id)  # type: ignore[union-attr]
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    total_count = _validate_base_event_tree(request.events)
    if total_count > _MAX_TOTAL_EVENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Total event count {total_count} exceeds maximum of {_MAX_TOTAL_EVENTS}",
        )

    contexts = _flatten_base_event_tree(request.events, agent_id)

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
    """List base events for an agent."""
    storage = get_storage()

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

    all_contexts = []
    for ct_value in query_types:
        all_contexts.extend(result.get(ct_value, []))

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
    for ct_value in _ALL_AGENT_BASE_TYPES:
        success = await storage.delete_processed_context(event_id, ct_value)  # type: ignore[union-attr]
        if success:
            return convert_resp(message="Event deleted")
    raise HTTPException(status_code=404, detail="Event not found")
