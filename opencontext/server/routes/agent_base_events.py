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
from opencontext.storage.redis_cache import get_cache
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
# Tree-reconstruction helpers (used by DELETE)
# ============================================================================


def _collect_subtree_ids(
    ctx_by_id: dict[str, ProcessedContext],
    root_id: str,
) -> set[str]:
    """BFS over downward refs (refs pointing to lower hierarchy_level).

    Returns {root_id} plus all descendant ids reachable via downward refs.
    Safe against cycles via visited set.
    """
    if root_id not in ctx_by_id:
        return set()

    visited: set[str] = {root_id}
    queue: list[str] = [root_id]

    while queue:
        current_id = queue.pop(0)
        current = ctx_by_id.get(current_id)
        if current is None or not current.properties.refs:
            continue
        current_level = current.properties.hierarchy_level
        for _ref_type, ref_ids in current.properties.refs.items():
            for rid in ref_ids:
                if rid in visited:
                    continue
                child = ctx_by_id.get(rid)
                if child is None:
                    continue
                # Downward-only: child's level must be strictly less than current's
                if child.properties.hierarchy_level >= current_level:
                    continue
                visited.add(rid)
                queue.append(rid)

    return visited


def _find_parent_id(
    ctx_by_id: dict[str, ProcessedContext],
    event_id: str,
) -> str | None:
    """Return the parent id for event_id, or None if event is a root / unknown.

    A ref points upward when the referenced context exists in ctx_by_id and has
    a strictly higher hierarchy_level than event_id.
    """
    event = ctx_by_id.get(event_id)
    if event is None or not event.properties.refs:
        return None
    event_level = event.properties.hierarchy_level
    for _ref_type, ref_ids in event.properties.refs.items():
        for rid in ref_ids:
            parent = ctx_by_id.get(rid)
            if parent is None:
                continue
            if parent.properties.hierarchy_level > event_level:
                return rid
    return None


def _scrub_parent_refs(
    parent_ctx: ProcessedContext,
    child_id: str,
    child_context_type: ContextType,
) -> None:
    """Remove child_id from parent_ctx.properties.refs[child_context_type.value] in place.

    No-op if the ref key is missing or the id isn't present.
    """
    key = child_context_type.value
    ids = parent_ctx.properties.refs.get(key)
    if not ids:
        return
    parent_ctx.properties.refs[key] = [rid for rid in ids if rid != child_id]


async def _fetch_existing_base_events(
    storage,
    agent_id: str,
) -> list[ProcessedContext]:
    """Return all AGENT_BASE_* contexts for the given agent_id under user_id='__base__'."""
    result = await storage.get_all_processed_contexts(
        context_types=_ALL_AGENT_BASE_TYPES,
        user_id="__base__",
        agent_id=agent_id,
        limit=_MAX_TOTAL_EVENTS,
    )
    all_contexts: list[ProcessedContext] = []
    for ct_value in _ALL_AGENT_BASE_TYPES:
        all_contexts.extend(result.get(ct_value, []))
    return all_contexts


async def _replace_base_events_impl(
    storage,
    agent_id: str,
    new_contexts: list[ProcessedContext],
) -> dict:
    """Replace all AGENT_BASE_* contexts for an agent with new_contexts.

    Order: upsert first (fail-safe: old tree intact on upsert failure),
    then delete the ids that no longer exist in new_contexts.

    A delete failure is logged but does not fail the request — stragglers
    will be cleaned up on the next replace.

    Returns:
        {
            "upserted": N,
            "deleted": M,                      # count of ids confirmed deleted
            "stragglers": K,                   # count that failed to delete
            "deleted_ids": [...],              # ids confirmed deleted (len == M)
            "straggler_ids": [...],            # ids that failed to delete (len == K)
        }

    Raises RuntimeError on upsert failure.
    """
    existing = await _fetch_existing_base_events(storage, agent_id)
    existing_ids = {c.id for c in existing}
    new_ids = {c.id for c in new_contexts}
    to_delete = list(existing_ids - new_ids)

    # Upsert FIRST — if this fails, old tree is intact.
    # Skip when empty: some backends reject empty upsert payloads, and there's
    # nothing to write anyway (used by DELETE when the entire tree is pruned).
    if new_contexts:
        upsert_result = await storage.batch_upsert_processed_context(new_contexts)
        if upsert_result is None:
            raise RuntimeError(f"Failed to upsert base events for agent={agent_id}")

    # Delete SECOND — dispatch via delete_batch_by_type so the backend owns
    # physical routing (VikingDB coalesces into 1 HTTP call; Qdrant runs
    # per-collection deletes in parallel). Per-type success map preserves
    # stragglers granularity. Upserts already succeeded; stragglers will
    # be cleaned up on the next replace.
    deleted_ids: list[str] = []
    straggler_ids: list[str] = []
    if to_delete:
        ct_by_id: dict[str, str] = {
            c.id: c.extracted_data.context_type.value
            for c in existing
            if c.extracted_data and c.extracted_data.context_type
        }
        ids_by_type: dict[str, list[str]] = {}
        for del_id in to_delete:
            ct_value = ct_by_id.get(del_id)
            if ct_value is None:
                # Shouldn't happen — existing contexts always have a type — but
                # if it does, we can't route the delete, so count as straggler.
                straggler_ids.append(del_id)
                logger.warning(
                    f"delete straggler (no context_type in existing map) "
                    f"id={del_id} agent={agent_id}"
                )
                continue
            ids_by_type.setdefault(ct_value, []).append(del_id)

        if ids_by_type:
            try:
                per_type_ok = await storage.delete_batch_by_type(ids_by_type)
            except Exception as e:
                per_type_ok = {ct: False for ct in ids_by_type}
                logger.warning(f"delete_batch_by_type raised for agent={agent_id}: {e}")
            for ct_value, ids in ids_by_type.items():
                if per_type_ok.get(ct_value, False):
                    deleted_ids.extend(ids)
                else:
                    straggler_ids.extend(ids)
                    logger.warning(
                        f"delete failed for type={ct_value} agent={agent_id} "
                        f"({len(ids)} stragglers)"
                    )

    return {
        "upserted": len(new_contexts),
        "deleted": len(deleted_ids),
        "stragglers": len(straggler_ids),
        "deleted_ids": deleted_ids,
        "straggler_ids": straggler_ids,
    }


# ============================================================================
# Endpoints (migrated as-is; rewritten in later tasks)
# ============================================================================


@router.post("/{agent_id}/base/events")
async def push_base_events(agent_id: str, request: BaseEventsRequest, _auth: str = auth_dependency):
    """Push the agent's base-event tree with REPLACE semantics.

    Diffs the payload against existing AGENT_BASE_* contexts for this agent;
    upserts all new ids, then deletes any existing ids not present in the payload.
    Serializes per-agent via Redis lock.
    """
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

    new_contexts = _flatten_base_event_tree(request.events, agent_id)

    cache = await get_cache()
    lock_key = f"agent_base_edit:{agent_id}"
    lock_token = await cache.acquire_lock(
        lock_key,
        timeout=_LOCK_TIMEOUT_SECONDS,
        blocking=True,
        blocking_timeout=_LOCK_BLOCKING_TIMEOUT_SECONDS,
    )
    if not lock_token:
        raise HTTPException(
            status_code=503,
            detail=f"Another edit is in progress for agent {agent_id}",
        )

    try:
        result = await _replace_base_events_impl(storage, agent_id, new_contexts)
    finally:
        await cache.release_lock(lock_key, lock_token)

    return convert_resp(
        data={
            "upserted": result["upserted"],
            "deleted": result["deleted"],
            "stragglers": result["stragglers"],
            "deleted_ids": result["deleted_ids"],
            "straggler_ids": result["straggler_ids"],
            "ids": [c.id for c in new_contexts],
        },
        message="Base events replaced",
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
    """Delete a base event and its entire subtree via replace semantics.

    Fetches the current tree, computes the subtree rooted at event_id,
    scrubs the parent's downward refs (if any parent exists), and invokes
    the same replace core as POST. The empty-parent summary is preserved
    (not cascade-deleted) — users decide whether to keep or remove
    summaries in their next POST.
    """
    storage = get_storage()
    cache = await get_cache()
    lock_key = f"agent_base_edit:{agent_id}"
    lock_token = await cache.acquire_lock(
        lock_key,
        timeout=_LOCK_TIMEOUT_SECONDS,
        blocking=True,
        blocking_timeout=_LOCK_BLOCKING_TIMEOUT_SECONDS,
    )
    if not lock_token:
        raise HTTPException(
            status_code=503,
            detail=f"Another edit is in progress for agent {agent_id}",
        )

    try:
        existing = await _fetch_existing_base_events(storage, agent_id)
        ctx_by_id = {c.id: c for c in existing}

        if event_id not in ctx_by_id:
            raise HTTPException(status_code=404, detail="Event not found")

        subtree_ids = _collect_subtree_ids(ctx_by_id, event_id)
        parent_id = _find_parent_id(ctx_by_id, event_id)

        kept = [c for c in existing if c.id not in subtree_ids]

        # Scrub parent's downward ref to the deleted root (in the kept list).
        if parent_id and parent_id in ctx_by_id:
            target_ctx = ctx_by_id[event_id]
            target_type = target_ctx.extracted_data.context_type
            parent_ctx = next((c for c in kept if c.id == parent_id), None)
            if parent_ctx is not None:
                _scrub_parent_refs(parent_ctx, event_id, target_type)

        result = await _replace_base_events_impl(storage, agent_id, kept)
    finally:
        await cache.release_lock(lock_key, lock_token)

    return convert_resp(
        data={
            "deleted_ids": result["deleted_ids"],
            "straggler_ids": result["straggler_ids"],
            "updated_parent_id": parent_id,
            "stragglers": result["stragglers"],
        },
        message="Event deleted",
    )


@router.delete("/{agent_id}/base/events")
async def delete_all_base_events(agent_id: str, _auth: str = auth_dependency):
    """Clear all base events for an agent via replace-with-empty semantics.

    Fetches existing AGENT_BASE_* contexts to report deleted ids, then invokes
    the same replace core as POST/DELETE-by-id with an empty list. Serializes
    per-agent via Redis lock. No-op (returns empty result) if the agent has no
    existing base events.
    """
    storage = get_storage()
    agent = await storage.get_agent(agent_id)  # type: ignore[union-attr]
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    cache = await get_cache()
    lock_key = f"agent_base_edit:{agent_id}"
    lock_token = await cache.acquire_lock(
        lock_key,
        timeout=_LOCK_TIMEOUT_SECONDS,
        blocking=True,
        blocking_timeout=_LOCK_BLOCKING_TIMEOUT_SECONDS,
    )
    if not lock_token:
        raise HTTPException(
            status_code=503,
            detail=f"Another edit is in progress for agent {agent_id}",
        )

    try:
        result = await _replace_base_events_impl(storage, agent_id, [])
    finally:
        await cache.release_lock(lock_key, lock_token)

    return convert_resp(
        data={
            "deleted_ids": result["deleted_ids"],
            "straggler_ids": result["straggler_ids"],
            "deleted": result["deleted"],
            "stragglers": result["stragglers"],
        },
        message="All base events deleted",
    )
