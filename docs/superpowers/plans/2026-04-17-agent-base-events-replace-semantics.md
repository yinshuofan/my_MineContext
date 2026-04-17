# Agent Base Events Replace-Tree Semantics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-node base-event DELETE + client-side cascade with server-side tree-replace semantics, eliminating dangling refs structurally and making edits atomic.

**Architecture:** `POST /api/agents/{id}/base/events` becomes replace-tree (fetch existing, diff, delete missing, upsert new). `DELETE /api/agents/{id}/base/events/{event_id}` becomes a server-side wrapper that fetches the tree, prunes the subtree rooted at `event_id`, scrubs the parent's refs, and performs the same upsert+delete under a Redis lock. Base-event code moves out of `routes/agents.py` into new `routes/agent_base_events.py`.

**Tech Stack:** Python 3.10+ / FastAPI / VikingDB (via `UnifiedStorage`) / Redis lock (via `RedisCache.acquire_lock`) / pytest (`@pytest.mark.unit`).

**Spec:** `docs/superpowers/specs/2026-04-17-agent-base-events-replace-semantics.md`

---

## File Structure

### New files

- `opencontext/server/routes/agent_base_events.py` — new sub-router holding all base-event constants, pydantic models, helpers, and three endpoints. Receives moved-in code from `routes/agents.py` + new replace/prune logic.
- `tests/server/routes/test_agent_base_events.py` — unit tests for new helpers, replace logic, POST/DELETE endpoints.

### Modified files

- `opencontext/server/routes/agents.py` — remove `_BASE_HIERARCHY_LEVEL_TO_TYPE`, `_ALL_AGENT_BASE_TYPES`, `BaseEventItem`, `BaseEventsRequest`, `_validate_base_event_tree`, `_parse_event_time`, `_flatten_base_event_tree`, and three base-event handlers (`push_base_events`, `list_base_events`, `delete_base_event`). Shrinks from 498 to ~230 lines. Keeps only agent CRUD + base profile.
- `opencontext/server/api.py` — register new `agent_base_events.router` alongside `agents.router` at line 49.
- `opencontext/web/templates/agents.html` — simplify `deleteEvent` (lines 449–488); remove `_collectDescendantIds` (lines 438–447).
- `docs/api_reference.md` — document POST replace semantics.
- `docs/curls.sh` — update example request if needed.

---

## Task 1: Create new route file, migrate base-event code as-is

**Goal:** Zero-behavior-change move. Extract all base-event code into a new file and register it. Verify existing behavior preserved.

**Files:**
- Create: `opencontext/server/routes/agent_base_events.py`
- Modify: `opencontext/server/routes/agents.py` (remove migrated code, lines 57–65, 68–203, 206–287, 400–498)
- Modify: `opencontext/server/api.py` (register new router around line 49)

- [ ] **Step 1: Create the new route file with migrated code**

Create `opencontext/server/routes/agent_base_events.py` with the following content (this is the complete file; do not abbreviate):

```python
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


def _parse_event_time(value: str | None, node_path: str, field_name: str) -> datetime.datetime | None:
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
```

- [ ] **Step 2: Remove migrated code from `opencontext/server/routes/agents.py`**

In `opencontext/server/routes/agents.py`, **delete** the following blocks (the line ranges are from the current state):

1. Lines 56–65 — constants `_BASE_HIERARCHY_LEVEL_TO_TYPE` and `_ALL_AGENT_BASE_TYPES`
2. Lines 68–81 — `BaseEventItem` class and `BaseEventsRequest` class
3. Lines 84–203 — `_validate_base_event_tree` and `_parse_event_time` functions
4. Lines 206–287 — `_flatten_base_event_tree` function
5. Lines 399–498 — the `# ============ Agent Base Memory — Events ============` section header plus three handlers (`push_base_events`, `list_base_events`, `delete_base_event`)

Also remove unused imports: `ProcessedContextModel` and `Vectorize` (if not used elsewhere in the file); `get_timezone`; `Path`. Keep `ContextProperties`, `ExtractedData`, `ProcessedContext`, `ContentFormat`, `ContextType` if still used.

Verify with: `uv run python -c "from opencontext.server.routes.agents import router; print('ok')"`.

- [ ] **Step 3: Register new router in `opencontext/server/api.py`**

In `opencontext/server/api.py` around line 49 (where `agents.router` is included), add an import and registration:

```python
# Near the top, with the other route imports (around lines 15-33):
from opencontext.server.routes import agent_base_events

# Near line 49, with other router.include_router calls:
router.include_router(agent_base_events.router)
```

- [ ] **Step 4: Smoke-test — import and route registration**

Run: `uv run python -c "from opencontext.server.api import router; paths = sorted({r.path for r in router.routes}); [print(p) for p in paths if 'base/events' in p]"`

Expected output: three paths ending in `/base/events` and `/base/events/{event_id}`:
```
/api/agents/{agent_id}/base/events
/api/agents/{agent_id}/base/events/{event_id}
```

- [ ] **Step 5: Commit**

```bash
git add opencontext/server/routes/agent_base_events.py opencontext/server/routes/agents.py opencontext/server/api.py
git commit -m "refactor(agents): extract base-event routes to own module

Move _BASE_HIERARCHY_LEVEL_TO_TYPE, _ALL_AGENT_BASE_TYPES, BaseEventItem,
BaseEventsRequest, _validate_base_event_tree, _parse_event_time,
_flatten_base_event_tree, push_base_events, list_base_events, and
delete_base_event from routes/agents.py to routes/agent_base_events.py.
No behavior change. agents.py retains only agent CRUD + base profile."
```

---

## Task 2: Add tree-reconstruction helpers (pure functions, TDD)

**Goal:** Add three pure helpers needed by DELETE to reason about the current tree state from a flat context list. All take no storage — pure Python.

**Files:**
- Modify: `opencontext/server/routes/agent_base_events.py` (add helpers near bottom)
- Test: `tests/server/routes/test_agent_base_events.py` (new file)

- [ ] **Step 1: Write failing test for `_collect_subtree_ids`**

Create `tests/server/routes/test_agent_base_events.py`:

```python
# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent base-event routes."""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import pytest

from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    Vectorize,
)
from opencontext.models.enums import ContentFormat, ContextType

TZ = ZoneInfo("Asia/Shanghai")


def _make_base_ctx(
    ctx_id: str,
    context_type: ContextType,
    hierarchy_level: int,
    refs: dict[str, list[str]] | None = None,
    agent_id: str = "agent-x",
) -> ProcessedContext:
    """Build a minimal AGENT_BASE_* ProcessedContext."""
    ct = datetime.datetime(2026, 4, 10, 12, 0, 0, tzinfo=TZ)
    return ProcessedContext(
        id=ctx_id,
        properties=ContextProperties(
            raw_properties=[],
            create_time=ct,
            update_time=ct,
            event_time_start=ct,
            event_time_end=ct,
            is_processed=True,
            user_id="__base__",
            agent_id=agent_id,
            hierarchy_level=hierarchy_level,
            refs=refs or {},
        ),
        extracted_data=ExtractedData(
            title=f"t-{ctx_id}",
            summary=f"s-{ctx_id}",
            context_type=context_type,
            confidence=10,
            importance=5,
        ),
        vectorize=Vectorize(vector=[0.1], content_format=ContentFormat.TEXT),
    )


@pytest.mark.unit
class TestCollectSubtreeIds:
    """Tests for _collect_subtree_ids."""

    def test_single_leaf_returns_only_root(self):
        from opencontext.server.routes.agent_base_events import _collect_subtree_ids

        leaf = _make_base_ctx("leaf-1", ContextType.AGENT_BASE_EVENT, 0)
        ctx_by_id = {"leaf-1": leaf}

        result = _collect_subtree_ids(ctx_by_id, "leaf-1")

        assert result == {"leaf-1"}

    def test_collects_all_descendants_of_l1(self):
        from opencontext.server.routes.agent_base_events import _collect_subtree_ids

        # L1 summary with two L0 children
        l1 = _make_base_ctx(
            "l1-1",
            ContextType.AGENT_BASE_L1_SUMMARY,
            1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["l0-a", "l0-b"]},
        )
        l0_a = _make_base_ctx(
            "l0-a",
            ContextType.AGENT_BASE_EVENT,
            0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-1"]},
        )
        l0_b = _make_base_ctx(
            "l0-b",
            ContextType.AGENT_BASE_EVENT,
            0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-1"]},
        )
        ctx_by_id = {"l1-1": l1, "l0-a": l0_a, "l0-b": l0_b}

        result = _collect_subtree_ids(ctx_by_id, "l1-1")

        assert result == {"l1-1", "l0-a", "l0-b"}

    def test_ignores_upward_refs(self):
        from opencontext.server.routes.agent_base_events import _collect_subtree_ids

        # L0 has only an upward ref — BFS should not follow upward
        l0 = _make_base_ctx(
            "l0-x",
            ContextType.AGENT_BASE_EVENT,
            0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-parent"]},
        )
        l1 = _make_base_ctx("l1-parent", ContextType.AGENT_BASE_L1_SUMMARY, 1)
        ctx_by_id = {"l0-x": l0, "l1-parent": l1}

        result = _collect_subtree_ids(ctx_by_id, "l0-x")

        assert result == {"l0-x"}

    def test_cycle_terminates(self):
        from opencontext.server.routes.agent_base_events import _collect_subtree_ids

        # Synthetic cycle — guard against pathological data
        a = _make_base_ctx(
            "a", ContextType.AGENT_BASE_L1_SUMMARY, 1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["b"]},
        )
        b = _make_base_ctx(
            "b", ContextType.AGENT_BASE_EVENT, 0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["a"]},
        )
        # Force a downward-looking cycle by also adding b's refs to a:
        b.properties.refs[ContextType.AGENT_BASE_L1_SUMMARY.value] = ["a"]
        # Pretend a also has a downward ref to b (already set); b now has a
        # matching-level ref back to a — the function should not loop.
        ctx_by_id = {"a": a, "b": b}

        result = _collect_subtree_ids(ctx_by_id, "a")

        assert "a" in result and "b" in result
        assert len(result) == 2  # terminated, no duplicates
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py::TestCollectSubtreeIds -v`

Expected: FAIL with `ImportError: cannot import name '_collect_subtree_ids'`.

- [ ] **Step 3: Implement `_collect_subtree_ids`**

In `opencontext/server/routes/agent_base_events.py`, add at the end of the file (after `_flatten_base_event_tree`, before the endpoint definitions):

```python
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
        for ref_type, ref_ids in current.properties.refs.items():
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py::TestCollectSubtreeIds -v`

Expected: all four tests PASS.

- [ ] **Step 5: Write failing tests for `_find_parent_id` and `_scrub_parent_refs`**

Append to `tests/server/routes/test_agent_base_events.py`:

```python
@pytest.mark.unit
class TestFindParentId:
    """Tests for _find_parent_id."""

    def test_returns_parent_for_l0_with_upward_ref(self):
        from opencontext.server.routes.agent_base_events import _find_parent_id

        l0 = _make_base_ctx(
            "l0-1",
            ContextType.AGENT_BASE_EVENT,
            0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-parent"]},
        )
        l1 = _make_base_ctx("l1-parent", ContextType.AGENT_BASE_L1_SUMMARY, 1)
        ctx_by_id = {"l0-1": l0, "l1-parent": l1}

        assert _find_parent_id(ctx_by_id, "l0-1") == "l1-parent"

    def test_returns_none_for_root(self):
        from opencontext.server.routes.agent_base_events import _find_parent_id

        # L3 at the top of the tree has no parent
        l3 = _make_base_ctx("l3-1", ContextType.AGENT_BASE_L3_SUMMARY, 3)
        ctx_by_id = {"l3-1": l3}

        assert _find_parent_id(ctx_by_id, "l3-1") is None

    def test_returns_none_for_unknown_id(self):
        from opencontext.server.routes.agent_base_events import _find_parent_id

        assert _find_parent_id({}, "does-not-exist") is None

    def test_ignores_downward_refs(self):
        from opencontext.server.routes.agent_base_events import _find_parent_id

        # L1 that has only downward refs (to its L0 children) but no upward ref
        l1 = _make_base_ctx(
            "l1-top",
            ContextType.AGENT_BASE_L1_SUMMARY,
            1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["l0-a"]},
        )
        l0 = _make_base_ctx("l0-a", ContextType.AGENT_BASE_EVENT, 0)
        ctx_by_id = {"l1-top": l1, "l0-a": l0}

        assert _find_parent_id(ctx_by_id, "l1-top") is None


@pytest.mark.unit
class TestScrubParentRefs:
    """Tests for _scrub_parent_refs."""

    def test_removes_child_id_from_parent_refs(self):
        from opencontext.server.routes.agent_base_events import _scrub_parent_refs

        parent = _make_base_ctx(
            "p",
            ContextType.AGENT_BASE_L1_SUMMARY,
            1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["c1", "c2", "c3"]},
        )

        _scrub_parent_refs(parent, "c2", ContextType.AGENT_BASE_EVENT)

        assert parent.properties.refs[ContextType.AGENT_BASE_EVENT.value] == ["c1", "c3"]

    def test_missing_id_is_no_op(self):
        from opencontext.server.routes.agent_base_events import _scrub_parent_refs

        parent = _make_base_ctx(
            "p",
            ContextType.AGENT_BASE_L1_SUMMARY,
            1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["c1"]},
        )

        _scrub_parent_refs(parent, "c999", ContextType.AGENT_BASE_EVENT)

        assert parent.properties.refs[ContextType.AGENT_BASE_EVENT.value] == ["c1"]

    def test_missing_key_is_no_op(self):
        from opencontext.server.routes.agent_base_events import _scrub_parent_refs

        parent = _make_base_ctx("p", ContextType.AGENT_BASE_L1_SUMMARY, 1, refs={})

        _scrub_parent_refs(parent, "c1", ContextType.AGENT_BASE_EVENT)

        assert parent.properties.refs == {}
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py::TestFindParentId tests/server/routes/test_agent_base_events.py::TestScrubParentRefs -v`

Expected: FAIL with ImportError for the two new helpers.

- [ ] **Step 7: Implement `_find_parent_id` and `_scrub_parent_refs`**

In `opencontext/server/routes/agent_base_events.py`, append to the helpers section added in Step 3:

```python
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
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py -v`

Expected: all tests in `TestCollectSubtreeIds`, `TestFindParentId`, `TestScrubParentRefs` PASS.

- [ ] **Step 9: Commit**

```bash
git add opencontext/server/routes/agent_base_events.py tests/server/routes/test_agent_base_events.py
git commit -m "feat(agent-base-events): add pure tree-reconstruction helpers

Add _collect_subtree_ids (BFS over downward refs with cycle guard),
_find_parent_id (upward ref lookup via hierarchy_level), and
_scrub_parent_refs (in-place list removal). Unit tests cover leaf,
multi-level descent, upward-ref non-traversal, cycle termination,
missing ids, and no-op edge cases."
```

---

## Task 3: Add `_replace_base_events_impl` (shared core, TDD)

**Goal:** Core replace logic — fetch existing, diff, upsert new, delete missing. Takes a preflattened `new_contexts` list so it can be reused by both POST and DELETE.

**Files:**
- Modify: `opencontext/server/routes/agent_base_events.py`
- Modify: `tests/server/routes/test_agent_base_events.py`

- [ ] **Step 1: Write failing tests for `_replace_base_events_impl`**

Append to `tests/server/routes/test_agent_base_events.py`:

```python
@pytest.mark.unit
class TestReplaceBaseEventsImpl:
    """Tests for _replace_base_events_impl (shared replace logic)."""

    @pytest.fixture
    def mock_storage(self):
        from unittest.mock import AsyncMock, MagicMock

        storage = MagicMock()
        storage.get_all_processed_contexts = AsyncMock(return_value={})
        storage.batch_upsert_processed_context = AsyncMock(return_value=["id-1"])
        storage.delete_batch_processed_contexts = AsyncMock(return_value=True)
        return storage

    async def test_fresh_agent_upserts_all_no_delete(self, mock_storage):
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        new_contexts = [_make_base_ctx("new-1", ContextType.AGENT_BASE_EVENT, 0)]
        mock_storage.batch_upsert_processed_context.return_value = ["new-1"]

        result = await _replace_base_events_impl(mock_storage, "agent-x", new_contexts)

        assert result["upserted"] == 1
        assert result["deleted"] == 0
        assert result["stragglers"] == 0
        mock_storage.batch_upsert_processed_context.assert_awaited_once_with(new_contexts)
        mock_storage.delete_batch_processed_contexts.assert_not_awaited()

    async def test_diff_deletes_removed_ids(self, mock_storage):
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        # Existing: a, b, c. New: a, d. Should delete: b, c.
        existing = {
            ContextType.AGENT_BASE_EVENT.value: [
                _make_base_ctx("a", ContextType.AGENT_BASE_EVENT, 0),
                _make_base_ctx("b", ContextType.AGENT_BASE_EVENT, 0),
                _make_base_ctx("c", ContextType.AGENT_BASE_EVENT, 0),
            ],
        }
        mock_storage.get_all_processed_contexts.return_value = existing
        mock_storage.batch_upsert_processed_context.return_value = ["a", "d"]

        new_contexts = [
            _make_base_ctx("a", ContextType.AGENT_BASE_EVENT, 0),
            _make_base_ctx("d", ContextType.AGENT_BASE_EVENT, 0),
        ]

        result = await _replace_base_events_impl(mock_storage, "agent-x", new_contexts)

        assert result["upserted"] == 2
        assert result["deleted"] == 2
        deleted_ids_arg = mock_storage.delete_batch_processed_contexts.call_args[0][0]
        assert set(deleted_ids_arg) == {"b", "c"}

    async def test_empty_new_deletes_all_existing(self, mock_storage):
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        existing = {
            ContextType.AGENT_BASE_EVENT.value: [
                _make_base_ctx("a", ContextType.AGENT_BASE_EVENT, 0),
            ],
        }
        mock_storage.get_all_processed_contexts.return_value = existing
        mock_storage.batch_upsert_processed_context.return_value = []

        result = await _replace_base_events_impl(mock_storage, "agent-x", [])

        assert result["upserted"] == 0
        assert result["deleted"] == 1
        deleted_ids_arg = mock_storage.delete_batch_processed_contexts.call_args[0][0]
        assert deleted_ids_arg == ["a"]

    async def test_upsert_failure_propagates(self, mock_storage):
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        mock_storage.batch_upsert_processed_context.return_value = None  # indicates failure

        with pytest.raises(RuntimeError, match="Failed to upsert"):
            await _replace_base_events_impl(
                mock_storage, "agent-x",
                [_make_base_ctx("x", ContextType.AGENT_BASE_EVENT, 0)],
            )
        mock_storage.delete_batch_processed_contexts.assert_not_awaited()

    async def test_delete_failure_logged_returns_stragglers(self, mock_storage, caplog):
        from opencontext.server.routes.agent_base_events import _replace_base_events_impl

        existing = {
            ContextType.AGENT_BASE_EVENT.value: [
                _make_base_ctx("old", ContextType.AGENT_BASE_EVENT, 0),
            ],
        }
        mock_storage.get_all_processed_contexts.return_value = existing
        mock_storage.batch_upsert_processed_context.return_value = ["new"]
        mock_storage.delete_batch_processed_contexts.side_effect = Exception("viking down")

        result = await _replace_base_events_impl(
            mock_storage, "agent-x",
            [_make_base_ctx("new", ContextType.AGENT_BASE_EVENT, 0)],
        )

        assert result["upserted"] == 1
        assert result["deleted"] == 0
        assert result["stragglers"] == 1
        assert any("delete_batch_processed_contexts failed" in rec.message for rec in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py::TestReplaceBaseEventsImpl -v`

Expected: FAIL with `ImportError: cannot import name '_replace_base_events_impl'`.

- [ ] **Step 3: Implement `_replace_base_events_impl`**

In `opencontext/server/routes/agent_base_events.py`, append to the helpers section:

```python
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
) -> dict[str, int]:
    """Replace all AGENT_BASE_* contexts for an agent with new_contexts.

    Order: upsert first (fail-safe: old tree intact on upsert failure),
    then delete the ids that no longer exist in new_contexts.

    A delete failure is logged but does not fail the request — stragglers
    will be cleaned up on the next replace.

    Returns {"upserted": N, "deleted": M, "stragglers": K}.
    Raises RuntimeError on upsert failure.
    """
    existing = await _fetch_existing_base_events(storage, agent_id)
    existing_ids = {c.id for c in existing}
    new_ids = {c.id for c in new_contexts}
    to_delete = list(existing_ids - new_ids)

    # Upsert FIRST — if this fails, old tree is intact.
    if new_contexts:
        upsert_result = await storage.batch_upsert_processed_context(new_contexts)
        if upsert_result is None:
            raise RuntimeError(f"Failed to upsert base events for agent={agent_id}")

    # Delete SECOND — if this fails, new tree is written but stragglers remain.
    deleted = 0
    stragglers = 0
    if to_delete:
        try:
            await storage.delete_batch_processed_contexts(to_delete, _ALL_AGENT_BASE_TYPES[0])
            deleted = len(to_delete)
        except Exception as e:
            logger.warning(
                f"delete_batch_processed_contexts failed after upsert for agent={agent_id}: {e}"
            )
            stragglers = len(to_delete)

    return {
        "upserted": len(new_contexts),
        "deleted": deleted,
        "stragglers": stragglers,
    }
```

Note on `delete_batch_processed_contexts`: per `unified_storage.py:262`, the signature is `(ids: list[str], context_type: str)`. The `context_type` argument is ignored by the VikingDB backend (all types share one collection), so passing any value from `_ALL_AGENT_BASE_TYPES` is fine.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py::TestReplaceBaseEventsImpl -v`

Expected: all five tests PASS.

- [ ] **Step 5: Commit**

```bash
git add opencontext/server/routes/agent_base_events.py tests/server/routes/test_agent_base_events.py
git commit -m "feat(agent-base-events): add replace-semantics core

Add _fetch_existing_base_events + _replace_base_events_impl. Upsert
first, delete second (failure-safe ordering: upsert failure preserves
old tree; delete failure leaves recoverable stragglers). Unit tests
cover fresh-agent, diff-delete, empty-replace, upsert failure
(propagates), and delete failure (logs + returns straggler count)."
```

---

## Task 4: Rewrite POST handler with Redis lock

**Goal:** Wire the new `_replace_base_events_impl` into `push_base_events`, wrapped in an agent-level Redis lock.

**Files:**
- Modify: `opencontext/server/routes/agent_base_events.py` (replace `push_base_events` body)
- Modify: `tests/server/routes/test_agent_base_events.py`

- [ ] **Step 1: Write failing test for the new POST semantics**

Append to `tests/server/routes/test_agent_base_events.py`:

```python
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_app():
    from opencontext.server.routes.agent_base_events import router
    app = FastAPI()
    app.include_router(router)
    return app


@contextlib.contextmanager
def _patched_client(mock_storage, mock_cache=None):
    """TestClient with get_storage + get_cache + auth patched."""
    app = _build_app()

    if mock_cache is None:
        mock_cache = MagicMock()
        mock_cache.acquire_lock = AsyncMock(return_value="lock-token-123")
        mock_cache.release_lock = AsyncMock(return_value=True)

    with (
        patch(
            "opencontext.server.routes.agent_base_events.get_storage",
            return_value=mock_storage,
        ),
        patch(
            "opencontext.server.routes.agent_base_events.get_cache",
            new=AsyncMock(return_value=mock_cache),
        ),
        patch(
            "opencontext.server.routes.agent_base_events.auth_dependency",
            return_value="test-token",
        ),
    ):
        yield TestClient(app), mock_cache


@pytest.mark.unit
class TestPushBaseEventsReplace:
    """Tests for POST /api/agents/{agent_id}/base/events — replace semantics."""

    def _mock_storage_with_existing(self, existing_ids):
        storage = MagicMock()
        storage.get_agent = AsyncMock(return_value={"agent_id": "a1", "name": "A1"})
        existing_dict = {
            ContextType.AGENT_BASE_EVENT.value: [
                _make_base_ctx(eid, ContextType.AGENT_BASE_EVENT, 0) for eid in existing_ids
            ],
        }
        storage.get_all_processed_contexts = AsyncMock(return_value=existing_dict)
        storage.batch_upsert_processed_context = AsyncMock(return_value=["new-id"])
        storage.delete_batch_processed_contexts = AsyncMock(return_value=True)
        return storage

    def test_404_when_agent_missing(self):
        storage = self._mock_storage_with_existing([])
        storage.get_agent = AsyncMock(return_value=None)

        with _patched_client(storage) as (client, _):
            resp = client.post(
                "/api/agents/a1/base/events",
                json={"events": [{"title": "t", "summary": "s"}]},
            )
            assert resp.status_code == 404

    def test_replace_deletes_ids_missing_from_payload(self):
        storage = self._mock_storage_with_existing(["old-1", "old-2"])

        with _patched_client(storage) as (client, _):
            resp = client.post(
                "/api/agents/a1/base/events",
                json={"events": [{"title": "t", "summary": "s"}]},  # one new event
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["data"]["upserted"] == 1
            assert body["data"]["deleted"] == 2
            deleted_args = storage.delete_batch_processed_contexts.call_args[0][0]
            assert set(deleted_args) == {"old-1", "old-2"}

    def test_acquires_and_releases_lock(self):
        storage = self._mock_storage_with_existing([])

        with _patched_client(storage) as (client, mock_cache):
            client.post(
                "/api/agents/a1/base/events",
                json={"events": [{"title": "t", "summary": "s"}]},
            )
            mock_cache.acquire_lock.assert_awaited_once()
            lock_args = mock_cache.acquire_lock.call_args
            assert lock_args[0][0] == "agent_base_edit:a1"
            mock_cache.release_lock.assert_awaited_once_with(
                "agent_base_edit:a1", "lock-token-123"
            )

    def test_returns_503_when_lock_unavailable(self):
        storage = self._mock_storage_with_existing([])

        mock_cache = MagicMock()
        mock_cache.acquire_lock = AsyncMock(return_value=None)  # acquisition failed
        mock_cache.release_lock = AsyncMock(return_value=True)

        with _patched_client(storage, mock_cache) as (client, _):
            resp = client.post(
                "/api/agents/a1/base/events",
                json={"events": [{"title": "t", "summary": "s"}]},
            )
            assert resp.status_code == 503
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py::TestPushBaseEventsReplace -v`

Expected: FAIL — existing handler doesn't take a lock and doesn't do replace semantics.

- [ ] **Step 3: Rewrite `push_base_events` to use lock + replace impl**

In `opencontext/server/routes/agent_base_events.py`, replace the entire existing `push_base_events` function (the one added in Task 1, Step 1) with:

```python
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
            "ids": [c.id for c in new_contexts],
        },
        message="Base events replaced",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py::TestPushBaseEventsReplace -v`

Expected: all four tests PASS.

- [ ] **Step 5: Run the full test file to confirm no regressions**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add opencontext/server/routes/agent_base_events.py tests/server/routes/test_agent_base_events.py
git commit -m "feat(agent-base-events): replace-semantics POST handler

POST now diffs existing vs new contexts and deletes missing ids after
upsert. Wrapped in Redis lock 'agent_base_edit:{agent_id}' (60s lease,
10s blocking) to serialize per-agent edits. Returns 503 on lock
acquisition timeout."
```

---

## Task 5: Rewrite DELETE handler with prune + replace

**Goal:** DELETE fetches current contexts, computes subtree rooted at `event_id`, scrubs the parent's refs, and invokes the same replace core.

**Files:**
- Modify: `opencontext/server/routes/agent_base_events.py`
- Modify: `tests/server/routes/test_agent_base_events.py`

- [ ] **Step 1: Write failing tests for the new DELETE semantics**

Append to `tests/server/routes/test_agent_base_events.py`:

```python
@pytest.mark.unit
class TestDeleteBaseEvent:
    """Tests for DELETE /api/agents/{agent_id}/base/events/{event_id}."""

    def _tree_storage(self):
        """Build mock storage with a small L0/L1 tree."""
        l1 = _make_base_ctx(
            "l1-1", ContextType.AGENT_BASE_L1_SUMMARY, 1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["l0-a", "l0-b"]},
        )
        l0_a = _make_base_ctx(
            "l0-a", ContextType.AGENT_BASE_EVENT, 0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-1"]},
        )
        l0_b = _make_base_ctx(
            "l0-b", ContextType.AGENT_BASE_EVENT, 0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-1"]},
        )
        existing_dict = {
            ContextType.AGENT_BASE_L1_SUMMARY.value: [l1],
            ContextType.AGENT_BASE_EVENT.value: [l0_a, l0_b],
        }
        storage = MagicMock()
        storage.get_all_processed_contexts = AsyncMock(return_value=existing_dict)
        storage.batch_upsert_processed_context = AsyncMock(
            return_value=["l1-1", "l0-b"],
        )
        storage.delete_batch_processed_contexts = AsyncMock(return_value=True)
        return storage

    def test_404_when_event_not_found(self):
        storage = self._tree_storage()

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events/does-not-exist")
            assert resp.status_code == 404

    def test_delete_leaf_prunes_single_id_and_scrubs_parent(self):
        storage = self._tree_storage()

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events/l0-a")
            assert resp.status_code == 200
            body = resp.json()
            assert body["data"]["deleted_ids"] == ["l0-a"]
            assert body["data"]["updated_parent_id"] == "l1-1"

            # Verify parent ctx passed to upsert has l0-a removed from refs
            upserted = storage.batch_upsert_processed_context.call_args[0][0]
            parent_ctx = next(c for c in upserted if c.id == "l1-1")
            assert parent_ctx.properties.refs[
                ContextType.AGENT_BASE_EVENT.value
            ] == ["l0-b"]

            # Verify delete_batch called with l0-a
            deleted_ids = storage.delete_batch_processed_contexts.call_args[0][0]
            assert deleted_ids == ["l0-a"]

    def test_delete_l1_prunes_entire_subtree(self):
        storage = self._tree_storage()

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events/l1-1")
            assert resp.status_code == 200
            body = resp.json()
            assert set(body["data"]["deleted_ids"]) == {"l1-1", "l0-a", "l0-b"}
            assert body["data"]["updated_parent_id"] is None  # L1 had no parent

            # Upsert called with empty list (tree fully pruned)
            upserted = storage.batch_upsert_processed_context.call_args[0][0]
            assert upserted == []

            deleted_ids = storage.delete_batch_processed_contexts.call_args[0][0]
            assert set(deleted_ids) == {"l1-1", "l0-a", "l0-b"}

    def test_delete_leaf_preserves_empty_parent_summary(self):
        """Regression guard: deleting the only child of an L1 leaves L1 intact."""
        # Build tree with single L0 child under L1
        l1 = _make_base_ctx(
            "l1-only", ContextType.AGENT_BASE_L1_SUMMARY, 1,
            refs={ContextType.AGENT_BASE_EVENT.value: ["l0-only"]},
        )
        l0 = _make_base_ctx(
            "l0-only", ContextType.AGENT_BASE_EVENT, 0,
            refs={ContextType.AGENT_BASE_L1_SUMMARY.value: ["l1-only"]},
        )
        existing = {
            ContextType.AGENT_BASE_L1_SUMMARY.value: [l1],
            ContextType.AGENT_BASE_EVENT.value: [l0],
        }
        storage = MagicMock()
        storage.get_all_processed_contexts = AsyncMock(return_value=existing)
        storage.batch_upsert_processed_context = AsyncMock(return_value=["l1-only"])
        storage.delete_batch_processed_contexts = AsyncMock(return_value=True)

        with _patched_client(storage) as (client, _):
            resp = client.delete("/api/agents/a1/base/events/l0-only")
            assert resp.status_code == 200

            # L1 still in the upsert list (preserved) with empty downward ref list
            upserted = storage.batch_upsert_processed_context.call_args[0][0]
            assert any(c.id == "l1-only" for c in upserted)
            l1_after = next(c for c in upserted if c.id == "l1-only")
            assert l1_after.properties.refs[
                ContextType.AGENT_BASE_EVENT.value
            ] == []

    def test_delete_acquires_and_releases_lock(self):
        storage = self._tree_storage()

        with _patched_client(storage) as (client, mock_cache):
            client.delete("/api/agents/a1/base/events/l0-a")
            mock_cache.acquire_lock.assert_awaited_once()
            lock_args = mock_cache.acquire_lock.call_args
            assert lock_args[0][0] == "agent_base_edit:a1"
            mock_cache.release_lock.assert_awaited_once()

    def test_delete_returns_503_when_lock_unavailable(self):
        storage = self._tree_storage()
        mock_cache = MagicMock()
        mock_cache.acquire_lock = AsyncMock(return_value=None)
        mock_cache.release_lock = AsyncMock(return_value=True)

        with _patched_client(storage, mock_cache) as (client, _):
            resp = client.delete("/api/agents/a1/base/events/l0-a")
            assert resp.status_code == 503
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py::TestDeleteBaseEvent -v`

Expected: FAIL — existing DELETE handler does single point-delete, not prune+replace.

- [ ] **Step 3: Rewrite the DELETE handler**

In `opencontext/server/routes/agent_base_events.py`, replace the entire existing `delete_base_event` function (added in Task 1, Step 1) with:

```python
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
            "deleted_ids": list(subtree_ids),
            "updated_parent_id": parent_id,
            "stragglers": result["stragglers"],
        },
        message="Event deleted",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py::TestDeleteBaseEvent -v`

Expected: all six tests PASS.

- [ ] **Step 5: Run the full test file**

Run: `uv run pytest tests/server/routes/test_agent_base_events.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add opencontext/server/routes/agent_base_events.py tests/server/routes/test_agent_base_events.py
git commit -m "feat(agent-base-events): server-side cascade DELETE via prune+replace

DELETE now fetches the tree, computes the subtree rooted at event_id,
scrubs the surviving parent's downward refs, and invokes the same
replace core as POST — producing the same atomic guarantees.

Empty parent summaries (whose only child was deleted) are PRESERVED
by design, not cascade-deleted. Regression test covers this."
```

---

## Task 6: Simplify frontend `deleteEvent`

**Goal:** Replace the client-side cascade loop with a single DELETE request; use the server response to update local state.

**Files:**
- Modify: `opencontext/web/templates/agents.html` (lines 438-488)

- [ ] **Step 1: Read the current deleteEvent implementation**

Run: `grep -n "deleteEvent\|_collectDescendantIds" opencontext/web/templates/agents.html`

Confirm the line ranges of `_collectDescendantIds` (should be ~438-447) and `deleteEvent` (should be ~449-488) before editing.

- [ ] **Step 2: Replace the two functions**

In `opencontext/web/templates/agents.html`, locate `_collectDescendantIds` function and the `deleteEvent` function. Replace both with:

```javascript
        function _estimateSubtreeSize(eventId) {
            const node = _eventTreeMap?.get(eventId);
            if (!node) return 1;
            let count = 1;
            const stack = [...(node.children || [])];
            while (stack.length) {
                const cur = stack.pop();
                count += 1;
                if (cur.children) stack.push(...cur.children);
            }
            return count;
        }

        async function deleteEvent(eventId) {
            const subtreeSize = _estimateSubtreeSize(eventId);
            const confirmMsg = subtreeSize > 1
                ? `确定删除此事件及其 ${subtreeSize - 1} 个子级事件？`
                : '确定删除此事件？';
            if (!confirm(confirmMsg)) return;

            try {
                const resp = await fetch(
                    `/api/agents/${_currentAgentId}/base/events/${eventId}`,
                    { method: 'DELETE' },
                );
                if (!resp.ok) {
                    const body = await resp.json().catch(() => ({}));
                    alert(`删除失败：${body.detail || resp.statusText}`);
                    return;
                }
                const body = await resp.json();
                const deleted = new Set(body.data?.deleted_ids || []);

                // Update local state based on server response.
                // VikingDB indexing can lag — rely on the authoritative server set.
                _currentEvents = _currentEvents.filter(e => !deleted.has(e.id));

                renderEvents();
            } catch (err) {
                console.error('deleteEvent failed', err);
                alert(`删除失败：${err.message}`);
            }
        }
```

Notes:
- `_estimateSubtreeSize` replaces `_collectDescendantIds` — cheaper, just a count for the prompt.
- The single `fetch` replaces the previous sequential loop.
- Trust `body.data.deleted_ids` for local filtering; the server is authoritative.
- No parent-refs client-side patch needed — the server rewrote them and the next `renderEvents` reads the `_currentEvents` fresh state (any derived children_count will recompute from surviving refs).

- [ ] **Step 3: Manual verification**

Start the server (or use an existing running instance) and browse to the agents page. Manually test:

1. Delete a leaf L0 event — UI shows the event removed; parent summary remains with children_count reduced by 1.
2. Delete an L1 summary — UI shows the summary and all its L0 children removed.
3. Attempt to delete when server is unreachable — UI shows an error alert instead of partial state.

Record the result in the commit message. If any UI breakage is observed, note it and fix before proceeding.

- [ ] **Step 4: Commit**

```bash
git add opencontext/web/templates/agents.html
git commit -m "refactor(agents-ui): replace client cascade loop with single DELETE

_collectDescendantIds replaced by _estimateSubtreeSize (confirm prompt
only). deleteEvent issues a single DELETE and uses the server's
deleted_ids response to update _currentEvents. Trusts the server as
authoritative — no more optimistic local-state patches racing
VikingDB indexing lag."
```

---

## Task 7: Update API documentation

**Goal:** Document the breaking POST semantic change and the new DELETE response shape.

**Files:**
- Modify: `docs/api_reference.md`
- Modify: `docs/curls.sh`

- [ ] **Step 1: Locate the relevant sections**

Run: `grep -n "base/events\|base_event\|Base Events" docs/api_reference.md docs/curls.sh`

Note the line ranges covering base events in each file.

- [ ] **Step 2: Update `docs/api_reference.md`**

Find the section covering `POST /api/agents/{agent_id}/base/events`. Add a callout immediately under the endpoint heading:

```markdown
> **⚠️ Breaking change (2026-04-17):** This endpoint now uses **replace semantics**. Each request treats the submitted tree as the agent's complete base-event state. Any existing `AGENT_BASE_*` context not present in the new tree will be deleted. Clients that previously relied on additive/incremental uploads must now merge locally before posting.
```

Update the response schema for POST to include the new fields:

```markdown
**Response** (`200 OK`):
```json
{
  "code": 0,
  "status": 200,
  "message": "Base events replaced",
  "data": {
    "upserted": 12,
    "deleted": 4,
    "stragglers": 0,
    "ids": ["<new_id_1>", "..."]
  }
}
```
- `upserted` — count of contexts written
- `deleted` — count of previously-existing contexts removed
- `stragglers` — count of contexts that should have been deleted but whose deletion failed (non-fatal; cleaned on next POST)
- `ids` — ids of upserted contexts
```

Also add `503 Service Unavailable — another edit is in progress for this agent` to the POST error responses.

Find the section covering `DELETE /api/agents/{agent_id}/base/events/{event_id}`. Replace its behavior/response description with:

```markdown
Deletes an event and its entire subtree (all descendants reachable via
downward refs). The surviving parent's ref list is scrubbed of the
deleted root id. Empty parent summaries (whose only child was just
deleted) are preserved — to remove them, POST a new tree without them.

**Response** (`200 OK`):
```json
{
  "code": 0,
  "status": 200,
  "message": "Event deleted",
  "data": {
    "deleted_ids": ["<root_id>", "<child_id_1>", "..."],
    "updated_parent_id": "<parent_id_or_null>",
    "stragglers": 0
  }
}
```

Also document `503 Service Unavailable` for the DELETE endpoint.

- [ ] **Step 3: Update `docs/curls.sh`**

Find the existing curl examples for base events. No change needed to the curl commands themselves (the URLs and methods are unchanged), but add a comment above the POST example:

```bash
# NOTE: POST uses REPLACE semantics. Each request treats the payload as the
# agent's complete base-event state — ids not in the payload are deleted.
```

- [ ] **Step 4: Commit**

```bash
git add docs/api_reference.md docs/curls.sh
git commit -m "docs(api): document base-events replace semantics

POST: add breaking-change callout, new response fields
(upserted/deleted/stragglers/ids), and 503 error case.
DELETE: document subtree cascade and updated response shape."
```

---

## Self-Review Checklist (completed inline by plan author)

**Spec coverage:**
- POST replace semantics → Task 4 ✓
- DELETE as server-side wrapper → Task 5 ✓
- Redis lock `agent_base_edit:{agent_id}` → Tasks 4, 5 ✓
- Upsert-first-delete-second ordering → Task 3 ✓
- Tree helpers (collect subtree, find parent, scrub refs) → Task 2 ✓
- `_flatten_base_event_tree` reuse → Task 1 (migrated) ✓
- `ALL_AGENT_BASE_TYPES` filter for `user_id="__base__"` → Task 3 (`_fetch_existing_base_events`) ✓
- Frontend simplification → Task 6 ✓
- `docs/api_reference.md` + `docs/curls.sh` updates → Task 7 ✓
- Empty-summary preservation (non-goal enforced by regression test) → Task 5, `test_delete_leaf_preserves_empty_parent_summary` ✓
- Cache invalidation NOT performed (non-goal, deliberately) — no task, correct ✓

**Placeholder scan:** No TBD/TODO/"similar to" references. Each step shows concrete code.

**Type consistency:** `_collect_subtree_ids`, `_find_parent_id`, `_scrub_parent_refs`, `_fetch_existing_base_events`, `_replace_base_events_impl` — names match across tasks. Parameter shapes (`ctx_by_id: dict[str, ProcessedContext]`, `child_context_type: ContextType`, `new_contexts: list[ProcessedContext]`) are consistent across tests and implementations.

**Lock API signature:** Verified — `cache.acquire_lock(name, timeout, blocking, blocking_timeout)` returns `str | None`; `cache.release_lock(name, token)` returns `bool`. Matches pattern in `memory_cache_manager.py:175-197`.

**Storage method name:** Tests and implementation use `delete_batch_processed_contexts` (matching `unified_storage.py:262`), not the backend-level `delete_contexts` and not the single-id `delete_processed_context`.
