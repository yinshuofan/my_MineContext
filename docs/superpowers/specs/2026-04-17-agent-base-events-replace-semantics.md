# Agent Base Events — Replace-Tree Semantics

## Goal

Fix two real bugs in the agent base-events edit flow:

1. **Dangling refs**: deleting a base-event node leaves stale IDs in the surviving parent's `refs` field.
2. **Non-atomic cascade**: the frontend deletes nodes one-by-one via a client-side loop, so partial failures produce inconsistent state and no rollback.

The fix changes `POST /api/agents/{agent_id}/base/events` from upsert to **replace-tree** semantics (every POST rebuilds the agent's entire base-event tree from scratch). `DELETE /api/agents/{agent_id}/base/events/{event_id}` becomes a thin server-side wrapper that prunes the subtree and calls the same replace logic. Refs consistency becomes a structural invariant, not an algorithmic contract.

## Non-Goals

- **Cache invalidation**: `memory_cache:accessed:*` may serve stale base-event metadata for up to 7 days (`accessed_ttl`). Accepted — base-event edits are infrequent admin actions and the staleness is bounded.
- **Orphan summary cleanup**: an L1/L2/L3 summary whose downward children were all deleted is **preserved**. Users decide explicitly what to keep when editing the tree. An empty summary is an "abstract memory without specifics" — a legitimate user state, not a bug.
- **Agent-ownership enforcement on DELETE**: the existing handler accepts any `event_id` without checking it belongs to `agent_id`. Out of scope for this spec.

## Architecture

Current write path is already tree-shaped: `_flatten_base_event_tree` (agents.py:206-287) accepts a nested event tree and emits flat `ProcessedContext` list with bidirectional `refs` baked in. The POST endpoint already calls this then upserts the flat list.

The change: in the same request, also **diff against existing contexts** for that agent and **delete the ones no longer present** in the new tree. Because `_flatten_base_event_tree` rebuilds `refs` from the input tree every time, the result is always internally consistent. There is no sequence of operations where a "surviving parent" could be left with a dangling reference — refs are regenerated, not patched.

DELETE becomes a convenience operation on top of replace: server fetches the current tree, computes the subtree rooted at `event_id`, builds a pruned version, and runs the same replace flow.

A single Redis lock (`agent_base_edit:{agent_id}`) serializes all writes for an agent. Cross-agent writes parallelize.

## Files to Modify

| File | Change |
|---|---|
| `opencontext/server/routes/agent_base_events.py` | **New file**. Holds all base-event constants, helpers, and three endpoints (POST/GET/DELETE). Includes `flatten_tree`, tree reconstruction helpers, and the replace logic. |
| `opencontext/server/routes/agents.py` | Remove `_BASE_HIERARCHY_LEVEL_TO_TYPE`, `_ALL_AGENT_BASE_TYPES`, `_flatten_base_event_tree`, and three base-event handlers. Shrinks from ~498 to ~230 lines. Keep agent CRUD only. |
| `opencontext/server/api.py` (or wherever the main router is composed) | Include new `agent_base_events` sub-router. |
| `opencontext/web/templates/agents.html` | Remove `_collectDescendantIds` (438-447); simplify `deleteEvent` (449-488) to a single `DELETE` with local-state update driven by the response. |
| `docs/api_reference.md` | Document POST semantic change from upsert to replace. |
| `docs/curls.sh` | Update example payloads if needed. |
| `tests/server/routes/test_agent_base_events.py` | **New**. Unit + integration tests for replace and delete. |

## Design

### 1. POST — replace semantics

```python
@router.post("/api/agents/{agent_id}/base/events")
async def push_base_events(agent_id: str, request: BaseEventsPushRequest, _=Depends(auth)):
    cache = await get_cache()
    lock_key = f"agent_base_edit:{agent_id}"
    lock_token = await cache.acquire_lock(lock_key, timeout=60, blocking=True, blocking_timeout=10)
    if not lock_token:
        raise HTTPException(503, "Another edit is in progress for this agent")
    try:
        new_contexts = flatten_tree(request.events, agent_id)
        new_ids = {c.id for c in new_contexts}

        existing = await _fetch_all_base_events(agent_id)
        existing_ids = {c.id for c in existing}

        to_delete = list(existing_ids - new_ids)

        # Upsert FIRST, delete SECOND — failure-safe ordering.
        # If upsert fails: no data lost (old tree intact), exception propagates → 500.
        # If delete fails: new tree written, old stragglers remain, logged + 200 returned.
        await get_storage().batch_upsert_processed_context(new_contexts)

        delete_ok = True
        if to_delete:
            try:
                await get_storage().delete_contexts(to_delete, ALL_AGENT_BASE_TYPES[0])  # type ignored by backend
            except Exception as e:
                logger.warning(f"delete_contexts failed after upsert for agent={agent_id}: {e}")
                delete_ok = False

        return convert_resp(data={
            "upserted": len(new_contexts),
            "deleted": len(to_delete) if delete_ok else 0,
            "stragglers": 0 if delete_ok else len(to_delete),
            "ids": [c.id for c in new_contexts],
        })
    finally:
        await cache.release_lock(lock_key, lock_token)
```

**Fetch existing contexts**: `_fetch_all_base_events(agent_id)` calls `storage.get_all_processed_contexts()` with `context_types=ALL_AGENT_BASE_TYPES`, `filter={"user_id": "__base__", "agent_id": agent_id}`. Pagination through all results — the agent's base-event corpus is expected to be hundreds of nodes max, so a single query with a generous `limit` is fine.

**Why upsert before delete**: if `batch_upsert_processed_context` fails, the old tree remains intact (zero data loss). If `delete_contexts` fails, the new tree is written but some stragglers from the old tree remain; next POST cleans them up (idempotent recovery). The reverse order would lose data if upsert fails after delete.

**VikingDB eventual consistency note**: sync upserts are visible within ~1s via `fetch_in_collection`-style queries. If a user does two POSTs in rapid succession, the second's `_fetch_all_base_events` may miss nodes just written by the first → they get treated as "not existing in old set" and may escape diffing. This is a known limitation; the agent-level lock prevents concurrent POSTs, but not rapid sequential POSTs across requests. Document as caveat.

### 2. DELETE — server-side prune + replace

```python
@router.delete("/api/agents/{agent_id}/base/events/{event_id}")
async def delete_base_event(agent_id: str, event_id: str, _=Depends(auth)):
    cache = await get_cache()
    lock_key = f"agent_base_edit:{agent_id}"
    lock_token = await cache.acquire_lock(lock_key, timeout=60, blocking=True, blocking_timeout=10)
    if not lock_token:
        raise HTTPException(503, "Another edit is in progress for this agent")
    try:
        existing = await _fetch_all_base_events(agent_id)
        ctx_by_id = {c.id: c for c in existing}

        if event_id not in ctx_by_id:
            raise HTTPException(404, "Event not found")

        subtree_ids = _collect_subtree_ids(ctx_by_id, event_id)
        parent_id = _find_parent_id(ctx_by_id, event_id)

        kept = [c for c in existing if c.id not in subtree_ids]

        # Scrub parent's refs in the kept list before write-back.
        if parent_id and parent_id in ctx_by_id:
            parent_ctx = next(c for c in kept if c.id == parent_id)
            _scrub_parent_refs(parent_ctx, event_id, ctx_by_id[event_id])

        # Same failure-safe ordering as POST.
        await get_storage().batch_upsert_processed_context(kept)
        try:
            await get_storage().delete_contexts(list(subtree_ids), ALL_AGENT_BASE_TYPES[0])
        except Exception as e:
            logger.warning(f"delete_contexts failed after parent scrub for agent={agent_id}: {e}")
            # Parent refs already scrubbed and written; descendant stragglers remain
            # but are unreachable via refs and will be cleaned on next POST.

        return convert_resp(data={
            "deleted_ids": list(subtree_ids),
            "updated_parent_id": parent_id,
        })
    finally:
        await cache.release_lock(lock_key, lock_token)
```

**`_collect_subtree_ids(ctx_by_id, root_id) -> set[str]`**: BFS from `root_id` using each node's downward `refs[child_type]`. Downward vs upward is distinguished by `hierarchy_level` comparison (child has smaller level). Maintain a `visited` set for cycle safety. Returns `{root_id, all descendant ids}`.

**`_find_parent_id(ctx_by_id, event_id) -> str | None`**: read `event.properties.refs[parent_type]` where `parent_type` is the context type one level up. Returns first parent id or `None` if root.

**`_scrub_parent_refs(parent_ctx, child_id, child_ctx)`**: modifies `parent_ctx.properties.refs[child_ctx_type]` in place, removing `child_id`. Idempotent (safe if already removed).

### 3. `flatten_tree` (moved and renamed)

Move `_flatten_base_event_tree` from `routes/agents.py:206-287` into `routes/agent_base_events.py`, rename to `flatten_tree` (drop the underscore prefix since it's the public interface of this module). Behavior unchanged.

### 4. Tree helpers (new)

All module-level pure functions in `routes/agent_base_events.py`. No class.

```python
def _collect_subtree_ids(ctx_by_id: dict[str, ProcessedContext], root_id: str) -> set[str]:
    """BFS over downward refs. Returns {root_id} ∪ descendants."""

def _find_parent_id(ctx_by_id: dict[str, ProcessedContext], event_id: str) -> str | None:
    """Read event.refs[parent_type]. Returns first parent id or None."""

def _scrub_parent_refs(parent_ctx: ProcessedContext, child_id: str, child_ctx: ProcessedContext) -> None:
    """Remove child_id from parent.refs[child_context_type] (in place)."""
```

`_collect_subtree_ids` distinguishes child vs parent refs by comparing `hierarchy_level` — a ref entry pointing to a lower level is a child, a ref pointing higher is a parent. This matches the existing logic in `event_search_service.collect_descendants()` (event_search_service.py:304-311).

### 5. `CascadeResult` DTO

Local to `agent_base_events.py` — a simple `dataclass`, not exposed in public models:

```python
@dataclass
class CascadeResult:
    deleted_ids: list[str]
    updated_parent_id: str | None
```

Return via `convert_resp(data=asdict(result))`.

### 6. Locking — `agent_base_edit:{agent_id}`

- Uses existing `RedisCache.acquire_lock` / `release_lock` (`opencontext/storage/redis_cache.py`).
- `timeout=60` (lease), `blocking_timeout=10`. If acquisition fails within 10s, return HTTP 503.
- **Both POST and DELETE take the same lock** — they are mutually exclusive per agent.
- Cross-agent edits run in parallel (different lock keys).
- Lock release in `finally` block. Lock TTL protects against crashed holders.

### 7. Frontend changes (`agents.html`)

Remove:
- `_collectDescendantIds` (438-447)

Modify `deleteEvent(eventId)` (449-488):

```javascript
async function deleteEvent(eventId) {
    const subtreeSize = _estimateSubtreeSize(eventId);  // use _eventTreeMap locally
    const confirmMsg = subtreeSize > 1
        ? `确定删除此事件及其 ${subtreeSize - 1} 个子级事件？`
        : '确定删除此事件？';
    if (!confirm(confirmMsg)) return;

    const resp = await fetch(`/api/agents/${_currentAgentId}/base/events/${eventId}`, {
        method: 'DELETE',
    });
    if (!resp.ok) { alert('删除失败'); return; }

    const { data } = await resp.json();
    const deleted = new Set(data.deleted_ids);
    _currentEvents = _currentEvents.filter(e => !deleted.has(e.id));

    // Patch parent's displayed children_count if rendered
    if (data.updated_parent_id) {
        const parent = _currentEvents.find(e => e.id === data.updated_parent_id);
        if (parent?.refs) { /* recompute count from refs */ }
    }

    renderEvents();
}
```

`_estimateSubtreeSize(eventId)` uses the existing `_eventTreeMap` to count descendants for the confirmation prompt. Server-side is authoritative.

### 8. Error handling

| Failure | Response | State |
|---|---|---|
| Lock acquisition timeout | 503 Service Unavailable | No change |
| `_fetch_all_base_events` fails | 500 | No change |
| `batch_upsert_processed_context` fails | 500 | Old tree intact |
| `delete_contexts` fails after successful upsert | Log warning, return 200 with partial result | Some stragglers from old tree remain; next POST cleans up |
| `event_id` not found (DELETE) | 404 Not Found | No change |

The "delete-after-upsert" partial failure is the most benign failure mode — data is never lost, only accumulated. An idempotent retry of the same POST converges the state.

## Testing

`tests/server/routes/test_agent_base_events.py` (new file).

**Unit tests (fake storage, no infra needed):**

- `test_replace_deletes_removed_ids` — POST tree A, then POST tree B that shares some but not all ids with A. Verify ids-only-in-A are deleted.
- `test_replace_preserves_kept_ids` — POST tree A, then POST tree A' differing only in one subtree. Verify unchanged nodes retain their data.
- `test_replace_reconstructs_refs` — POST tree, fetch, verify every node's `refs` exactly matches the tree structure (no dangling ids, bidirectional consistency).
- `test_delete_prunes_subtree` — build tree in fake storage, DELETE middle node, verify all descendants are gone.
- `test_delete_scrubs_parent_refs` — after DELETE, verify `parent.refs[child_type]` no longer contains deleted id.
- `test_delete_preserves_empty_summary` — DELETE the only L0 child of an L1 summary, verify the L1 summary is still present (Issue 6 regression guard).
- `test_delete_root_has_no_parent` — DELETE a top-level L3 summary, verify no parent scrub attempted, no crash.
- `test_delete_nonexistent_returns_404`.
- `test_collect_subtree_ids_cycle_safety` — inject a bad tree with a cycle, verify BFS terminates.
- `test_flatten_tree_unchanged` — behavioral parity with old `_flatten_base_event_tree`.

**Integration test (optional, `@pytest.mark.integration`):**

- `test_concurrent_post_serialized` — two concurrent POSTs to the same agent complete without data loss; verify final state matches one of the two payloads (last-writer-wins semantics).

## Migration & Rollout

- **Breaking API change**. Clients that relied on additive POST semantics (incremental tree building) must now fetch-merge-post. No in-tree scripts currently rely on this pattern.
- No data migration needed. The new POST handles an existing tree transparently — first call after deployment will have `to_delete = ∅` since all existing ids are in the tree.
- Docs: `docs/api_reference.md` and `docs/curls.sh` updated in the same commit.

## Caveats (documented, accepted)

1. **Cache staleness**: base-event deletions do not invalidate `memory_cache:accessed:*`; stale metadata may surface in "recently accessed" lists for up to 7 days.
2. **Empty summaries preserved**: deleting an L0 child whose L1 parent has no other children leaves the L1 in place. Users explicitly prune it in their next POST if unwanted.
3. **VikingDB index lag**: two rapid sequential POSTs may see inconsistent "existing" snapshots, causing short-lived duplicates. Self-heals on the next stable POST.
4. **No cross-request atomicity**: a crash between upsert and delete leaves the new tree plus old stragglers. Next POST resolves.
