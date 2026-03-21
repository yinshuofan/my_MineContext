# Base Event Hierarchy Upload Design

## Goal

Extend the existing `POST /api/agents/{agent_id}/base/events` endpoint to support uploading a complete hierarchy tree of agent base events (L0-L3), with server-side validation and bidirectional reference construction.

## Architecture

The existing flat-list base event upload is extended with a recursive nested structure. Each event node can contain `children`, forming a tree. The server validates the tree structure, builds bidirectional `refs`, and batch-writes all nodes to the vector DB in one operation.

A new family of context types (`AGENT_BASE_EVENT`, `AGENT_BASE_L1/L2/L3_SUMMARY`) replaces the current `AGENT_EVENT` + `user_id="__base__"` convention, cleanly separating base data from user-generated agent events at the type level.

## Request Model

`BaseEventItem` is extended with new optional fields. When none are provided, behavior is identical to the current API.

```python
class BaseEventItem(BaseModel):
    title: str
    summary: str
    event_time_start: Optional[str] = None    # ISO 8601, defaults to current time
    event_time_end: Optional[str] = None      # Required for hierarchy_level > 0
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    importance: int = 5
    hierarchy_level: int = 0                   # 0/1/2/3, pure hierarchy depth
    children: Optional[List["BaseEventItem"]] = None  # Nested child events
```

Field changes from the old model:
- New: `event_time_start` (aligned with `ContextProperties`)
- New: `event_time_end` (required for summaries)
- New: `hierarchy_level` (default 0)
- New: `children` (recursive nesting)

### Example Request

```json
{
  "events": [
    {
      "title": "Standalone raw event",
      "summary": "An orphan L0 event with no parent",
      "event_time_start": "2026-03-15T10:00:00+08:00"
    },
    {
      "title": "Week 11 Summary",
      "summary": "Summary of the week...",
      "event_time_start": "2026-03-09T00:00:00+08:00",
      "event_time_end": "2026-03-15T23:59:59+08:00",
      "hierarchy_level": 2,
      "children": [
        {
          "title": "Monday Summary",
          "summary": "Summary of Monday...",
          "event_time_start": "2026-03-09T00:00:00+08:00",
          "event_time_end": "2026-03-09T23:59:59+08:00",
          "hierarchy_level": 1,
          "children": [
            {
              "title": "Morning standup",
              "summary": "Discussed sprint progress",
              "event_time_start": "2026-03-09T09:00:00+08:00",
              "keywords": ["standup", "sprint"],
              "importance": 6
            },
            {
              "title": "Code review session",
              "summary": "Reviewed auth module PR",
              "event_time_start": "2026-03-09T14:00:00+08:00",
              "keywords": ["code review", "auth"],
              "importance": 5
            }
          ]
        },
        {
          "title": "Tuesday Summary",
          "summary": "Summary of Tuesday...",
          "event_time_start": "2026-03-10T00:00:00+08:00",
          "event_time_end": "2026-03-10T23:59:59+08:00",
          "hierarchy_level": 1,
          "children": [
            {
              "title": "Design meeting",
              "summary": "Finalized API design",
              "event_time_start": "2026-03-10T10:00:00+08:00",
              "keywords": ["design", "API"],
              "importance": 7
            }
          ]
        }
      ]
    }
  ]
}
```

### Request Size Limits

- Maximum total events per request (across all levels): 500
- Maximum `children` per node: no explicit limit (constrained by total count)
- If the total flattened count exceeds the limit, return HTTP 400

## Validation Rules

All validation is performed before any storage operation. On failure, the entire request is rejected with HTTP 400 and a path-based error message (e.g., `"events[1].children[0]: hierarchy_level 0 cannot have children"`).

### 1. Structure Validation
- `hierarchy_level > 0` requires `event_time_end` to be present and `children` to be non-empty
- `hierarchy_level == 0` requires `children` to be absent or null
- Each child's `hierarchy_level` must equal `parent.hierarchy_level - 1`
- Maximum `hierarchy_level` is 3; maximum nesting depth is 4 levels (L3 → L2 → L1 → L0)

### 2. Time Range Validation
- For summaries (`hierarchy_level > 0`): `event_time_start` must be <= the minimum `event_time_start` of all **direct** children
- For summaries (`hierarchy_level > 0`): `event_time_end` must be >= the maximum `event_time_end` (or `event_time_start` for L0 children) of all **direct** children
- Transitive containment is guaranteed by induction — if every parent-child pair satisfies the constraint, the entire tree is valid

### 3. Single-Parent Constraint
- Guaranteed by the nested structure itself — each node appears under exactly one parent

### 4. Agent Existence Check
- The agent specified by `agent_id` must exist (existing validation, unchanged)

## New Context Types

Four new context types are added. `hierarchy_level` is purely a depth indicator, not bound to daily/weekly/monthly semantics.

```python
AGENT_BASE_EVENT = "agent_base_event"            # L0
AGENT_BASE_L1_SUMMARY = "agent_base_l1_summary"  # L1
AGENT_BASE_L2_SUMMARY = "agent_base_l2_summary"  # L2
AGENT_BASE_L3_SUMMARY = "agent_base_l3_summary"  # L3
```

### Context Type Mapping

| hierarchy_level | context_type | update_strategy | storage_backend |
|---|---|---|---|
| 0 | `AGENT_BASE_EVENT` | APPEND | VECTOR_DB |
| 1 | `AGENT_BASE_L1_SUMMARY` | APPEND | VECTOR_DB |
| 2 | `AGENT_BASE_L2_SUMMARY` | APPEND | VECTOR_DB |
| 3 | `AGENT_BASE_L3_SUMMARY` | APPEND | VECTOR_DB |

### MEMORY_OWNER_TYPES Registration

A new memory owner entry `"agent_base"` is added to `MEMORY_OWNER_TYPES`:

```python
"agent_base": [
    ContextType.AGENT_BASE_EVENT,          # index 0 = L0
    ContextType.AGENT_BASE_L1_SUMMARY,     # index 1 = L1
    ContextType.AGENT_BASE_L2_SUMMARY,     # index 2 = L2
    ContextType.AGENT_BASE_L3_SUMMARY,     # index 3 = L3
],
```

This ensures `EventSearchService`, `MemoryCacheManager`, and `HierarchicalEventTool` can discover and query base events. The existing `"agent"` entry remains unchanged.

### SYSTEM_GENERATED_TYPES Exclusion

The 4 new types must NOT be added to `SYSTEM_GENERATED_TYPES`. They are user-uploaded base knowledge, not system-generated summaries. Adding them would cause `get_context_type_for_analysis()` to misclassify them during LLM processing.

### Auto-Generated Hierarchy Exclusion

The `HierarchySummaryTask` periodic task does NOT apply to `AGENT_BASE_*` types. Base events come with pre-built hierarchies uploaded by the caller. No auto-generation of summaries should be triggered for these types.

### Vector DB Collection Side Effect

Adding 4 new `ContextType` enum members will automatically create 4 new collections/indexes in the vector DB backends (Qdrant creates one collection per type at init; VikingDB may behave similarly). This is expected and acceptable — each base type gets its own collection, consistent with how other context types are stored.

### Migration from AGENT_EVENT

The current base event code uses `ContextType.AGENT_EVENT` with `user_id="__base__"` sentinel. This is replaced:
- All base event write paths change to `AGENT_BASE_EVENT`
- All base event query/delete paths filter by `AGENT_BASE_*` types
- `user_id="__base__"` is retained as the user_id value (base data doesn't belong to any real user) but is no longer the primary mechanism for distinguishing base vs user data

## Server-Side Processing Flow

### 1. Recursive Flatten + ID Generation
- Depth-first traversal of the nested tree
- Generate a unique ID for each node
- Flatten into `List[ProcessedContext]`

### 2. Bidirectional Ref Construction (in-memory)
- **Downward refs** (parent → children): summary node's `refs` stores child IDs, keyed by child's context_type value
  - Example: L1 node `refs = {"agent_base_event": ["id1", "id2"]}`
- **Upward refs** (child → parent): child node's `refs` stores parent ID, keyed by parent's context_type value
  - Example: L0 node `refs = {"agent_base_l1_summary": ["parent_id"]}`
- All refs are built in memory before storage — no post-write backfill needed

### 3. Batch Vectorization
- Concatenate `title + "\n" + summary + "\n" + keywords.join(", ")` for each node
- Use `do_vectorize_batch()` from `global_embedding_client` to generate all embeddings in one batched call, not per-event sequential `do_vectorize()`

### 4. Batch Write
- Single call to `storage.batch_upsert_processed_context(all_contexts)`
- All nodes (L0 through L3) written in one batch

## Delete Semantics

When deleting a base event via `DELETE /api/agents/{agent_id}/base/events/{event_id}`:

- **Single-node delete only**: deletes the specified node, does not cascade to children or clean up parent refs
- Orphaned refs (a parent pointing to a deleted child, or a child pointing to a deleted parent) are acceptable — the system already handles missing refs gracefully during retrieval
- For bulk cleanup, the caller should delete nodes individually or use the existing agent delete flow which removes all agent data

## GET Endpoint Behavior

`GET /api/agents/{agent_id}/base/events` is updated to:
- Query all 4 `AGENT_BASE_*` types by default (not just L0)
- Accept an optional `hierarchy_level` query parameter to filter by level (e.g., `?hierarchy_level=0` for L0 only)
- Response includes `hierarchy_level` and `refs` fields for each event

## Impact on Existing Code

### Files to Modify

1. **`opencontext/models/enums.py`**
   - Add 4 new context types (`AGENT_BASE_EVENT`, `AGENT_BASE_L1/L2/L3_SUMMARY`)
   - Update `CONTEXT_UPDATE_STRATEGIES` (all APPEND)
   - Update `CONTEXT_STORAGE_BACKENDS` (all VECTOR_DB)
   - Update `ContextDescriptions` / `ContextSimpleDescriptions`
   - Add `"agent_base"` entry to `MEMORY_OWNER_TYPES`
   - Do NOT add to `SYSTEM_GENERATED_TYPES`

2. **`opencontext/server/routes/agents.py`**
   - Extend `BaseEventItem` model (add `event_time_start`, `event_time_end`, `hierarchy_level`, `children`)
   - Rewrite `push_base_events` handler: recursive validate → flatten → build refs → batch write
   - Update `list_base_events` to query all `AGENT_BASE_*` types, add `hierarchy_level` filter
   - Update `delete_base_event` to handle all `AGENT_BASE_*` types

3. **`config/prompts_zh.yaml` + `config/prompts_en.yaml`**
   - Add descriptions for new context types if prompts enumerate context types

4. **`docs/api_reference.md` + `docs/curls.sh`**
   - Update base event API documentation with new request format, hierarchy examples, and query parameters

### Paths to Audit (AGENT_EVENT + __base__ → AGENT_BASE_EVENT)

5. Memory cache — any base-event-related cache logic
6. Search / retrieval — any filters involving base events

### No Changes Required

- Vector DB backends (Qdrant/VikingDB) — `hierarchy_level` and `refs` fields already supported; new collections auto-created
- `batch_upsert_processed_context` — generic method, type-agnostic
- User-level `AGENT_EVENT` write/query paths — unaffected
- `HierarchySummaryTask` — does not process `AGENT_BASE_*` types

## Backward Compatibility

- Requests with no `hierarchy_level` or `children` fields behave identically to the current API
- The context type migration from `AGENT_EVENT` to `AGENT_BASE_EVENT` for base events affects stored data queries. No old data compatibility is needed (confirmed by user)
