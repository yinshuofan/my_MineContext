# Search Events API Design

**Date**: 2026-03-04
**Status**: Implemented

## Problem

The existing `/api/search` endpoint searched across all 5 context types (profile, entity, document, event, knowledge) with two strategy modes (fast, intelligent). This was overly broad for the primary use case of event retrieval with hierarchy context.

## Solution

Replace `/api/search` with an event-only search endpoint that supports:

1. **Three search paths**: semantic query, event ID lookup, or filter-only browsing
2. **Upward hierarchy drill-up**: recursively fetches ancestor summaries (L0->L1->L2->L3)
3. **Flexible filtering**: by time range, hierarchy level, or both
4. **No strategy selection**: single direct implementation, no LLM reasoning calls

## API

### Request: `EventSearchRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | `str` | No* | Semantic search text |
| `event_ids` | `List[str]` | No* | Exact ID lookup |
| `time_range` | `{start, end}` | No* | Unix timestamps |
| `hierarchy_levels` | `List[int]` | No* | Filter by level [0,1,2,3] |
| `drill_up` | `bool` | No | Fetch ancestor chain (default: true) |
| `top_k` | `int` | No | Max results (default: 20) |
| `user_id/device_id/agent_id` | `str` | No | User identifiers |

*At least one of query/event_ids/time_range/hierarchy_levels must be provided.

### Response: `EventSearchResponse`

```json
{
  "success": true,
  "events": [
    {
      "id": "...",
      "title": "...",
      "summary": "...",
      "content": "...",
      "score": 0.92,
      "hierarchy_level": 0,
      "parent_id": "...",
      "ancestors": [
        {"id": "...", "hierarchy_level": 1, "summary": "..."},
        {"id": "...", "hierarchy_level": 2, "summary": "..."}
      ]
    }
  ],
  "metadata": {"query": "...", "total_results": 1, "search_time_ms": 402.5}
}
```

## Search Logic

Priority: `event_ids` > `query` > filters-only.

### Drill-up Algorithm

- Batch-iterative: collect all parent_ids per round, fetch in one call
- Max 3 rounds (L0->L1->L2->L3)
- Shared `seen` cache across results (sibling L0 events share L1 parent)
- Respects `max_level = max(hierarchy_levels)` cap

## Files Changed

| File | Change |
|------|--------|
| `opencontext/server/search/models.py` | Rewritten with new models |
| `opencontext/server/routes/search.py` | Rewritten with new handler |
| `opencontext/server/search/__init__.py` | Updated exports |
| `opencontext/server/search/base_strategy.py` | Deleted |
| `opencontext/server/search/fast_strategy.py` | Deleted |
| `opencontext/server/search/intelligent_strategy.py` | Deleted |
| `docs/curls.sh` | Updated search examples |
| `opencontext/server/MODULE.md` | Updated documentation |
