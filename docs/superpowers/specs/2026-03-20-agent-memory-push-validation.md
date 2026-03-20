# Agent Memory Push Validation

**Date:** 2026-03-20
**Status:** Approved

## Problem

When `POST /api/push/chat` is called with `processors: ["agent_memory"]` and an `agent_id` that hasn't been registered via `/api/agents`, the agent memory processor silently skips processing (`logger.debug` + return `[]`). The caller receives a 202 response with no indication that agent memory extraction was skipped.

## Design

Add a synchronous validation check in `push_chat()` (in `opencontext/server/routes/push.py`) before chat batch creation.

### Validation Logic

**Trigger conditions** (all three must be true):
1. `"agent_memory"` is in `request.processors`
2. `request.agent_id` is not `None`
3. `request.agent_id != "default"`

**When triggered:** Call `storage.get_agent(request.agent_id)`. If it returns `None`, raise `HTTPException(status_code=400)` with a message directing the user to register the agent first.

**When not triggered:** If `agent_id` is `"default"` or absent, the agent memory processor's existing internal skip logic handles it silently — this is expected behavior, not an error.

### Insertion Point

After existing `user_id` validation (the `__base__` reserved check), before chat batch persistence and background task dispatch.

### Error Response

```json
{
  "detail": "Agent '<agent_id>' is not registered. Please register the agent via POST /api/agents before using agent_memory processor."
}
```

HTTP status: **400 Bad Request**.

### What Does NOT Change

- `AgentMemoryProcessor._process_async()` internal validation remains as defensive fallback
- Other processors' behavior is unaffected
- Response format uses FastAPI's default `HTTPException` JSON structure

## Files to Modify

| File | Change |
|------|--------|
| `opencontext/server/routes/push.py` | Add agent existence check in `push_chat()` |
