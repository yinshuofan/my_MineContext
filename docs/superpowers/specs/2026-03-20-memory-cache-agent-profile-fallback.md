# Memory Cache: Agent Profile `__base__` Fallback

**Date**: 2026-03-20
**Status**: Approved
**Scope**: Single-file change in `memory_cache_manager.py`

## Problem

When fetching agent memory cache (`memory_owner="agent"`), if a per-user agent profile hasn't been generated yet, the profile section returns empty. The `__base__` profile (set via `/api/agents/{id}/base/profile`) exists as the default but isn't used as fallback in the cache read path.

The `__base__` fallback already exists in:
- `profile_processor.py` (lines 63-66) — profile merge/update
- `agent_memory_processor.py` (lines 77-84) — memory extraction

Missing from: `memory_cache_manager.py` `_build_snapshot()` — the cache read path.

## Design

In `_build_snapshot()`, replace the profile assembly block (lines 473-483) with a three-step linear flow:

```python
# Profile — with __base__ fallback for agent memory owner
profile_data = results_map.get("profile")
if isinstance(profile_data, Exception):
    profile_data = None
if not profile_data and memory_owner == "agent":
    profile_data = await storage.get_profile(
        "__base__", device_id, agent_id, context_type=profile_context_type
    )
    if profile_data:
        logger.debug(f"[memory-cache] Using __base__ fallback profile for user={user_id}, agent={agent_id}")
if profile_data:
    snapshot["profile"] = {
        "user_id": user_id,
        "device_id": device_id,
        "agent_id": agent_id,
        "factual_profile": profile_data.get("factual_profile", ""),
        "behavioral_profile": profile_data.get("behavioral_profile"),
        "metadata": profile_data.get("metadata", {}),
    }
```

Steps:
1. **Normalize**: exception → `None` (error already logged in the loop above)
2. **Fallback**: if no per-user profile and `memory_owner == "agent"`, query `__base__`; log when used
3. **Assemble**: if data exists, build snapshot profile section — always use the requesting user's identity fields (not `__base__`'s)

## Constraints

- Fallback only triggers for `memory_owner == "agent"`, never for user memory cache
- No new fields in response — caller doesn't need to distinguish `__base__` from per-user
- `get_profile()` returns `None` on both not-found and DB errors (exceptions are caught internally by storage backends); the fallback query follows the same contract
- When fallback fires, identity fields (`user_id`, `device_id`, `agent_id`) in the snapshot are always set to the requesting user's values, not `__base__`'s
- Fallback query is sequential (after parallel batch), adding latency only when no per-user profile exists
- No concurrency concern: this is a read-only fallback query within an already-locked snapshot build

## Files Changed

| File | Change |
|------|--------|
| `opencontext/server/cache/memory_cache_manager.py` | Replace profile assembly block with 3-step flow including `__base__` fallback |
