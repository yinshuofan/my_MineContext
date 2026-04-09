# Agent Memory Processing Redesign

## Problem

The current agent memory processing has three architectural issues:

1. **TextChatProcessor produces `agent_event`** — The `chat_analyze` prompt includes `agent_event` as a classification option, but this processor has no agent persona context, so it cannot produce meaningful agent-perspective content.

2. **AgentMemoryProcessor produces `agent_profile`** — Each chat can trigger a profile overwrite via `refresh_profile()`. This means a single conversation's partial impression can rewrite the agent's entire understanding of a user. Profile should be derived from accumulated events, not extracted from individual chats.

3. **AgentMemoryProcessor produces disconnected `agent_event`** — The standalone `agent_event` records have no association (refs, shared IDs) with the `event` records they comment on. An agent memory like "the user seemed frustrated" cannot be traced back to which event it refers to.

## Design

### 1. Data Model Change

Add a field to `ExtractedData`:

```python
agent_commentary: Optional[str] = None
```

- Only populated on `event` type contexts
- Contains the agent's subjective commentary on the event, written from the agent's persona
- Does NOT participate in embedding/vectorization — the event's own content remains the embedding source
- `ProcessedContextModel` (API response model) adds the same field in `from_processed_context()`

### 2. Processor Orchestration — DAG-based Dependencies

**Current state:** `process_batch()` runs all processors in parallel via `asyncio.gather()`. No processor can consume another's output.

**Change:** Introduce a dependency configuration and topological-sort execution.

#### Dependency Configuration

```python
# processor_manager.py
PROCESSOR_DEPENDENCIES = {
    "agent_memory": ["user_memory"],
}
```

- Processors not listed have no dependencies
- Dependencies reference processor names (keys in `BATCH_PROCESSOR_MAP`)

#### Execution Model

`process_batch()` performs topological sort on the dependency graph, grouping processors into layers:

- **Layer 0**: Processors with no dependencies → run in parallel
- **Layer 1**: Processors depending only on layer 0 → run in parallel after layer 0 completes
- **Layer N**: Processors depending on layer N-1 or earlier → run after dependencies complete

Each layer's processors receive accumulated results from all previous layers via a new `prior_results` parameter:

```python
# BaseContextProcessor
async def process(
    self,
    context: RawContextProperties,
    prior_results: List[ProcessedContext] = None,
) -> List[ProcessedContext]:
```

- Processors without dependencies ignore `prior_results` (it will be `None` or empty)
- Processors with dependencies use `prior_results` to find and work with earlier outputs

#### Result Accumulation

`process_batch()` accumulates results across layers. When a processor returns contexts with the same `id` as an existing context in the accumulated results, the returned version **replaces** the existing one. New IDs are **appended**. This allows post-processors to modify prior results and return them alongside any new contexts.

### 3. AgentMemoryProcessor Rewrite

Transform from an independent extractor to a post-processor that annotates events.

#### Input

- `prior_results`: filters for `context_type == EVENT` contexts
- `RawContextProperties`: for user_id, agent_id, chat content

#### Processing Flow

1. Validate `agent_id` (skip if missing or "default")
2. Verify agent is registered via `storage.get_agent(agent_id)`
3. Fetch agent persona: `get_profile(context_type="agent_profile")`, fallback to `agent_base_profile`
4. Fetch related past memories via `EventSearchService.semantic_search()` (searches EVENT + agent base types automatically after memory_owner removal)
5. Build LLM prompt with: agent persona, related memories, event list (titles + summaries), chat content
6. LLM returns commentary for each event (keyed by event index or title)
7. Fill `agent_commentary` field on each matched event
8. Return the modified events

#### Output

Returns the modified event contexts (same IDs, with `agent_commentary` populated). Does NOT produce:
- `agent_event` type contexts
- `agent_profile` type contexts

#### Prompt Change

`agent_memory_analyze` prompt changes from "extract memories" to "write commentary on given events from your perspective". Input includes the event list; output is a mapping of event index to commentary string.

### 4. agent_profile Periodic Update

Move agent_profile updates from per-chat extraction to a scheduled task.

#### New Task Type: `agent_profile_update`

- **Trigger**: `user_activity` mode (same as `hierarchy_summary`)
- **Scheduling**: In the push endpoint, after `agent_memory` processor validation passes (agent_id is registered), schedule this task. If agent_id is not registered or is "default", do NOT schedule.
- **Execution flow**:
  1. Fetch today's events for this (user_id, device_id, agent_id) from vector DB
  2. Fetch current `agent_profile` (per-user)
  3. Fetch `agent_base_profile` (base persona — serves as anchor to prevent personality drift)
  4. Call LLM: given the base persona as constraint, today's events as new information, and current profile as baseline, produce an updated profile
  5. Call `refresh_profile(context_type="agent_profile")` to store

#### Design Rationale

- **base_profile as anchor**: The agent's personality should evolve based on interactions, but must not drift too far from the original character design. The LLM prompt should treat base_profile as a hard constraint.
- **Daily granularity**: Updates run at L1 (daily) frequency, providing a reasonable cadence for profile evolution without excessive churn.

### 5. Cleanup

#### Remove from `chat_analyze` prompt

Remove `agent_event` from the classification options in `processing.extraction.chat_analyze` (both `prompts_en.yaml` and `prompts_zh.yaml`). TextChatProcessor should not produce agent-perspective content.

#### Deprecate `AGENT_EVENT` and Related Types

Remove from all locations:
- `ContextType` enum: `AGENT_EVENT`, `AGENT_DAILY_SUMMARY`, `AGENT_WEEKLY_SUMMARY`, `AGENT_MONTHLY_SUMMARY`
- `CONTEXT_UPDATE_STRATEGIES`
- `CONTEXT_STORAGE_BACKENDS`
- `MEMORY_OWNER_TYPES`
- `ContextDescriptions`
- `ContextSimpleDescriptions`
- Both prompt files (any references)

#### Remove `memory_owner` from Search API

With `AGENT_EVENT` removed, the `memory_owner` parameter on the search API is no longer needed. Search should automatically include both user interaction events and agent base memories:

- `EVENT`, `DAILY_SUMMARY`, `WEEKLY_SUMMARY`, `MONTHLY_SUMMARY` — per-user interaction history (events now include `agent_commentary` when an agent was involved)
- `AGENT_BASE_EVENT`, `AGENT_BASE_L1/L2/L3_SUMMARY` — agent's base memories

The search endpoint combines both sets of context types in a single query. Callers no longer specify `memory_owner`.

`MEMORY_OWNER_TYPES` mapping can be removed or simplified — the search layer directly uses the combined type list.

#### Adjust Memory Cache

The memory cache (`memory_cache_manager.py`) similarly removes `memory_owner` parameterization. Snapshots include:
- Profile: `agent_profile` (with fallback to `agent_base_profile`)
- Events: `EVENT` type (recent events with `agent_commentary`)
- Summaries: `DAILY_SUMMARY` etc.
- Base memories: `AGENT_BASE_EVENT` + base summaries
- Docs/knowledge: included as before

## Migration

- Existing `agent_event` records in vector DB become orphaned. They can be left in place (searches won't match them after the type is removed from `MEMORY_OWNER_TYPES`) or cleaned up via a one-time migration script.
- Existing `agent_profile` records in the profiles table remain valid — the periodic update task will merge with them on next execution.

## Files Affected

| Area | Files |
|------|-------|
| Data model | `opencontext/models/context.py`, `opencontext/models/enums.py` |
| Processor orchestration | `opencontext/context_processing/processor/processor_manager.py`, `opencontext/context_processing/processor/base_processor.py` |
| Agent memory processor | `opencontext/context_processing/processor/agent_memory_processor.py` |
| Periodic task | New file in `opencontext/periodic_task/` for agent_profile_update |
| Push endpoint | `opencontext/server/routes/push.py` (schedule agent_profile_update) |
| Prompts | `config/prompts_en.yaml`, `config/prompts_zh.yaml` |
| Search/cache | `opencontext/server/cache/memory_cache_manager.py`, `opencontext/server/search/event_search_service.py` |
| API response model | `opencontext/models/context.py` (ProcessedContextModel) |
| MODULE.md files | `opencontext/context_processing/MODULE.md`, `opencontext/models/MODULE.md`, `opencontext/periodic_task/MODULE.md` |
