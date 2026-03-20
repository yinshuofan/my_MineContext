# Agent Memory Context-Aware Generation

## Problem

The `AgentMemoryProcessor` currently generates agent memories in a stateless manner — the LLM only receives the agent's name/description and the current chat messages. It has no access to the agent's existing memories or persona, resulting in memories that lack continuity and depth. Agent memories should be generated like a real person's: informed by who they are (persona) and what they remember (past experiences).

## Design

### Overview

Three changes:
1. **Extract search logic into a shared service** (`EventSearchService`) so both the search route and the processor can use semantic search with drill-up.
2. **Restructure `AgentMemoryProcessor`** to gather context (persona + related past memories) before generating new memories.
3. **Update prompts** to accept the new context inputs.

### 1. EventSearchService — Search Logic Extraction

**New file:** `opencontext/server/search/event_search_service.py`

Extract core search logic from `opencontext/server/routes/search.py` into a stateless service class. The route becomes a thin HTTP wrapper.

**Class design:**

```python
@dataclass
class SearchResult:
    """Return type for semantic_search."""
    hits: List[Tuple[ProcessedContext, float]]       # (context, score) search matches
    ancestors: Dict[str, ProcessedContext]            # drill-up parent summaries (id → context)

class EventSearchService:
    """Stateless service. Access storage via get_storage() property (never cache in __init__)."""

    @property
    def storage(self):
        return get_storage()

    async def semantic_search(
        self,
        query: List[Dict],          # OpenAI content parts format
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        memory_owner: str = "user",
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        time_range: Optional[Dict] = None,
        drill_up: bool = False,
    ) -> SearchResult:
        """Semantic search with optional drill-up.
        Handles vectorization internally — caller provides raw query content."""
        ...

    async def filter_search(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        memory_owner: str = "user",
        hierarchy_levels: Optional[List[int]] = None,
        time_range: Optional[Dict] = None,
        top_k: int = 20,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Filter-only search (no semantic vector)."""
        ...
```

**Migrated helpers** (from search.py to service):
- `_collect_ancestors()` — drill-up logic
- `_build_filters()` — filter construction
- `_get_context_types_for_levels()` — context type resolution by memory_owner
- `_get_l0_type()` — L0 type resolution

**Route change:** `search.py` delegates to `EventSearchService`, handling only HTTP request/response.

### 2. AgentMemoryProcessor Restructure

**New processing flow** (was: steps 4-5 only):

```
1. Parallel: gather context (asyncio.gather)
   ├─ 1a. Get agent persona
   │      storage.get_profile(context_type="agent_profile") → factual_profile
   └─ 1b. Extract query from chat (new LLM call)
          messages → LLM (agent_memory_query prompt) → query_text

2. Search related past memories (new, depends on 1b)
   query_text → wrap as [{"type": "text", "text": query_text}]
   → EventSearchService.semantic_search(
       memory_owner="agent", drill_up=True
   ) → related agent_events + hierarchy summaries

4. Format related memories as text
   ProcessedContext list → text, sorted by time_bucket ascending, each entry:
   ```
   [{time_bucket}] {title}
   {summary}
   ```
   Hits and ancestors are merged and deduplicated before formatting.

5. Generate memories (modified LLM call)
   LLM receives:
   - {agent_persona}: factual_profile content
   - {related_memories}: formatted search results
   - {chat_history}: push messages (existing)
   → memories JSON (output format unchanged)

6. Build ProcessedContext list (unchanged)
   → agent_profile → relational DB
   → agent_event → vector DB
```

**Search scope:** Scoped to current user — `user_id` is passed to `semantic_search` so the agent only recalls memories from interactions with this specific user. Cross-user memory is not included.

**Error handling:**
- No profile → error (not handled, by design — agent must have a profile set up before agent memory processing)
- No search results → continue with empty `{related_memories}` (degrades gracefully to current behavior)

### 3. Prompt Changes

**New prompt** — `processing.extraction.agent_memory_query`:

```yaml
agent_memory_query:
  system: |
    你是一个AI助手。从以下对话中，以你（AI）的视角提取核心话题，
    输出简短的摘要文本，用于搜索你过去与该用户相关的记忆。
    只输出摘要文本，不要其他内容。
  user: |
    {chat_history}
```

**Modified prompt** — `processing.extraction.agent_memory_analyze`:

Replace `{agent_description}` with `{agent_persona}` (from factual_profile) and add `{related_memories}`:

```yaml
agent_memory_analyze:
  system: |
    你是{agent_name}。

    你的人设:
    {agent_persona}

    以下是与本次对话相关的你的过去记忆:
    {related_memories}

    基于以上背景和下面的对话，从你的视角提取记忆...
    (output format requirements unchanged)
  user: |
    当前时间: {current_time}
    对话内容:
    {chat_history}
```

Both `prompts_en.yaml` and `prompts_zh.yaml` must be updated in sync.

### Data Flow

```
POST /api/push/chat (processors=["agent_memory"])
  │
  ▼
AgentMemoryProcessor._process_async()
  │
  ├─ 1. asyncio.gather:
  │   ├─ 1a. get_profile(context_type="agent_profile") → factual_profile
  │   └─ 1b. LLM(agent_memory_query + messages) → query_text
  │
  ├─ 2. EventSearchService.semantic_search(
  │       query=query_text, memory_owner="agent", drill_up=True
  │   ) → related_memories
  │
  ├─ 3. Format related_memories → text
  │
  ├─ 4. LLM(agent_memory_analyze
  │       + agent_persona + related_memories + messages
  │   ) → memories JSON
  │
  └─ 5. Build ProcessedContext list (unchanged)
         → agent_profile → relational DB
         → agent_event → vector DB
```

## Change Summary

| File | Change |
|------|--------|
| `opencontext/server/search/event_search_service.py` | New — core search logic extracted from search.py |
| `opencontext/server/routes/search.py` | Refactor to thin wrapper over EventSearchService |
| `opencontext/context_processing/processor/agent_memory_processor.py` | Restructure flow: add steps 1-4 |
| `config/prompts_en.yaml` | Add agent_memory_query, modify agent_memory_analyze |
| `config/prompts_zh.yaml` | Sync updates |
| `opencontext/context_processing/MODULE.md` | Update processor documentation |
| `opencontext/server/MODULE.md` | Update search architecture documentation |

## What Does NOT Change

- Push endpoint request format and agent validation logic
- ProcessedContext output structure and storage routing
- agent_profile LLM merge in `refresh_profile()`
- agent_event append-only storage
