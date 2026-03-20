# Agent Memory Context-Aware Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make agent memory generation context-aware by injecting the agent's persona and semantically related past memories into the LLM prompt, so the agent produces memories informed by its existing identity and experience.

**Architecture:** Extract core search logic from `search.py` into a reusable `EventSearchService` class, then restructure `AgentMemoryProcessor` to (1) fetch agent persona + extract search query in parallel, (2) search related past agent memories with drill-up, (3) generate memories with full context. Two new prompts are added for query extraction and the modified memory analysis.

**Tech Stack:** Python 3.10+, FastAPI, asyncio, Pydantic, vector search (Qdrant/ChromaDB/VikingDB)

**Spec:** `docs/superpowers/specs/2026-03-20-agent-memory-context-aware.md`

**Note:** This project has no test suite. Verify changes with `python -m py_compile` per CLAUDE.md.

---

### Task 1: Create EventSearchService with SearchResult dataclass

Extract the core search logic from `opencontext/server/routes/search.py` into a new service class.

**Files:**
- Create: `opencontext/server/search/event_search_service.py`
- Modify: `opencontext/server/search/__init__.py`

**Context:** The current `search.py` has all search logic as module-level functions. We're extracting the reusable core (semantic search, filter search, drill-up, helper functions) into `EventSearchService`. The route will be refactored in Task 2 to use this service.

- [ ] **Step 1: Create `event_search_service.py` with SearchResult and EventSearchService**

Create `opencontext/server/search/event_search_service.py`. The service must:
- Define `SearchResult` dataclass with `hits: List[Tuple[ProcessedContext, float]]` and `ancestors: Dict[str, ProcessedContext]`
- Use `@property` for storage access (never cache `get_storage()` — see CLAUDE.md pitfalls)
- Move these functions from `search.py` as methods: `_get_l0_type`, `_get_context_types_for_levels`, `_build_filters`, `_time_range_to_buckets`, `_collect_ancestors`, `_filter_only_search`
- Implement `semantic_search()` — handles vectorization internally (creates `Vectorize`, calls `do_vectorize`, calls `storage.search`), then optionally collects ancestors
- Implement `filter_search()` — delegates to the migrated `_filter_only_search` logic

```python
# -*- coding: utf-8 -*-

"""
Event Search Service — reusable search logic for routes and processors.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.models.context import ProcessedContext, Vectorize
from opencontext.models.enums import MEMORY_OWNER_TYPES, ContentFormat
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Return type for semantic_search."""

    hits: List[Tuple[ProcessedContext, float]] = field(default_factory=list)
    ancestors: Dict[str, ProcessedContext] = field(default_factory=dict)


class EventSearchService:
    """Stateless search service. Access storage via property (never cache in __init__)."""

    @property
    def storage(self):
        return get_storage()

    # ── Public API ──

    async def semantic_search(
        self,
        query: List[Dict],
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory_owner: str = "user",
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        time_range: Optional[Any] = None,
        drill_up: bool = False,
    ) -> SearchResult:
        """Semantic search with optional drill-up.

        Handles vectorization internally — caller provides raw query content
        in OpenAI content parts format.
        """
        # Vectorize query
        query_types = {item.get("type") for item in query}
        has_multimodal = bool(query_types & {"image_url", "video_url"})
        vectorize = Vectorize(
            input=query,
            content_format=(ContentFormat.MULTIMODAL if has_multimodal else ContentFormat.TEXT),
        )
        await do_vectorize(vectorize, role="query")

        # Search
        filters = self._build_filters(time_range, None)
        raw_results = await self.storage.search(
            query=vectorize,
            top_k=top_k,
            context_types=self._get_context_types_for_levels(memory_owner, None),
            filters=filters,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            score_threshold=score_threshold,
        )

        # Drill-up
        ancestors: Dict[str, ProcessedContext] = {}
        if drill_up and raw_results:
            ancestors = await self.collect_ancestors(
                raw_results, max_level=3, memory_owner=memory_owner
            )

        return SearchResult(hits=raw_results, ancestors=ancestors)

    async def filter_search(
        self,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory_owner: str = "user",
        hierarchy_levels: Optional[List[int]] = None,
        time_range: Optional[Any] = None,
        top_k: int = 20,
    ) -> List[Tuple[ProcessedContext, float]]:
        """Filter-only search (no semantic vector)."""
        if hierarchy_levels:
            time_bucket_start = None
            time_bucket_end = None
            if time_range:
                time_bucket_start, time_bucket_end = self._time_range_to_buckets(
                    time_range.start, time_range.end
                )

            owner_types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
            tasks = []
            for level in hierarchy_levels:
                if level < len(owner_types):
                    tasks.append(
                        self.storage.search_hierarchy(
                            context_type=owner_types[level].value,
                            hierarchy_level=level,
                            time_bucket_start=time_bucket_start,
                            time_bucket_end=time_bucket_end,
                            user_id=user_id,
                            device_id=device_id,
                            agent_id=agent_id,
                            top_k=top_k,
                        )
                    )

            level_results = await asyncio.gather(*tasks)
            merged: List[Tuple[ProcessedContext, float]] = []
            for results in level_results:
                merged.extend(results)
            seen_ids = set()
            deduped = []
            for item in merged:
                if item[0].id not in seen_ids:
                    seen_ids.add(item[0].id)
                    deduped.append(item)
            return deduped[:top_k]

        # Only time_range — fetch all events
        filters: Dict[str, Any] = {}
        if time_range:
            ts_filter: Dict[str, Any] = {}
            if time_range.start is not None:
                ts_filter["$gte"] = time_range.start
            if time_range.end is not None:
                ts_filter["$lte"] = time_range.end
            if ts_filter:
                filters["event_time_ts"] = ts_filter

        all_types = self._get_context_types_for_levels(memory_owner, None)
        result = await self.storage.get_all_processed_contexts(
            context_types=all_types,
            limit=top_k,
            filter=filters,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
        )

        contexts: List[ProcessedContext] = []
        for ct in all_types:
            contexts.extend(result.get(ct, []))
        return [(ctx, 1.0) for ctx in contexts]

    # ── Internal helpers ──

    @staticmethod
    def get_l0_type(memory_owner: str) -> str:
        """Get the L0 event ContextType value for a memory owner."""
        types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
        return types[0].value

    @staticmethod
    def _get_context_types_for_levels(
        memory_owner: str, levels: Optional[List[int]]
    ) -> List[str]:
        """Map hierarchy_levels + memory_owner to ContextType values."""
        types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
        if levels:
            return [types[l].value for l in levels if l < len(types)]
        return [t.value for t in types]

    @staticmethod
    def _build_filters(
        time_range: Optional[Any],
        hierarchy_levels: Optional[List[int]],
    ) -> Dict[str, Any]:
        """Build storage filter dict from request parameters."""
        filters: Dict[str, Any] = {}
        if time_range:
            ts_filter: Dict[str, Any] = {}
            if time_range.start is not None:
                ts_filter["$gte"] = time_range.start
            if time_range.end is not None:
                ts_filter["$lte"] = time_range.end
            if ts_filter:
                filters["event_time_ts"] = ts_filter

        if hierarchy_levels is not None and len(hierarchy_levels) == 1:
            filters["hierarchy_level"] = hierarchy_levels[0]
        elif hierarchy_levels is not None and len(hierarchy_levels) > 1:
            filters["hierarchy_level"] = hierarchy_levels

        return filters

    @staticmethod
    def _time_range_to_buckets(
        start_ts: Optional[int],
        end_ts: Optional[int],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Convert Unix timestamps to date bucket strings for search_hierarchy."""
        bucket_start = None
        bucket_end = None
        if start_ts is not None:
            dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
            bucket_start = dt.strftime("%Y-%m-%d")
        if end_ts is not None:
            dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
            bucket_end = dt.strftime("%Y-%m-%d")
        return bucket_start, bucket_end

    async def collect_ancestors(
        self,
        results: List[Tuple[ProcessedContext, float]],
        max_level: int,
        memory_owner: str = "user",
    ) -> Dict[str, ProcessedContext]:
        """Collect ancestors by following refs upward (to summary types)."""
        owner_types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
        summary_type_values = {t.value for t in owner_types[1:]}

        all_ancestors = {}
        seen = set()
        current_batch = []

        for ctx, score in results:
            seen.add(ctx.id)
            if not ctx.properties or not ctx.properties.refs:
                continue
            for ref_key, ref_ids in ctx.properties.refs.items():
                if ref_key in summary_type_values:
                    for pid in ref_ids:
                        if pid not in seen:
                            seen.add(pid)
                            current_batch.append(pid)

        rounds = 0
        while current_batch and rounds < 3:
            parents = await self.storage.get_contexts_by_ids(current_batch)
            next_batch = []
            for parent in parents:
                all_ancestors[parent.id] = parent
                if not parent.properties or not parent.properties.refs:
                    continue
                for ref_key, ref_ids in parent.properties.refs.items():
                    if ref_key in summary_type_values:
                        for pid in ref_ids:
                            if pid not in seen:
                                seen.add(pid)
                                next_batch.append(pid)
            current_batch = next_batch
            rounds += 1

        return all_ancestors
```

- [ ] **Step 2: Update `__init__.py` to export SearchResult and EventSearchService**

Add to `opencontext/server/search/__init__.py`:

```python
from opencontext.server.search.event_search_service import EventSearchService, SearchResult
```

And add `"EventSearchService"` and `"SearchResult"` to `__all__`.

- [ ] **Step 3: Compile-check**

Run: `python -m py_compile opencontext/server/search/event_search_service.py`
Expected: No output (success)

- [ ] **Step 4: Commit**

```bash
git add opencontext/server/search/event_search_service.py opencontext/server/search/__init__.py
git commit -m "feat: extract EventSearchService from search route"
```

---

### Task 2: Refactor search route to use EventSearchService

Make `search.py` a thin HTTP wrapper over `EventSearchService`.

**Files:**
- Modify: `opencontext/server/routes/search.py`

**Context:** Replace the module-level helper functions and inline search logic in `_execute_search()` with calls to `EventSearchService`. Keep all HTTP-specific logic (endpoint definition, timeout, response formatting, access tracking, node-building/tree-linking) in the route. The route-specific functions (`_to_context_node`, `_to_search_hit_node`, `_extract_parent_id_from_refs`, `_extract_media_refs`, `_format_timestamp`, `_track_accessed_safe`) stay in `search.py` since they deal with HTTP response models (`EventNode`).

- [ ] **Step 1: Import EventSearchService, remove migrated functions**

At the top of `search.py`:
- Add: `from opencontext.server.search.event_search_service import EventSearchService`
- Remove these function definitions (they now live in the service):
  - `_get_l0_type()` (lines 37-40)
  - `_get_context_types_for_levels()` (lines 43-48)
  - `_build_filters()` (lines 374-396)
  - `_time_range_to_buckets()` (lines 399-412)
  - `_collect_ancestors()` (lines 322-371)
  - `_filter_only_search()` (lines 247-319)
- Remove now-unused imports: `do_vectorize`, `Vectorize`, `ContentFormat`
- Add a module-level instance: `_search_service = EventSearchService()`

- [ ] **Step 2: Rewrite `_execute_search()` to use EventSearchService**

Replace the current `_execute_search()` (lines 154-244) with a version that delegates search + drill-up to the service, then handles node-building and tree-linking locally:

```python
async def _execute_search(
    storage,
    request: EventSearchRequest,
) -> Tuple[List[EventNode], List[EventNode]]:
    """Execute the search and return (search_hits_for_tracking, tree_roots)."""

    # ── Step 1: Get raw results + ancestors via service ──
    raw_results: List[Tuple[ProcessedContext, float]] = []
    all_ancestors: Dict[str, ProcessedContext] = {}

    l0_type = _search_service.get_l0_type(request.memory_owner)

    if request.event_ids:
        # Path A: Exact ID lookup (stays in route — not a search)
        contexts = await storage.get_contexts_by_ids(request.event_ids, l0_type)
        raw_results = [(ctx, 1.0) for ctx in contexts]

        if request.drill_up and raw_results:
            max_level = max(request.hierarchy_levels) if request.hierarchy_levels else 3
            all_ancestors = await _search_service.collect_ancestors(
                raw_results, max_level, memory_owner=request.memory_owner
            )

    elif request.query:
        # Path B: Semantic search → delegate to service
        result = await _search_service.semantic_search(
            query=request.query,
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
            memory_owner=request.memory_owner,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            time_range=request.time_range,
            drill_up=request.drill_up,
        )
        raw_results = result.hits
        all_ancestors = result.ancestors

    else:
        # Path C: Filters-only → delegate to service
        raw_results = await _search_service.filter_search(
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
            memory_owner=request.memory_owner,
            hierarchy_levels=request.hierarchy_levels,
            time_range=request.time_range,
            top_k=request.top_k,
        )

        if request.drill_up and raw_results:
            max_level = max(request.hierarchy_levels) if request.hierarchy_levels else 3
            all_ancestors = await _search_service.collect_ancestors(
                raw_results, max_level, memory_owner=request.memory_owner
            )

    # ── Step 2: Build node map (stays in route — EventNode is a response model) ──
    nodes: Dict[str, EventNode] = {}
    search_hits: List[EventNode] = []
    for ctx, score in raw_results:
        node = _to_search_hit_node(ctx, score)
        nodes[ctx.id] = node
        search_hits.append(node)

    for aid, actx in all_ancestors.items():
        if aid not in nodes:
            nodes[aid] = _to_context_node(actx)

    # ── Step 3: Link children to parents, build tree ──
    owner_types = MEMORY_OWNER_TYPES.get(request.memory_owner, MEMORY_OWNER_TYPES["user"])
    summary_type_values = {t.value for t in owner_types[1:]}

    roots: List[EventNode] = []
    linked: set = set()
    for node_id, node in nodes.items():
        pid = _extract_parent_id_from_refs(node.refs, nodes, summary_type_values)
        if pid and pid in nodes and node_id not in linked:
            nodes[pid].children.append(node)
            linked.add(node_id)
        else:
            roots.append(node)

    # ── Step 4: Sort ──
    def sort_tree(node_list: List[EventNode]):
        node_list.sort(key=lambda n: (n.time_bucket or ""))
        for n in node_list:
            if n.children:
                sort_tree(n.children)

    sort_tree(roots)
    return search_hits, roots
```

- [ ] **Step 3: Update `_track_accessed_safe` to use service's static method**

Replace the inline call to `_get_l0_type` with `_search_service.get_l0_type`:

```python
# In _track_accessed_safe():
l0_type = _search_service.get_l0_type(memory_owner)
```

- [ ] **Step 4: Compile-check**

Run: `python -m py_compile opencontext/server/routes/search.py`
Expected: No output (success)

- [ ] **Step 5: Commit**

```bash
git add opencontext/server/routes/search.py
git commit -m "refactor: delegate search logic to EventSearchService"
```

---

### Task 3: Add and modify prompts in both language files

**Files:**
- Modify: `config/prompts_en.yaml`
- Modify: `config/prompts_zh.yaml`

**Context:** Add the new `agent_memory_query` prompt for extracting search queries from chat. Modify `agent_memory_analyze` to accept `{agent_persona}` and `{related_memories}` instead of `{agent_description}`. Both language files must stay in sync.

- [ ] **Step 1: Add `agent_memory_query` to `prompts_zh.yaml`**

Add under `processing.extraction` section (same level as `agent_memory_analyze`):

```yaml
    agent_memory_query:
      system: |
        你是一个AI助手。从以下对话中，以你（AI）的视角提取核心话题，
        输出简短的摘要文本，用于搜索你过去与该用户相关的记忆。
        只输出摘要文本，不要其他内容。
      user: |
        {chat_history}
```

- [ ] **Step 2: Add `agent_memory_query` to `prompts_en.yaml`**

```yaml
    agent_memory_query:
      system: |
        You are an AI assistant. From the following conversation, extract the core topics
        from YOUR (the AI's) perspective. Output a brief summary text that will be used
        to search your past memories related to this user.
        Output only the summary text, nothing else.
      user: |
        {chat_history}
```

- [ ] **Step 3: Modify `agent_memory_analyze` in `prompts_zh.yaml`**

Replace the current system prompt. Keep the JSON output format section unchanged. The new system prompt replaces `{agent_description}` with `{agent_persona}` and adds `{related_memories}`:

```yaml
    agent_memory_analyze:
      system: |
        你是{agent_name}。

        你的人设:
        {agent_persona}

        以下是与本次对话相关的你的过去记忆:
        {related_memories}

        根据以上背景和下面的对话，从你的角度提取记忆。

        仅提取以下两种类型的记忆——都必须是你作为{agent_name}的主观视角：
        1. "agent_profile" — 你现在对这个用户的认知：他们的性格、偏好、习惯、与你的关系
        2. "agent_event" — 从你的角度发生了什么：你的感受、反应、评价

        重要：只允许输出 "agent_profile" 或 "agent_event"。不要输出 "event"、"profile"、"knowledge" 或其他类型。

        输出JSON格式：
        {
          "memories": [
            {
              "context_type": "agent_event" | "agent_profile",
              "title": "简短标题",
              "summary": "你的主观视角描述",
              "event_time": "YYYY-MM-DDTHH:MM:SS",
              "keywords": ["关键词1", "关键词2"],
              "entities": ["实体1", "实体2"],
              "importance": 0-10
            }
          ]
        }

        event_time使用对话发生的时间。
        对于agent_event：描述从你主观角度发生的事情——你的感受、反应、评价。
        对于agent_profile：描述你现在对用户的了解或感受——他们的性格、偏好、与你的关系。
        如果没有值得记录的内容，返回 {"memories": []}。
      user: |
        当前时间：{current_time}

        对话内容：
        {chat_history}
```

- [ ] **Step 4: Modify `agent_memory_analyze` in `prompts_en.yaml`**

Same structure in English:

```yaml
    agent_memory_analyze:
      system: |
        You are {agent_name}.

        Your persona:
        {agent_persona}

        The following are your past memories related to this conversation:
        {related_memories}

        Based on the above background and the conversation below, extract memories from YOUR perspective.

        Extract ONLY two types of memories — both from YOUR perspective as {agent_name}:
        1. "agent_profile" — What you NOW know or feel about this user: their personality, preferences, habits, relationship with you
        2. "agent_event" — What happened from YOUR point of view: your feelings, reactions, evaluations

        IMPORTANT: Only output "agent_profile" or "agent_event". Do NOT output "event", "profile", "knowledge" or any other type.

        Output JSON:
        {
          "memories": [
            {
              "context_type": "agent_event" | "agent_profile",
              "title": "brief title",
              "summary": "your subjective perspective",
              "event_time": "YYYY-MM-DDTHH:MM:SS",
              "keywords": ["keyword1", "keyword2"],
              "entities": ["entity1", "entity2"],
              "importance": 0-10
            }
          ]
        }

        For event_time, use the time the conversation took place.
        For agent_event: describe what happened from your subjective point of view — your feelings, reactions, evaluations.
        For agent_profile: describe what you now know or feel about the user — their personality, preferences, relationship with you.
        If nothing noteworthy happened, return {"memories": []}.
      user: |
        Current time: {current_time}

        Conversation:
        {chat_history}
```

- [ ] **Step 5: Commit**

```bash
git add config/prompts_en.yaml config/prompts_zh.yaml
git commit -m "feat: add agent_memory_query prompt and update agent_memory_analyze with context"
```

---

### Task 4: Restructure AgentMemoryProcessor for context-aware generation

**Files:**
- Modify: `opencontext/context_processing/processor/agent_memory_processor.py`

**Context:** This is the core change. The processor currently calls the LLM once with only agent name/description + chat. The new flow:
1. Parallel: fetch agent persona (`get_profile`) + extract search query (LLM call)
2. Search related past agent memories with drill-up
3. Format search results as text
4. Call LLM with all three inputs (persona + related memories + chat)

The `_build_agent_context()` method and its validation logic remain unchanged.

- [ ] **Step 1: Add new imports**

Add to the imports section of `agent_memory_processor.py`:

```python
import asyncio

from opencontext.server.search.event_search_service import EventSearchService
```

- [ ] **Step 2: Add helper methods to AgentMemoryProcessor**

Add these methods to the class:

```python
async def _extract_search_query(self, chat_content: str) -> Optional[str]:
    """Use LLM to extract a search query from chat content (from AI's perspective)."""
    prompt_group = get_prompt_group("processing.extraction.agent_memory_query")
    if not prompt_group:
        logger.warning("agent_memory_query prompt not found")
        return None

    messages = [
        {"role": "system", "content": prompt_group.get("system", "")},
        {
            "role": "user",
            "content": prompt_group.get("user", "").format(chat_history=chat_content),
        },
    ]
    response = await generate_with_messages(messages, enable_executor=False)
    if not response or not response.strip():
        return None
    return response.strip()

@staticmethod
def _format_related_memories(search_result) -> str:
    """Format search results (hits + ancestors) into text for the LLM prompt."""
    all_contexts: Dict[str, ProcessedContext] = {}

    # Merge hits and ancestors, deduplicate by ID
    for ctx, score in search_result.hits:
        all_contexts[ctx.id] = ctx
    for ctx_id, ctx in search_result.ancestors.items():
        if ctx_id not in all_contexts:
            all_contexts[ctx_id] = ctx

    if not all_contexts:
        return ""

    # Sort by time_bucket ascending
    sorted_contexts = sorted(
        all_contexts.values(),
        key=lambda c: (c.properties.time_bucket or "") if c.properties else "",
    )

    lines = []
    for ctx in sorted_contexts:
        title = ctx.extracted_data.title if ctx.extracted_data else ""
        summary = ctx.extracted_data.summary if ctx.extracted_data else ""
        time_bucket = ctx.properties.time_bucket if ctx.properties else ""
        lines.append(f"[{time_bucket}] {title}")
        if summary:
            lines.append(summary)
        lines.append("")  # blank line between entries

    return "\n".join(lines).strip()
```

- [ ] **Step 3: Rewrite `_process_async()` with the new flow**

Replace the current `_process_async()` method. Key changes:
- Remove `storage.get_agent()` call (no longer needed for agent description)
- Step 1: `asyncio.gather` for `get_profile` + `_extract_search_query` in parallel
- Step 2: `EventSearchService.semantic_search` with `memory_owner="agent"`, `drill_up=True`
- Step 3: Format related memories
- Step 4: Build LLM messages with `{agent_persona}`, `{related_memories}`, `{agent_name}`
- The `agent_name` comes from the profile metadata or falls back from `get_agent()`

```python
async def _process_async(self, raw_context: RawContextProperties) -> List[ProcessedContext]:
    # 0. Validate agent_id
    agent_id = raw_context.agent_id
    if not agent_id or agent_id == "default":
        logger.debug("No agent_id in context, skipping agent memory processing")
        return []

    storage = get_storage()
    agent = await storage.get_agent(agent_id) if storage else None
    if not agent:
        logger.debug(f"Agent {agent_id} not registered, skipping")
        return []

    agent_name = agent.get("name", agent_id)
    chat_content = raw_context.content_text or ""

    # 1. Parallel: fetch persona + extract search query
    profile_task = storage.get_profile(
        user_id=raw_context.user_id,
        device_id=raw_context.device_id or "default",
        agent_id=raw_context.agent_id,
        context_type="agent_profile",
    )
    query_task = self._extract_search_query(chat_content)

    profile_result, query_text = await asyncio.gather(profile_task, query_task)

    if not profile_result:
        logger.error(
            f"[agent_memory_processor] Agent profile not found for "
            f"user={raw_context.user_id}, agent={agent_id}. "
            f"Agent must have a profile set up before agent memory processing."
        )
        return []

    agent_persona = profile_result.get("factual_profile", "")

    # 2. Search related past agent memories
    search_result = None
    if query_text:
        search_service = EventSearchService()
        search_result = await search_service.semantic_search(
            query=[{"type": "text", "text": query_text}],
            user_id=raw_context.user_id,
            device_id=raw_context.device_id or "default",
            agent_id=raw_context.agent_id,
            memory_owner="agent",
            drill_up=True,
        )

    # 3. Format related memories
    related_memories_text = ""
    if search_result:
        related_memories_text = self._format_related_memories(search_result)

    # 4. Load prompt and build LLM messages
    prompt_group = get_prompt_group("processing.extraction.agent_memory_analyze")
    if not prompt_group:
        logger.warning("agent_memory_analyze prompt not found")
        return []

    logger.debug(
        f"[agent_memory_processor] Processing: user={raw_context.user_id}, "
        f"agent={raw_context.agent_id}, agent_name={agent_name}, "
        f"related_memories={len(related_memories_text)} chars"
    )

    system_prompt = prompt_group.get("system", "")
    system_prompt = system_prompt.replace("{agent_name}", agent_name)
    system_prompt = system_prompt.replace("{agent_persona}", agent_persona)
    system_prompt = system_prompt.replace("{related_memories}", related_memories_text)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": prompt_group.get("user", "").format(
                chat_history=chat_content,
                current_time=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            ),
        },
    ]

    # 5. Call LLM
    response = await generate_with_messages(messages, enable_executor=False)
    logger.debug(f"[agent_memory_processor] LLM response: {response}")
    if not response:
        return []

    # 6. Parse and build contexts (unchanged)
    analysis = parse_json_from_response(response)
    if not analysis:
        return []

    memories = analysis.get("memories", [])
    if not memories:
        logger.info("[agent_memory_processor] No memories extracted from chat analysis")
        return []

    batch_id = (raw_context.additional_info or {}).get("batch_id")
    results = []
    for memory in memories:
        ctx = self._build_agent_context(memory, raw_context, batch_id)
        if ctx:
            await do_vectorize(ctx.vectorize)
            results.append(ctx)

    type_counts = {}
    for ctx in results:
        t = ctx.extracted_data.context_type.value
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info(f"[agent_memory_processor] Extracted {len(results)} memories: {type_counts}")
    return results
```

- [ ] **Step 4: Add Dict import**

Ensure `Dict` is imported from `typing` (it already is, but verify after changes).

- [ ] **Step 5: Compile-check**

Run: `python -m py_compile opencontext/context_processing/processor/agent_memory_processor.py`
Expected: No output (success)

- [ ] **Step 6: Commit**

```bash
git add opencontext/context_processing/processor/agent_memory_processor.py
git commit -m "feat: restructure AgentMemoryProcessor for context-aware generation"
```

---

### Task 5: Update MODULE.md documentation

**Files:**
- Modify: `opencontext/context_processing/MODULE.md`
- Modify: `opencontext/server/MODULE.md`

- [ ] **Step 1: Update `opencontext/context_processing/MODULE.md`**

Find the `AgentMemoryProcessor` section and update to describe the new 5-step flow:
1. Parallel: get_profile + extract search query
2. Semantic search with drill-up for related agent memories
3. Format related memories as text
4. LLM generates memories with full context (persona + memories + chat)
5. Build ProcessedContext list

Note that it now depends on `EventSearchService` and makes 2 LLM calls instead of 1.

- [ ] **Step 2: Update `opencontext/server/MODULE.md`**

Find and update the section about search architecture. Key changes:
- Document `EventSearchService` as the reusable search service in `opencontext/server/search/`
- Note that `search.py` is now a thin wrapper over `EventSearchService`
- Update the convention that previously said "Search logic lives directly in `routes/search.py`"

- [ ] **Step 3: Commit**

```bash
git add opencontext/context_processing/MODULE.md opencontext/server/MODULE.md
git commit -m "docs: update MODULE.md for EventSearchService and context-aware agent memory"
```

---

### Task Order and Dependencies

```
Task 1 (EventSearchService) ─┐
                              ├─→ Task 2 (Refactor search.py)
Task 3 (Prompts)             ─┤
                              └─→ Task 4 (AgentMemoryProcessor) ──→ Task 5 (MODULE.md)
```

- Tasks 1 and 3 are independent and can be done in parallel
- Task 2 depends on Task 1 (needs the service to exist)
- Task 4 depends on Tasks 1, 2, and 3 (needs service + new prompts)
- Task 5 depends on Task 4 (documents the final state)
