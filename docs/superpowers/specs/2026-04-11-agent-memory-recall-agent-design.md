# Agent Memory Recall Agent — Design Spec

**Date**: 2026-04-11
**Author**: brainstormed via Claude Code
**Status**: approved, awaiting plan

## 1. Motivation

The current `AgentMemoryProcessor` (`opencontext/context_processing/processor/agent_memory_processor.py`) generates an agent's subjective commentary on each extracted event. Before generating commentary it fetches "related past memories" by:

1. Calling an LLM once to extract a search query from the chat (`_extract_search_query`, prompt `agent_memory_query`).
2. Calling `EventSearchService.search()` once with that query.
3. Formatting the hits + ancestors into a flat text blob fed to the commentary prompt as `{related_memories}`.

Three problems with this single-shot recall:

- **Recall insufficiency** — one query often misses relevant memories that a different angle would have caught.
- **No self-judged sufficiency** — the recall step runs exactly once regardless of whether one query was enough or ten would have been better.
- **Doesn't feel like human recall** — a human remembers one thing, which reminds them of another, which they then look up. The current flow has no way to "pull on a thread."

We want to replace the single-shot recall with a **multi-turn agentic loop** that plans, searches, observes, and self-terminates.

## 2. Goals and non-goals

**Goals**

- Replace the single-shot query+search with a 1-to-N turn LLM-driven loop backed by the existing `EventSearchService`.
- Let the LLM decide when it has recalled enough, subject to a hard turn cap.
- Extract the recall loop into its own module so the commentary processor stays focused.
- Allow the recall step to run on a different (typically cheaper/faster) model than the commentary step, configurable via `config.yaml` and the admin settings UI.
- Preserve the downstream commentary pipeline unchanged — the only difference the commentary prompt sees is that `{related_memories}` is richer.

**Non-goals**

- Introducing new retrieval tools. The hierarchical event search already returns `hits` + `ancestors`, which is enough to let the agent "drill up" for context. No knowledge-search or profile-search tools are added.
- Building a general-purpose agent framework (langgraph, langchain, etc.). The codebase has no LangChain ecosystem today; a 60-line hand-written loop consistent with `GlobalVLMClient.generate_with_messages`'s existing tool-call loop is simpler and has zero new dependencies.
- Letting the recall agent choose its own `base_url` / `api_key` / `provider`. Only the model name is configurable; everything else is inherited from the VLM client.
- A runtime feature flag / legacy mode toggle. The old `_extract_search_query` path is deleted. Rollback is a git revert.

## 3. Architecture overview

**Changed module**: `opencontext/context_processing/processor/agent_memory_processor.py` loses its `_extract_search_query`, `_format_related_memories`, and direct `EventSearchService` usage.

**New module**: `opencontext/context_processing/processor/recall_agent.py` contains a `RecallAgent` class that owns the multi-turn loop.

**Module boundaries**:

| Responsibility | Owner |
|---|---|
| agent_id validation, persona fetch with base fallback, event filter from `prior_results`, commentary generation LLM call, writing `agent_commentary` back onto events | `AgentMemoryProcessor` (existing file) |
| Multi-turn recall loop, action JSON parsing, `EventSearchService` calls, dedup + formatting memories text | `RecallAgent` (new file) |

**Call count comparison** (per chat push):

| Stage | Before | After (typical) | After (worst case) |
|---|---|---|---|
| Query extraction LLM | 1 | 0 | 0 |
| Recall loop LLM | 0 | 1–2 | 3 |
| Recall loop searches | 1 | 1–2 | 3 |
| Commentary LLM | 1 | 1 | 1 |
| **Total LLM** | **2** | **2–3** | **4** |
| **Total searches** | **1** | **1–2** | **3** |

Worst case is 2 extra LLM calls + 2 extra searches vs. the current flow. Typical case (agent stops after turn 2) is 1 extra LLM + 1 extra search.

## 4. The `RecallAgent` module

### Interface

```python
# opencontext/context_processing/processor/recall_agent.py

class RecallAgent:
    """Multi-turn memory recall loop driven by an LLM-selected search action."""

    def __init__(self) -> None:
        # No config cached — read fresh in recall() for hot reload
        pass

    async def recall(
        self,
        chat_content: str,
        agent_name: str,
        agent_persona: str,
        user_id: str,
        device_id: str,
        agent_id: str,
    ) -> str:
        """Run the recall loop. Returns a formatted related_memories text.

        Returns an empty string if nothing was recalled (including when the
        loop aborts on the first turn). Downstream commentary generation
        degrades gracefully on an empty string.
        """
```

### State machine

Internal per-call state:

```python
turn = 0
max_turns: int               # from config, default 3
recall_model: str | None     # from config, None = use VLM default
seen_ids: set[str] = set()
accumulated_contexts: list[ProcessedContext] = []  # insertion order
previous_actions: list[dict] = []    # for next-turn LLM context
consecutive_empty_searches = 0       # safety brake
```

Per turn:

```
1. Build messages via _build_turn_messages(chat_content, agent_name, agent_persona,
                                           previous_actions, accumulated_brief, max_turns)
2. action = await _decide_action(messages)              # one LLM call
3. If action is None or action.type not in {"search","done"}:
       log warning; break     # error bail-out → return what we have
4. If action.type == "done":
       break                  # normal termination
5. new_contexts = await _execute_search(action.query, user_id, device_id, agent_id)
       # on exception → log warning, break with what we have
6. new_hits = [c for c in new_contexts if c.id not in seen_ids]
   seen_ids.update(c.id for c in new_hits)
   accumulated_contexts.extend(new_hits)
7. previous_actions.append({"turn": turn, "query": action.query,
                            "reason": action.reason, "new_hits": len(new_hits)})
8. If len(new_hits) == 0:
       consecutive_empty_searches += 1
       if consecutive_empty_searches >= 2:
           break              # safety brake
   else:
       consecutive_empty_searches = 0
9. turn += 1
10. if turn >= max_turns: break
```

After the loop: `return self._format_memories(accumulated_contexts)`.

### Action schema

LLM must return one of:

```json
{"action": "search", "query": "<short query>", "reason": "<why>"}
{"action": "done", "reason": "<why>"}
```

`reason` is observability-only (logged, not used for control flow).

### Config read

In `recall()`, at the top:

```python
from opencontext.config.global_config import get_config
cfg = get_config("processing.agent_memory_processor.recall_agent") or {}
max_turns = int(cfg.get("max_turns", 3))
recall_model = cfg.get("model") or None
```

Reading per-call costs one dict lookup and enables config hot reload without restart.

### LLM call helper

```python
async def _decide_action(self, messages: list[dict]) -> dict | None:
    kwargs = {"enable_executor": False}
    if self._recall_model:
        kwargs["model"] = self._recall_model
    response = await generate_with_messages(messages, **kwargs)
    return parse_json_from_response(response)
```

## 5. `AgentMemoryProcessor` changes

```python
from opencontext.context_processing.processor.recall_agent import RecallAgent

class AgentMemoryProcessor(BaseContextProcessor):
    def __init__(self) -> None:
        super().__init__({})
        self._recall_agent = RecallAgent()

    async def _process_async(self, raw_context, prior_results):
        # ... agent validation, event filter (unchanged)
        # ... persona fetch with __base__ fallback (unchanged)

        related_memories_text = await self._recall_agent.recall(
            chat_content=chat_content,
            agent_name=agent_name,
            agent_persona=agent_persona,
            user_id=raw_context.user_id,
            device_id=raw_context.device_id or "default",
            agent_id=raw_context.agent_id,
        )

        # ... build commentary messages with {related_memories} = related_memories_text
        # ... call commentary LLM
        # ... write agent_commentary back onto events (unchanged)
```

**Deleted from the file**:

- `_extract_search_query()` method
- `_format_related_memories()` static method (logic moves to `RecallAgent._format_memories`)
- Direct import of `EventSearchService`
- Direct use of `agent_memory_query` prompt

**Unchanged**:

- `get_name`, `get_description`, `can_process`, `process` public surface
- Agent validation / persona fallback / commentary prompt wiring / event annotation

## 6. LLM model configuration

### `config/config.yaml`

Extend the existing `processing.agent_memory_processor` section:

```yaml
processing:
  agent_memory_processor:
    enabled: "${AGENT_MEMORY_PROCESSOR_ENABLED:true}"
    recall_agent:
      max_turns: ${AGENT_MEMORY_RECALL_MAX_TURNS:3}
      model: "${AGENT_MEMORY_RECALL_MODEL:}"   # empty = use VLM default
```

### `LLMClient._openai_chat_completion`

File: `opencontext/llm/llm_client.py`. Respect a `model` override from kwargs so the recall step can swap models without creating a private `LLMClient` instance.

```python
create_params = {
    "model": kwargs.get("model") or self.model,
    "messages": messages,
}
```

One-line change. Non-breaking (callers that don't pass `model` get the configured default).

### Call site in `RecallAgent`

`_decide_action` passes `model=` kwarg into `generate_with_messages` when the config has a non-empty `model` field. When the config is empty, the recall step runs on the default VLM model — same as commentary. This means users can optionally split: recall on a cheap fast model, commentary on a stronger one. Non-goal: cross-provider switching.

### Why VLM client, not `llm` client

The `_decide_action` call needs to handle chat content that may already be multimodal (images / PDF pages in the incoming chat log). VLM client handles that; text-only `llm` client doesn't. The user-visible model override is still just the model name.

## 7. Settings UI

### API

**Zero backend changes.** The existing `GET/POST /api/settings/general` endpoint already handles the top-level `processing` key (it's in `_GENERAL_SETTINGS_KEYS` in `opencontext/server/routes/settings.py`). The new subkey `processing.agent_memory_processor.recall_agent.{max_turns, model}` is persisted through the existing flow into `system_settings`.

### HTML fields

File: `opencontext/web/templates/settings.html`. Add a new card in the processors section alongside "文本对话处理器":

- **Card title**: "Agent 记忆处理器"
- **Enabled toggle** (`agent_memory_proc_enabled`) — wires to `processing.agent_memory_processor.enabled`. This field already exists in config but had no UI; add it now.
- **Model input** (`agent_memory_recall_model`) — wires to `processing.agent_memory_processor.recall_agent.model`. Placeholder: "留空使用 VLM 默认模型". Help tooltip: "只修改模型名，api_key 和 base_url 沿用 VLM 配置".
- **Max turns input** (`agent_memory_recall_max_turns`) — number input, min 1, max 10, wires to `processing.agent_memory_processor.recall_agent.max_turns`. Help tooltip: "recall agent 最多搜索的轮数，超过后强制停止".

### JS

File: `opencontext/web/static/js/settings.js`. Extend the existing `loadGeneralSettings` / `saveGeneralSettings` functions to read/write the three new fields into `config.processing.agent_memory_processor.*`. The existing save flow handles persistence and takes effect on the next push (because `RecallAgent` reads config per-call).

## 8. Prompts

### Retained

`processing.extraction.agent_memory_analyze` — commentary generation. No changes.

### New

`processing.extraction.agent_memory_recall` — used by `RecallAgent._decide_action`. Add to both `config/prompts_zh.yaml` and `config/prompts_en.yaml` (the codebase requires both language files to stay key-compatible).

### Deleted

`processing.extraction.agent_memory_query` — no longer needed.

### Prompt contract (placeholders)

| Placeholder | Source | Injected via |
|---|---|---|
| `{agent_name}` | from `agent_memory_processor` → passed to `recall()` | system `.replace()` |
| `{agent_persona}` | ditto | system `.replace()` (persona may contain literal `{`) |
| `{max_turns}` | from config | system `.replace()` |
| `{chat_history}` | raw chat content | user `.format()` |
| `{previous_actions}` | formatted log of earlier turns, empty on turn 1 | user `.format()` |
| `{accumulated_brief}` | `[date] title` list (no summaries) of memories found so far | user `.format()` |

### Brief format rationale

`{accumulated_brief}` intentionally excludes summaries to keep recall-loop context growth slow. Full summaries are only rendered in the final `related_memories` text that goes to the commentary prompt.

### Chinese draft

```yaml
    agent_memory_recall:
      system: |
        你是{agent_name}的"回忆系统"。你的任务是帮助 agent 想起
        与当前对话相关的过去事件。你每轮要决定：再搜一个新的 query
        去挖更多相关记忆，还是已经够了。

        你的人设提示（用于理解 agent 关心什么）:
        {agent_persona}

        规则：
        - 每轮输出一个 JSON 对象，格式严格如下：
          {"action": "search", "query": "<简短搜索词>", "reason": "<为什么搜这个>"}
          或
          {"action": "done", "reason": "<为什么觉得够了>"}
        - query 应该是短句或关键词组合，不要照搬用户原话
        - 参考"已搜过的 query"避免重复
        - 参考"已回忆到的事件"，如果足够丰富就果断 done
        - 最多允许 {max_turns} 轮，到达上限会自动停止
        - 不要输出 Markdown 标记或任何解释文本，只输出 JSON
      user: |
        当前对话片段：
        {chat_history}

        已搜过的 query（按时间先后）:
        {previous_actions}

        已回忆到的事件简报：
        {accumulated_brief}

        请给出本轮的 action JSON。
```

The English version mirrors this content exactly, per the codebase's prompt-parity rule.

## 9. Error handling and fallback

Any failure in the recall loop returns whatever was accumulated (possibly an empty string). Downstream commentary generation degrades gracefully: `{related_memories}` becomes empty, and the commentary LLM still produces commentary based on persona + event list alone.

| Failure | Behavior | Loop state returned |
|---|---|---|
| Turn-1 LLM call raises | log warning, abort | `""` |
| Turn-1 action JSON unparseable | log warning, abort | `""` |
| Turn-1 action type unknown | log warning, abort | `""` |
| Turn-N (N>1) LLM call raises | log warning, break | accumulated so far |
| Turn-N action JSON unparseable | log warning, break | accumulated so far |
| Turn-N action type unknown | log warning, break | accumulated so far |
| `EventSearchService.search` raises | log warning, break | accumulated so far |
| 2 consecutive turns with `new_hits == 0` | break (safety brake, not an error) | accumulated so far |
| Reached `max_turns` | normal exit | accumulated so far |

No legacy-mode flag, no fallback to the old single-query path. Deleting `_extract_search_query` is intentional.

## 10. Testing

### New file: `tests/context_processing/test_recall_agent.py`

Unit tests (`@pytest.mark.unit`). All tests use fake LLM and fake `EventSearchService` — no real network, no real storage.

| Case | What it proves |
|---|---|
| `test_first_turn_done_returns_empty` | Agent that says done on turn 1 → return `""` |
| `test_single_search_then_done` | Turn 1 search, turn 2 done → memories from turn 1 only |
| `test_max_turns_hard_cap` | Agent that never says done → exits at `max_turns`, runs exactly 3 searches |
| `test_llm_returns_invalid_json_turn_one` | Turn-1 LLM returns garbage → abort with `""` |
| `test_llm_returns_unknown_action` | Action type is neither search nor done → abort |
| `test_llm_call_raises_turn_n` | Turn-2 LLM raises → returns turn-1 accumulated |
| `test_search_service_raises` | Search raises on turn 2 → returns turn-1 accumulated |
| `test_two_consecutive_empty_searches_stops` | Turns 2 and 3 return 0 new hits → break |
| `test_dedup_across_turns` | Turn 1 and turn 2 search return overlapping IDs → final list deduped |
| `test_config_hot_reload` | Change `max_turns` in `get_config` stub → next `recall()` uses new value |
| `test_model_override_kwarg_applied` | Config has `model: "foo"` → `generate_with_messages` called with `model="foo"` |
| `test_model_override_absent_no_kwarg` | Config has empty model → `generate_with_messages` called without `model=` |

### Modified file: `tests/context_processing/test_agent_memory_processor.py`

- Delete all tests touching `_extract_search_query`.
- Add/rewrite: mock `RecallAgent.recall` to return a fixed string, then assert that the commentary prompt's `{related_memories}` placeholder is filled with that string.
- Keep all existing tests for: persona `__base__` fallback, event filter from `prior_results`, commentary JSON parse, `agent_commentary` write-back, agent_id validation branches.

### `LLMClient` test addition

Add a small test to `tests/llm/test_llm_client.py` (create if missing) proving that `generate_with_messages(model="override-name")` passes `"override-name"` into the openai `create()` call while omitting `model` kwarg preserves `self.model` behavior. This locks the one-line change in `_openai_chat_completion` against regressions.

## 11. Out of scope / future work

- **Knowledge-search / profile-search tools in the recall loop.** Deliberately excluded — the motivation is recall depth, not breadth. Can be added later if profile queries start making sense in the recall context.
- **Cross-provider LLM swapping for recall.** Only model name is configurable. Swapping `base_url` / `api_key` would require a private `LLMClient` per purpose and a reload mechanism; not justified by current requirements.
- **Streaming recall updates to the UI.** The recall loop runs in the push-chat background task; there's no UI surface for live updates today.
- **Global agent-framework migration (e.g., to langgraph).** Explicitly rejected for this feature. If the project later decides to adopt such a framework, it should be a separate spec that migrates all existing agentic code points together.

## 12. Affected files (checklist)

- `opencontext/context_processing/processor/recall_agent.py` — **new**
- `opencontext/context_processing/processor/agent_memory_processor.py` — modify: delete query extraction + direct search, inject `RecallAgent`
- `opencontext/context_processing/MODULE.md` — update to document the new module and the processor change
- `opencontext/llm/llm_client.py` — one-line: honor `model` kwarg in `_openai_chat_completion`
- `config/config.yaml` — add `recall_agent` subkey
- `config/prompts_zh.yaml` — add `agent_memory_recall`, delete `agent_memory_query`
- `config/prompts_en.yaml` — add `agent_memory_recall`, delete `agent_memory_query`
- `opencontext/web/templates/settings.html` — new "Agent 记忆处理器" card
- `opencontext/web/static/js/settings.js` — load/save new fields through existing general-settings flow
- `tests/context_processing/test_recall_agent.py` — **new**
- `tests/context_processing/test_agent_memory_processor.py` — modify
- `tests/llm/test_llm_client.py` — add model-override test (create file if absent)
