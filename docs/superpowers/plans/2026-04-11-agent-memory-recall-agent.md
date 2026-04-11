# Agent Memory Recall Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-shot query+search in `AgentMemoryProcessor` with a multi-turn LLM-driven recall loop that lives in its own `RecallAgent` module, and make the recall-step model configurable via config + admin settings UI.

**Architecture:** Introduce a new `RecallAgent` class in `opencontext/context_processing/processor/recall_agent.py` that runs a bounded while-loop (default 3 turns) over an LLM-chosen action schema (`search` / `done`) backed by `EventSearchService`. `AgentMemoryProcessor` delegates the "related memories" fetch step to a `RecallAgent` instance and keeps its other responsibilities (persona fetch, commentary generation, write-back) unchanged. `LLMClient._openai_chat_completion` gets a non-breaking `model` kwarg override so the recall step can use a different model from commentary generation.

**Tech Stack:** Python 3.11+, pytest + pytest-asyncio (`asyncio_mode = "auto"`), existing OpenAI-compatible `LLMClient`, existing `EventSearchService`, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-11-agent-memory-recall-agent-design.md`

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `opencontext/context_processing/processor/recall_agent.py` | `RecallAgent` class — multi-turn recall loop, action parsing, search execution, memories formatting |
| `tests/llm/__init__.py` | Test package marker |
| `tests/llm/test_llm_client.py` | Unit tests for `LLMClient` model-override kwarg |
| `tests/context_processing/__init__.py` | Test package marker |
| `tests/context_processing/test_recall_agent.py` | Unit tests for `RecallAgent` behavior |
| `tests/context_processing/test_agent_memory_processor.py` | Unit tests for `AgentMemoryProcessor` integration with `RecallAgent` |

**Modified files:**

| Path | Change |
|---|---|
| `opencontext/llm/llm_client.py` | Respect `kwargs.get("model")` override in `_openai_chat_completion` (one-line change at `create_params`) |
| `opencontext/context_processing/processor/agent_memory_processor.py` | Delete `_extract_search_query`, `_format_related_memories`, direct `EventSearchService` import; inject `RecallAgent` and delegate the recall step to it |
| `opencontext/context_processing/MODULE.md` | Document `RecallAgent` and updated `AgentMemoryProcessor` flow |
| `config/config.yaml` | Add `processing.agent_memory_processor.recall_agent` subkey |
| `config/prompts_en.yaml` | Add `processing.extraction.agent_memory_recall`; delete `processing.extraction.agent_memory_query` |
| `config/prompts_zh.yaml` | Same changes as English |
| `opencontext/web/templates/settings.html` | Add "Agent 记忆处理器" card with enabled toggle + recall model input + max turns input |
| `opencontext/web/static/js/settings.js` | Load/save the three new fields through existing general-settings flow |

---

## Task 1: Add `model` kwarg override to `LLMClient._openai_chat_completion`

**Files:**
- Modify: `opencontext/llm/llm_client.py:98-101`
- Create: `tests/llm/__init__.py`
- Create: `tests/llm/test_llm_client.py`

- [ ] **Step 1: Create the test package marker**

Create `tests/llm/__init__.py` as an empty file:

```python
```

- [ ] **Step 2: Write the failing tests**

Create `tests/llm/test_llm_client.py`:

```python
"""Unit tests for LLMClient._openai_chat_completion model override."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from opencontext.llm.llm_client import LLMClient, LLMType


def _make_client() -> LLMClient:
    config = {
        "provider": "openai",
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "model": "default-model",
        "max_concurrent": 1,
    }
    return LLMClient(llm_type=LLMType.CHAT, config=config)


@pytest.mark.unit
async def test_model_override_via_kwarg_applied():
    """Passing model=<name> as kwarg overrides self.model in create_params."""
    client = _make_client()
    fake_create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))],
        usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    ))
    client.client.chat.completions.create = fake_create

    await client._openai_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="override-model",
    )

    assert fake_create.await_count == 1
    kwargs = fake_create.await_args.kwargs
    assert kwargs["model"] == "override-model"


@pytest.mark.unit
async def test_model_override_absent_uses_default():
    """Without model kwarg, create_params.model is self.model."""
    client = _make_client()
    fake_create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))],
        usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    ))
    client.client.chat.completions.create = fake_create

    await client._openai_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
    )

    kwargs = fake_create.await_args.kwargs
    assert kwargs["model"] == "default-model"
```

- [ ] **Step 3: Run the tests to verify both fail**

Run: `uv run pytest tests/llm/test_llm_client.py -v`
Expected: `test_model_override_via_kwarg_applied` FAILS (the override is ignored, so `kwargs["model"] == "default-model"`). `test_model_override_absent_uses_default` PASSES already.

- [ ] **Step 4: Apply the one-line fix in `LLMClient._openai_chat_completion`**

In `opencontext/llm/llm_client.py`, replace:

```python
                create_params = {
                    "model": self.model,
                    "messages": messages,
                }
```

with:

```python
                create_params = {
                    "model": kwargs.get("model") or self.model,
                    "messages": messages,
                }
```

- [ ] **Step 5: Run the tests to verify both pass**

Run: `uv run pytest tests/llm/test_llm_client.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add opencontext/llm/llm_client.py tests/llm/__init__.py tests/llm/test_llm_client.py
git commit -m "feat(llm): honor model kwarg override in _openai_chat_completion"
```

---

## Task 2: Add `recall_agent` config block to `config.yaml`

**Files:**
- Modify: `config/config.yaml:86-108`

- [ ] **Step 1: Add the new subsection**

In `config/config.yaml`, replace:

```yaml
  # Agent 记忆处理器（为事件添加 agent 视角的评论注释）
  agent_memory_processor:
    enabled: "${AGENT_MEMORY_PROCESSOR_ENABLED:true}"
```

with:

```yaml
  # Agent 记忆处理器（为事件添加 agent 视角的评论注释）
  agent_memory_processor:
    enabled: "${AGENT_MEMORY_PROCESSOR_ENABLED:true}"
    # Recall agent: multi-turn LLM-driven search loop that collects past memories
    # before commentary generation.
    recall_agent:
      max_turns: ${AGENT_MEMORY_RECALL_MAX_TURNS:3}
      # Optional: override the model used for the recall step. Leave empty to
      # use the VLM default. Only the model name is configurable; api_key,
      # base_url, and provider are inherited from vlm_model.
      model: "${AGENT_MEMORY_RECALL_MODEL:}"
```

- [ ] **Step 2: Verify the config parses**

Run: `uv run python -c "from opencontext.config.global_config import GlobalConfig; gc = GlobalConfig.get_instance(); gc._auto_initialize(); print(gc.get_config('processing.agent_memory_processor.recall_agent'))"`
Expected: prints a dict with `max_turns: 3` and `model: ''`.

- [ ] **Step 3: Commit**

```bash
git add config/config.yaml
git commit -m "feat(config): add recall_agent block under agent_memory_processor"
```

---

## Task 3: Swap `agent_memory_query` for `agent_memory_recall` in both prompt files

**Files:**
- Modify: `config/prompts_en.yaml:597-604`
- Modify: `config/prompts_zh.yaml:597-603`

- [ ] **Step 1: Replace the English prompt block**

In `config/prompts_en.yaml`, replace:

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

with:

```yaml
    agent_memory_recall:
      system: |
        You are the "memory recall system" for {agent_name}. Your job is to help
        the agent remember past events related to the current conversation. On each
        turn you decide whether to search for more memories with a new query, or
        stop because you've remembered enough.

        The agent's persona (for understanding what they care about):
        {agent_persona}

        RULES:
        - Output exactly one JSON object per turn in one of these forms:
          {"action": "search", "query": "<short search terms>", "reason": "<why>"}
          or
          {"action": "done", "reason": "<why it's enough>"}
        - The query should be a short phrase or keyword combination — do not copy
          the user's wording verbatim
        - Review "previous queries" to avoid repeating yourself
        - Review "memories recalled so far" — stop decisively when it's rich enough
        - At most {max_turns} turns are allowed; the loop stops automatically at the cap
        - No Markdown, no explanation — output ONLY the JSON object
      user: |
        Current conversation:
        {chat_history}

        Previous queries (chronological):
        {previous_actions}

        Memories recalled so far:
        {accumulated_brief}

        Give the action JSON for this turn.
```

- [ ] **Step 2: Replace the Chinese prompt block**

In `config/prompts_zh.yaml`, replace:

```yaml
    agent_memory_query:
      system: |
        你是一个AI助手。从以下对话中，以你（AI）的视角提取核心话题，
        输出简短的摘要文本，用于搜索你过去与该用户相关的记忆。
        只输出摘要文本，不要其他内容。
      user: |
        {chat_history}
```

with:

```yaml
    agent_memory_recall:
      system: |
        你是 {agent_name} 的"回忆系统"。你的任务是帮助 agent 想起与当前对话
        相关的过去事件。你每轮要决定：再搜一个新的 query 去挖更多相关记忆，
        还是已经够了。

        agent 的人设提示（用于理解 agent 关心什么）:
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

- [ ] **Step 3: Verify both files parse and the new key is visible**

Run: `uv run python -c "import yaml; y = yaml.safe_load(open('config/prompts_en.yaml', encoding='utf-8')); print('recall' in y['processing']['extraction']); print('query' not in y['processing']['extraction'] or 'agent_memory_query' not in y['processing']['extraction'])"`
Expected: `True` then `True`.

Run the same for `config/prompts_zh.yaml`.

- [ ] **Step 4: Commit**

```bash
git add config/prompts_en.yaml config/prompts_zh.yaml
git commit -m "feat(prompts): replace agent_memory_query with agent_memory_recall"
```

---

## Task 4: Scaffold `RecallAgent` class with stub methods

**Files:**
- Create: `opencontext/context_processing/processor/recall_agent.py`
- Create: `tests/context_processing/__init__.py`

- [ ] **Step 1: Create the test package marker**

Create `tests/context_processing/__init__.py` as an empty file:

```python
```

- [ ] **Step 2: Create the scaffold module**

Create `opencontext/context_processing/processor/recall_agent.py`:

```python
"""Recall Agent — multi-turn LLM-driven loop that gathers past memories for commentary generation."""

import datetime
from dataclasses import dataclass, field
from typing import Any

from opencontext.config.global_config import get_config, get_prompt_group
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.context import ProcessedContext
from opencontext.server.search.event_search_service import EventSearchService
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class _RecallState:
    """Per-call state for the recall loop."""

    max_turns: int
    recall_model: str | None
    turn: int = 0
    seen_ids: set[str] = field(default_factory=set)
    accumulated: list[ProcessedContext] = field(default_factory=list)
    previous_actions: list[dict[str, Any]] = field(default_factory=list)
    consecutive_empty: int = 0


class RecallAgent:
    """Multi-turn memory recall loop driven by an LLM-selected search action."""

    def __init__(self) -> None:
        # No config cached — read fresh in recall() for hot reload support.
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
        """Run the recall loop. Returns formatted related_memories text.

        Returns an empty string when nothing was recalled (including first-turn
        aborts). Downstream commentary generation handles empty gracefully.
        """
        state = self._read_config()
        # Loop body is filled in by subsequent tasks.
        return self._format_memories(state.accumulated)

    def _read_config(self) -> _RecallState:
        cfg = get_config("processing.agent_memory_processor.recall_agent") or {}
        return _RecallState(
            max_turns=int(cfg.get("max_turns", 3)),
            recall_model=(cfg.get("model") or None),
        )

    async def _decide_action(
        self, messages: list[dict[str, Any]], recall_model: str | None
    ) -> dict[str, Any] | None:
        """One LLM call. Returns parsed action dict or None on parse failure."""
        kwargs: dict[str, Any] = {"enable_executor": False}
        if recall_model:
            kwargs["model"] = recall_model
        try:
            response = await generate_with_messages(messages, **kwargs)
        except Exception as exc:
            logger.warning(f"[recall_agent] LLM call failed: {exc}")
            return None
        if not response:
            return None
        parsed = parse_json_from_response(response)
        if not isinstance(parsed, dict):
            return None
        return parsed

    async def _execute_search(
        self, query: str, user_id: str, device_id: str, agent_id: str
    ) -> list[ProcessedContext]:
        """One search call. Returns flat list of hits + ancestors."""
        try:
            service = EventSearchService()
            result = await service.search(
                query=[{"type": "text", "text": query}],
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            )
        except Exception as exc:
            logger.warning(f"[recall_agent] Search failed: {exc}")
            raise
        if not result:
            return []
        collected: dict[str, ProcessedContext] = {}
        for ctx, _score in result.hits:
            collected[ctx.id] = ctx
        for ctx_id, ctx in result.ancestors.items():
            if ctx_id not in collected:
                collected[ctx_id] = ctx
        return list(collected.values())

    def _build_turn_messages(
        self,
        chat_content: str,
        agent_name: str,
        agent_persona: str,
        max_turns: int,
        previous_actions: list[dict[str, Any]],
        accumulated: list[ProcessedContext],
    ) -> list[dict[str, Any]] | None:
        prompt_group = get_prompt_group("processing.extraction.agent_memory_recall")
        if not prompt_group:
            logger.warning("[recall_agent] agent_memory_recall prompt not found")
            return None
        system_template: str = prompt_group.get("system", "")
        system = (
            system_template.replace("{agent_name}", agent_name)
            .replace("{agent_persona}", agent_persona)
            .replace("{max_turns}", str(max_turns))
        )
        user_template: str = prompt_group.get("user", "")
        user = user_template.format(
            chat_history=chat_content,
            previous_actions=self._format_previous_actions(previous_actions),
            accumulated_brief=self._format_brief(accumulated),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _format_previous_actions(previous_actions: list[dict[str, Any]]) -> str:
        if not previous_actions:
            return "(none yet)"
        lines = []
        for entry in previous_actions:
            lines.append(
                f"[turn {entry['turn']}] query={entry['query']!r}, "
                f"new_hits={entry['new_hits']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_brief(accumulated: list[ProcessedContext]) -> str:
        if not accumulated:
            return "(none yet)"
        lines = []
        for ctx in accumulated:
            title = (ctx.extracted_data.title if ctx.extracted_data else "") or "Untitled"
            event_time = (
                ctx.properties.event_time_start.strftime("%Y-%m-%d")
                if ctx.properties and ctx.properties.event_time_start
                else ""
            )
            lines.append(f"[{event_time}] {title}")
        return "\n".join(lines)

    @staticmethod
    def _format_memories(contexts: list[ProcessedContext]) -> str:
        """Final output — full [date] title + summary text for commentary prompt."""
        if not contexts:
            return ""
        sorted_ctxs = sorted(
            contexts,
            key=lambda c: c.properties.event_time_start
            if c.properties and c.properties.event_time_start
            else datetime.datetime.min,
        )
        lines = []
        for ctx in sorted_ctxs:
            title = (ctx.extracted_data.title if ctx.extracted_data else "") or ""
            summary = (ctx.extracted_data.summary if ctx.extracted_data else "") or ""
            event_time = (
                ctx.properties.event_time_start.strftime("%Y-%m-%d")
                if ctx.properties and ctx.properties.event_time_start
                else ""
            )
            lines.append(f"[{event_time}] {title}")
            if summary:
                lines.append(summary)
            lines.append("")
        return "\n".join(lines).strip()
```

- [ ] **Step 3: Verify the module imports cleanly**

Run: `uv run python -c "from opencontext.context_processing.processor.recall_agent import RecallAgent; print(RecallAgent().__class__.__name__)"`
Expected: prints `RecallAgent`.

- [ ] **Step 4: Commit**

```bash
git add opencontext/context_processing/processor/recall_agent.py tests/context_processing/__init__.py
git commit -m "feat(recall_agent): scaffold RecallAgent with helper methods"
```

---

## Task 5: Test — first-turn `done` returns empty string

**Files:**
- Create: `tests/context_processing/test_recall_agent.py`
- Modify: `opencontext/context_processing/processor/recall_agent.py:recall`

- [ ] **Step 1: Write the first failing test**

Create `tests/context_processing/test_recall_agent.py`:

```python
"""Unit tests for RecallAgent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from opencontext.context_processing.processor.recall_agent import RecallAgent


def _make_ctx(ctx_id: str, title: str, summary: str = "", date: str = "2026-04-01"):
    """Build a minimal ProcessedContext-like object for formatting assertions."""
    import datetime
    return SimpleNamespace(
        id=ctx_id,
        extracted_data=SimpleNamespace(title=title, summary=summary),
        properties=SimpleNamespace(
            event_time_start=datetime.datetime.fromisoformat(date)
        ),
    )


_FAKE_CONFIG = {"max_turns": 3, "model": ""}

_FAKE_PROMPT_GROUP = {
    "system": "You are {agent_name}. Persona: {agent_persona}. Max turns: {max_turns}.",
    "user": "Chat: {chat_history}\nPrev: {previous_actions}\nBrief: {accumulated_brief}",
}


def _patch_config_and_prompt():
    """Context manager stack: patches get_config and get_prompt_group."""
    return [
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
    ]


async def _run_recall(agent: RecallAgent) -> str:
    return await agent.recall(
        chat_content="hello",
        agent_name="Bot",
        agent_persona="helpful",
        user_id="u1",
        device_id="d1",
        agent_id="a1",
    )


@pytest.mark.unit
async def test_first_turn_done_returns_empty():
    """Agent says 'done' on turn 1 → recall() returns empty string, no search called."""
    agent = RecallAgent()

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value='{"action": "done", "reason": "not relevant"}'),
        ) as llm,
        patch.object(
            RecallAgent, "_execute_search", new=AsyncMock(return_value=[])
        ) as search,
    ):
        result = await _run_recall(agent)

    assert result == ""
    assert llm.await_count == 1
    assert search.await_count == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/context_processing/test_recall_agent.py::test_first_turn_done_returns_empty -v`
Expected: FAIL — `recall()` currently skips the loop entirely; `llm.await_count` is 0, assertion fails.

- [ ] **Step 3: Implement the minimal loop shell**

Replace `RecallAgent.recall` in `opencontext/context_processing/processor/recall_agent.py` with:

```python
    async def recall(
        self,
        chat_content: str,
        agent_name: str,
        agent_persona: str,
        user_id: str,
        device_id: str,
        agent_id: str,
    ) -> str:
        state = self._read_config()
        while state.turn < state.max_turns:
            messages = self._build_turn_messages(
                chat_content=chat_content,
                agent_name=agent_name,
                agent_persona=agent_persona,
                max_turns=state.max_turns,
                previous_actions=state.previous_actions,
                accumulated=state.accumulated,
            )
            if messages is None:
                logger.warning("[recall_agent] Prompt missing — aborting")
                break

            action = await self._decide_action(messages, state.recall_model)
            if action is None:
                logger.warning(
                    f"[recall_agent] turn={state.turn}: action parse failed, breaking"
                )
                break

            action_type = action.get("action")
            if action_type == "done":
                logger.info(
                    f"[recall_agent] turn={state.turn}: agent stopped "
                    f"(reason={action.get('reason')!r})"
                )
                break
            if action_type != "search":
                logger.warning(
                    f"[recall_agent] turn={state.turn}: unknown action {action_type!r}, breaking"
                )
                break

            # Search execution is wired in by later tasks.
            state.turn += 1

        return self._format_memories(state.accumulated)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/context_processing/test_recall_agent.py::test_first_turn_done_returns_empty -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opencontext/context_processing/processor/recall_agent.py tests/context_processing/test_recall_agent.py
git commit -m "feat(recall_agent): minimal loop shell with done-action termination"
```

---

## Task 6: Test — single search then done

**Files:**
- Modify: `tests/context_processing/test_recall_agent.py`
- Modify: `opencontext/context_processing/processor/recall_agent.py:recall`

- [ ] **Step 1: Add the new test**

Append to `tests/context_processing/test_recall_agent.py`:

```python
@pytest.mark.unit
async def test_single_search_then_done():
    """Turn 1 searches and finds 2 hits, turn 2 says done → formatted memories."""
    agent = RecallAgent()

    ctx1 = _make_ctx("c1", "meeting with alice", "discussed Q2 plans", "2026-04-01")
    ctx2 = _make_ctx("c2", "demo feedback", "team liked the UI", "2026-04-03")

    llm_responses = [
        '{"action": "search", "query": "alice demo", "reason": "new keywords"}',
        '{"action": "done", "reason": "got what I need"}',
    ]

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(side_effect=llm_responses),
        ) as llm,
        patch.object(
            RecallAgent, "_execute_search", new=AsyncMock(return_value=[ctx1, ctx2])
        ) as search,
    ):
        result = await _run_recall(agent)

    assert llm.await_count == 2
    assert search.await_count == 1
    assert "meeting with alice" in result
    assert "demo feedback" in result
    assert "2026-04-01" in result
    assert "2026-04-03" in result
    # Memories are sorted ascending by date, so alice appears before demo
    assert result.index("meeting with alice") < result.index("demo feedback")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/context_processing/test_recall_agent.py::test_single_search_then_done -v`
Expected: FAIL — search is not called yet (the loop body has no search branch).

- [ ] **Step 3: Wire the search branch into the loop**

In `opencontext/context_processing/processor/recall_agent.py`, replace the comment `# Search execution is wired in by later tasks.` with the following block (inside the `while` loop, right after the `action_type != "search"` check):

```python
            query = action.get("query") or ""
            if not query:
                logger.warning(
                    f"[recall_agent] turn={state.turn}: search action missing query, breaking"
                )
                break

            try:
                new_contexts = await self._execute_search(
                    query=query,
                    user_id=user_id,
                    device_id=device_id,
                    agent_id=agent_id,
                )
            except Exception:
                logger.warning(f"[recall_agent] turn={state.turn}: search raised, breaking")
                break

            new_hits = [c for c in new_contexts if c.id not in state.seen_ids]
            state.seen_ids.update(c.id for c in new_hits)
            state.accumulated.extend(new_hits)

            state.previous_actions.append(
                {
                    "turn": state.turn,
                    "query": query,
                    "reason": action.get("reason", ""),
                    "new_hits": len(new_hits),
                }
            )

            state.turn += 1
```

**Important**: delete the old `state.turn += 1` line that was below the comment — it's replaced by the one at the end of the new block.

- [ ] **Step 4: Run both tests to verify they pass**

Run: `uv run pytest tests/context_processing/test_recall_agent.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add opencontext/context_processing/processor/recall_agent.py tests/context_processing/test_recall_agent.py
git commit -m "feat(recall_agent): execute search action and accumulate results"
```

---

## Task 7: Test — `max_turns` hard cap

**Files:**
- Modify: `tests/context_processing/test_recall_agent.py`

No implementation change expected — the loop already caps on `state.turn < state.max_turns`. This task verifies that existing logic with a dedicated test.

- [ ] **Step 1: Add the test**

Append to `tests/context_processing/test_recall_agent.py`:

```python
@pytest.mark.unit
async def test_max_turns_hard_cap():
    """Agent never says done → loop stops at max_turns (default 3)."""
    agent = RecallAgent()

    always_search = '{"action": "search", "query": "q", "reason": "more"}'

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value=always_search),
        ) as llm,
        patch.object(
            RecallAgent,
            "_execute_search",
            new=AsyncMock(return_value=[_make_ctx("unique", "title", "sum", "2026-04-01")]),
        ) as search,
    ):
        # Use unique IDs per turn so the empty-search brake (Task 8) doesn't fire.
        ctxs_per_turn = [
            [_make_ctx(f"c{i}", f"title{i}", "sum", "2026-04-01")] for i in range(10)
        ]
        search.side_effect = ctxs_per_turn

        result = await _run_recall(agent)

    assert llm.await_count == 3
    assert search.await_count == 3
```

- [ ] **Step 2: Run to verify it passes immediately**

Run: `uv run pytest tests/context_processing/test_recall_agent.py::test_max_turns_hard_cap -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/context_processing/test_recall_agent.py
git commit -m "test(recall_agent): verify max_turns hard cap"
```

---

## Task 8: Test + feature — two consecutive empty searches stops the loop

**Files:**
- Modify: `tests/context_processing/test_recall_agent.py`
- Modify: `opencontext/context_processing/processor/recall_agent.py:recall`

- [ ] **Step 1: Add the test**

Append to `tests/context_processing/test_recall_agent.py`:

```python
@pytest.mark.unit
async def test_two_consecutive_empty_searches_stops():
    """Two empty searches in a row → safety brake breaks the loop."""
    agent = RecallAgent()

    config_five_turns = {"max_turns": 5, "model": ""}
    always_search = '{"action": "search", "query": "q", "reason": "more"}'

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=config_five_turns,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value=always_search),
        ) as llm,
        patch.object(
            RecallAgent,
            "_execute_search",
            new=AsyncMock(side_effect=[[], []]),
        ) as search,
    ):
        result = await _run_recall(agent)

    # Loop: turn 0 empty (counter=1, continue), turn 1 empty (counter=2, break).
    assert search.await_count == 2
    assert llm.await_count == 2
    assert result == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/context_processing/test_recall_agent.py::test_two_consecutive_empty_searches_stops -v`
Expected: FAIL — without the brake, the loop runs all 5 turns.

- [ ] **Step 3: Add the empty-search brake**

In `opencontext/context_processing/processor/recall_agent.py`, inside `recall()`, replace:

```python
            new_hits = [c for c in new_contexts if c.id not in state.seen_ids]
            state.seen_ids.update(c.id for c in new_hits)
            state.accumulated.extend(new_hits)

            state.previous_actions.append(
                {
                    "turn": state.turn,
                    "query": query,
                    "reason": action.get("reason", ""),
                    "new_hits": len(new_hits),
                }
            )

            state.turn += 1
```

with:

```python
            new_hits = [c for c in new_contexts if c.id not in state.seen_ids]
            state.seen_ids.update(c.id for c in new_hits)
            state.accumulated.extend(new_hits)

            state.previous_actions.append(
                {
                    "turn": state.turn,
                    "query": query,
                    "reason": action.get("reason", ""),
                    "new_hits": len(new_hits),
                }
            )

            if len(new_hits) == 0:
                state.consecutive_empty += 1
                if state.consecutive_empty >= 2:
                    logger.info(
                        f"[recall_agent] turn={state.turn}: two consecutive empty "
                        "searches, stopping"
                    )
                    state.turn += 1
                    break
            else:
                state.consecutive_empty = 0

            state.turn += 1
```

- [ ] **Step 4: Run all RecallAgent tests**

Run: `uv run pytest tests/context_processing/test_recall_agent.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add opencontext/context_processing/processor/recall_agent.py tests/context_processing/test_recall_agent.py
git commit -m "feat(recall_agent): brake after two consecutive empty searches"
```

---

## Task 9: Test — dedup across turns

**Files:**
- Modify: `tests/context_processing/test_recall_agent.py`

Already implemented in Task 6. This task adds a dedicated test.

- [ ] **Step 1: Add the test**

Append to `tests/context_processing/test_recall_agent.py`:

```python
@pytest.mark.unit
async def test_dedup_across_turns():
    """Turn 1 and turn 2 return overlapping ctx IDs → final output dedupes."""
    agent = RecallAgent()

    ctx_a = _make_ctx("c1", "shared event", "sum", "2026-04-01")
    ctx_b = _make_ctx("c2", "turn 1 extra", "sum", "2026-04-02")
    ctx_c = _make_ctx("c3", "turn 2 extra", "sum", "2026-04-03")

    llm_responses = [
        '{"action": "search", "query": "first", "reason": "r"}',
        '{"action": "search", "query": "second", "reason": "r"}',
        '{"action": "done", "reason": "enough"}',
    ]

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(side_effect=llm_responses),
        ),
        patch.object(
            RecallAgent,
            "_execute_search",
            new=AsyncMock(side_effect=[[ctx_a, ctx_b], [ctx_a, ctx_c]]),
        ),
    ):
        result = await _run_recall(agent)

    # Shared event appears exactly once.
    assert result.count("shared event") == 1
    assert "turn 1 extra" in result
    assert "turn 2 extra" in result
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/context_processing/test_recall_agent.py::test_dedup_across_turns -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/context_processing/test_recall_agent.py
git commit -m "test(recall_agent): verify dedup across search turns"
```

---

## Task 10: Tests — error paths (invalid JSON, unknown action, search raises, LLM raises)

**Files:**
- Modify: `tests/context_processing/test_recall_agent.py`

All of these behaviors already exist in the loop; this task adds coverage.

- [ ] **Step 1: Add four tests**

Append to `tests/context_processing/test_recall_agent.py`:

```python
@pytest.mark.unit
async def test_turn_one_invalid_json_returns_empty():
    """Unparseable LLM output on turn 1 → abort with empty string."""
    agent = RecallAgent()

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value="not json at all"),
        ),
        patch.object(RecallAgent, "_execute_search", new=AsyncMock()) as search,
    ):
        result = await _run_recall(agent)

    assert result == ""
    assert search.await_count == 0


@pytest.mark.unit
async def test_unknown_action_breaks():
    """Unknown action type → break, return accumulated (empty here)."""
    agent = RecallAgent()

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value='{"action": "fetch", "query": "x"}'),
        ),
        patch.object(RecallAgent, "_execute_search", new=AsyncMock()) as search,
    ):
        result = await _run_recall(agent)

    assert result == ""
    assert search.await_count == 0


@pytest.mark.unit
async def test_search_raises_returns_accumulated():
    """Turn 1 search ok, turn 2 search raises → return turn 1's memories."""
    agent = RecallAgent()

    ctx1 = _make_ctx("c1", "first hit", "sum", "2026-04-01")
    llm_responses = [
        '{"action": "search", "query": "q1", "reason": "r"}',
        '{"action": "search", "query": "q2", "reason": "r"}',
    ]

    async def search_side_effect(*args, **kwargs):
        if search_side_effect.call_count == 0:
            search_side_effect.call_count += 1
            return [ctx1]
        raise RuntimeError("search down")

    search_side_effect.call_count = 0

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(side_effect=llm_responses),
        ),
        patch.object(RecallAgent, "_execute_search", new=search_side_effect),
    ):
        result = await _run_recall(agent)

    assert "first hit" in result


@pytest.mark.unit
async def test_turn_one_llm_raises_returns_empty():
    """Turn 1 LLM raises → return empty string."""
    agent = RecallAgent()

    async def boom(*args, **kwargs):
        raise RuntimeError("LLM outage")

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=boom,
        ),
        patch.object(RecallAgent, "_execute_search", new=AsyncMock()) as search,
    ):
        result = await _run_recall(agent)

    assert result == ""
    assert search.await_count == 0
```

- [ ] **Step 2: Run the full RecallAgent suite**

Run: `uv run pytest tests/context_processing/test_recall_agent.py -v`
Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/context_processing/test_recall_agent.py
git commit -m "test(recall_agent): cover error paths (bad JSON, unknown action, raises)"
```

---

## Task 11: Tests — config `model` override passed as kwarg

**Files:**
- Modify: `tests/context_processing/test_recall_agent.py`

- [ ] **Step 1: Add two tests**

Append to `tests/context_processing/test_recall_agent.py`:

```python
@pytest.mark.unit
async def test_model_override_passed_when_configured():
    """Config.model set → generate_with_messages called with model kwarg."""
    agent = RecallAgent()
    config_with_model = {"max_turns": 3, "model": "cheap-model"}
    done_once = '{"action": "done", "reason": "quick"}'

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=config_with_model,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value=done_once),
        ) as llm,
        patch.object(RecallAgent, "_execute_search", new=AsyncMock(return_value=[])),
    ):
        await _run_recall(agent)

    assert llm.await_count == 1
    call_kwargs = llm.await_args.kwargs
    assert call_kwargs.get("model") == "cheap-model"
    assert call_kwargs.get("enable_executor") is False


@pytest.mark.unit
async def test_model_override_omitted_when_not_configured():
    """Config.model empty → generate_with_messages called WITHOUT model kwarg."""
    agent = RecallAgent()
    done_once = '{"action": "done", "reason": "quick"}'

    with (
        patch(
            "opencontext.context_processing.processor.recall_agent.get_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.get_prompt_group",
            return_value=_FAKE_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.recall_agent.generate_with_messages",
            new=AsyncMock(return_value=done_once),
        ) as llm,
        patch.object(RecallAgent, "_execute_search", new=AsyncMock(return_value=[])),
    ):
        await _run_recall(agent)

    assert llm.await_count == 1
    assert "model" not in llm.await_args.kwargs
```

- [ ] **Step 2: Run to verify both pass**

Run: `uv run pytest tests/context_processing/test_recall_agent.py -v`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/context_processing/test_recall_agent.py
git commit -m "test(recall_agent): verify model override kwarg behavior"
```

---

## Task 12: Wire `RecallAgent` into `AgentMemoryProcessor`

**Files:**
- Modify: `opencontext/context_processing/processor/agent_memory_processor.py` (significant rewrite of `_process_async` only — top-level class structure preserved)

- [ ] **Step 1: Rewrite imports and delete stale code**

Open `opencontext/context_processing/processor/agent_memory_processor.py`. Replace the file contents with:

```python
"""Agent Memory Processor — post-processor that annotates events with agent commentary."""

from typing import Any

from opencontext.config.global_config import get_prompt_group
from opencontext.context_processing.processor.base_processor import BaseContextProcessor
from opencontext.context_processing.processor.recall_agent import RecallAgent
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.context import ProcessedContext, RawContextProperties
from opencontext.models.enums import ContextSource, ContextType
from opencontext.storage.global_storage import get_storage
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.time_utils import now as tz_now

logger = get_logger(__name__)


class AgentMemoryProcessor(BaseContextProcessor):
    """Post-processor that writes agent commentary onto events from prior_results."""

    def __init__(self):
        super().__init__({})
        self._recall_agent = RecallAgent()

    def get_name(self) -> str:
        return "agent_memory_processor"

    def get_description(self) -> str:
        return "Annotates events with agent's subjective commentary."

    def can_process(self, context: Any) -> bool:
        if not isinstance(context, RawContextProperties):
            return False
        return context.source == ContextSource.CHAT_LOG

    async def process(
        self,
        context: RawContextProperties,
        prior_results: list[ProcessedContext] | None = None,
    ) -> list[ProcessedContext]:
        try:
            return await self._process_async(context, prior_results or [])
        except Exception as e:
            logger.error(f"Agent memory processing failed: {e}")
            return []

    async def _process_async(
        self,
        raw_context: RawContextProperties,
        prior_results: list[ProcessedContext],
    ) -> list[ProcessedContext]:
        agent_id = raw_context.agent_id
        if not agent_id or agent_id == "default":
            logger.debug("No agent_id in context, skipping agent memory processing")
            return []

        storage = get_storage()
        agent = await storage.get_agent(agent_id) if storage else None
        if not agent:
            logger.debug(f"Agent {agent_id} not registered, skipping")
            return []

        events = [
            ctx for ctx in prior_results if ctx.extracted_data.context_type == ContextType.EVENT
        ]
        if not events:
            logger.debug("[agent_memory_processor] No events in prior_results, skipping")
            return []

        agent_name = agent.get("name", agent_id)
        chat_content = raw_context.content_text or ""

        profile_result = await storage.get_profile(
            user_id=raw_context.user_id,
            device_id=raw_context.device_id or "default",
            agent_id=raw_context.agent_id,
            context_type="agent_profile",
        )

        if not profile_result:
            profile_result = await storage.get_profile(
                user_id="__base__",
                device_id=raw_context.device_id or "default",
                agent_id=raw_context.agent_id,
                context_type="agent_base_profile",
            )

        if not profile_result:
            logger.error(
                f"[agent_memory_processor] Agent profile not found for "
                f"user={raw_context.user_id}, agent={agent_id} (also checked __base__). "
                f"Agent must have a profile set up before agent memory processing."
            )
            return []

        agent_persona = profile_result.get("factual_profile", "")

        related_memories_text = await self._recall_agent.recall(
            chat_content=chat_content,
            agent_name=agent_name,
            agent_persona=agent_persona,
            user_id=raw_context.user_id,
            device_id=raw_context.device_id or "default",
            agent_id=raw_context.agent_id,
        )

        event_list = self._format_event_list(events)

        prompt_group = get_prompt_group("processing.extraction.agent_memory_analyze")
        if not prompt_group:
            logger.warning("agent_memory_analyze prompt not found")
            return []

        logger.debug(
            f"[agent_memory_processor] Processing: user={raw_context.user_id}, "
            f"agent={raw_context.agent_id}, events={len(events)}"
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
                    current_time=tz_now().isoformat(),
                    event_list=event_list,
                    chat_history=chat_content,
                ),
            },
        ]

        response = await generate_with_messages(messages, enable_executor=False)
        logger.debug(f"[agent_memory_processor] LLM response: {response}")
        if not response:
            return []

        analysis = parse_json_from_response(response)
        if not analysis:
            return []

        commentaries = analysis.get("commentaries", {})
        if not commentaries:
            logger.info("[agent_memory_processor] No commentaries from LLM")
            return []

        annotated_count = 0
        for idx_str, commentary in commentaries.items():
            try:
                idx = int(idx_str)
            except (ValueError, TypeError):
                continue
            if idx < 0 or idx >= len(events):
                continue
            if commentary and commentary != "null":
                events[idx].extracted_data.agent_commentary = str(commentary).strip()
                annotated_count += 1

        logger.info(f"[agent_memory_processor] Annotated {annotated_count}/{len(events)} events")
        return events

    @staticmethod
    def _format_event_list(events: list[ProcessedContext]) -> str:
        lines = []
        for i, ctx in enumerate(events):
            title = ctx.extracted_data.title or "Untitled"
            summary = ctx.extracted_data.summary or ""
            lines.append(f"[Event {i}] {title}")
            if summary:
                lines.append(summary)
            lines.append("")
        return "\n".join(lines).strip()
```

Note what was removed: `_extract_search_query`, `_format_related_memories`, `asyncio` import (no longer used), `datetime` import (moved to recall_agent), direct `EventSearchService` import, the parallel `asyncio.gather(profile_task, query_task)` pattern.

- [ ] **Step 2: Verify the module still imports**

Run: `uv run python -c "from opencontext.context_processing.processor.agent_memory_processor import AgentMemoryProcessor; p = AgentMemoryProcessor(); print(p.get_name())"`
Expected: prints `agent_memory_processor`.

- [ ] **Step 3: Run the existing test suite to make sure nothing else broke**

Run: `uv run pytest -m unit -q`
Expected: existing tests still PASS. (No tests currently reference the deleted methods.)

- [ ] **Step 4: Commit**

```bash
git add opencontext/context_processing/processor/agent_memory_processor.py
git commit -m "refactor(agent_memory): delegate recall to RecallAgent module"
```

---

## Task 13: Integration test for `AgentMemoryProcessor` with mocked `RecallAgent`

**Files:**
- Create: `tests/context_processing/test_agent_memory_processor.py`

- [ ] **Step 1: Write the integration test**

Create `tests/context_processing/test_agent_memory_processor.py`:

```python
"""Unit tests for AgentMemoryProcessor integration with RecallAgent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opencontext.context_processing.processor.agent_memory_processor import (
    AgentMemoryProcessor,
)
from opencontext.models.enums import ContextSource, ContextType


def _make_raw_context():
    return SimpleNamespace(
        user_id="u1",
        device_id="d1",
        agent_id="bot1",
        source=ContextSource.CHAT_LOG,
        content_text="hello bot",
    )


def _make_event_ctx(idx: int):
    return SimpleNamespace(
        id=f"evt{idx}",
        extracted_data=SimpleNamespace(
            context_type=ContextType.EVENT,
            title=f"Event {idx}",
            summary=f"summary {idx}",
            agent_commentary=None,
        ),
    )


_RECALL_PROMPT_GROUP = {
    "system": (
        "You are {agent_name}.\nPersona: {agent_persona}\n"
        "Past memories: {related_memories}\n"
    ),
    "user": (
        "Current time: {current_time}\n"
        "Events: {event_list}\n"
        "Conversation: {chat_history}"
    ),
}


@pytest.mark.unit
async def test_processor_uses_recall_agent_output_in_commentary_prompt():
    """RecallAgent.recall result is injected into {related_memories} placeholder."""
    processor = AgentMemoryProcessor()
    raw = _make_raw_context()
    events = [_make_event_ctx(0), _make_event_ctx(1)]

    fake_storage = MagicMock()
    fake_storage.get_agent = AsyncMock(return_value={"name": "Bot"})
    fake_storage.get_profile = AsyncMock(
        return_value={"factual_profile": "helpful agent"}
    )

    commentary_response = (
        '{"commentaries": {"0": "felt nice", "1": "also interesting"}}'
    )

    captured_messages: list = []

    async def fake_generate(messages, **kwargs):
        captured_messages.append(messages)
        return commentary_response

    with (
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.get_storage",
            return_value=fake_storage,
        ),
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.get_prompt_group",
            return_value=_RECALL_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.generate_with_messages",
            new=fake_generate,
        ),
        patch.object(
            processor._recall_agent,
            "recall",
            new=AsyncMock(return_value="[2026-04-10] some past event\nsum"),
        ),
    ):
        result = await processor._process_async(raw, events)

    assert len(captured_messages) == 1
    system_content = captured_messages[0][0]["content"]
    assert "[2026-04-10] some past event" in system_content
    assert events[0].extracted_data.agent_commentary == "felt nice"
    assert events[1].extracted_data.agent_commentary == "also interesting"
    assert len(result) == 2


@pytest.mark.unit
async def test_processor_falls_back_to_base_profile_when_user_profile_missing():
    """If user-specific profile missing, processor tries __base__ agent_base_profile."""
    processor = AgentMemoryProcessor()
    raw = _make_raw_context()
    events = [_make_event_ctx(0)]

    fake_storage = MagicMock()
    fake_storage.get_agent = AsyncMock(return_value={"name": "Bot"})
    fake_storage.get_profile = AsyncMock(
        side_effect=[None, {"factual_profile": "base persona"}]
    )

    with (
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.get_storage",
            return_value=fake_storage,
        ),
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.get_prompt_group",
            return_value=_RECALL_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.generate_with_messages",
            new=AsyncMock(return_value='{"commentaries": {"0": "ok"}}'),
        ),
        patch.object(
            processor._recall_agent, "recall", new=AsyncMock(return_value="")
        ),
    ):
        await processor._process_async(raw, events)

    # Second call was for __base__.
    second_call_kwargs = fake_storage.get_profile.await_args_list[1].kwargs
    assert second_call_kwargs["user_id"] == "__base__"
    assert second_call_kwargs["context_type"] == "agent_base_profile"


@pytest.mark.unit
async def test_processor_skips_when_agent_id_is_default():
    """agent_id == 'default' → early return, RecallAgent not called."""
    processor = AgentMemoryProcessor()
    raw = SimpleNamespace(
        user_id="u1",
        device_id="d1",
        agent_id="default",
        source=ContextSource.CHAT_LOG,
        content_text="hi",
    )

    with patch.object(
        processor._recall_agent, "recall", new=AsyncMock()
    ) as recall_mock:
        result = await processor._process_async(raw, [_make_event_ctx(0)])

    assert result == []
    assert recall_mock.await_count == 0


@pytest.mark.unit
async def test_processor_skips_when_no_events_in_prior_results():
    """No EVENT contexts in prior_results → early return, RecallAgent not called."""
    processor = AgentMemoryProcessor()
    raw = _make_raw_context()

    fake_storage = MagicMock()
    fake_storage.get_agent = AsyncMock(return_value={"name": "Bot"})

    with (
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.get_storage",
            return_value=fake_storage,
        ),
        patch.object(
            processor._recall_agent, "recall", new=AsyncMock()
        ) as recall_mock,
    ):
        result = await processor._process_async(raw, [])

    assert result == []
    assert recall_mock.await_count == 0
```

- [ ] **Step 2: Run the new test file**

Run: `uv run pytest tests/context_processing/test_agent_memory_processor.py -v`
Expected: all four tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/context_processing/test_agent_memory_processor.py
git commit -m "test(agent_memory): integration tests for RecallAgent delegation"
```

---

## Task 14: Update `context_processing/MODULE.md`

**Files:**
- Modify: `opencontext/context_processing/MODULE.md`

- [ ] **Step 1: Locate the `AgentMemoryProcessor` section**

Run: `uv run python -c "print(open('opencontext/context_processing/MODULE.md', encoding='utf-8').read())" | head -80`

Find the section that documents `AgentMemoryProcessor` and the `agent_memory_query` / `agent_memory_analyze` prompts.

- [ ] **Step 2: Apply a targeted edit**

Replace any mention of the `_extract_search_query` helper, the `agent_memory_query` prompt, and the two-phase (extract query → single search) description with the new architecture. Concretely:

- In the class responsibility table / file overview, add a row for `recall_agent.py` with responsibility "Multi-turn LLM-driven recall loop backing `AgentMemoryProcessor`".
- In the `AgentMemoryProcessor` flow description, replace the "extract search query → single search → format" steps with "delegate to `RecallAgent.recall()` → returns formatted memories text".
- Mention that `RecallAgent` reads `processing.agent_memory_processor.recall_agent.{max_turns, model}` from config per call (supports hot reload).
- Note that the `agent_memory_query` prompt has been removed, replaced by `agent_memory_recall`.

If the file does not currently document `AgentMemoryProcessor` at all (check first), add a short section with the same information.

- [ ] **Step 3: Commit**

```bash
git add opencontext/context_processing/MODULE.md
git commit -m "docs(context_processing): document RecallAgent and updated AgentMemoryProcessor flow"
```

---

## Task 15: Settings UI — add "Agent 记忆处理器" card in `settings.html`

**Files:**
- Modify: `opencontext/web/templates/settings.html`

- [ ] **Step 1: Locate the insertion point**

Find the "文本对话处理器" card in `settings.html`. Based on the file at the time of planning, the toggle input id is `text_chat_proc_enabled` on line 306. The entire card wraps around it.

Run: `uv run python -c "import re; s=open('opencontext/web/templates/settings.html', encoding='utf-8').read(); idx=s.find('text_chat_proc_enabled'); print(s[max(0,idx-400):idx+400])"`

This shows the surrounding card structure so you can mimic it for the new card.

- [ ] **Step 2: Add the new card directly after the text chat processor card**

Add a sibling card with the same visual structure. The three form fields should map to:
- `agent_memory_proc_enabled` — checkbox, binds to `processing.agent_memory_processor.enabled`
- `agent_memory_recall_max_turns` — number input, min=1, max=10, step=1, binds to `processing.agent_memory_processor.recall_agent.max_turns`
- `agent_memory_recall_model` — text input, placeholder "留空使用 VLM 默认模型", binds to `processing.agent_memory_processor.recall_agent.model`

Use a card structure like:

```html
<!-- Agent 记忆处理器 -->
<div class="card mb-3">
    <div class="card-header">
        <strong>Agent 记忆处理器 <i class="bi bi-info-circle settings-tip" data-bs-toggle="tooltip" title="为事件添加 agent 视角的评论注释。启用后，每次聊天 push 后会运行一次多轮 LLM 驱动的 recall 循环来提取相关过去记忆，再生成 commentary。"></i></strong>
        <div class="form-check form-switch float-end">
            <input class="form-check-input" type="checkbox" id="agent_memory_proc_enabled">
            <label class="form-check-label" for="agent_memory_proc_enabled">启用</label>
        </div>
    </div>
    <div class="card-body">
        <div class="row g-3">
            <div class="col-md-6">
                <label for="agent_memory_recall_max_turns" class="form-label">回忆最大轮数 <i class="bi bi-info-circle settings-tip" data-bs-toggle="tooltip" title="recall agent 最多搜索的轮数，超过后强制停止。推荐 3。"></i></label>
                <input type="number" class="form-control" id="agent_memory_recall_max_turns" min="1" max="10" step="1">
            </div>
            <div class="col-md-6">
                <label for="agent_memory_recall_model" class="form-label">回忆 agent 模型 <i class="bi bi-info-circle settings-tip" data-bs-toggle="tooltip" title="只修改模型名，api_key 和 base_url 沿用 VLM 配置。留空则使用 VLM 默认模型。"></i></label>
                <input type="text" class="form-control" id="agent_memory_recall_model" placeholder="留空使用 VLM 默认模型">
            </div>
        </div>
    </div>
</div>
```

- [ ] **Step 3: Verify the template still renders**

Run: `uv run python -c "import jinja2; env = jinja2.Environment(loader=jinja2.FileSystemLoader('opencontext/web/templates')); env.get_template('settings.html').render()"`
Expected: no exception (empty output is fine; we just want a parse check).

- [ ] **Step 4: Commit**

```bash
git add opencontext/web/templates/settings.html
git commit -m "feat(settings-ui): add Agent 记忆处理器 card with recall config fields"
```

---

## Task 16: Settings JS — wire the three new fields into load/save

**Files:**
- Modify: `opencontext/web/static/js/settings.js`

- [ ] **Step 1: Locate the general-settings load and save functions**

Run: `uv run python -c "s=open('opencontext/web/static/js/settings.js', encoding='utf-8').read(); print(s[s.find('text_chat_proc_enabled')-500:s.find('text_chat_proc_enabled')+800])"`

This shows how `text_chat_proc_enabled` is read on load and written on save. Mirror the same pattern for the three new fields.

- [ ] **Step 2: Add load logic**

In the general-settings load function (the one that fetches `/api/settings/general` and populates form fields), add next to the existing `text_chat_proc_enabled` wiring:

```javascript
const agentMemProc = processing.agent_memory_processor || {};
document.getElementById('agent_memory_proc_enabled').checked = agentMemProc.enabled !== false;
const recallAgent = agentMemProc.recall_agent || {};
document.getElementById('agent_memory_recall_max_turns').value = recallAgent.max_turns ?? 3;
document.getElementById('agent_memory_recall_model').value = recallAgent.model || '';
```

(Replace `processing` with the variable name actually used in the existing function.)

- [ ] **Step 3: Add save logic**

In the general-settings save function (the one that POSTs to `/api/settings/general`), include the new fields in the `processing.agent_memory_processor` payload:

```javascript
payload.processing = payload.processing || {};
payload.processing.agent_memory_processor = {
    enabled: document.getElementById('agent_memory_proc_enabled').checked,
    recall_agent: {
        max_turns: parseInt(document.getElementById('agent_memory_recall_max_turns').value, 10) || 3,
        model: document.getElementById('agent_memory_recall_model').value.trim(),
    },
};
```

(Merge this into the existing payload-building logic — don't overwrite other `processing` subkeys the save function already fills in.)

- [ ] **Step 4: Manually verify in the browser**

Start the dev server: `uv run opencontext start`
Navigate to `http://localhost:1733/settings`
- Confirm the new card renders with the three fields
- Set recall model to `"test-model-abc"`, max turns to `2`, save
- Refresh the page — confirm values persist
- Run `uv run python -c "from opencontext.config.global_config import get_config; GlobalConfig = __import__('opencontext.config.global_config', fromlist=['GlobalConfig']).GlobalConfig; GlobalConfig.get_instance()._auto_initialize(); print(get_config('processing.agent_memory_processor.recall_agent'))"` — expect `{'max_turns': 2, 'model': 'test-model-abc'}`

If any step fails, debug and fix before committing.

- [ ] **Step 5: Commit**

```bash
git add opencontext/web/static/js/settings.js
git commit -m "feat(settings-ui): load and save recall_agent config fields"
```

---

## Task 17: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit test suite**

Run: `uv run pytest -m unit -v`
Expected: all tests PASS, no warnings about unregistered markers.

- [ ] **Step 2: Check the prompt files for key parity**

Run: `uv run python -c "import yaml; en=yaml.safe_load(open('config/prompts_en.yaml', encoding='utf-8')); zh=yaml.safe_load(open('config/prompts_zh.yaml', encoding='utf-8')); assert set(en['processing']['extraction'].keys()) == set(zh['processing']['extraction'].keys()); print('prompt key parity OK')"`
Expected: `prompt key parity OK`.

- [ ] **Step 3: Grep for any leftover references to the deleted artifacts**

Run: `uv run python -c "import subprocess; out = subprocess.check_output(['git', 'grep', '-n', '_extract_search_query'], text=True, stderr=subprocess.DEVNULL).strip() or '(none)'; print(out)"`
Expected: `(none)`.

Run: `uv run python -c "import subprocess; out = subprocess.check_output(['git', 'grep', '-n', 'agent_memory_query'], text=True, stderr=subprocess.DEVNULL).strip() or '(none)'; print(out)"`
Expected: `(none)`.

- [ ] **Step 4: Manual smoke test of the recall loop (optional but recommended)**

If there's a running test environment with a real agent registered and storage populated, push a chat message via `/api/push/chat` with `processors: ["user_memory", "agent_memory"]` and watch the logs for entries tagged `[recall_agent]`. You should see one or more `turn=N` log lines followed by either `agent stopped` or `two consecutive empty searches`.

If no test environment is available, skip this step and note in the PR description that manual smoke testing is deferred.

- [ ] **Step 5: No commit here** — this task is pure verification. If any step fails, go back to the failing task.

---

## Self-Review Checklist

Before finishing, verify:

- [x] Every spec section mapped to at least one task:
  - Spec §3 Architecture → Tasks 4, 12
  - Spec §4 RecallAgent module → Tasks 4–11
  - Spec §5 AgentMemoryProcessor changes → Task 12
  - Spec §6 Model configuration → Tasks 1, 2, 11
  - Spec §7 Settings UI → Tasks 15, 16
  - Spec §8 Prompts → Task 3
  - Spec §9 Error handling → Tasks 8, 10
  - Spec §10 Testing → Tasks 5–11, 13
  - Spec §12 Affected files → covered across tasks
- [x] No `TBD` / `TODO` / `implement later` placeholders.
- [x] Type consistency: `RecallAgent.recall(chat_content, agent_name, agent_persona, user_id, device_id, agent_id) -> str` is the same signature in Tasks 4, 12, 13.
- [x] Config key is `processing.agent_memory_processor.recall_agent.{max_turns, model}` in Tasks 2, 4, 11, 15, 16.
- [x] Prompt key is `processing.extraction.agent_memory_recall` in Tasks 3, 4.
- [x] Commit messages follow conventional-commits style consistent with the repo's recent history.
