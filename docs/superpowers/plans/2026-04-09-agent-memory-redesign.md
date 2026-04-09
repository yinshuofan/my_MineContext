# Agent Memory Processing Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign agent memory processing so that agent commentary annotates events in-place, processor orchestration supports DAG dependencies, agent_profile updates via periodic tasks, and AGENT_EVENT type is removed.

**Architecture:** AgentMemoryProcessor becomes a post-processor that fills `agent_commentary` on events produced by TextChatProcessor. `process_batch()` gains DAG-based topological sort execution. A new `agent_profile_update` scheduled task replaces per-chat profile extraction. The `memory_owner` parameter is removed from search/cache APIs — base memories are included automatically.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, asyncio, Redis (scheduler), vector DB

---

### Task 1: Add `agent_commentary` Field to Data Model

**Files:**
- Modify: `opencontext/models/context.py:66-96` (ExtractedData), `opencontext/models/context.py:317-400` (ProcessedContextModel)

- [ ] **Step 1: Add field to `ExtractedData`**

In `opencontext/models/context.py`, add `agent_commentary` to `ExtractedData` (after `importance` field, line ~96):

```python
class ExtractedData(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    context_type: ContextType  # context type
    confidence: int = Field(default=0)  # confidence (0-10)
    importance: int = Field(default=0)  # importance (0-10)
    agent_commentary: Optional[str] = None  # agent's subjective commentary on this event
```

- [ ] **Step 2: Add field to `ProcessedContextModel`**

In `ProcessedContextModel` class (line ~317), add the field among the other extracted_data fields:

```python
agent_commentary: Optional[str] = None
```

In `from_processed_context()` method (line ~351), add the mapping. Find the line that sets `importance` and add after it:

```python
agent_commentary=ctx.extracted_data.agent_commentary,
```

- [ ] **Step 3: Compile-check**

Run:
```bash
python -m py_compile opencontext/models/context.py
```
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add opencontext/models/context.py
git commit -m "feat: add agent_commentary field to ExtractedData and ProcessedContextModel"
```

---

### Task 2: DAG-based Processor Orchestration

**Files:**
- Modify: `opencontext/context_processing/processor/base_processor.py:120-131`
- Modify: `opencontext/managers/processor_manager.py:20-23, 140-187`

- [ ] **Step 1: Add `prior_results` parameter to `BaseContextProcessor.process()`**

In `opencontext/context_processing/processor/base_processor.py`, change the `process()` abstract method signature (line ~120):

```python
@abc.abstractmethod
async def process(
    self,
    context: Any,
    prior_results: Optional[List] = None,
) -> list:
    """
    Process the context and return results.

    Args:
        context: The context to process
        prior_results: Results from processors that ran in earlier dependency layers.
                       None for processors with no dependencies.

    Returns:
        List of processed results
    """
    pass
```

Add the import at the top of the file:

```python
from typing import Any, Callable, Dict, List, Optional
```

(Verify `Optional` and `List` are already imported; add if not.)

- [ ] **Step 2: Add `prior_results=None` to existing processor `process()` signatures**

Update these files to accept the new parameter (they ignore it):

**`opencontext/context_processing/processor/text_chat_processor.py`** — find the `process()` method and add the parameter:

```python
async def process(self, context: RawContextProperties, prior_results=None) -> List[ProcessedContext]:
```

**`opencontext/context_processing/processor/document_processor.py`** — same change:

```python
async def process(self, context: Any, prior_results=None) -> list:
```

**`opencontext/context_processing/processor/agent_memory_processor.py`** — same change (will be rewritten in Task 3, but keep it compiling):

```python
async def process(self, context: RawContextProperties, prior_results=None) -> List[ProcessedContext]:
```

- [ ] **Step 3: Add dependency config and rewrite `process_batch()` in `processor_manager.py`**

In `opencontext/managers/processor_manager.py`, replace `BATCH_PROCESSOR_MAP` (lines 20-23) with:

```python
BATCH_PROCESSOR_MAP = {
    "user_memory": "text_chat_processor",
    "agent_memory": "agent_memory_processor",
}

# DAG dependency configuration: processor_name -> list of processor_names it depends on.
# Processors not listed here have no dependencies and run in the first layer.
PROCESSOR_DEPENDENCIES = {
    "agent_memory": ["user_memory"],
}
```

Replace the `process_batch()` method (lines 140-187) with:

```python
async def process_batch(
    self, raw_context: RawContextProperties, processor_names: List[str]
) -> List[ProcessedContext]:
    """
    Process a raw context through multiple processors with DAG-based ordering.

    Processors are grouped into layers via topological sort on PROCESSOR_DEPENDENCIES.
    Each layer runs in parallel; layers execute sequentially.
    Later layers receive accumulated results from all prior layers via `prior_results`.
    If a returned context has the same ID as an existing one, it replaces it.
    """
    if not processor_names:
        return []

    # Resolve external names to (external_name, processor_instance) pairs
    resolved = []
    for ext_name in processor_names:
        internal_name = BATCH_PROCESSOR_MAP.get(ext_name, ext_name)
        processor = self._processors.get(internal_name)
        if not processor:
            logger.warning(f"Processor '{internal_name}' not found, skipping")
            continue
        if not processor.can_process(raw_context):
            logger.debug(f"Processor '{internal_name}' cannot process this context, skipping")
            continue
        resolved.append((ext_name, processor))

    if not resolved:
        return []

    # Build layers via topological sort
    layers = self._topological_sort([name for name, _ in resolved])
    proc_map = {name: proc for name, proc in resolved}

    # Execute layer by layer
    accumulated: Dict[str, ProcessedContext] = {}  # id -> context

    for layer in layers:
        layer_processors = [(name, proc_map[name]) for name in layer if name in proc_map]
        if not layer_processors:
            continue

        prior = list(accumulated.values()) if accumulated else None

        tasks = []
        task_names = []
        for name, proc in layer_processors:
            tasks.append(proc.process(raw_context, prior_results=prior))
            task_names.append(name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Processor '{name}' failed: {result}")
                continue
            if not result:
                continue
            logger.debug(f"Processor '{name}' produced {len(result)} contexts")
            for ctx in result:
                accumulated[ctx.id] = ctx  # replace if same ID, append if new

    all_contexts = list(accumulated.values())

    if all_contexts:
        type_summary = {}
        for ctx in all_contexts:
            t = ctx.extracted_data.context_type.value
            type_summary[t] = type_summary.get(t, 0) + 1
        logger.info(f"process_batch produced {len(all_contexts)} contexts: {type_summary}")

    if all_contexts and self._callback:
        await self._callback(all_contexts)

    return all_contexts

@staticmethod
def _topological_sort(processor_names: List[str]) -> List[List[str]]:
    """
    Sort processor names into execution layers based on PROCESSOR_DEPENDENCIES.

    Returns a list of layers, where each layer is a list of processor names
    that can run in parallel. Earlier layers complete before later ones start.
    """
    name_set = set(processor_names)
    # Filter dependencies to only include processors in this batch
    deps = {}
    for name in processor_names:
        raw_deps = PROCESSOR_DEPENDENCIES.get(name, [])
        deps[name] = [d for d in raw_deps if d in name_set]

    layers = []
    placed = set()

    while len(placed) < len(processor_names):
        # Find all processors whose dependencies are satisfied
        layer = [
            name for name in processor_names
            if name not in placed and all(d in placed for d in deps[name])
        ]
        if not layer:
            # Circular dependency — break by placing remaining
            remaining = [n for n in processor_names if n not in placed]
            logger.error(f"Circular dependency detected among: {remaining}. Running them together.")
            layers.append(remaining)
            break
        layers.append(layer)
        placed.update(layer)

    return layers
```

Add `import asyncio` at the top if not already imported. Also add `Dict` to typing imports if missing.

- [ ] **Step 4: Compile-check all modified files**

Run:
```bash
python -m py_compile opencontext/context_processing/processor/base_processor.py
python -m py_compile opencontext/managers/processor_manager.py
python -m py_compile opencontext/context_processing/processor/text_chat_processor.py
python -m py_compile opencontext/context_processing/processor/document_processor.py
python -m py_compile opencontext/context_processing/processor/agent_memory_processor.py
```
Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add opencontext/context_processing/processor/base_processor.py opencontext/managers/processor_manager.py opencontext/context_processing/processor/text_chat_processor.py opencontext/context_processing/processor/document_processor.py opencontext/context_processing/processor/agent_memory_processor.py
git commit -m "feat: add DAG-based processor orchestration with topological sort execution"
```

---

### Task 3: Rewrite AgentMemoryProcessor as Post-Processor

**Files:**
- Modify: `opencontext/context_processing/processor/agent_memory_processor.py` (full rewrite)
- Modify: `config/prompts_en.yaml:607-647`
- Modify: `config/prompts_zh.yaml:606-645`

- [ ] **Step 1: Rewrite the prompt — `agent_memory_analyze` in `config/prompts_en.yaml`**

Replace the `agent_memory_analyze` section (lines 607-647) with:

```yaml
    agent_memory_analyze:
      system: |
        You are {agent_name}.

        Your persona:
        {agent_persona}

        The following are your past memories related to this conversation:
        {related_memories}

        Below is a conversation you just had with a user, along with a list of events extracted from it.
        For each event, write a brief commentary from YOUR subjective perspective as {agent_name}.
        Express your feelings, reactions, observations, or evaluations about what happened.

        IMPORTANT:
        - Write commentary for EACH event in the list
        - Each commentary should be 1-3 sentences, in first person
        - If an event is mundane and you have nothing meaningful to say, write "null" for that event
        - Output a JSON object mapping event index (0-based) to your commentary string (or null)

        Example output (no Markdown markers):
        {
          "commentaries": {
            "0": "I noticed he was quite enthusiastic about this topic — reminds me of our earlier discussions about architecture.",
            "1": null,
            "2": "She seemed hesitant when asking about this. I should be more encouraging next time."
          }
        }
      user: |
        Current time: {current_time}

        Events extracted from this conversation:
        {event_list}

        Full conversation:
        {chat_history}
```

- [ ] **Step 2: Rewrite the prompt — `agent_memory_analyze` in `config/prompts_zh.yaml`**

Replace the `agent_memory_analyze` section (lines 606-645) with:

```yaml
    agent_memory_analyze:
      system: |
        你是{agent_name}。

        你的人设:
        {agent_persona}

        以下是与本次对话相关的你的过去记忆:
        {related_memories}

        下面是你刚刚与用户的一段对话，以及从中提取的事件列表。
        请以{agent_name}的主观视角，为每个事件写一段简短的评论。
        表达你对发生的事情的感受、反应、观察或评价。

        重要：
        - 为列表中的每个事件写评论
        - 每条评论1-3句话，使用第一人称
        - 如果某个事件很平淡，你没有什么有意义的话要说，对该事件写"null"
        - 输出一个JSON对象，将事件索引（从0开始）映射到你的评论字符串（或null）

        输出示例（不要Markdown标记）：
        {
          "commentaries": {
            "0": "我注意到他对这个话题非常热情——让我想起了我们之前关于架构的讨论。",
            "1": null,
            "2": "她问这个问题时似乎有些犹豫。下次我应该更鼓励一些。"
          }
        }
      user: |
        当前时间：{current_time}

        从本次对话中提取的事件：
        {event_list}

        完整对话：
        {chat_history}
```

- [ ] **Step 3: Rewrite `agent_memory_processor.py`**

Replace the entire file content of `opencontext/context_processing/processor/agent_memory_processor.py`:

```python
"""Agent Memory Processor — post-processor that annotates events with agent commentary."""

import asyncio
from typing import Any, List, Optional

from opencontext.config.global_config import get_prompt_group
from opencontext.context_processing.processor.base_processor import BaseContextProcessor
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.context import ProcessedContext, RawContextProperties
from opencontext.models.enums import ContextSource, ContextType
from opencontext.server.search.event_search_service import EventSearchService
from opencontext.storage.global_storage import get_storage
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.time_utils import now as tz_now

logger = get_logger(__name__)


class AgentMemoryProcessor(BaseContextProcessor):
    """Post-processor that writes agent commentary onto events from prior_results."""

    def __init__(self):
        super().__init__({})

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
        prior_results: Optional[List[ProcessedContext]] = None,
    ) -> List[ProcessedContext]:
        """Annotate events from prior_results with agent commentary.

        Returns the modified event contexts (same IDs, with agent_commentary populated).
        Does NOT produce new contexts.
        """
        try:
            return await self._process_async(context, prior_results or [])
        except Exception as e:
            logger.error(f"Agent memory processing failed: {e}")
            return []

    async def _process_async(
        self,
        raw_context: RawContextProperties,
        prior_results: List[ProcessedContext],
    ) -> List[ProcessedContext]:
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

        # 1. Filter events from prior_results
        events = [
            ctx for ctx in prior_results
            if ctx.extracted_data.context_type == ContextType.EVENT
        ]
        if not events:
            logger.debug("[agent_memory_processor] No events in prior_results, skipping")
            return []

        agent_name = agent.get("name", agent_id)
        chat_content = raw_context.content_text or ""

        # 2. Parallel: fetch persona + extract search query
        profile_task = storage.get_profile(
            user_id=raw_context.user_id,
            device_id=raw_context.device_id or "default",
            agent_id=raw_context.agent_id,
            context_type="agent_profile",
        )
        query_task = self._extract_search_query(chat_content)

        profile_result, query_text = await asyncio.gather(profile_task, query_task)

        # Fallback to base profile if per-user profile doesn't exist
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

        # 3. Search related past memories
        related_memories_text = ""
        if query_text:
            search_service = EventSearchService()
            search_result = await search_service.semantic_search(
                query=[{"type": "text", "text": query_text}],
                user_id=raw_context.user_id,
                device_id=raw_context.device_id or "default",
                agent_id=raw_context.agent_id,
            )
            if search_result:
                related_memories_text = self._format_related_memories(search_result)

        # 4. Build event list for prompt
        event_list = self._format_event_list(events)

        # 5. Load prompt and build LLM messages
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

        # 6. Call LLM
        response = await generate_with_messages(messages, enable_executor=False)
        logger.debug(f"[agent_memory_processor] LLM response: {response}")
        if not response:
            return []

        # 7. Parse and apply commentaries
        analysis = parse_json_from_response(response)
        if not analysis:
            return []

        commentaries = analysis.get("commentaries", {})
        if not commentaries:
            logger.info("[agent_memory_processor] No commentaries from LLM")
            return []

        modified = []
        for idx_str, commentary in commentaries.items():
            try:
                idx = int(idx_str)
            except (ValueError, TypeError):
                continue
            if idx < 0 or idx >= len(events):
                continue
            if commentary and commentary != "null":
                events[idx].extracted_data.agent_commentary = str(commentary).strip()
                modified.append(events[idx])

        logger.info(
            f"[agent_memory_processor] Annotated {len(modified)}/{len(events)} events"
        )
        return modified

    async def _extract_search_query(self, chat_content: str) -> Optional[str]:
        """Use LLM to extract a search query from chat content (from AI's perspective)."""
        try:
            prompt_group = get_prompt_group("processing.extraction.agent_memory_query")
            if not prompt_group:
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
        except Exception as e:
            logger.warning(f"[agent_memory_processor] Query extraction failed: {e}")
            return None

    @staticmethod
    def _format_related_memories(search_result) -> str:
        """Format search results into text for the LLM prompt."""
        import datetime

        all_contexts = {}
        for ctx, score in search_result.hits:
            all_contexts[ctx.id] = ctx
        for ctx_id, ctx in search_result.ancestors.items():
            if ctx_id not in all_contexts:
                all_contexts[ctx_id] = ctx

        if not all_contexts:
            return ""

        sorted_contexts = sorted(
            all_contexts.values(),
            key=lambda c: c.properties.event_time_start if c.properties else datetime.datetime.min,
        )

        lines = []
        for ctx in sorted_contexts:
            title = ctx.extracted_data.title if ctx.extracted_data else ""
            summary = ctx.extracted_data.summary if ctx.extracted_data else ""
            event_time_str = (
                ctx.properties.event_time_start.strftime("%Y-%m-%d") if ctx.properties else ""
            )
            lines.append(f"[{event_time_str}] {title}")
            if summary:
                lines.append(summary)
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _format_event_list(events: List[ProcessedContext]) -> str:
        """Format events into a numbered list for the LLM prompt."""
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

- [ ] **Step 4: Compile-check**

Run:
```bash
python -m py_compile opencontext/context_processing/processor/agent_memory_processor.py
```
Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add opencontext/context_processing/processor/agent_memory_processor.py config/prompts_en.yaml config/prompts_zh.yaml
git commit -m "feat: rewrite AgentMemoryProcessor as post-processor that annotates events with commentary"
```

---

### Task 4: Add `agent_profile_update` Periodic Task

**Files:**
- Create: `opencontext/periodic_task/agent_profile_update.py`
- Modify: `opencontext/periodic_task/__init__.py`
- Modify: `opencontext/server/component_initializer.py:255-259`
- Modify: `opencontext/server/routes/push.py:483-496`
- Modify: `config/config.yaml:308-350`
- Modify: `config/prompts_en.yaml`
- Modify: `config/prompts_zh.yaml`

- [ ] **Step 1: Add the prompt — `agent_profile_update` in `config/prompts_en.yaml`**

Add after the `agent_memory_analyze` section (after line ~647):

```yaml
    agent_profile_update:
      system: |
        You are {agent_name}.

        Your base persona (this is your core character — do NOT deviate significantly from it):
        {base_profile}

        Your current understanding of this user:
        {current_profile}

        Below are today's events from your interactions with this user.
        Based on these events, update your understanding of the user.

        Rules:
        - Your updated profile should reflect new insights from today's events
        - Stay true to your base persona — your personality may evolve slightly but must not contradict your core character
        - Keep the profile concise and factual from your perspective
        - Use the same section format as the current profile if one exists
        - If today's events don't change your understanding, return the current profile unchanged

        Output ONLY the updated profile text, no JSON wrapping, no explanation.
      user: |
        Current time: {current_time}

        Today's events:
        {today_events}
```

- [ ] **Step 2: Add the prompt — `agent_profile_update` in `config/prompts_zh.yaml`**

Add the Chinese version at the same position:

```yaml
    agent_profile_update:
      system: |
        你是{agent_name}。

        你的基础人设（这是你的核心角色——不要显著偏离）：
        {base_profile}

        你目前对这个用户的了解：
        {current_profile}

        以下是今天你与这个用户互动中的事件。
        根据这些事件，更新你对用户的了解。

        规则：
        - 更新后的档案应反映今天事件中的新认知
        - 忠于你的基础人设——你的性格可以微调但不能与核心角色矛盾
        - 保持档案简洁，从你的视角陈述事实
        - 如果已有档案，沿用相同的分节格式
        - 如果今天的事件没有改变你的认知，原样返回当前档案

        仅输出更新后的档案文本，不要JSON包装，不要解释。
      user: |
        当前时间：{current_time}

        今天的事件：
        {today_events}
```

- [ ] **Step 3: Create `opencontext/periodic_task/agent_profile_update.py`**

```python
"""Agent Profile Update Task — updates agent_profile from daily events."""

from typing import List, Optional

from opencontext.config.global_config import get_prompt_group
from opencontext.context_processing.processor.profile_processor import refresh_profile
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.enums import ContextType
from opencontext.periodic_task.base import BasePeriodicTask, TaskContext, TaskResult
from opencontext.scheduler.base import TriggerMode
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.time_utils import now as tz_now, tz_today_start

logger = get_logger(__name__)


class AgentProfileUpdateTask(BasePeriodicTask):
    """
    Updates agent_profile based on today's events.

    Triggered via user_activity (scheduled from push endpoint when agent_memory is used).
    Skipped if agent_id is not registered or is "default".
    """

    def __init__(self, interval: int = 86400, timeout: int = 300):
        super().__init__(
            name="agent_profile_update",
            description="Update agent profile from daily interaction events",
            trigger_mode=TriggerMode.USER_ACTIVITY,
            interval=interval,
            timeout=timeout,
            task_ttl=14400,
            max_retries=2,
        )

    async def execute(self, context: TaskContext) -> TaskResult:
        user_id = context.user_id
        device_id = context.device_id or "default"
        agent_id = context.agent_id or "default"

        # Should not happen (filtered at scheduling time), but guard anyway
        if not agent_id or agent_id == "default":
            return TaskResult.ok("Skipped: no agent_id")

        storage = get_storage()
        if not storage:
            return TaskResult.fail("Storage not initialized")

        # Verify agent is still registered
        agent = await storage.get_agent(agent_id)
        if not agent:
            return TaskResult.ok(f"Skipped: agent {agent_id} not registered")

        agent_name = agent.get("name", agent_id)

        # 1. Fetch today's events
        today_start = tz_today_start()
        today_start_ts = float(int(today_start.timestamp()))

        events = await storage.get_all_processed_contexts(
            context_types=[ContextType.EVENT.value],
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            time_start=today_start_ts,
            limit=50,
        )

        if not events:
            return TaskResult.ok("Skipped: no events today")

        # 2. Fetch current agent_profile and base_profile
        current_profile = await storage.get_profile(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            context_type="agent_profile",
        )
        base_profile = await storage.get_profile(
            user_id="__base__",
            device_id=device_id,
            agent_id=agent_id,
            context_type="agent_base_profile",
        )

        if not base_profile:
            return TaskResult.fail(f"No base profile for agent {agent_id}")

        base_persona = base_profile.get("factual_profile", "")
        current_persona = current_profile.get("factual_profile", "") if current_profile else ""

        # 3. Format today's events
        events_text = self._format_events(events)

        # 4. Load prompt and call LLM
        prompt_group = get_prompt_group("processing.extraction.agent_profile_update")
        if not prompt_group:
            return TaskResult.fail("agent_profile_update prompt not found")

        system_prompt = prompt_group.get("system", "")
        system_prompt = system_prompt.replace("{agent_name}", agent_name)
        system_prompt = system_prompt.replace("{base_profile}", base_persona)
        system_prompt = system_prompt.replace("{current_profile}", current_persona or "(No existing profile)")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt_group.get("user", "").format(
                    current_time=tz_now().isoformat(),
                    today_events=events_text,
                ),
            },
        ]

        response = await generate_with_messages(messages, enable_executor=False)
        if not response or not response.strip():
            return TaskResult.fail("LLM returned empty response")

        # 5. Store updated profile
        updated_profile = response.strip()
        success = await refresh_profile(
            new_factual_profile=updated_profile,
            new_entities=None,
            new_importance=7,
            new_metadata=None,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            context_type="agent_profile",
        )

        if success:
            logger.info(
                f"[agent_profile_update] Updated profile for user={user_id}, agent={agent_id}"
            )
            return TaskResult.ok(
                f"Profile updated for user={user_id}, agent={agent_id}",
                data={"events_count": len(events)},
            )
        else:
            return TaskResult.fail("refresh_profile returned False")

    @staticmethod
    def _format_events(events: list) -> str:
        """Format events into text for the LLM prompt."""
        lines = []
        for ctx in events:
            title = ctx.extracted_data.title or "Untitled"
            summary = ctx.extracted_data.summary or ""
            commentary = ctx.extracted_data.agent_commentary or ""
            time_str = (
                ctx.properties.event_time_start.strftime("%H:%M")
                if ctx.properties and ctx.properties.event_time_start
                else ""
            )
            lines.append(f"[{time_str}] {title}")
            if summary:
                lines.append(f"  {summary}")
            if commentary:
                lines.append(f"  My thoughts: {commentary}")
            lines.append("")
        return "\n".join(lines).strip()


def create_agent_profile_update_handler():
    """Create handler function for the scheduler."""
    task = AgentProfileUpdateTask()

    async def handler(
        user_id: str,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        context = TaskContext(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            task_type="agent_profile_update",
        )
        if not task.validate_context(context):
            return False
        result = await task.execute(context)
        return result.success

    return handler
```

- [ ] **Step 4: Register in `__init__.py`**

In `opencontext/periodic_task/__init__.py`, add the import and export:

```python
from opencontext.periodic_task.agent_profile_update import (
    AgentProfileUpdateTask,
    create_agent_profile_update_handler,
)
```

Add to `__all__`:

```python
"AgentProfileUpdateTask",
"create_agent_profile_update_handler",
```

- [ ] **Step 5: Register handler in `component_initializer.py`**

In `opencontext/server/component_initializer.py`, after the hierarchy_summary registration (line ~259), add:

```python
            # Agent profile update handler
            if tasks_config.get("agent_profile_update", {}).get("enabled", False):
                from opencontext.periodic_task import create_agent_profile_update_handler

                agent_profile_handler = create_agent_profile_update_handler()
                scheduler.register_handler("agent_profile_update", agent_profile_handler)
                logger.info("Registered agent_profile_update task handler")
```

- [ ] **Step 6: Add config entry in `config/config.yaml`**

In the `scheduler.tasks` section (after `hierarchy_summary` block, line ~350), add:

```yaml
    agent_profile_update:
      enabled: "${AGENT_PROFILE_UPDATE_ENABLED:true}"
      trigger_mode: "user_activity"
      interval: 86400       # 24h delay after user push
      timeout: 300          # LLM call timeout (seconds)
      task_ttl: 14400       # 4h task state TTL
```

- [ ] **Step 7: Schedule task in push endpoint**

In `opencontext/server/routes/push.py`, after the hierarchy_summary scheduling block (line ~496), add:

```python
        # Schedule agent profile update (only for registered agents)
        if (
            "agent_memory" in request.processors
            and request.agent_id is not None
            and request.agent_id != "default"
        ):
            background_tasks.add_task(
                _schedule_user_task,
                task_type="agent_profile_update",
                user_id=request.user_id,
                device_id=request.device_id,
                agent_id=request.agent_id,
            )
```

This reuses the agent validation already done earlier in the endpoint (lines 405-418) — if we reach this point, the agent is verified as registered.

- [ ] **Step 8: Compile-check**

Run:
```bash
python -m py_compile opencontext/periodic_task/agent_profile_update.py
python -m py_compile opencontext/periodic_task/__init__.py
python -m py_compile opencontext/server/component_initializer.py
python -m py_compile opencontext/server/routes/push.py
```
Expected: No errors.

- [ ] **Step 9: Commit**

```bash
git add opencontext/periodic_task/agent_profile_update.py opencontext/periodic_task/__init__.py opencontext/server/component_initializer.py opencontext/server/routes/push.py config/config.yaml config/prompts_en.yaml config/prompts_zh.yaml
git commit -m "feat: add agent_profile_update periodic task triggered from push endpoint"
```

---

### Task 5: Remove `agent_event` from `chat_analyze` Prompt

**Files:**
- Modify: `config/prompts_en.yaml:429-434`
- Modify: `config/prompts_zh.yaml:429-434`

- [ ] **Step 1: Remove `agent_event` from classification in English prompt**

In `config/prompts_en.yaml`, find the classification section (around line 429-434). Remove the `agent_event` line:

```
        - `agent_event`: Agent's subjective experience — Records the agent's feelings, reactions, and evaluations about user events from the agent's own perspective. Use this when the memory captures what the agent observed, felt, or thought about an interaction, rather than the user's own activities.
```

Remove this entire line. Keep the other types (`event`, `profile`, `knowledge`, `document`).

Note: After this change, if the LLM still returns `"agent_event"` as a context_type, `get_context_type_for_analysis()` in `enums.py` will fail `validate_context_type()` (since the enum value no longer exists after Task 6) and fall back to `KNOWLEDGE`. This is acceptable — the prompt no longer instructs the LLM to use this type, and the fallback is safe.

- [ ] **Step 2: Remove `agent_event` from classification in Chinese prompt**

In `config/prompts_zh.yaml`, find the same classification section and remove the `agent_event` line (the Chinese equivalent).

- [ ] **Step 3: Commit**

```bash
git add config/prompts_en.yaml config/prompts_zh.yaml
git commit -m "fix: remove agent_event from chat_analyze prompt classification options"
```

---

### Task 6: Remove `AGENT_EVENT` and Related Types from Enums

**Files:**
- Modify: `opencontext/models/enums.py`

- [ ] **Step 1: Remove enum members from `ContextType`**

In `opencontext/models/enums.py`, remove these four lines from the `ContextType` enum (lines 88-91):

```python
    AGENT_EVENT = "agent_event"
    AGENT_DAILY_SUMMARY = "agent_daily_summary"
    AGENT_WEEKLY_SUMMARY = "agent_weekly_summary"
    AGENT_MONTHLY_SUMMARY = "agent_monthly_summary"
```

- [ ] **Step 2: Remove from `CONTEXT_UPDATE_STRATEGIES`**

Remove these four entries (lines 117-120):

```python
    ContextType.AGENT_EVENT: UpdateStrategy.APPEND,
    ContextType.AGENT_DAILY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.AGENT_WEEKLY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.AGENT_MONTHLY_SUMMARY: UpdateStrategy.APPEND,
```

- [ ] **Step 3: Remove from `CONTEXT_STORAGE_BACKENDS`**

Remove these four entries (lines 138-141):

```python
    ContextType.AGENT_EVENT: "vector_db",  # Vector DB
    ContextType.AGENT_DAILY_SUMMARY: "vector_db",  # Vector DB
    ContextType.AGENT_WEEKLY_SUMMARY: "vector_db",  # Vector DB
    ContextType.AGENT_MONTHLY_SUMMARY: "vector_db",  # Vector DB
```

- [ ] **Step 4: Remove `"agent"` key from `MEMORY_OWNER_TYPES`**

Remove the entire `"agent"` entry (lines 157-162):

```python
    "agent": [
        ContextType.AGENT_EVENT,
        ContextType.AGENT_DAILY_SUMMARY,
        ContextType.AGENT_WEEKLY_SUMMARY,
        ContextType.AGENT_MONTHLY_SUMMARY,
    ],
```

- [ ] **Step 5: Remove from `ContextSimpleDescriptions`**

Remove the entries for `AGENT_EVENT`, `AGENT_DAILY_SUMMARY`, `AGENT_WEEKLY_SUMMARY`, `AGENT_MONTHLY_SUMMARY` (lines 217-236).

- [ ] **Step 6: Remove from `ContextDescriptions`**

Remove the `AGENT_EVENT` entry (lines 337-350).

- [ ] **Step 7: Remove from `SYSTEM_GENERATED_TYPES`**

Remove these three entries (lines 379-381):

```python
    ContextType.AGENT_DAILY_SUMMARY,
    ContextType.AGENT_WEEKLY_SUMMARY,
    ContextType.AGENT_MONTHLY_SUMMARY,
```

- [ ] **Step 8: Compile-check enums only (search/cache will be fixed in the same task below)**

Run:
```bash
python -m py_compile opencontext/models/enums.py
```
Expected: No errors. Note: search and cache code will break until we fix them in the next steps.

**IMPORTANT: Do NOT commit yet. Removing AGENT_EVENT types from MEMORY_OWNER_TYPES breaks the search and cache layers that still reference `memory_owner="agent"`. The following steps (originally Task 7) must be completed in the same commit to avoid an intermediate broken state.**

#### Part B: Remove `memory_owner` from Search API

**Files:**
- Modify: `opencontext/server/search/models.py:73-77`
- Modify: `opencontext/server/search/event_search_service.py`
- Modify: `opencontext/server/routes/search.py`

- [ ] **Step 9: Remove `memory_owner` field from `EventSearchRequest`**

In `opencontext/server/search/models.py`, remove lines 73-77:

```python
    memory_owner: str = Field(
        default="user",
        pattern="^(user|agent)$",
        description="Memory owner: 'user' or 'agent'",
    )
```

- [ ] **Step 10: Update `EventSearchService` — remove `memory_owner` parameter**

In `opencontext/server/search/event_search_service.py`:

**Replace `_get_context_types_for_levels()` (lines 184-191)** — remove `memory_owner` param, return combined user + agent_base types:

```python
@staticmethod
def _get_context_types_for_levels(levels: Optional[List[int]] = None) -> List[str]:
    """Map hierarchy_levels to ContextType values.

    Combines user event types and agent_base types for unified search.
    """
    user_types = [
        ContextType.EVENT,
        ContextType.DAILY_SUMMARY,
        ContextType.WEEKLY_SUMMARY,
        ContextType.MONTHLY_SUMMARY,
    ]
    base_types = [
        ContextType.AGENT_BASE_EVENT,
        ContextType.AGENT_BASE_L1_SUMMARY,
        ContextType.AGENT_BASE_L2_SUMMARY,
        ContextType.AGENT_BASE_L3_SUMMARY,
    ]
    if levels:
        result = []
        for l in levels:
            if l < len(user_types):
                result.append(user_types[l].value)
            if l < len(base_types):
                result.append(base_types[l].value)
        return result
    return [t.value for t in user_types + base_types]
```

**Replace `get_l0_type()` (lines 178-181)** — return user EVENT type:

```python
@staticmethod
def get_l0_type() -> str:
    """Get the L0 event ContextType value."""
    return ContextType.EVENT.value
```

**Update `semantic_search()` signature (line 37-106)** — remove `memory_owner` param, update calls to `_get_context_types_for_levels()` and `collect_ancestors()`.

**Update `filter_search()` signature (line 108-173)** — remove `memory_owner` param.

**Update `collect_ancestors()` (lines 217-259)** — remove `memory_owner` param. Change summary type resolution to use combined types:

```python
async def collect_ancestors(
    self,
    results: List[Tuple[ProcessedContext, float]],
    max_level: int,
) -> Dict[str, ProcessedContext]:
    """Collect ancestors by following refs upward (to summary types)."""
    user_summary_types = {
        ContextType.DAILY_SUMMARY.value,
        ContextType.WEEKLY_SUMMARY.value,
        ContextType.MONTHLY_SUMMARY.value,
    }
    base_summary_types = {
        ContextType.AGENT_BASE_L1_SUMMARY.value,
        ContextType.AGENT_BASE_L2_SUMMARY.value,
        ContextType.AGENT_BASE_L3_SUMMARY.value,
    }
    summary_type_values = user_summary_types | base_summary_types
    # ... rest of method uses summary_type_values unchanged
```

- [ ] **Step 11: Update `search.py` route — remove `memory_owner` references**

In `opencontext/server/routes/search.py`:

- Remove all `memory_owner=request.memory_owner` arguments from `_search_service` calls (lines 101, 152, 162, 172, 187, 196)
- Replace `l0_type = _search_service.get_l0_type(request.memory_owner)` with `l0_type = _search_service.get_l0_type()` (line 152)
- Replace the `owner_types` / `summary_type_values` block (lines 212-213) with the combined types:

```python
    user_summary_types = {
        ContextType.DAILY_SUMMARY.value,
        ContextType.WEEKLY_SUMMARY.value,
        ContextType.MONTHLY_SUMMARY.value,
    }
    base_summary_types = {
        ContextType.AGENT_BASE_L1_SUMMARY.value,
        ContextType.AGENT_BASE_L2_SUMMARY.value,
        ContextType.AGENT_BASE_L3_SUMMARY.value,
    }
    summary_type_values = user_summary_types | base_summary_types
```

- Update `_track_accessed_safe()` — remove `memory_owner` param, update `get_l0_type()` call.

#### Part C: Remove `memory_owner` from Memory Cache

- [ ] **Step 12: Update `memory_cache_manager.py` — remove `memory_owner` from all methods**

In `opencontext/server/cache/memory_cache_manager.py`:

**`_snapshot_key()`** — remove `memory_owner` param:

```python
@staticmethod
def _snapshot_key(user_id: str, device_id: str, agent_id: str) -> str:
    return f"memory_cache:snapshot:{user_id}:{device_id}:{agent_id}"
```

**`invalidate_snapshot()`** — remove `memory_owner` param, update key and log.

**`refresh_snapshot()`** — remove `memory_owner` param, update key, lock key, and `_build_snapshot()` call.

**`get_user_memory_cache()`** — remove `memory_owner` param, update all key and call references.

**`_build_snapshot()`** — remove `memory_owner` param. Change context type resolution:

Replace the `memory_owner`-based type resolution (lines 382-386) and the conditional doc/knowledge skip (lines 433-453):

```python
# Context types — always include user events + agent base events
l0_type = ContextType.EVENT.value
l1_type = ContextType.DAILY_SUMMARY.value
summary_type_values = {
    ContextType.DAILY_SUMMARY.value,
    ContextType.WEEKLY_SUMMARY.value,
    ContextType.MONTHLY_SUMMARY.value,
}

# Profile — use agent_profile if agent_id is not default, else user profile
if agent_id and agent_id != "default":
    profile_context_type = ContextType.AGENT_PROFILE.value
else:
    profile_context_type = ContextType.PROFILE.value
```

The parallel query tasks should include both user events and agent base events. Add base event queries to the tasks dict:

```python
tasks["base_events"] = storage.get_all_processed_contexts(
    context_types=[ContextType.AGENT_BASE_EVENT.value],
    user_id="__base__",
    device_id=device_id,
    agent_id=agent_id,
    limit=20,
)
```

Always include docs/knowledge (remove the `if memory_owner != "agent"` conditional).

In the profile fallback section (lines 480-483), keep the fallback to `agent_base_profile` but remove the `memory_owner` check — instead check if `agent_id != "default"`:

```python
if not profile_data and agent_id and agent_id != "default":
    profile_data = await storage.get_profile(
        "__base__", device_id, agent_id, context_type="agent_base_profile"
    )
```

- [ ] **Step 13: Update `memory_cache.py` route — remove `memory_owner` parameter**

In `opencontext/server/routes/memory_cache.py`:

- Remove `memory_owner` query parameter from `get_user_memory_cache()` (lines 45-48)
- Remove the validation check `if memory_owner not in ("user", "agent")` (lines 68-69)
- Remove `memory_owner=memory_owner` from the `get_user_memory_cache()` call (line 86)
- Remove `memory_owner` query parameter from `invalidate_user_memory_cache()` (lines 111-114)
- Remove `memory_owner=memory_owner` from `invalidate_snapshot()` call (line 119)
- Update response messages to remove `owner=` references

- [ ] **Step 14: Update internal callers of `invalidate_snapshot` and `refresh_snapshot`**

Search for any other callers that pass `memory_owner` to these methods. In `opencontext/server/opencontext.py`, the `_invalidate_user_cache()` method likely calls `invalidate_snapshot()` — remove `memory_owner` argument from that call.

#### Part D: Snapshot assembly for base events

- [ ] **Step 15: Merge base events into snapshot response**

In `_build_snapshot()`, after results are gathered, merge `base_events` into the snapshot's `recent_memories` section. Add base events to `today_events` list (they are always included regardless of date since they are background knowledge):

```python
# After results_map is built
base_events_result = results_map.get("base_events")
if isinstance(base_events_result, Exception):
    base_events_result = []
base_events_result = base_events_result or []

# Add base events as a separate section in recent_memories
```

In the snapshot dict assembly, add:

```python
"base_memories": [self._ctx_to_recent_item(ctx) for ctx in base_events_result],
```

- [ ] **Step 16: Compile-check all files in this task**

Run:
```bash
python -m py_compile opencontext/models/enums.py
python -m py_compile opencontext/server/search/models.py
python -m py_compile opencontext/server/search/event_search_service.py
python -m py_compile opencontext/server/routes/search.py
python -m py_compile opencontext/server/cache/memory_cache_manager.py
python -m py_compile opencontext/server/routes/memory_cache.py
python -m py_compile opencontext/server/opencontext.py
```
Expected: All pass with no errors.

- [ ] **Step 17: Commit**

```bash
git add opencontext/models/enums.py opencontext/server/search/models.py opencontext/server/search/event_search_service.py opencontext/server/routes/search.py opencontext/server/cache/memory_cache_manager.py opencontext/server/routes/memory_cache.py opencontext/server/opencontext.py
git commit -m "refactor: remove AGENT_EVENT types, memory_owner param; include base memory automatically"
```

---

### Task 7: Fix Remaining References and Update Docs

**Files:**
- Modify: `opencontext/web/templates/chat_batches.html`
- Modify: `opencontext/web/templates/contexts.html`
- Modify: `opencontext/context_processing/MODULE.md`
- Modify: `opencontext/models/MODULE.md`
- Modify: `opencontext/periodic_task/MODULE.md`
- Modify: `opencontext/server/MODULE.md`
- Modify: `docs/api_reference.md`
- Modify: `docs/curls.sh`

- [ ] **Step 1: Search for remaining `AGENT_EVENT` and `memory_owner` references**

Run:
```bash
cd /d/尹硕范/repositories/my_MineContext
grep -rn "AGENT_EVENT\|AGENT_DAILY_SUMMARY\|AGENT_WEEKLY_SUMMARY\|AGENT_MONTHLY_SUMMARY" opencontext/ --include="*.py" | grep -v __pycache__
grep -rn "memory_owner" opencontext/ --include="*.py" | grep -v __pycache__
grep -rn "agent_event\|AGENT_EVENT\|memory_owner" opencontext/web/templates/ --include="*.html"
```

Fix any remaining references found.

- [ ] **Step 2: Update web templates**

In `opencontext/web/templates/chat_batches.html` and `opencontext/web/templates/contexts.html`:
- Remove `agent_event`, `agent_daily_summary`, `agent_weekly_summary`, `agent_monthly_summary` from any type filter dropdowns or display logic
- Remove `memory_owner` from any query parameter handling
- Add `agent_commentary` to event display if applicable

- [ ] **Step 3: Update MODULE.md files**

Update these MODULE.md files to reflect the changes:

- `opencontext/context_processing/MODULE.md` — document AgentMemoryProcessor's new role as post-processor, the DAG dependency model
- `opencontext/models/MODULE.md` — document `agent_commentary` field, removed types
- `opencontext/periodic_task/MODULE.md` — document `AgentProfileUpdateTask`
- `opencontext/server/MODULE.md` — document removal of `memory_owner` from search and cache APIs

- [ ] **Step 4: Update API docs**

In `docs/api_reference.md`:
- Remove `memory_owner` parameter from search and memory-cache endpoint docs
- Document `agent_commentary` field in response schemas

In `docs/curls.sh`:
- Remove `memory_owner` from search and memory-cache curl examples

- [ ] **Step 5: Compile-check all Python files that were changed**

Run:
```bash
python -m py_compile opencontext/server/opencontext.py
```
Plus any other files modified in step 1.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: update MODULE.md, templates, and API docs for agent memory redesign"
```

---

### Task 8: Full Integration Verification

- [ ] **Step 1: Run full compile check on all modified modules**

```bash
python -m py_compile opencontext/models/context.py
python -m py_compile opencontext/models/enums.py
python -m py_compile opencontext/context_processing/processor/base_processor.py
python -m py_compile opencontext/context_processing/processor/agent_memory_processor.py
python -m py_compile opencontext/context_processing/processor/text_chat_processor.py
python -m py_compile opencontext/context_processing/processor/document_processor.py
python -m py_compile opencontext/managers/processor_manager.py
python -m py_compile opencontext/periodic_task/agent_profile_update.py
python -m py_compile opencontext/periodic_task/__init__.py
python -m py_compile opencontext/server/component_initializer.py
python -m py_compile opencontext/server/routes/push.py
python -m py_compile opencontext/server/routes/search.py
python -m py_compile opencontext/server/routes/memory_cache.py
python -m py_compile opencontext/server/search/event_search_service.py
python -m py_compile opencontext/server/search/models.py
python -m py_compile opencontext/server/cache/memory_cache_manager.py
python -m py_compile opencontext/server/opencontext.py
```

Expected: All pass with no errors.

- [ ] **Step 2: Verify no remaining references to removed types**

```bash
grep -rn "AGENT_EVENT\|agent_event\|AGENT_DAILY_SUMMARY\|AGENT_WEEKLY_SUMMARY\|AGENT_MONTHLY_SUMMARY" opencontext/ --include="*.py" | grep -v __pycache__ | grep -v "# removed" | grep -v ".pyc"
```

Expected: No matches (or only in migration notes/comments).

- [ ] **Step 3: Run formatter**

```bash
uv run black opencontext --line-length 100
uv run isort opencontext
```

- [ ] **Step 4: Commit formatting if needed**

```bash
git add -A
git commit -m "style: format code after agent memory redesign"
```
