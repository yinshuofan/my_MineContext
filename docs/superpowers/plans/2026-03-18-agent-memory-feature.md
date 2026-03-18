# Agent Memory Feature — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable AI agents (e.g., novel characters) to have their own memory. Agents have base memories (background profile + events pushed via API) and interaction memories (extracted from conversations via LLM). Memory is served through the existing search and cache endpoints with `memory_owner` parameterization.

**Architecture:** Agent CRUD is in `server/routes/agents.py`. Agent base memory (profile + events) uses direct storage writes (no LLM). Agent interaction memory is extracted by `AgentMemoryProcessor` — a new processor registered in `BATCH_PROCESSOR_MAP` and invoked via `processors: ["agent_memory"]` on the push endpoint. Profile merge for agents falls back to the base profile on first interaction.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, MySQL/SQLite, Qdrant/VikingDB, LLM (VLM client)

**Spec:** `docs/superpowers/specs/2026-03-18-agent-memory-design.md` (Sections 2-6)

**Prerequisites:** Plan 1 (refs + ContextType) and Plan 2 (buffer removal + chat batch) must be completed first.

**Verification:** No test suite. Use `python -m py_compile opencontext/path/to/file.py`.

---

### Task 1: Add agent_registry Table

**Files:**
- Modify: `opencontext/storage/backends/mysql_backend.py` (DDL + CRUD)
- Modify: `opencontext/storage/backends/sqlite_backend.py` (DDL + CRUD)
- Modify: `opencontext/storage/base_storage.py` (interface)
- Modify: `opencontext/storage/unified_storage.py` (delegation)

- [ ] **Step 1: Add agent_registry DDL to MySQL backend**

In `_create_tables()`, add:

```sql
CREATE TABLE IF NOT EXISTS agent_registry (
    agent_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

- [ ] **Step 2: Add agent CRUD methods to MySQL backend**

```python
async def create_agent(self, agent_id: str, name: str, description: str = "") -> bool:
    """Register a new agent."""

async def get_agent(self, agent_id: str) -> Optional[Dict]:
    """Get agent by ID (excludes soft-deleted)."""

async def list_agents(self) -> List[Dict]:
    """List all active agents."""

async def update_agent(self, agent_id: str, name: Optional[str] = None,
                       description: Optional[str] = None) -> bool:
    """Update agent info."""

async def delete_agent(self, agent_id: str) -> bool:
    """Soft delete agent (set is_deleted=TRUE)."""
```

- [ ] **Step 3: Mirror in SQLite backend**

Same DDL (SQLite syntax) and methods.

- [ ] **Step 4: Add interface methods to base_storage.py**

In `IDocumentStorageBackend`, add abstract methods for all 5 agent CRUD operations.

- [ ] **Step 5: Add delegation to unified_storage.py**

Delegate all 5 methods to `self._document_backend`.

- [ ] **Step 6: Compile check**

```bash
python -m py_compile opencontext/storage/backends/mysql_backend.py && \
python -m py_compile opencontext/storage/backends/sqlite_backend.py && \
python -m py_compile opencontext/storage/base_storage.py && \
python -m py_compile opencontext/storage/unified_storage.py
```

- [ ] **Step 7: Commit**

```bash
git add opencontext/storage/
git commit -m "feat(storage): add agent_registry table with soft delete"
```

---

### Task 2: Create Agent CRUD Routes

**Files:**
- Create: `opencontext/server/routes/agents.py`
- Modify: `opencontext/server/api.py:42-55` (register router)

- [ ] **Step 1: Create agents.py route module**

Create `opencontext/server/routes/agents.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from opencontext.server.utils import get_context_lab
from opencontext.server.middleware.auth import auth_dependency
from opencontext.storage.global_storage import get_storage

router = APIRouter(prefix="/api/agents", tags=["agents"])


class CreateAgentRequest(BaseModel):
    agent_id: str = Field(..., max_length=100)
    name: str = Field(..., max_length=255)
    description: str = ""


class UpdateAgentRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None


@router.post("", dependencies=[Depends(auth_dependency)])
async def create_agent(request: CreateAgentRequest):
    storage = get_storage()
    success = await storage.create_agent(
        request.agent_id, request.name, request.description
    )
    if not success:
        raise HTTPException(400, "Agent creation failed (ID may already exist)")
    return {"success": True, "agent_id": request.agent_id}


@router.get("", dependencies=[Depends(auth_dependency)])
async def list_agents():
    storage = get_storage()
    agents = await storage.list_agents()
    return {"success": True, "agents": agents}


@router.get("/{agent_id}", dependencies=[Depends(auth_dependency)])
async def get_agent(agent_id: str):
    storage = get_storage()
    agent = await storage.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return {"success": True, "agent": agent}


@router.put("/{agent_id}", dependencies=[Depends(auth_dependency)])
async def update_agent(agent_id: str, request: UpdateAgentRequest):
    storage = get_storage()
    success = await storage.update_agent(agent_id, request.name, request.description)
    if not success:
        raise HTTPException(404, "Agent not found or update failed")
    return {"success": True}


@router.delete("/{agent_id}", dependencies=[Depends(auth_dependency)])
async def delete_agent(agent_id: str):
    storage = get_storage()
    success = await storage.delete_agent(agent_id)
    if not success:
        raise HTTPException(404, "Agent not found")
    return {"success": True}
```

- [ ] **Step 2: Register router in api.py**

At `opencontext/server/api.py`, add import and include:

```python
from .routes import agents
router.include_router(agents.router)
```

- [ ] **Step 3: Compile check**

```bash
python -m py_compile opencontext/server/routes/agents.py && \
python -m py_compile opencontext/server/api.py
```

- [ ] **Step 4: Commit**

```bash
git add opencontext/server/routes/agents.py opencontext/server/api.py
git commit -m "feat(api): add agent CRUD routes (/api/agents)"
```

---

### Task 3: Add Agent Base Memory Routes

**Files:**
- Modify: `opencontext/server/routes/agents.py` (add base profile + events endpoints)

- [ ] **Step 1: Add base profile endpoints**

```python
class BaseProfileRequest(BaseModel):
    factual_profile: str
    behavioral_profile: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    importance: int = 0


@router.post("/{agent_id}/base/profile", dependencies=[Depends(auth_dependency)])
async def set_base_profile(agent_id: str, request: BaseProfileRequest):
    storage = get_storage()
    agent = await storage.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    success = await storage.upsert_profile(
        user_id="__base__", device_id="default", agent_id=agent_id,
        owner_type="agent",
        factual_profile=request.factual_profile,
        behavioral_profile=request.behavioral_profile,
        entities=request.entities,
        importance=request.importance,
    )
    if not success:
        raise HTTPException(500, "Failed to save base profile")
    return {"success": True}


@router.get("/{agent_id}/base/profile", dependencies=[Depends(auth_dependency)])
async def get_base_profile(agent_id: str):
    storage = get_storage()
    profile = await storage.get_profile(
        user_id="__base__", device_id="default", agent_id=agent_id,
        owner_type="agent",
    )
    if not profile:
        raise HTTPException(404, "Base profile not found")
    return {"success": True, "profile": profile}
```

- [ ] **Step 2: Add base events endpoints**

```python
class BaseEventItem(BaseModel):
    title: str
    summary: str
    event_time: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    importance: int = 5


class BaseEventsRequest(BaseModel):
    events: List[BaseEventItem] = Field(..., min_length=1)


@router.post("/{agent_id}/base/events", dependencies=[Depends(auth_dependency)])
async def push_base_events(agent_id: str, request: BaseEventsRequest):
    """Push base events (structured, no LLM). Generates embeddings and stores."""
    from opencontext.models.context import (
        ProcessedContext, ContextProperties, ExtractedData, Vectorize,
        RawContextProperties,
    )
    from opencontext.models.enums import ContextType, ContextSource, ContentFormat
    from opencontext.llm.global_embedding_client import do_vectorize
    import datetime

    storage = get_storage()
    agent = await storage.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    contexts = []
    for event in request.events:
        event_time = (
            datetime.datetime.fromisoformat(event.event_time)
            if event.event_time
            else datetime.datetime.now(tz=datetime.timezone.utc)
        )
        text_for_embedding = f"{event.title}\n{event.summary}\n{', '.join(event.keywords)}"
        vectorize = Vectorize(
            input=[{"type": "text", "text": text_for_embedding}],
            content_format=ContentFormat.TEXT,
        )
        await do_vectorize(vectorize)

        ctx = ProcessedContext(
            properties=ContextProperties(
                raw_properties=[],
                create_time=datetime.datetime.now(tz=datetime.timezone.utc),
                update_time=datetime.datetime.now(tz=datetime.timezone.utc),
                event_time=event_time,
                time_bucket=event_time.strftime("%Y-%m-%dT%H:%M:%S"),
                is_processed=True,
                agent_id=agent_id,
            ),
            extracted_data=ExtractedData(
                title=event.title,
                summary=event.summary,
                keywords=event.keywords,
                entities=event.entities,
                context_type=ContextType.AGENT_EVENT,
                importance=event.importance,
                confidence=10,
            ),
            vectorize=vectorize,
        )
        contexts.append(ctx)

    success = await storage.batch_upsert_processed_context(contexts)
    return {"success": success, "count": len(contexts)}
```

- [ ] **Step 3: Add base events list/edit/delete endpoints**

```python
@router.get("/{agent_id}/base/events", dependencies=[Depends(auth_dependency)])
async def list_base_events(agent_id: str, limit: int = 50, offset: int = 0):
    storage = get_storage()
    contexts = await storage.get_all_processed_contexts(
        context_types=[ContextType.AGENT_EVENT.value],
        agent_id=agent_id,
        top_k=limit,
    )
    # Filter to base events (no user_id or user_id is None)
    base_events = [c for c in contexts if not c.properties.user_id]
    return {"success": True, "events": [
        ProcessedContextModel.from_processed_context(c, Path("."))
        for c in base_events[offset:offset+limit]
    ]}


@router.delete("/{agent_id}/base/events/{event_id}", dependencies=[Depends(auth_dependency)])
async def delete_base_event(agent_id: str, event_id: str):
    storage = get_storage()
    success = await storage.delete_context(event_id, ContextType.AGENT_EVENT.value)
    if not success:
        raise HTTPException(404, "Event not found")
    return {"success": True}
```

- [ ] **Step 4: Compile check**

Run: `python -m py_compile opencontext/server/routes/agents.py`

- [ ] **Step 5: Commit**

```bash
git add opencontext/server/routes/agents.py
git commit -m "feat(api): add agent base memory routes (profile + events)"
```

---

### Task 4: Add Agent Memory Processor

**Files:**
- Create: `opencontext/context_processing/processor/agent_memory_processor.py`
- Modify: `opencontext/context_processing/processor/processor_factory.py:41-51`
- Modify: `opencontext/managers/processor_manager.py` (BATCH_PROCESSOR_MAP)

- [ ] **Step 1: Create AgentMemoryProcessor**

Create `opencontext/context_processing/processor/agent_memory_processor.py`:

```python
"""Agent Memory Processor — extracts memories from the agent's perspective."""

import json
import datetime
from typing import Any, List, Optional, Dict

from opencontext.context_processing.processor.base_processor import BaseContextProcessor
from opencontext.models.context import (
    ProcessedContext, ContextProperties, ExtractedData, Vectorize,
    RawContextProperties,
)
from opencontext.models.enums import ContextType, ContextSource, ContentFormat
from opencontext.config.global_config import GlobalConfig
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.storage.global_storage import get_storage
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AgentMemoryProcessor(BaseContextProcessor):
    def __init__(self):
        super().__init__({})

    def get_name(self) -> str:
        return "agent_memory_processor"

    def get_description(self) -> str:
        return "Extracts memories from the agent's subjective perspective."

    def can_process(self, context: Any) -> bool:
        if not isinstance(context, RawContextProperties):
            return False
        return context.source == ContextSource.CHAT_LOG

    async def process(self, context: RawContextProperties) -> List[ProcessedContext]:
        try:
            return await self._process_async(context)
        except Exception as e:
            logger.error(f"Agent memory processing failed: {e}")
            return []

    async def _process_async(self, raw_context: RawContextProperties) -> List[ProcessedContext]:
        # 1. Load agent info
        agent_id = raw_context.agent_id
        if not agent_id:
            logger.debug("No agent_id in context, skipping agent memory processing")
            return []

        storage = get_storage()
        agent = await storage.get_agent(agent_id) if storage else None
        if not agent:
            logger.debug(f"Agent {agent_id} not registered, skipping")
            return []

        agent_name = agent.get("name", agent_id)
        agent_description = agent.get("description", "")

        # 2. Load prompt
        config = GlobalConfig.get_instance()
        prompt_group = config.get_prompt_group("processing.extraction.agent_memory_analyze")
        if not prompt_group:
            logger.warning("agent_memory_analyze prompt not found")
            return []

        # 3. Build LLM messages
        system_prompt = prompt_group.get("system", "")
        system_prompt = system_prompt.replace("{agent_name}", agent_name)
        system_prompt = system_prompt.replace("{agent_description}", agent_description)

        chat_history = raw_context.content_text or ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_history},
        ]

        # 4. Call LLM
        response = await generate_with_messages(messages)
        if not response:
            return []

        # 5. Parse response
        analysis = parse_json_from_response(response)
        if not analysis:
            return []

        memories = analysis.get("memories", [])
        if not memories:
            return []

        # 6. Build ProcessedContext for each memory
        batch_id = (raw_context.additional_info or {}).get("batch_id")
        results = []
        for memory in memories:
            ctx = self._build_agent_context(memory, raw_context, batch_id)
            if ctx:
                results.append(ctx)

        return results

    def _build_agent_context(
        self, memory: Dict, raw_context: RawContextProperties, batch_id: Optional[str]
    ) -> Optional[ProcessedContext]:
        """Build a ProcessedContext from an agent memory extraction."""
        try:
            mem_type = memory.get("type", "agent_event")
            title = str(memory.get("title", "Untitled"))[:500]
            summary = str(memory.get("summary", ""))
            keywords = memory.get("keywords", [])[:20]
            entities = memory.get("entities", [])[:50]
            importance = max(0, min(10, int(memory.get("importance", 5))))

            # Parse event_time
            event_time_str = memory.get("event_time")
            try:
                event_time = datetime.datetime.fromisoformat(event_time_str) if event_time_str else raw_context.create_time
            except (ValueError, TypeError):
                event_time = raw_context.create_time

            # Determine context_type
            if mem_type == "profile":
                context_type = ContextType.PROFILE
            else:
                context_type = ContextType.AGENT_EVENT

            extracted_data = ExtractedData(
                title=title,
                summary=summary,
                keywords=keywords,
                entities=entities,
                context_type=context_type,
                importance=importance,
                confidence=7,
            )

            properties = ContextProperties(
                raw_properties=[raw_context],
                create_time=datetime.datetime.now(tz=datetime.timezone.utc),
                update_time=datetime.datetime.now(tz=datetime.timezone.utc),
                event_time=event_time,
                time_bucket=event_time.strftime("%Y-%m-%dT%H:%M:%S"),
                is_processed=True,
                user_id=raw_context.user_id,
                device_id=raw_context.device_id,
                agent_id=raw_context.agent_id,
                raw_type="chat_batch" if batch_id else None,
                raw_id=batch_id,
            )

            text = f"{title}\n{summary}\n{', '.join(keywords)}"
            vectorize = Vectorize(
                input=[{"type": "text", "text": text}],
                content_format=ContentFormat.TEXT,
            )

            return ProcessedContext(
                properties=properties,
                extracted_data=extracted_data,
                vectorize=vectorize,
            )

        except Exception as e:
            logger.warning(f"Failed to build agent context: {e}")
            return None
```

- [ ] **Step 2: Register in ProcessorFactory**

At `opencontext/context_processing/processor/processor_factory.py:41-51`, add:

```python
from opencontext.context_processing.processor.agent_memory_processor import AgentMemoryProcessor

def _register_built_in_processors(self) -> None:
    self.register_processor_type("document_processor", DocumentProcessor)
    self.register_processor_type("text_chat_processor", TextChatProcessor)
    self.register_processor_type("agent_memory_processor", AgentMemoryProcessor)
```

- [ ] **Step 3: Add to BATCH_PROCESSOR_MAP**

At `opencontext/managers/processor_manager.py`, uncomment/add the agent_memory entry:

```python
BATCH_PROCESSOR_MAP = {
    "user_memory": "text_chat_processor",
    "agent_memory": "agent_memory_processor",
}
```

- [ ] **Step 4: Compile check**

```bash
python -m py_compile opencontext/context_processing/processor/agent_memory_processor.py && \
python -m py_compile opencontext/context_processing/processor/processor_factory.py && \
python -m py_compile opencontext/managers/processor_manager.py
```

- [ ] **Step 5: Commit**

```bash
git add opencontext/context_processing/processor/agent_memory_processor.py \
        opencontext/context_processing/processor/processor_factory.py \
        opencontext/managers/processor_manager.py
git commit -m "feat(processor): add AgentMemoryProcessor for agent-perspective extraction

Extracts agent events and profile updates from conversations using
agent-specific LLM prompts. Registered in BATCH_PROCESSOR_MAP as
'agent_memory'."
```

---

### Task 5: Add Agent Memory Prompts

**Files:**
- Modify: `config/prompts_en.yaml`
- Modify: `config/prompts_zh.yaml`

- [ ] **Step 1: Add English prompt**

In `config/prompts_en.yaml`, under `processing.extraction`, add:

```yaml
  agent_memory_analyze:
    system: |
      You are {agent_name}. {agent_description}

      Based on the following conversation with a user, extract memories from YOUR perspective.

      Extract two types:
      1. "profile" — What you learned about this user, how your perception of them changed
      2. "agent_event" — What happened from your point of view, your feelings and reactions

      Output JSON:
      {
        "memories": [
          {
            "type": "agent_event" | "profile",
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
      For events: describe what happened from your subjective point of view — your feelings, reactions, evaluations.
      For profile: describe what you now know or feel about the user — their personality, preferences, relationship with you.
      If nothing noteworthy happened, return {"memories": []}.
```

- [ ] **Step 2: Add Chinese prompt**

Mirror in `config/prompts_zh.yaml` with Chinese translation.

- [ ] **Step 3: Commit**

```bash
git add config/prompts_en.yaml config/prompts_zh.yaml
git commit -m "feat(prompts): add agent_memory_analyze prompt for agent perspective extraction"
```

---

### Task 6: Handle Agent Profile in _handle_processed_context

**Files:**
- Modify: `opencontext/server/opencontext.py:133-181` (_handle_processed_context)

- [ ] **Step 1: Route agent profile updates through refresh_profile with owner_type**

In `_handle_processed_context()`, where profile contexts are processed (the `_store_profile` call), detect if the context came from the agent memory processor (check if `agent_id` is a registered agent) and pass `owner_type="agent"`:

```python
async def _store_profile(self, ctx: ProcessedContext):
    # Determine owner_type: if agent_id is a registered agent, this is an agent profile
    owner_type = "user"
    if ctx.properties.agent_id:
        storage = get_storage()
        agent = await storage.get_agent(ctx.properties.agent_id)
        if agent:
            owner_type = "agent"

    await refresh_profile(
        new_factual_profile=ctx.extracted_data.summary or "",
        new_entities=ctx.extracted_data.entities,
        new_importance=ctx.extracted_data.importance,
        new_metadata=ctx.metadata,
        user_id=ctx.properties.user_id or "default",
        device_id=ctx.properties.device_id or "default",
        agent_id=ctx.properties.agent_id or "default",
        owner_type=owner_type,
    )
```

This uses the `owner_type` parameter added in Plan 1, Task 4. The base profile fallback (for first interaction) is already handled in `refresh_profile()`.

- [ ] **Step 2: Compile check**

Run: `python -m py_compile opencontext/server/opencontext.py`

- [ ] **Step 3: Commit**

```bash
git add opencontext/server/opencontext.py
git commit -m "feat(routing): route agent profile updates with owner_type='agent'

Detects agent profile contexts by checking agent_registry. Uses
refresh_profile with owner_type='agent' for LLM merge + base profile
fallback."
```

---

### Task 7: Update Documentation

**Files:**
- Modify: `opencontext/server/MODULE.md`
- Modify: `opencontext/context_processing/MODULE.md`
- Modify: `opencontext/models/MODULE.md`
- Modify: `docs/api_reference.md`
- Modify: `docs/curls.sh`

- [ ] **Step 1: Update server MODULE.md**

Document agent CRUD routes, base memory routes, `memory_owner` parameter on search/cache.

- [ ] **Step 2: Update context_processing MODULE.md**

Document `AgentMemoryProcessor` — its role, prompt usage, output types.

- [ ] **Step 3: Update API documentation**

In `docs/api_reference.md` and `docs/curls.sh`, add all new agent endpoints with request/response examples.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: add Agent Memory feature documentation"
```
