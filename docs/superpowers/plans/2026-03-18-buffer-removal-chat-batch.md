# Buffer Removal + Chat Batch Persistence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Redis-based chat message buffer, persist raw chat messages in a `chat_batches` table, and add multi-processor parallel dispatch via a `processors` parameter on the push endpoint.

**Architecture:** The buffer mode (`process_mode="buffer"`) is removed entirely. Every push is direct processing. Raw messages are persisted to `chat_batches` before processing, ensuring no message loss even if processing fails. `ProcessorManager.process_batch()` enables parallel dispatch to multiple processors (e.g., `user_memory` + `agent_memory`). This is a prerequisite for the Agent Memory feature (Plan 3).

**Tech Stack:** Python 3.10+, FastAPI, Redis, MySQL/SQLite, asyncio

**Spec:** `docs/superpowers/specs/2026-03-18-agent-memory-design.md` (Sections 1.1, 1.2, 3.1-3.3)

**Verification:** No test suite. Use `python -m py_compile opencontext/path/to/file.py`.

**Important:** Only the chat message buffer in `TextChatCapture` is removed. All other Redis cache usage (memory cache snapshots, recently accessed tracking, scheduler locks, distributed locks) is untouched.

---

### Task 1: Add chat_batches Table to Storage Backends

**Files:**
- Modify: `opencontext/storage/backends/mysql_backend.py` (initialize + new methods)
- Modify: `opencontext/storage/backends/sqlite_backend.py` (initialize + new methods)
- Modify: `opencontext/storage/base_storage.py` (IDocumentStorageBackend interface)
- Modify: `opencontext/storage/unified_storage.py` (delegation)

- [ ] **Step 1: Add chat_batches table DDL to MySQL backend**

In `opencontext/storage/backends/mysql_backend.py`, find `_create_tables()` (called from `initialize()` at line 91). Add:

```sql
CREATE TABLE IF NOT EXISTS chat_batches (
    batch_id VARCHAR(36) PRIMARY KEY,
    messages JSON NOT NULL,
    user_id VARCHAR(255),
    device_id VARCHAR(100) DEFAULT 'default',
    agent_id VARCHAR(100) DEFAULT 'default',
    message_count INT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_chat_batches_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

- [ ] **Step 2: Add create_chat_batch method to MySQL backend**

```python
async def create_chat_batch(
    self, batch_id: str, messages: List[Dict], user_id: Optional[str],
    device_id: str = "default", agent_id: str = "default",
) -> bool:
    """Persist a chat batch. batch_id is app-generated UUID."""
    conn = self._get_connection()
    try:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """INSERT INTO chat_batches
                   (batch_id, messages, user_id, device_id, agent_id, message_count)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (batch_id, json.dumps(messages), user_id, device_id,
                 agent_id, len(messages)),
            )
            await conn.commit()
            return True
    except Exception as e:
        logger.error(f"create_chat_batch failed: {e}")
        return False
```

- [ ] **Step 3: Add cleanup_chat_batches method to MySQL backend**

```python
async def cleanup_chat_batches(self, retention_days: int = 90) -> int:
    """Delete chat batches older than retention_days."""
    conn = self._get_connection()
    try:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM chat_batches WHERE created_at < DATE_SUB(NOW(), INTERVAL %s DAY)",
                (retention_days,),
            )
            await conn.commit()
            return cursor.rowcount
    except Exception as e:
        logger.error(f"cleanup_chat_batches failed: {e}")
        return 0
```

- [ ] **Step 4: Mirror in SQLite backend**

Add the same DDL (with SQLite syntax) and methods to `opencontext/storage/backends/sqlite_backend.py`.

- [ ] **Step 5: Add interface methods to base_storage.py**

In `IDocumentStorageBackend`, add:

```python
async def create_chat_batch(self, batch_id: str, messages: List[Dict],
                            user_id: Optional[str], device_id: str = "default",
                            agent_id: str = "default") -> bool:
    raise NotImplementedError

async def cleanup_chat_batches(self, retention_days: int = 90) -> int:
    raise NotImplementedError
```

- [ ] **Step 6: Add delegation to unified_storage.py**

```python
async def create_chat_batch(self, batch_id: str, messages: List[Dict],
                            user_id: Optional[str], device_id: str = "default",
                            agent_id: str = "default") -> bool:
    return await self._document_backend.create_chat_batch(
        batch_id, messages, user_id, device_id, agent_id
    )

async def cleanup_chat_batches(self, retention_days: int = 90) -> int:
    return await self._document_backend.cleanup_chat_batches(retention_days)
```

- [ ] **Step 7: Compile check**

```bash
python -m py_compile opencontext/storage/backends/mysql_backend.py && \
python -m py_compile opencontext/storage/backends/sqlite_backend.py && \
python -m py_compile opencontext/storage/base_storage.py && \
python -m py_compile opencontext/storage/unified_storage.py
```

- [ ] **Step 8: Commit**

```bash
git add opencontext/storage/
git commit -m "feat(storage): add chat_batches table and CRUD methods

Stores raw chat messages with batch_id for processor reference.
Includes 90-day cleanup method."
```

---

### Task 2: Add chat_batches Cleanup Periodic Task

**Files:**
- Modify: `opencontext/periodic_task/data_cleanup.py` (or create new file)
- Modify: `opencontext/server/component_initializer.py`

- [ ] **Step 1: Add chat batch cleanup to DataCleanupTask**

In `opencontext/periodic_task/data_cleanup.py`, extend the `execute()` method (around line 72). Add after existing cleanup logic:

```python
# Clean up old chat batches
try:
    storage = get_storage()
    if storage:
        deleted = await storage.cleanup_chat_batches(retention_days=90)
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} chat batches older than 90 days")
except Exception as e:
    logger.warning(f"Chat batch cleanup failed: {e}")
```

- [ ] **Step 2: Compile check**

Run: `python -m py_compile opencontext/periodic_task/data_cleanup.py`

- [ ] **Step 3: Commit**

```bash
git add opencontext/periodic_task/data_cleanup.py
git commit -m "feat(cleanup): add chat_batches cleanup to data cleanup task"
```

---

### Task 3: Add ProcessorManager.process_batch()

**Files:**
- Modify: `opencontext/managers/processor_manager.py:21-133`

- [ ] **Step 1: Add BATCH_PROCESSOR_MAP constant**

At top of `opencontext/managers/processor_manager.py`, add:

```python
BATCH_PROCESSOR_MAP = {
    "user_memory": "text_chat_processor",
    # "agent_memory": "agent_memory_processor",  # Added in Plan 3
}
```

- [ ] **Step 2: Add process_batch() method**

After the existing `process()` method (line 133), add:

```python
async def process_batch(
    self, raw_context: RawContextProperties, processor_names: List[str]
) -> List[ProcessedContext]:
    """Run multiple processors in parallel on the same input, merge outputs."""
    import asyncio

    # Resolve processor names to instances
    tasks = []
    resolved_names = []
    for name in processor_names:
        internal_name = BATCH_PROCESSOR_MAP.get(name)
        if not internal_name:
            logger.warning(f"Unknown processor name: {name}, skipping")
            continue
        processor = self._processors.get(internal_name)
        if not processor:
            logger.warning(f"Processor not registered: {internal_name}, skipping")
            continue
        if not processor.can_process(raw_context):
            logger.debug(f"Processor {internal_name} cannot process this input, skipping")
            continue
        tasks.append(processor.process(raw_context))
        resolved_names.append(internal_name)

    if not tasks:
        logger.warning("No processors available for batch processing")
        return []

    # Run in parallel, tolerant of individual failures
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_contexts = []
    for name, result in zip(resolved_names, results):
        if isinstance(result, Exception):
            logger.error(f"Processor {name} failed: {result}")
            continue
        if result:
            all_contexts.extend(result)

    logger.info(
        f"process_batch: {len(processor_names)} processors → "
        f"{len(all_contexts)} contexts"
    )

    # Invoke callback to route to storage
    if all_contexts and self._callback:
        await self._callback(all_contexts)

    return all_contexts
```

- [ ] **Step 3: Compile check**

Run: `python -m py_compile opencontext/managers/processor_manager.py`

- [ ] **Step 4: Commit**

```bash
git add opencontext/managers/processor_manager.py
git commit -m "feat(processor): add process_batch for parallel multi-processor dispatch

BATCH_PROCESSOR_MAP resolves request-level names to registered
processors. asyncio.gather with return_exceptions=True ensures
individual processor failures don't block others."
```

---

### Task 4: Remove Buffer Mode from TextChatCapture

**Files:**
- Modify: `opencontext/context_capture/text_chat.py`

**Important:** Only remove chat buffer logic. Do NOT remove or modify `RedisCache` methods (`rpush_expire_llen`, `lrange_json`, etc.) — they may be used by other components.

- [ ] **Step 1: Remove buffer-related methods**

Remove these methods from `TextChatCapture`:
- `push_message()` (lines 264-312) — the buffer push method
- `_safe_flush_buffer()` (lines 314-325)
- `_flush_buffer()` (lines 327-368)
- `_flush_all_buffers()` (lines 195-231)
- `_make_buffer_key()` (lines 94-104) — buffer key generation

- [ ] **Step 2: Remove buffer constants and init state**

Remove:
- `BUFFER_KEY_PREFIX = "chat:buffer:"` (line 29)
- Any buffer-related state from `__init__` (lines 31-43): buffer_size, buffer_ttl config reads

- [ ] **Step 3: Simplify _stop_impl**

At lines 84-88, `_stop_impl` calls `_flush_all_buffers()` for graceful shutdown. Remove this call — messages are now persisted in `chat_batches` before processing, so no buffered data can be lost:

```python
def _stop_impl(self, graceful: bool = True) -> bool:
    return True
```

- [ ] **Step 4: Keep process_messages_directly and _create_and_send_context**

These methods (lines 177-193 and 119-175) are the "direct processing" path and must remain. `process_messages_directly()` is the method the push endpoint will call.

- [ ] **Step 5: Compile check**

Run: `python -m py_compile opencontext/context_capture/text_chat.py`

- [ ] **Step 6: Commit**

```bash
git add opencontext/context_capture/text_chat.py
git commit -m "refactor(capture): remove chat message buffer from TextChatCapture

Only direct processing remains. Buffer keys, flush logic, Lua scripts
all removed. RedisCache generic methods untouched."
```

---

### Task 5: Update Push Endpoint

**Files:**
- Modify: `opencontext/server/routes/push.py:72-97` (PushChatRequest)
- Modify: `opencontext/server/routes/push.py:388-490` (push_chat endpoint)

- [ ] **Step 1: Update PushChatRequest model**

At `opencontext/server/routes/push.py:72-97`, remove `process_mode` and `flush_immediately` fields. Add `processors` field:

```python
class PushChatRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., min_length=1, max_length=100)
    user_id: Optional[str] = Field(None, max_length=255)
    device_id: Optional[str] = Field(None, max_length=100)
    agent_id: Optional[str] = Field(None, max_length=100)
    processors: List[str] = Field(
        default=["user_memory"],
        description="Processors to run: 'user_memory', 'agent_memory', etc.",
    )
```

- [ ] **Step 2: Rewrite push_chat endpoint**

At `opencontext/server/routes/push.py:388-490`, replace the buffer/direct branching with:

```python
@router.post("/api/push/chat", ...)
async def push_chat(
    request: PushChatRequest,
    background_tasks: BackgroundTasks,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    import uuid
    from opencontext.storage.global_storage import get_storage

    # 1. Validate messages (existing validation logic)
    messages = request.messages
    # ... multimodal processing if needed ...

    # 2. Persist to chat_batches
    batch_id = str(uuid.uuid4())
    storage = get_storage()
    await storage.create_chat_batch(
        batch_id=batch_id,
        messages=messages,
        user_id=request.user_id,
        device_id=request.device_id or "default",
        agent_id=request.agent_id or "default",
    )

    # 3. Build RawContextProperties
    raw_context = RawContextProperties(
        source=ContextSource.CHAT_LOG,
        content_format=...,  # detect from messages
        content_text=json.dumps(messages),
        user_id=request.user_id,
        device_id=request.device_id,
        agent_id=request.agent_id,
        additional_info={"batch_id": batch_id},
    )

    # 4. Dispatch to processors in background
    async def _process():
        await opencontext.processor_manager.process_batch(
            raw_context, request.processors
        )

    background_tasks.add_task(_process)

    # 5. Schedule tasks (hierarchy, compression) — keep existing logic
    background_tasks.add_task(
        opencontext._schedule_user_task,
        "hierarchy_summary", request.user_id, request.device_id, request.agent_id,
    )

    return {"success": True, "batch_id": batch_id}
```

Preserve existing multimodal message processing (`_process_multimodal_messages`) and validation logic. The key change is replacing the buffer/direct branch with: persist → build raw context → process_batch.

- [ ] **Step 3: Compile check**

Run: `python -m py_compile opencontext/server/routes/push.py`

- [ ] **Step 4: Commit**

```bash
git add opencontext/server/routes/push.py
git commit -m "feat(push): replace buffer mode with direct processing + processors param

process_mode and flush_immediately removed (breaking API change).
Messages persisted to chat_batches before processing. processors
parameter enables multi-processor dispatch."
```

---

### Task 6: Update MODULE.md and Documentation

**Files:**
- Modify: `opencontext/context_capture/MODULE.md`
- Modify: `opencontext/server/MODULE.md`
- Modify: `docs/api_reference.md`
- Modify: `docs/curls.sh`

- [ ] **Step 1: Update context_capture MODULE.md**

Document buffer removal. `TextChatCapture` is now stateless — only `process_messages_directly()` and `_create_and_send_context()` remain.

- [ ] **Step 2: Update server MODULE.md**

Document new `processors` parameter on push/chat. Document `ProcessorManager.process_batch()`. Note breaking change: `process_mode` and `flush_immediately` removed.

- [ ] **Step 3: Update API docs**

In `docs/api_reference.md` and `docs/curls.sh`, update the push/chat endpoint documentation to reflect the new request format (remove `process_mode`/`flush_immediately`, add `processors`).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: update documentation for buffer removal and processors parameter"
```
