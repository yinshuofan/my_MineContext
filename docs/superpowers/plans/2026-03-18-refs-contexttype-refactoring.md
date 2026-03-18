# refs + ContextType Refactoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `parent_id`/`children_ids` with a generic `refs` field, expand ContextType to include hierarchy summary types and agent event types, and add `owner_type` to the profiles table.

**Architecture:** This is a foundation-level refactoring. Changes flow bottom-up: models → storage → consumers (hierarchy generation, search, cache, tools). All new types and fields are additive first, then consumers migrate, then old fields are removed. The `MEMORY_OWNER_TYPES` mapping enables parameterized search/cache for the upcoming Agent Memory feature.

**Tech Stack:** Python 3.10+, Pydantic, MySQL/SQLite, Qdrant/VikingDB, FastAPI

**Spec:** `docs/superpowers/specs/2026-03-18-agent-memory-design.md` (Sections 1.3, 1.4, 1.5)

**Verification:** This project has no test suite. Verify each change with `python -m py_compile opencontext/path/to/file.py`. Manual API testing for integration verification.

---

### Task 1: Expand ContextType Enum and Mappings

**Files:**
- Modify: `opencontext/models/enums.py:78-84` (ContextType enum)
- Modify: `opencontext/models/enums.py:96-101` (CONTEXT_UPDATE_STRATEGIES)
- Modify: `opencontext/models/enums.py:104-109` (CONTEXT_STORAGE_BACKENDS)
- Modify: `opencontext/models/enums.py:120-141` (ContextSimpleDescriptions)
- Modify: `opencontext/models/enums.py:143-211` (ContextDescriptions)

- [ ] **Step 1: Add new members to ContextType enum**

At `opencontext/models/enums.py:78-84`, add after `KNOWLEDGE = "knowledge"`:

```python
class ContextType(str, Enum):
    """Context type enumeration"""
    PROFILE = "profile"
    DOCUMENT = "document"
    EVENT = "event"
    KNOWLEDGE = "knowledge"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_SUMMARY = "monthly_summary"
    AGENT_EVENT = "agent_event"
    AGENT_DAILY_SUMMARY = "agent_daily_summary"
    AGENT_WEEKLY_SUMMARY = "agent_weekly_summary"
    AGENT_MONTHLY_SUMMARY = "agent_monthly_summary"
```

- [ ] **Step 2: Add entries to CONTEXT_UPDATE_STRATEGIES**

At `opencontext/models/enums.py:96-101`, add entries for all new types:

```python
CONTEXT_UPDATE_STRATEGIES = {
    ContextType.PROFILE: UpdateStrategy.OVERWRITE,
    ContextType.DOCUMENT: UpdateStrategy.OVERWRITE,
    ContextType.EVENT: UpdateStrategy.APPEND,
    ContextType.KNOWLEDGE: UpdateStrategy.APPEND_MERGE,
    ContextType.DAILY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.WEEKLY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.MONTHLY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.AGENT_EVENT: UpdateStrategy.APPEND,
    ContextType.AGENT_DAILY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.AGENT_WEEKLY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.AGENT_MONTHLY_SUMMARY: UpdateStrategy.APPEND,
}
```

- [ ] **Step 3: Add entries to CONTEXT_STORAGE_BACKENDS**

At `opencontext/models/enums.py:104-109`:

```python
CONTEXT_STORAGE_BACKENDS = {
    ContextType.PROFILE: "document_db",
    ContextType.DOCUMENT: "vector_db",
    ContextType.EVENT: "vector_db",
    ContextType.KNOWLEDGE: "vector_db",
    ContextType.DAILY_SUMMARY: "vector_db",
    ContextType.WEEKLY_SUMMARY: "vector_db",
    ContextType.MONTHLY_SUMMARY: "vector_db",
    ContextType.AGENT_EVENT: "vector_db",
    ContextType.AGENT_DAILY_SUMMARY: "vector_db",
    ContextType.AGENT_WEEKLY_SUMMARY: "vector_db",
    ContextType.AGENT_MONTHLY_SUMMARY: "vector_db",
}
```

- [ ] **Step 4: Add MEMORY_OWNER_TYPES mapping**

Add after `CONTEXT_STORAGE_BACKENDS`:

```python
MEMORY_OWNER_TYPES = {
    "user": [ContextType.EVENT, ContextType.DAILY_SUMMARY, ContextType.WEEKLY_SUMMARY, ContextType.MONTHLY_SUMMARY],
    "agent": [ContextType.AGENT_EVENT, ContextType.AGENT_DAILY_SUMMARY, ContextType.AGENT_WEEKLY_SUMMARY, ContextType.AGENT_MONTHLY_SUMMARY],
}
# index 0=L0, 1=L1, 2=L2, 3=L3
```

- [ ] **Step 5: Add ContextSimpleDescriptions for new types**

At `opencontext/models/enums.py:120-141`, add entries for ALL new types (including summaries and agent types). Example for `DAILY_SUMMARY`:

```python
ContextSimpleDescriptions[ContextType.DAILY_SUMMARY] = {
    "name": "Daily Summary",
    "description": "Auto-generated daily summary of user events",
    "purpose": "Provides a condensed view of daily activity",
}
```

Add similar entries for `WEEKLY_SUMMARY`, `MONTHLY_SUMMARY`, `AGENT_EVENT`, `AGENT_DAILY_SUMMARY`, `AGENT_WEEKLY_SUMMARY`, `AGENT_MONTHLY_SUMMARY`.

- [ ] **Step 6: Add ContextDescriptions ONLY for AGENT_EVENT**

**CRITICAL**: Do NOT add summary types (`DAILY_SUMMARY`, `WEEKLY_SUMMARY`, `MONTHLY_SUMMARY`, `AGENT_DAILY_SUMMARY`, etc.) to `ContextDescriptions`. The `get_context_type_descriptions_for_extraction()` function iterates all entries in `ContextDescriptions` to build LLM classification prompts. If summary types are added, the LLM will incorrectly classify user content as summary types. Summary types are system-generated, never LLM-classified.

Only add `AGENT_EVENT` to `ContextDescriptions`:

```python
ContextDescriptions[ContextType.AGENT_EVENT] = {
    "name": "Agent Event",
    "description": "Agent's subjective experience — Records the agent's feelings, reactions, and evaluations about user events from the agent's own perspective.",
    "key_indicators": [
        "Contains the agent's subjective perspective or emotional reaction",
        "Describes what the agent observed, felt, or thought about an interaction",
        "Records the agent's evaluation of user behavior or events",
    ],
    "examples": [
        "He mentioned enjoying poetry today — I found his taste surprisingly refined",
        "The user seemed distracted during our conversation, I wonder what's troubling him",
    ],
    "classification_priority": 5,
}
```

- [ ] **Step 7: Add SYSTEM_GENERATED_TYPES guard to get_context_type_for_analysis**

At `opencontext/models/enums.py:235-253`, add a guard to reject system-generated types that should never come from LLM classification:

```python
SYSTEM_GENERATED_TYPES = {
    ContextType.DAILY_SUMMARY, ContextType.WEEKLY_SUMMARY, ContextType.MONTHLY_SUMMARY,
    ContextType.AGENT_DAILY_SUMMARY, ContextType.AGENT_WEEKLY_SUMMARY, ContextType.AGENT_MONTHLY_SUMMARY,
}

def get_context_type_for_analysis(context_type_str: str) -> ContextType:
    ...
    result = ...  # existing lookup logic
    if result in SYSTEM_GENERATED_TYPES:
        return ContextType.KNOWLEDGE  # fallback — summaries are never LLM-classified
    return result
```

- [ ] **Step 8: Compile check**

Run: `python -m py_compile opencontext/models/enums.py`
Expected: No errors

> **Deployment note (Qdrant)**: After this change, the Qdrant backend will create 7 new empty collections on first startup (one per new ContextType). This is expected behavior — `qdrant_backend.py` initialization iterates all `ContextType` values. Old summaries remain in the `event` collection (found by the old-format dedup check in Task 6). New summaries go to type-specific collections (`daily_summary`, etc.). VikingDB is unaffected (uses single collection with `context_type` field filtering).

- [ ] **Step 9: Commit**

```bash
git add opencontext/models/enums.py
git commit -m "feat(models): expand ContextType with hierarchy summary and agent event types

Add DAILY_SUMMARY, WEEKLY_SUMMARY, MONTHLY_SUMMARY, AGENT_EVENT, and
agent hierarchy types. Add MEMORY_OWNER_TYPES mapping. Update all
strategy/backend/description dicts."
```

---

### Task 2: Add refs Field to Models

**Files:**
- Modify: `opencontext/models/context.py:97-133` (ContextProperties)
- Modify: `opencontext/models/context.py:310-395` (ProcessedContextModel + from_processed_context)

- [ ] **Step 1: Add refs field to ContextProperties**

At `opencontext/models/context.py`, in the `ContextProperties` class (after `time_bucket` at line 133), add:

```python
    refs: Dict[str, List[str]] = Field(default_factory=dict)
```

Ensure `Dict` and `List` are imported from `typing` (they should already be).

- [ ] **Step 2: Add refs field to ProcessedContextModel**

At `opencontext/models/context.py:340-343`, add `refs` after `time_bucket`:

```python
    hierarchy_level: int = 0
    parent_id: Optional[str] = None          # deprecated, use refs
    children_ids: List[str] = Field(default_factory=list)  # deprecated, use refs
    time_bucket: Optional[str] = None
    refs: Dict[str, List[str]] = Field(default_factory=dict)
```

Note: Keep `parent_id` and `children_ids` for now (they'll be removed in a later task after all consumers migrate).

- [ ] **Step 3: Map refs in from_processed_context()**

At `opencontext/models/context.py:391-394`, add `refs` mapping:

```python
            hierarchy_level=pc.properties.hierarchy_level,
            parent_id=pc.properties.parent_id,
            children_ids=pc.properties.children_ids,
            time_bucket=pc.properties.time_bucket,
            refs=pc.properties.refs,
```

- [ ] **Step 4: Compile check**

Run: `python -m py_compile opencontext/models/context.py`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add opencontext/models/context.py
git commit -m "feat(models): add refs field to ContextProperties and ProcessedContextModel

Additive change only. parent_id/children_ids kept for backward
compatibility during migration."
```

---

### Task 3: Add owner_type to Profiles Table

**Files:**
- Modify: `opencontext/storage/backends/mysql_backend.py:191-207` (profiles CREATE TABLE)
- Modify: `opencontext/storage/backends/mysql_backend.py:635-708` (upsert_profile, get_profile)
- Modify: `opencontext/storage/backends/sqlite_backend.py:141-155` (profiles CREATE TABLE)
- Modify: `opencontext/storage/backends/sqlite_backend.py:791-877` (upsert_profile, get_profile)
- Modify: `opencontext/storage/base_storage.py:386-441` (IDocumentStorageBackend interface)
- Modify: `opencontext/storage/unified_storage.py:868-903` (profile delegation methods)

- [ ] **Step 1: Add owner_type to MySQL profiles table DDL and migration**

At `opencontext/storage/backends/mysql_backend.py:191-207`, add `owner_type` column to the CREATE TABLE DDL. Additionally, in the `initialize()` method (around line 91), add an ALTER TABLE migration for existing databases:

```python
# Migration: add owner_type to existing profiles table
try:
    await cursor.execute(
        "ALTER TABLE profiles ADD COLUMN owner_type VARCHAR(50) NOT NULL DEFAULT 'user'"
    )
    await cursor.execute(
        "CREATE INDEX idx_profiles_owner_type ON profiles(owner_type)"
    )
except Exception:
    pass  # Column already exists
```

```sql
CREATE TABLE IF NOT EXISTS profiles (
    user_id VARCHAR(255) NOT NULL,
    device_id VARCHAR(100) NOT NULL DEFAULT 'default',
    agent_id VARCHAR(100) NOT NULL DEFAULT 'default',
    owner_type VARCHAR(50) NOT NULL DEFAULT 'user',
    factual_profile LONGTEXT NOT NULL,
    behavioral_profile LONGTEXT,
    entities JSON,
    importance INT DEFAULT 0,
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, device_id, agent_id),
    INDEX idx_profiles_owner_type (owner_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

- [ ] **Step 2: Add owner_type parameter to MySQL get_profile**

At `opencontext/storage/backends/mysql_backend.py:679-708`, add `owner_type` parameter with default `"user"`:

```python
async def get_profile(
    self,
    user_id: str,
    device_id: str = "default",
    agent_id: str = "default",
    owner_type: str = "user",
) -> Optional[Dict]:
```

Add `owner_type` to the WHERE clause: `AND owner_type = %s`.

- [ ] **Step 3: Add owner_type parameter to MySQL upsert_profile**

At `opencontext/storage/backends/mysql_backend.py:635-677`, add `owner_type` parameter with default `"user"`. Include `owner_type` in the INSERT column list and VALUES. Ensure ON DUPLICATE KEY UPDATE does not update `owner_type`.

- [ ] **Step 4: Add owner_type to SQLite profiles table DDL**

At `opencontext/storage/backends/sqlite_backend.py:141-155`, mirror the MySQL change:

```sql
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT NOT NULL,
    device_id TEXT NOT NULL DEFAULT 'default',
    agent_id TEXT NOT NULL DEFAULT 'default',
    owner_type TEXT NOT NULL DEFAULT 'user',
    factual_profile TEXT NOT NULL,
    ...
    PRIMARY KEY (user_id, device_id, agent_id)
)
```

- [ ] **Step 5: Add owner_type to SQLite get_profile and upsert_profile**

Mirror the MySQL changes at `opencontext/storage/backends/sqlite_backend.py:791-877`.

- [ ] **Step 6: Update base storage interface**

At `opencontext/storage/base_storage.py:386-441`, add `owner_type: str = "user"` parameter to `upsert_profile()`, `get_profile()`, and `delete_profile()` abstract method signatures.

- [ ] **Step 7: Update unified storage delegation**

At `opencontext/storage/unified_storage.py:868-903`, pass `owner_type` through to backend calls in `upsert_profile()`, `get_profile()`, and `delete_profile()`.

- [ ] **Step 8: Compile check all modified files**

```bash
python -m py_compile opencontext/storage/backends/mysql_backend.py && \
python -m py_compile opencontext/storage/backends/sqlite_backend.py && \
python -m py_compile opencontext/storage/base_storage.py && \
python -m py_compile opencontext/storage/unified_storage.py
```

- [ ] **Step 9: Commit**

```bash
git add opencontext/storage/
git commit -m "feat(storage): add owner_type column to profiles table

Defaults to 'user' for backward compatibility. All profile CRUD methods
accept optional owner_type parameter."
```

---

### Task 4: Update refresh_profile for owner_type Support

**Files:**
- Modify: `opencontext/context_processing/processor/profile_processor.py:21-92`

- [ ] **Step 1: Add owner_type parameter to refresh_profile()**

At `opencontext/context_processing/processor/profile_processor.py:21`, add `owner_type: str = "user"` parameter. Pass it through to `storage.get_profile()` and `storage.upsert_profile()` calls.

- [ ] **Step 2: Add base profile fallback for agent interaction profiles**

After the `storage.get_profile()` call (around line 53-57), add fallback logic:

```python
existing_profile = await storage.get_profile(
    user_id, device_id, agent_id, owner_type=owner_type
)
# Fallback: first interaction profile for an agent inherits from base profile
if existing_profile is None and owner_type == "agent" and user_id != "__base__":
    existing_profile = await storage.get_profile(
        "__base__", device_id, agent_id, owner_type="agent"
    )
```

- [ ] **Step 3: Compile check**

Run: `python -m py_compile opencontext/context_processing/processor/profile_processor.py`

- [ ] **Step 4: Commit**

```bash
git add opencontext/context_processing/processor/profile_processor.py
git commit -m "feat(profile): add owner_type support to refresh_profile

Agent interaction profiles fall back to base profile on first write."
```

---

### Task 5: Add refs Support to Vector DB Backends

**Files:**
- Modify: `opencontext/storage/backends/qdrant_backend.py` (payload serialization, batch_update_refs)
- Modify: `opencontext/storage/backends/vikingdb_backend.py:82-85` (field constants), payload serialization, batch_update_refs
- Modify: `opencontext/storage/base_storage.py` (IVectorStorageBackend interface)
- Modify: `opencontext/storage/unified_storage.py` (delegation)

- [ ] **Step 1: Add refs to Qdrant payload serialization**

In `opencontext/storage/backends/qdrant_backend.py`, find the method that converts `ProcessedContext` to Qdrant payload format (look for where `parent_id` and `children_ids` are written to payload). Add alongside them:

```python
data["refs"] = json.dumps(context.properties.refs) if context.properties.refs else "{}"
```

Keep `parent_id` and `children_ids` in payload for now (backward compat). In the deserialization path (where contexts are read back), parse `refs`:

```python
if "refs" in payload:
    props.refs = json.loads(payload["refs"])
```

- [ ] **Step 2: Add batch_update_refs to Qdrant backend**

Add new method to Qdrant backend:

```python
async def batch_update_refs(
    self, context_ids: List[str], ref_key: str, ref_value: str, context_type: str
) -> int:
    """Add a ref entry to multiple contexts."""
    updated = 0
    for ctx_id in context_ids:
        try:
            # Fetch existing point
            points = await self._client.retrieve(
                collection_name=self._get_collection(context_type),
                ids=[ctx_id],
                with_payload=True,
            )
            if not points:
                continue
            existing_refs = json.loads(points[0].payload.get("refs", "{}"))
            if ref_key not in existing_refs:
                existing_refs[ref_key] = []
            if ref_value not in existing_refs[ref_key]:
                existing_refs[ref_key].append(ref_value)
            await self._client.set_payload(
                collection_name=self._get_collection(context_type),
                payload={"refs": json.dumps(existing_refs)},
                points=[ctx_id],
            )
            updated += 1
        except Exception as e:
            logger.warning(f"batch_update_refs failed for {ctx_id}: {e}")
    return updated
```

- [ ] **Step 3: Add refs to VikingDB payload serialization**

In `opencontext/storage/backends/vikingdb_backend.py`, find field constants (line 82-85). Add:

```python
FIELD_REFS = "refs"
```

Find the context-to-VikingDB conversion method. Add alongside `parent_id`/`children_ids`:

```python
data[FIELD_REFS] = json.dumps(context.properties.refs) if context.properties.refs else "{}"
```

In deserialization, parse `refs`:

```python
if FIELD_REFS in doc:
    props.refs = json.loads(doc[FIELD_REFS])
```

- [ ] **Step 4: Add batch_update_refs to VikingDB backend**

Similar pattern to Qdrant but using VikingDB upsert API. Fetch existing docs, update refs JSON, upsert back.

- [ ] **Step 5: Add batch_update_refs to base storage interface**

At `opencontext/storage/base_storage.py`, add abstract method to `IVectorStorageBackend`:

```python
async def batch_update_refs(
    self, context_ids: List[str], ref_key: str, ref_value: str, context_type: str
) -> int:
    """Add a ref entry (ref_key -> ref_value) to the refs dict of multiple contexts."""
    raise NotImplementedError
```

- [ ] **Step 6: Add batch_update_refs to unified storage**

At `opencontext/storage/unified_storage.py`, add delegation method:

```python
async def batch_update_refs(
    self, context_ids: List[str], ref_key: str, ref_value: str, context_type: str
) -> int:
    return await self._vector_backend.batch_update_refs(
        context_ids, ref_key, ref_value, context_type
    )
```

- [ ] **Step 7: Compile check all modified files**

```bash
python -m py_compile opencontext/storage/backends/qdrant_backend.py && \
python -m py_compile opencontext/storage/backends/vikingdb_backend.py && \
python -m py_compile opencontext/storage/base_storage.py && \
python -m py_compile opencontext/storage/unified_storage.py
```

- [ ] **Step 8: Commit**

```bash
git add opencontext/storage/
git commit -m "feat(storage): add refs field to vector DB payloads and batch_update_refs

Refs stored as JSON string alongside existing parent_id/children_ids
for backward compatibility during migration."
```

---

### Task 6: Migrate Hierarchy Summary Generation

**Files:**
- Modify: `opencontext/periodic_task/hierarchy_summary.py`

This is the write-side migration. After this task, new hierarchy summaries are stored with both `refs` AND the old `parent_id`/`children_ids` (dual-write for safety).

- [ ] **Step 1: Import new ContextTypes**

Add imports at top of file:

```python
from opencontext.models.enums import ContextType, MEMORY_OWNER_TYPES
```

- [ ] **Step 2: Update _store_summary to use new ContextType and refs**

At `opencontext/periodic_task/hierarchy_summary.py`, find `_store_summary` (around line 1250). Currently it sets `context_type=ContextType.EVENT` at line 1333. Change the `context_type` assignment based on the `level` parameter:

```python
# Map hierarchy level to new ContextType
LEVEL_TO_CONTEXT_TYPE = {
    1: ContextType.DAILY_SUMMARY,
    2: ContextType.WEEKLY_SUMMARY,
    3: ContextType.MONTHLY_SUMMARY,
}
summary_context_type = LEVEL_TO_CONTEXT_TYPE.get(level, ContextType.EVENT)
```

Set `context_type=summary_context_type` instead of `ContextType.EVENT`.

- [ ] **Step 3: Write refs on the summary context**

In `_store_summary`, where `children_ids` is set (around line 1353), also set `refs`:

```python
# Determine child ContextType for refs key
LEVEL_TO_CHILD_TYPE = {
    1: ContextType.EVENT,           # L1 contains L0 events
    2: ContextType.DAILY_SUMMARY,   # L2 contains L1 summaries
    3: ContextType.WEEKLY_SUMMARY,  # L3 contains L2 summaries (and possibly L1s)
}
child_type = LEVEL_TO_CHILD_TYPE.get(level, ContextType.EVENT)

properties = ContextProperties(
    ...
    children_ids=children_ids,  # keep for backward compat
    refs={child_type.value: children_ids},  # new refs format
    ...
)
```

- [ ] **Step 4: Replace batch_set_parent_id with batch_update_refs**

At line 1395-1409, replace the parent backfill logic:

```python
# Backfill refs on child contexts (pointing to this summary)
if children_ids:
    try:
        updated = await storage.batch_update_refs(
            children_ids,
            ref_key=summary_context_type.value,  # e.g., "daily_summary"
            ref_value=summary_id,
            context_type=child_type.value,
        )
        logger.info(
            f"Set refs on {updated}/{len(children_ids)} children "
            f"→ {summary_context_type.value}:{summary_id}"
        )
    except Exception as e:
        logger.warning(f"Failed to backfill refs for summary: {e}")
```

Also keep the old `batch_set_parent_id` call alongside for backward compat during migration.

- [ ] **Step 5: Update search_hierarchy calls in summary generation**

Find all calls to `storage.search_hierarchy()` in the file (lines 735, 805, 906, 930, 999, 1040, 1059). These currently pass `context_type=ContextType.EVENT.value` and `hierarchy_level=N`. Update to use new ContextType values:

- L1 queries: `context_type=ContextType.DAILY_SUMMARY.value` (was EVENT + hierarchy_level=1)
- L2 queries: `context_type=ContextType.WEEKLY_SUMMARY.value` (was EVENT + hierarchy_level=2)
- L0 queries: `context_type=ContextType.EVENT.value` (unchanged)

**Important — dedup check compatibility**: The hierarchy generation code calls `search_hierarchy()` to check if a summary already exists before generating. Currently it queries `context_type=EVENT, hierarchy_level=1`. After migration, new summaries use `context_type=DAILY_SUMMARY`. The dedup check must query BOTH the old format (existing data) and new format (new data) to avoid regenerating summaries that already exist in either format:

```python
# Dedup check: look for existing summary in both old and new format
existing_old = await storage.search_hierarchy(
    context_type=ContextType.EVENT.value, hierarchy_level=level, ...
)
existing_new = await storage.search_hierarchy(
    context_type=summary_context_type.value, hierarchy_level=level, ...
)
if existing_old or existing_new:
    logger.info("Summary already exists, skipping")
    return
```

The `search_hierarchy` method signature still expects `hierarchy_level`. Keep passing it — the storage backend uses it for filtering. After all existing data is migrated (manual script), the old-format check can be removed.

- [ ] **Step 6: Compile check**

Run: `python -m py_compile opencontext/periodic_task/hierarchy_summary.py`

- [ ] **Step 7: Commit**

```bash
git add opencontext/periodic_task/hierarchy_summary.py
git commit -m "feat(hierarchy): migrate summary generation to new ContextTypes and refs

Summaries now stored as DAILY_SUMMARY/WEEKLY_SUMMARY/MONTHLY_SUMMARY
instead of EVENT. Refs written alongside parent_id/children_ids for
backward compat."
```

---

### Task 7: Migrate Search to refs and New ContextTypes

**Files:**
- Modify: `opencontext/server/routes/search.py:37` (EVENT_TYPE constant)
- Modify: `opencontext/server/routes/search.py:296-350` (_collect_ancestors)
- Modify: `opencontext/server/routes/search.py:411-414` (_normalize_parent_id)
- Modify: `opencontext/server/search/models.py:44` (hierarchy_levels)
- Modify: `opencontext/server/search/models.py:90-103` (EventNode)

- [ ] **Step 1: Add memory_owner to EventSearchRequest**

At `opencontext/server/search/models.py`, add `memory_owner` field to `EventSearchRequest`:

```python
memory_owner: str = Field(default="user", description="Memory owner: 'user' or 'agent'")
```

- [ ] **Step 2: Update EventNode to include refs**

At `opencontext/server/search/models.py:90-103`, add `refs` and keep `parent_id` for now:

```python
class EventNode(BaseModel):
    ...
    parent_id: Optional[str] = None  # deprecated, kept for backward compat
    refs: Dict[str, List[str]] = Field(default_factory=dict)
    ...
```

- [ ] **Step 3: Replace EVENT_TYPE with dynamic type resolution**

At `opencontext/server/routes/search.py:37`, replace:

```python
EVENT_TYPE = ContextType.EVENT.value
```

With a helper function:

```python
from opencontext.models.enums import MEMORY_OWNER_TYPES

def _get_l0_type(memory_owner: str) -> str:
    """Get the L0 event ContextType value for a memory owner."""
    types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
    return types[0].value  # index 0 = L0

def _get_context_types_for_levels(memory_owner: str, levels: Optional[List[int]]) -> List[str]:
    """Map hierarchy_levels + memory_owner to ContextType values."""
    types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
    if levels:
        return [types[l].value for l in levels if l < len(types)]
    return [t.value for t in types]
```

Update all usages of `EVENT_TYPE` in the file to use `_get_l0_type(request.memory_owner)` or `_get_context_types_for_levels(...)`.

- [ ] **Step 4: Migrate _collect_ancestors to use refs**

At `opencontext/server/routes/search.py:296-350`, replace the `parent_id` traversal with `refs` traversal:

```python
async def _collect_ancestors(storage, results, max_level, memory_owner="user"):
    """Collect ancestors by following refs upward (to summary types).

    Refs key convention: a child points to its parent by storing the parent's
    ContextType.value as the key. E.g., an L0 event has refs={"daily_summary": ["ds_001"]}.
    Summary types (L1+) are the "upward" direction.
    """
    owner_types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
    # Summary types = indices 1+ (L1, L2, L3) — these are "upward" refs
    summary_type_values = {t.value for t in owner_types[1:]}

    ancestor_map = {}  # child_id -> [parent_ids]
    all_ancestors = {}  # ancestor_id -> ProcessedContext
    seen = set()
    current_batch = []

    # Seed: collect parent IDs from search hit refs
    for ctx, score in results:
        if not ctx.properties or not ctx.properties.refs:
            # Fallback to old parent_id field during migration
            if ctx.properties and ctx.properties.parent_id:
                pid = ctx.properties.parent_id
                if pid not in seen:
                    seen.add(pid)
                    current_batch.append(pid)
                    ancestor_map.setdefault(ctx.id, []).append(pid)
            continue
        for ref_key, ref_ids in ctx.properties.refs.items():
            if ref_key in summary_type_values:
                for pid in ref_ids:
                    if pid not in seen:
                        seen.add(pid)
                        current_batch.append(pid)
                        ancestor_map.setdefault(ctx.id, []).append(pid)

    # BFS upward: fetch parents, then follow their upward refs
    rounds = 0
    while current_batch and rounds < 3:
        parents = await storage.get_contexts_by_ids(current_batch)
        next_batch = []
        for parent in parents:
            all_ancestors[parent.id] = parent
            if not parent.properties or not parent.properties.refs:
                # Fallback to old parent_id
                if parent.properties and parent.properties.parent_id:
                    pid = parent.properties.parent_id
                    if pid not in seen:
                        seen.add(pid)
                        next_batch.append(pid)
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

**Important notes**:
- Includes a fallback to `parent_id` for contexts created before the refs migration.
- `get_contexts_by_ids(current_batch)` is called WITHOUT a `context_type` parameter because ancestors may span multiple collections (e.g., L0 in `event`, L1 in `daily_summary`). In Qdrant, this means scanning all collections per BFS round. This is a known performance trade-off that is acceptable: drill-up typically involves 1-3 rounds with a small batch of IDs, and the operation is already I/O-bound. If performance becomes an issue, the caller could pass candidate context types derived from `MEMORY_OWNER_TYPES` to narrow the search.
- `__base__` profiles referenced in Task 4 are created by the Agent Memory feature (Plan 3 — agent CRUD routes). During this plan, the fallback is a forward-looking hook that safely returns `None` when no base profile exists.

- [ ] **Step 5: Update EventNode construction to populate refs**

At `opencontext/server/routes/search.py`, find where `EventNode` objects are constructed (around lines 434, 458). Add `refs` field:

```python
EventNode(
    ...
    parent_id=_normalize_parent_id(props),  # keep for compat
    refs=props.refs if props else {},
    ...
)
```

- [ ] **Step 6: Compile check**

```bash
python -m py_compile opencontext/server/routes/search.py && \
python -m py_compile opencontext/server/search/models.py
```

- [ ] **Step 7: Commit**

```bash
git add opencontext/server/routes/search.py opencontext/server/search/models.py
git commit -m "feat(search): migrate to refs-based hierarchy traversal and memory_owner

Replace hardcoded EVENT_TYPE with dynamic type resolution via
MEMORY_OWNER_TYPES. _collect_ancestors now follows refs upward."
```

---

### Task 8: Migrate Cache Manager

**Files:**
- Modify: `opencontext/server/cache/memory_cache_manager.py`
- Modify: `opencontext/server/cache/models.py`
- Modify: `opencontext/server/routes/memory_cache.py`

- [ ] **Step 1: Rename UserMemoryCacheManager to MemoryCacheManager**

Rename the class. Update all internal references. Keep a type alias for backward compat if needed:

```python
class MemoryCacheManager:
    ...

UserMemoryCacheManager = MemoryCacheManager  # backward compat alias
```

- [ ] **Step 2: Add memory_owner parameter to cache methods**

Update `get_user_memory_cache()` (or renamed equivalent) to accept `memory_owner: str = "user"`. Update the snapshot key:

```python
@staticmethod
def _snapshot_key(memory_owner, user_id, device_id, agent_id) -> str:
    return f"memory_cache:snapshot:{memory_owner}:{user_id}:{device_id}:{agent_id}"
```

> **Deployment note**: Key format change orphans existing Redis cached snapshots. They will NOT be retrieved or invalidated, but will expire naturally via TTL (default 300s). No manual cleanup needed.

- [ ] **Step 3: Parameterize snapshot building by memory_owner**

At `_build_snapshot()` (around line 342-505), use `memory_owner` to determine which context types to query:

```python
types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
l0_type = types[0].value  # EVENT or AGENT_EVENT
l1_type = types[1].value  # DAILY_SUMMARY or AGENT_DAILY_SUMMARY

# Profile query
profile = await storage.get_profile(
    user_id, device_id, agent_id,
    owner_type="agent" if memory_owner == "agent" else "user"
)

# Today events
today_events = await storage.get_all_processed_contexts(
    context_types=[l0_type], ...
)

# Daily summaries
daily_summaries = await storage.search_hierarchy(
    context_type=l1_type, hierarchy_level=1, ...
)
```

For `memory_owner="agent"`, skip docs/knowledge queries.

- [ ] **Step 4: Update cache route to accept memory_owner**

At `opencontext/server/routes/memory_cache.py`, add `memory_owner: str = Query(default="user")` parameter. Pass through to cache manager.

- [ ] **Step 5: Update children_ids reference in cache manager**

At line 472, replace `ctx.properties.children_ids` with reading from `ctx.properties.refs`:

```python
child_count = 0
if ctx.properties.refs:
    for key, ids in ctx.properties.refs.items():
        if key != l1_type:  # don't count parent refs
            child_count += len(ids)
```

- [ ] **Step 6: Compile check**

```bash
python -m py_compile opencontext/server/cache/memory_cache_manager.py && \
python -m py_compile opencontext/server/routes/memory_cache.py
```

- [ ] **Step 7: Commit**

```bash
git add opencontext/server/cache/ opencontext/server/routes/memory_cache.py
git commit -m "feat(cache): parameterize memory cache by memory_owner

Rename UserMemoryCacheManager to MemoryCacheManager. Snapshot building
uses MEMORY_OWNER_TYPES for dynamic type resolution. Cache key includes
memory_owner prefix."
```

---

### Task 9: Migrate Hierarchical Event Tool

**Files:**
- Modify: `opencontext/tools/retrieval_tools/hierarchical_event_tool.py`

- [ ] **Step 1: Update drill-down to use refs instead of children_ids**

At line 259, replace:

```python
children_ids = getattr(parent_ctx.properties, "children_ids", None) or []
```

With:

```python
from opencontext.models.enums import MEMORY_OWNER_TYPES

# Derive summary type values dynamically (all L1+ types across all owners)
_ALL_SUMMARY_TYPES = {
    t.value for types in MEMORY_OWNER_TYPES.values() for t in types[1:]
}

# Get child IDs from refs (exclude upward/parent refs)
children_ids = []
if parent_ctx.properties and parent_ctx.properties.refs:
    for key, ids in parent_ctx.properties.refs.items():
        if key not in _ALL_SUMMARY_TYPES:
            children_ids.extend(ids)
# Fallback to old field
if not children_ids:
    children_ids = getattr(parent_ctx.properties, "children_ids", None) or []
```

- [ ] **Step 2: Update hierarchy_level references**

At lines 219, 314, 347, 363, update `search_hierarchy` calls to use new ContextType values where applicable. Keep `hierarchy_level` parameter for storage compatibility.

- [ ] **Step 3: Update result formatting**

At lines 365-366, add `refs` to output:

```python
"refs": props.refs if props else {},
"parent_id": props.parent_id,      # keep for compat
"children_ids": props.children_ids or [],  # keep for compat
```

- [ ] **Step 4: Compile check**

Run: `python -m py_compile opencontext/tools/retrieval_tools/hierarchical_event_tool.py`

- [ ] **Step 5: Commit**

```bash
git add opencontext/tools/retrieval_tools/hierarchical_event_tool.py
git commit -m "feat(tools): migrate hierarchical event tool to use refs

Drill-down reads from refs dict with fallback to children_ids."
```

---

### Task 10: Update Prompt Files

**Files:**
- Modify: `config/prompts_en.yaml`
- Modify: `config/prompts_zh.yaml`

- [ ] **Step 1: Add AGENT_EVENT type description to English prompts**

Find the section in `config/prompts_en.yaml` where context types are described for LLM classification. Add `agent_event` description alongside existing types.

- [ ] **Step 2: Add AGENT_EVENT type description to Chinese prompts**

Mirror in `config/prompts_zh.yaml`.

- [ ] **Step 3: Add summary type descriptions if referenced in classification prompts**

If the classification prompts list all context types, add brief descriptions for `daily_summary`, `weekly_summary`, `monthly_summary` and their agent counterparts. Mark them as system-generated (not for user content classification).

- [ ] **Step 4: Commit**

```bash
git add config/prompts_en.yaml config/prompts_zh.yaml
git commit -m "feat(prompts): add agent_event and summary type descriptions

Both language files updated in sync."
```

---

### Task 11: Remove Deprecated Fields (Cleanup)

**Files:**
- Modify: `opencontext/models/context.py` (ContextProperties, ProcessedContextModel)
- Modify: All files that still read `parent_id`/`children_ids`

**Prerequisite:** Tasks 1-10 must all be complete and verified working.

- [ ] **Step 1: Remove parent_id and children_ids from ContextProperties**

At `opencontext/models/context.py:130-132`, remove:

```python
    parent_id: Optional[str] = None             # REMOVE
    children_ids: List[str] = Field(default_factory=list)  # REMOVE
```

- [ ] **Step 2: Remove parent_id and children_ids from ProcessedContextModel**

At `opencontext/models/context.py:341-342`, remove:

```python
    parent_id: Optional[str] = None             # REMOVE
    children_ids: List[str] = Field(default_factory=list)  # REMOVE
```

Remove the corresponding lines from `from_processed_context()` (lines 392-393).

- [ ] **Step 3: Remove _normalize_parent_id from search.py**

Remove the helper function at lines 411-414 and any remaining `parent_id` references.

- [ ] **Step 4: Remove parent_id from EventNode**

At `opencontext/server/search/models.py:103`, remove `parent_id` field.

- [ ] **Step 5: Remove batch_set_parent_id from storage layer**

Remove from: `base_storage.py`, `unified_storage.py`, `qdrant_backend.py`, `vikingdb_backend.py`.

- [ ] **Step 6: Remove parent_id/children_ids from vector DB payload serialization**

In Qdrant and VikingDB backends, stop writing `parent_id` and `children_ids` to payloads. Only write `refs`.

- [ ] **Step 7: Remove old field constants from VikingDB**

At `opencontext/storage/backends/vikingdb_backend.py:83-85`, remove `FIELD_PARENT_ID` and `FIELD_CHILDREN_IDS`.

- [ ] **Step 8: Compile check all modified files**

```bash
python -m py_compile opencontext/models/context.py && \
python -m py_compile opencontext/server/routes/search.py && \
python -m py_compile opencontext/server/search/models.py && \
python -m py_compile opencontext/storage/base_storage.py && \
python -m py_compile opencontext/storage/unified_storage.py && \
python -m py_compile opencontext/storage/backends/qdrant_backend.py && \
python -m py_compile opencontext/storage/backends/vikingdb_backend.py && \
python -m py_compile opencontext/periodic_task/hierarchy_summary.py && \
python -m py_compile opencontext/tools/retrieval_tools/hierarchical_event_tool.py && \
python -m py_compile opencontext/server/cache/memory_cache_manager.py
```

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: remove deprecated parent_id/children_ids fields

All consumers now use refs. batch_set_parent_id removed from storage
layer. EventNode and ProcessedContextModel no longer expose parent_id
or children_ids."
```

---

### Task 12: Update MODULE.md Files

**Files:**
- Modify: `opencontext/models/MODULE.md`
- Modify: `opencontext/storage/MODULE.md`
- Modify: `opencontext/server/MODULE.md`
- Modify: `opencontext/periodic_task/MODULE.md`
- Modify: `opencontext/tools/MODULE.md`

- [ ] **Step 1: Update models MODULE.md**

Document the new ContextType members, refs field, MEMORY_OWNER_TYPES mapping.

- [ ] **Step 2: Update storage MODULE.md**

Document `batch_update_refs`, `owner_type` on profiles, removal of `batch_set_parent_id`.

- [ ] **Step 3: Update server MODULE.md**

Document `memory_owner` parameter on search and cache endpoints. Document `MemoryCacheManager` rename.

- [ ] **Step 4: Update periodic_task MODULE.md**

Document new ContextType usage in hierarchy generation.

- [ ] **Step 5: Update tools MODULE.md**

Document refs-based drill-down in hierarchical event tool.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: update MODULE.md files for refs and ContextType changes"
```
