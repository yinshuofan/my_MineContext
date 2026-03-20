# Unify Time Filtering Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `time_bucket` string filtering with unified numeric timestamp range overlap using `event_time_start` + `event_time_end`, remove `created_at`/`created_at_ts`, and remove LLM `event_time` extraction from processors.

**Architecture:** Rename `event_time` → `event_time_start`, add `event_time_end` (both `datetime`). Storage layers auto-generate `_ts` numeric suffixes. All search paths use range overlap: `event_time_start_ts <= query_end AND event_time_end_ts >= query_start`. Delete `time_bucket`, `created_at`, `created_at_ts` from vector DB.

**Tech Stack:** Python 3.10+, Pydantic v2, FastAPI, Qdrant, VikingDB

**Spec:** `docs/superpowers/specs/2026-03-20-unify-time-filtering-design.md`

**Verification:** No test suite. Use `python -m py_compile opencontext/path/to/file.py` after each task.

---

## Chunk 1: Model + Storage Foundation

### Task 1: Update ContextProperties and ProcessedContextModel

**Files:**
- Modify: `opencontext/models/context.py:97-132` (ContextProperties), `:213-249` (get_llm_context_string), `:309-392` (ProcessedContextModel)

- [ ] **Step 1: Update ContextProperties fields**

In `opencontext/models/context.py`, line 106: rename `event_time` → `event_time_start`. Line 131: delete `time_bucket`. Add `event_time_end` with `model_validator`:

```python
# Line 106 — rename:
event_time_start: datetime.datetime  # event time range start

# After line 106 — add:
event_time_end: datetime.datetime  # event time range end (equals start for point events)

# Line 131 — DELETE this line:
# time_bucket: Optional[str] = None

# Inside the ContextProperties class body, after field declarations, add model_validator:
@model_validator(mode="before")
@classmethod
def _default_event_time_end(cls, data):
    if isinstance(data, dict) and "event_time_end" not in data and "event_time_start" in data:
        data["event_time_end"] = data["event_time_start"]
    return data
```

Update the Pydantic import (line 18) to include `model_validator`:
```python
from pydantic import BaseModel, Field, field_validator, model_validator
```

- [ ] **Step 2: Update get_llm_context_string()**

Lines 238-247: replace `event_time` reference and delete `time_bucket` block:

```python
# Lines 238-239 — replace:
event_time_start = self.properties.event_time_start
parts.append(f"event time: {event_time_start.isoformat()}")
if self.properties.event_time_end != event_time_start:
    parts.append(f"event time end: {self.properties.event_time_end.isoformat()}")

# Lines 246-247 — DELETE:
# if self.properties.time_bucket:
#     parts.append(f"time bucket: {self.properties.time_bucket}")
```

Also update line 244-245 where `hierarchy_level` references `self.properties.hierarchy_level` — no change needed there.

- [ ] **Step 3: Update ProcessedContextModel**

Line 329: rename `event_time` → `event_time_start`. Line 340: delete `time_bucket`. Add `event_time_end`.

```python
# Line 329 — rename:
event_time_start: str

# After line 329 — add:
event_time_end: str

# Line 340 — DELETE:
# time_bucket: Optional[str] = None
```

- [ ] **Step 4: Update from_processed_context()**

Line 380: rename. Line 390: delete. Add `event_time_end`.

```python
# Line 380 — rename:
event_time_start=pc.properties.event_time_start.strftime("%Y-%m-%d %H:%M:%S"),

# After — add:
event_time_end=pc.properties.event_time_end.strftime("%Y-%m-%d %H:%M:%S"),

# Line 390 — DELETE:
# time_bucket=pc.properties.time_bucket,
```

- [ ] **Step 5: Verify**

Run: `python -m py_compile opencontext/models/context.py`

- [ ] **Step 6: Commit**

```bash
git add opencontext/models/context.py
git commit -m "refactor: rename event_time to event_time_start, add event_time_end, remove time_bucket from models"
```

---

### Task 2: Update storage interfaces

**Files:**
- Modify: `opencontext/storage/base_storage.py:233-258`
- Modify: `opencontext/storage/unified_storage.py:924-946`

- [ ] **Step 1: Update base_storage.py interface**

Lines 233-258: change `search_by_hierarchy` signature — replace `time_bucket_start/end: str` with `time_start/end: Optional[float]`:

```python
async def search_by_hierarchy(
    self,
    context_type: str,
    hierarchy_level: int,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    user_id: Optional[str] = None,
    device_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    top_k: int = 20,
) -> List[Tuple[ProcessedContext, float]]:
    """Search contexts by hierarchy level and time range.

    Uses numeric timestamp range overlap:
    event_time_start_ts <= time_end AND event_time_end_ts >= time_start

    Args:
        context_type: The context type to search
        hierarchy_level: Hierarchy level (0=original, 1=daily, 2=weekly, 3=monthly)
        time_start: Start of query range (Unix timestamp), or None for no lower bound
        time_end: End of query range (Unix timestamp), or None for no upper bound
        user_id: Filter by user
        device_id: Filter by device
        agent_id: Filter by agent
        top_k: Max results
    """
```

- [ ] **Step 2: Update unified_storage.py wrapper**

Lines 924-946: match new signature, pass through:

```python
@_require_backend("_vector_backend", default=[])
async def search_hierarchy(
    self,
    context_type: str,
    hierarchy_level: int,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    user_id: Optional[str] = None,
    device_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    top_k: int = 20,
) -> List[Tuple[ProcessedContext, float]]:
    """Search by hierarchy level and time range → vector DB"""
    return await self._vector_backend.search_by_hierarchy(
        context_type=context_type,
        hierarchy_level=hierarchy_level,
        time_start=time_start,
        time_end=time_end,
        user_id=user_id,
        device_id=device_id,
        agent_id=agent_id,
        top_k=top_k,
    )
```

- [ ] **Step 3: Verify**

Run: `python -m py_compile opencontext/storage/base_storage.py && python -m py_compile opencontext/storage/unified_storage.py`

- [ ] **Step 4: Commit**

```bash
git add opencontext/storage/base_storage.py opencontext/storage/unified_storage.py
git commit -m "refactor: update search_hierarchy interface to use numeric timestamps"
```

---

### Task 3: Update VikingDB backend

**Files:**
- Modify: `opencontext/storage/backends/vikingdb_backend.py`

- [ ] **Step 1: Update field constants**

Lines 42-47, 81: Delete old constants, add new ones.

```python
# Lines 42-43 — DELETE:
# FIELD_CREATED_AT = "created_at"
# FIELD_CREATED_AT_TS = "created_at_ts"

# Lines 46-47 — REPLACE with:
FIELD_EVENT_TIME_START = "event_time_start"
FIELD_EVENT_TIME_START_TS = "event_time_start_ts"
FIELD_EVENT_TIME_END = "event_time_end"
FIELD_EVENT_TIME_END_TS = "event_time_end_ts"

# Line 81 — DELETE:
# FIELD_TIME_BUCKET = "time_bucket"
```

- [ ] **Step 2: Update schema definitions**

Lines 617-622, 644: Update `_create_collection` field definitions.

```python
# Lines 617-618 — DELETE created_at fields:
# {"FieldName": FIELD_CREATED_AT, "FieldType": "string"},
# {"FieldName": FIELD_CREATED_AT_TS, "FieldType": "float32"},

# Lines 621-622 — REPLACE event_time fields:
{"FieldName": FIELD_EVENT_TIME_START, "FieldType": "string"},
{"FieldName": FIELD_EVENT_TIME_START_TS, "FieldType": "float32"},
{"FieldName": FIELD_EVENT_TIME_END, "FieldType": "string"},
{"FieldName": FIELD_EVENT_TIME_END_TS, "FieldType": "float32"},

# Line 644 — DELETE time_bucket:
# {"FieldName": FIELD_TIME_BUCKET, "FieldType": "string"},
```

- [ ] **Step 3: Update ScalarIndex fields**

In the ScalarIndex field list (near lines 700-708), replace `FIELD_CREATED_AT_TS` with nothing (remove it), replace `FIELD_EVENT_TIME_TS` with `FIELD_EVENT_TIME_START_TS` and add `FIELD_EVENT_TIME_END_TS`.

- [ ] **Step 4: Update RANGE_SUPPORTED_FIELDS and TIME_FIELD_MAPPING**

Lines 1278-1282: Clean up `TIME_FIELD_MAPPING` — remove `FIELD_CREATED_AT` references:

```python
TIME_FIELD_MAPPING = {
    "create_time": FIELD_CREATE_TIME_TS,
}
```

Lines 1284-1295: Replace `RANGE_SUPPORTED_FIELDS`:

```python
RANGE_SUPPORTED_FIELDS = {
    FIELD_CREATE_TIME_TS,
    FIELD_EVENT_TIME_START_TS,
    FIELD_EVENT_TIME_END_TS,
    FIELD_UPDATE_TIME_TS,
    FIELD_LAST_CALL_TIME_TS,
    FIELD_CONFIDENCE,
    FIELD_IMPORTANCE,
    FIELD_CALL_COUNT,
    FIELD_MERGE_COUNT,
    FIELD_DURATION_COUNT,
}
```

- [ ] **Step 5: Update _context_to_doc_format**

Lines 840-842: Delete `created_at` assignment.

```python
# DELETE these 3 lines:
# now = datetime.datetime.now()
# fields[FIELD_CREATED_AT] = now.isoformat()
# fields[FIELD_CREATED_AT_TS] = now.timestamp()
```

- [ ] **Step 6: Replace sort field in all query methods**

Lines 991, 1443, 1549: Replace `FIELD_CREATED_AT_TS` with `FIELD_CREATE_TIME_TS`.

```python
# Each location: change
"field": FIELD_CREATED_AT_TS,
# to
"field": FIELD_CREATE_TIME_TS,
```

- [ ] **Step 7: Rewrite search_by_hierarchy**

Lines 1469-1587: Replace entire method with numeric timestamp filtering.

```python
async def search_by_hierarchy(
    self,
    context_type: str,
    hierarchy_level: int,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    user_id: Optional[str] = None,
    device_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    top_k: int = 20,
) -> List[Tuple[ProcessedContext, float]]:
    if not self._initialized:
        return []

    try:
        conditions = [
            {"op": "must", "field": FIELD_DATA_TYPE, "conds": [DATA_TYPE_CONTEXT]},
            {"op": "must", "field": FIELD_CONTEXT_TYPE, "conds": [context_type]},
            {"op": "must", "field": FIELD_HIERARCHY_LEVEL, "conds": [int(hierarchy_level)]},
        ]

        if user_id:
            conditions.append({"op": "must", "field": FIELD_USER_ID, "conds": [user_id]})
        if device_id:
            conditions.append({"op": "must", "field": FIELD_DEVICE_ID, "conds": [device_id]})
        if agent_id:
            conditions.append({"op": "must", "field": FIELD_AGENT_ID, "conds": [agent_id]})

        # Numeric range overlap: event_time_start_ts <= time_end AND event_time_end_ts >= time_start
        if time_end is not None:
            conditions.append({
                "op": "range", "field": FIELD_EVENT_TIME_START_TS, "lte": float(time_end),
            })
        if time_start is not None:
            conditions.append({
                "op": "range", "field": FIELD_EVENT_TIME_END_TS, "gte": float(time_start),
            })

        filter_dict = {"op": "and", "conds": conditions} if len(conditions) > 1 else conditions[0]

        data = {
            "collection_name": self._collection_name,
            "index_name": self._index_name,
            "limit": top_k,
            "field": FIELD_CREATE_TIME_TS,
            "order": "desc",
            "filter": filter_dict,
        }

        result = await self._client.async_data_request(
            path="/api/vikingdb/data/search/scalar", data=data,
        )

        if result.get("code") != "Success":
            logger.error(f"Failed to search by hierarchy: {result.get('message')}")
            return []

        output = result.get("result", {}).get("data", [])
        results = []
        for item in output:
            doc = {"id": item.get("id")}
            doc.update(item.get("fields", {}))
            context = self._doc_to_context(doc, need_vector=False)
            if context:
                results.append((context, 1.0))

        return results[:top_k]

    except Exception as e:
        logger.exception(f"Failed to search by hierarchy: {e}")
        return []
```

- [ ] **Step 8: Verify**

Run: `python -m py_compile opencontext/storage/backends/vikingdb_backend.py`

- [ ] **Step 9: Commit**

```bash
git add opencontext/storage/backends/vikingdb_backend.py
git commit -m "refactor: VikingDB backend — numeric time filtering, remove time_bucket/created_at"
```

---

### Task 4: Update Qdrant backend

**Files:**
- Modify: `opencontext/storage/backends/qdrant_backend.py:665-753`

- [ ] **Step 1: Rewrite search_by_hierarchy**

Lines 665-753: Replace entire method with numeric timestamp filtering.

```python
async def search_by_hierarchy(
    self,
    context_type: str,
    hierarchy_level: int,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    user_id: Optional[str] = None,
    device_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    top_k: int = 20,
) -> List[Tuple[ProcessedContext, float]]:
    if not self._initialized:
        return []

    if context_type not in self._collections:
        logger.warning(f"Collection not found for context_type: {context_type}")
        return []

    collection_name = self._collections[context_type]

    must_conditions = [
        models.FieldCondition(
            key="hierarchy_level",
            match=models.MatchValue(value=hierarchy_level),
        )
    ]

    if user_id:
        must_conditions.append(
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        )
    if device_id:
        must_conditions.append(
            models.FieldCondition(key="device_id", match=models.MatchValue(value=device_id))
        )
    if agent_id:
        must_conditions.append(
            models.FieldCondition(key="agent_id", match=models.MatchValue(value=agent_id))
        )

    # Numeric range overlap: event_time_start_ts <= time_end AND event_time_end_ts >= time_start
    if time_end is not None:
        must_conditions.append(
            models.FieldCondition(
                key="event_time_start_ts", range=models.Range(lte=time_end),
            )
        )
    if time_start is not None:
        must_conditions.append(
            models.FieldCondition(
                key="event_time_end_ts", range=models.Range(gte=time_start),
            )
        )

    filter_condition = models.Filter(must=must_conditions)

    try:
        records, _ = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for point in records:
            context = self._qdrant_result_to_context(point, need_vector=False)
            if context:
                results.append((context, 1.0))

        return results

    except Exception as e:
        logger.exception(
            f"search_by_hierarchy failed for context_type={context_type}, "
            f"hierarchy_level={hierarchy_level}: {e}"
        )
        return []
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile opencontext/storage/backends/qdrant_backend.py`

- [ ] **Step 3: Commit**

```bash
git add opencontext/storage/backends/qdrant_backend.py
git commit -m "refactor: Qdrant backend — numeric time filtering, remove time_bucket workaround"
```

---

## Chunk 2: Write Path

### Task 5: Update processors

**Files:**
- Modify: `opencontext/context_processing/processor/text_chat_processor.py:270-377`
- Modify: `opencontext/context_processing/processor/agent_memory_processor.py:300-344`
- Modify: `opencontext/context_processing/processor/document_processor.py:243`

- [ ] **Step 1: Update text_chat_processor.py**

Lines 270-280: Delete LLM event_time extraction. Line 296: Delete time_bucket. Lines 364-377: Update ContextProperties.

```python
# Lines 270-280 — REPLACE entire block with:
event_time_start = raw_context.create_time

# Line 296 — DELETE:
# time_bucket = event_time.strftime("%Y-%m-%dT%H:%M:%S")

# Lines 364-377 — Update ContextProperties constructor:
# Change event_time=event_time to event_time_start=event_time_start
# Delete time_bucket=time_bucket
properties=ContextProperties(
    raw_properties=[raw_context],
    create_time=raw_context.create_time,
    update_time=datetime.datetime.now(),
    event_time_start=event_time_start,
    is_processed=True,
    enable_merge=enable_merge,
    user_id=raw_context.user_id,
    device_id=raw_context.device_id,
    agent_id=raw_context.agent_id,
    raw_type="chat_batch" if batch_id else None,
    raw_id=batch_id,
),
```

- [ ] **Step 2: Update agent_memory_processor.py**

Lines 300-313: Delete LLM event_time extraction. Line 329: Delete time_bucket. Lines 331-344: Update ContextProperties. Lines 220-222: Update display formatting.

```python
# Lines 300-313 — REPLACE entire block with:
event_time_start = raw_context.create_time or datetime.datetime.now(tz=datetime.timezone.utc)

# Line 329 — DELETE:
# time_bucket = event_time.strftime("%Y-%m-%dT%H:%M:%S")

# Lines 331-344 — Update ContextProperties constructor:
properties = ContextProperties(
    raw_properties=[raw_context],
    create_time=raw_context.create_time or datetime.datetime.now(tz=datetime.timezone.utc),
    update_time=datetime.datetime.now(tz=datetime.timezone.utc),
    event_time_start=event_time_start,
    is_processed=True,
    enable_merge=enable_merge,
    user_id=raw_context.user_id,
    device_id=raw_context.device_id,
    agent_id=raw_context.agent_id,
    raw_type="chat_batch" if batch_id else None,
    raw_id=batch_id,
)

# Lines 220-222 — Update display formatting:
event_time_str = ctx.properties.event_time_start.strftime("%Y-%m-%d") if ctx.properties else ""
lines.append(f"[{event_time_str}] {title}")
```

- [ ] **Step 3: Update document_processor.py**

Line 243: rename `event_time=now` → `event_time_start=now`. Search for any other `event_time` or `time_bucket` references in this file and update.

- [ ] **Step 4: Verify all three**

```bash
python -m py_compile opencontext/context_processing/processor/text_chat_processor.py
python -m py_compile opencontext/context_processing/processor/agent_memory_processor.py
python -m py_compile opencontext/context_processing/processor/document_processor.py
```

- [ ] **Step 5: Commit**

```bash
git add opencontext/context_processing/processor/
git commit -m "refactor: processors — remove LLM event_time extraction, use event_time_start"
```

---

### Task 6: Update merger

**Files:**
- Modify: `opencontext/context_processing/merger/context_merger.py:347,356-373,772`
- Modify: `opencontext/context_processing/merger/merge_strategies.py:237`

- [ ] **Step 1: Update context_merger.py**

```python
# Line 347 — rename:
event_time_start=target.properties.event_time_start,
event_time_end=target.properties.event_time_end,

# Lines 356-373 — rename assignment target:
# Change all `properties.event_time = ...` to `properties.event_time_start = ...`
# Also set event_time_end to match (knowledge contexts are point-in-time):
#   properties.event_time_start = datetime.datetime.fromisoformat(...)
#   properties.event_time_end = properties.event_time_start

# Line 772 — rename:
event_time_start=context.properties.event_time_start,
event_time_end=context.properties.event_time_end,
```

- [ ] **Step 2: Update merge_strategies.py**

```python
# Line 237 — rename:
event_time_start=target.properties.event_time_start,
event_time_end=target.properties.event_time_end,
```

- [ ] **Step 3: Verify**

```bash
python -m py_compile opencontext/context_processing/merger/context_merger.py
python -m py_compile opencontext/context_processing/merger/merge_strategies.py
```

- [ ] **Step 4: Commit**

```bash
git add opencontext/context_processing/merger/
git commit -m "refactor: merger — event_time → event_time_start/end"
```

---

### Task 7: Update hierarchy summary task

**Files:**
- Modify: `opencontext/periodic_task/hierarchy_summary.py`

- [ ] **Step 1: Update _store_summary**

Replace `time_bucket` parameter with `event_time_start`/`event_time_end` (datetime). Delete `_parse_event_time_from_bucket`. In the ContextProperties construction (around line 1462-1474):

```python
# Change signature to accept event_time_start and event_time_end instead of time_bucket
# Remove hierarchy_level from ContextProperties (keep it — it stays)
# Replace:
#   event_time=event_time,  (was derived from _parse_event_time_from_bucket)
#   time_bucket=time_bucket,
# With:
    event_time_start=event_time_start,
    event_time_end=event_time_end,
```

- [ ] **Step 2: Update daily summary callers**

In `_backfill_daily_summaries` / daily generation: compute `event_time_start` and `event_time_end` as datetime objects for the day boundaries:

```python
import datetime
day_start = datetime.datetime.combine(
    datetime.date.fromisoformat(date_str),
    datetime.time.min,
    tzinfo=datetime.timezone.utc,
)
day_end = datetime.datetime.combine(
    datetime.date.fromisoformat(date_str),
    datetime.time(23, 59, 59),
    tzinfo=datetime.timezone.utc,
)
```

Pass `event_time_start=day_start, event_time_end=day_end` to `_store_summary`.

- [ ] **Step 3: Update weekly summary callers**

Compute week boundaries:

```python
week_start_date = datetime.date.fromisocalendar(int(year), int(week), 1)
week_end_date = week_start_date + datetime.timedelta(days=6)
week_start = datetime.datetime.combine(week_start_date, datetime.time.min, tzinfo=datetime.timezone.utc)
week_end = datetime.datetime.combine(week_end_date, datetime.time(23, 59, 59), tzinfo=datetime.timezone.utc)
```

- [ ] **Step 4: Update monthly summary callers**

Compute month boundaries:

```python
import calendar
month_start_date = datetime.date(int(year), int(month), 1)
last_day = calendar.monthrange(int(year), int(month))[1]
month_end_date = datetime.date(int(year), int(month), last_day)
month_start = datetime.datetime.combine(month_start_date, datetime.time.min, tzinfo=datetime.timezone.utc)
month_end = datetime.datetime.combine(month_end_date, datetime.time(23, 59, 59), tzinfo=datetime.timezone.utc)
```

- [ ] **Step 5: Update ALL search_hierarchy calls (20+ occurrences)**

This file has 20+ `search_hierarchy` calls for existence-checking (dedup) and child-fetching. ALL must be converted from `time_bucket_start/end` strings to `time_start/end` float timestamps.

**Pattern for each call:** Replace keyword arguments:
```python
# OLD:
time_bucket_start=date_str, time_bucket_end=date_str,
# NEW:
time_start=day_start.timestamp(), time_end=day_end.timestamp(),
```

For daily existence checks: use `day_start.timestamp()` / `day_end.timestamp()` computed from `date_str`.
For weekly existence checks: use `week_start.timestamp()` / `week_end.timestamp()` computed from `week_str`.
For monthly existence checks: use `month_start.timestamp()` / `month_end.timestamp()` computed from `month_str`.

Grep for all occurrences: `grep -n "search_hierarchy" opencontext/periodic_task/hierarchy_summary.py` and convert each one.

- [ ] **Step 5b: Update get_all_processed_contexts filter**

Line 876 and similar L0 event queries use `"event_time_ts"` which will no longer exist. Replace with overlap pattern:

```python
# OLD:
"event_time_ts": {"$gte": day_start_ts, "$lte": day_end_ts},
# NEW:
"event_time_start_ts": {"$lte": day_end_ts},
"event_time_end_ts": {"$gte": day_start_ts},
```

Grep for all `event_time_ts` references: `grep -n "event_time_ts" opencontext/periodic_task/hierarchy_summary.py` and convert each one.

- [ ] **Step 6: Update formatting methods**

`_format_weekly_hierarchical` (line 343): replace `ctx.properties.time_bucket` with date derived from `ctx.properties.event_time_start`:

```python
tb = ctx.properties.event_time_start.strftime("%Y-%m-%d") if ctx.properties.event_time_start else ""
```

`_format_monthly_hierarchical` (line 383): same pattern, derive week string from `event_time_start`:

```python
tb = f"{ctx.properties.event_time_start.isocalendar()[0]}-W{ctx.properties.event_time_start.isocalendar()[1]:02d}"
```

`_format_l0_events` (lines 307, 315-316): rename `event_time` → `event_time_start`.

- [ ] **Step 7: Delete _parse_event_time_from_bucket helper**

Remove the method entirely.

- [ ] **Step 7b: Update metadata dict in _store_summary**

Around line 1503, `_store_summary` builds a metadata dict containing `"time_bucket": time_bucket`. After removing the `time_bucket` parameter, this reference will break. Remove or replace:

```python
# DELETE or replace:
# "time_bucket": time_bucket,
# With (if context is useful):
"period_start": event_time_start.isoformat(),
"period_end": event_time_end.isoformat(),
```

- [ ] **Step 8: Verify**

Run: `python -m py_compile opencontext/periodic_task/hierarchy_summary.py`

- [ ] **Step 9: Commit**

```bash
git add opencontext/periodic_task/hierarchy_summary.py
git commit -m "refactor: hierarchy summary — use event_time_start/end, remove time_bucket"
```

---

### Task 8: Update push_base_events route

**Files:**
- Modify: `opencontext/server/routes/agents.py:196-234`

- [ ] **Step 1: Update ContextProperties construction**

Lines 213-233: rename `event_time` → `event_time_start`, delete `time_bucket`:

```python
ctx = ProcessedContext(
    properties=ContextProperties(
        raw_properties=[],
        create_time=now,
        update_time=now,
        event_time_start=event_time,
        is_processed=True,
        agent_id=agent_id,
    ),
    # ... extracted_data stays the same
)
```

Also rename the local variable parsing (lines 199-204): `event_time` can stay as local variable name, it's just the input parsing.

- [ ] **Step 2: Verify**

Run: `python -m py_compile opencontext/server/routes/agents.py`

- [ ] **Step 3: Commit**

```bash
git add opencontext/server/routes/agents.py
git commit -m "refactor: push_base_events — use event_time_start, remove time_bucket"
```

---

## Chunk 3: Search/Retrieval + Cache

### Task 9: Update EventSearchService

**Files:**
- Modify: `opencontext/server/search/event_search_service.py:109-238`

- [ ] **Step 1: Update _build_filters**

Lines 201-222: Replace single-field `event_time_ts` with range overlap on two fields.

```python
@staticmethod
def _build_filters(
    time_range: Optional[Any],
    hierarchy_levels: Optional[List[int]],
) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if time_range:
        # Range overlap: event_time_start_ts <= end AND event_time_end_ts >= start
        if time_range.end is not None:
            filters["event_time_start_ts"] = {"$lte": time_range.end}
        if time_range.start is not None:
            filters["event_time_end_ts"] = {"$gte": time_range.start}

    if hierarchy_levels is not None and len(hierarchy_levels) == 1:
        filters["hierarchy_level"] = hierarchy_levels[0]
    elif hierarchy_levels is not None and len(hierarchy_levels) > 1:
        filters["hierarchy_level"] = hierarchy_levels

    return filters
```

- [ ] **Step 2: Update filter_search hierarchy branch**

Lines 120-155: Replace `_time_range_to_buckets` with timestamp pass-through to `search_hierarchy`.

```python
if hierarchy_levels:
    time_start = None
    time_end = None
    if time_range:
        time_start = time_range.start
        time_end = time_range.end

    owner_types = MEMORY_OWNER_TYPES.get(memory_owner, MEMORY_OWNER_TYPES["user"])
    tasks = []
    for level in hierarchy_levels:
        if level < len(owner_types):
            tasks.append(
                self.storage.search_hierarchy(
                    context_type=owner_types[level].value,
                    hierarchy_level=level,
                    time_start=time_start,
                    time_end=time_end,
                    user_id=user_id,
                    device_id=device_id,
                    agent_id=agent_id,
                    top_k=top_k,
                )
            )
    # ... rest stays the same
```

- [ ] **Step 3: Update filter_search non-hierarchy branch**

Lines 157-181: The non-hierarchy branch also builds its own `filters` dict with `"event_time_ts"`. Update to use overlap pattern:

```python
# Lines 160-166 — REPLACE:
if time_range:
    ts_filter: Dict[str, Any] = {}
    if time_range.start is not None:
        filters["event_time_end_ts"] = {"$gte": time_range.start}
    if time_range.end is not None:
        filters["event_time_start_ts"] = {"$lte": time_range.end}
```

- [ ] **Step 4: Delete _time_range_to_buckets**

Lines 224-238: Delete entire method.

- [ ] **Step 5: Verify**

Run: `python -m py_compile opencontext/server/search/event_search_service.py`

- [ ] **Step 6: Commit**

```bash
git add opencontext/server/search/event_search_service.py
git commit -m "refactor: EventSearchService — unified numeric range overlap filtering"
```

---

### Task 10: Update search models and routes

**Files:**
- Modify: `opencontext/server/search/models.py:107,111`
- Modify: `opencontext/server/routes/search.py:226-232,270-308,331`

- [ ] **Step 1: Update EventNode model**

```python
# Line 107 — DELETE:
# time_bucket: Optional[str] = None

# Line 111 — RENAME:
event_time_start: Optional[str] = None

# ADD after:
event_time_end: Optional[str] = None
```

- [ ] **Step 2: Update search route — tree sorting**

Lines 226-232: Replace `time_bucket` sort with `event_time_start`:

```python
def sort_tree(node_list: List[EventNode]):
    node_list.sort(key=lambda n: (n.event_time_start or ""))
    for n in node_list:
        if n.children:
            sort_tree(n.children)
```

- [ ] **Step 3: Update search route — node construction**

`_to_context_node` (lines 270-286): Replace `time_bucket` and `event_time` with new fields.
`_to_search_hit_node` (lines 289-308): Same changes.

```python
# In both methods, replace:
#   time_bucket=props.time_bucket if props else None,
#   event_time=_format_timestamp(props.event_time if props else None),
# With:
    event_time_start=_format_timestamp(props.event_time_start if props else None),
    event_time_end=_format_timestamp(props.event_time_end if props else None),
```

- [ ] **Step 4: Update _track_accessed_safe**

Line 331: Replace `"event_time": er.event_time` with `"event_time_start": er.event_time_start`.

- [ ] **Step 5: Verify**

```bash
python -m py_compile opencontext/server/search/models.py
python -m py_compile opencontext/server/routes/search.py
```

- [ ] **Step 6: Commit**

```bash
git add opencontext/server/search/models.py opencontext/server/routes/search.py
git commit -m "refactor: search models/routes — event_time_start/end, remove time_bucket"
```

---

### Task 11: Update retrieval tools

**Files:**
- Modify: `opencontext/tools/retrieval_tools/hierarchical_event_tool.py:184-455`
- Modify: `opencontext/tools/retrieval_tools/base_context_retrieval_tool.py:31,170-172,240-241`
- Modify: `opencontext/tools/retrieval_tools/knowledge_retrieval_tool.py:79`

- [ ] **Step 1: Update hierarchical_event_tool.py**

Delete `_ts_to_day_bucket`, `_ts_to_week_bucket`, `_ts_to_month_bucket` (lines 184-206).

Update `_search_summaries` (lines 208-232): change signature and pass-through.

```python
async def _search_summaries(
    self,
    level: int,
    time_start: Optional[float],
    time_end: Optional[float],
    user_id: Optional[str],
    top_k: int,
    device_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> List[Tuple[ProcessedContext, float]]:
    try:
        return await self.storage.search_hierarchy(
            context_type=self.CONTEXT_TYPE.value,
            hierarchy_level=level,
            time_start=time_start,
            time_end=time_end,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            top_k=top_k,
        )
    except Exception as e:
        logger.warning(f"search_hierarchy L{level} failed: {e}")
        return []
```

Update lines 440-455: Replace per-level bucket conversion with unified timestamps.

```python
# Replace the entire for loop with:
for level in [1, 2, 3]:
    hits = await self._search_summaries(
        level=level,
        time_start=ts_start,
        time_end=ts_end,
        user_id=user_id,
        top_k=top_k,
        device_id=device_id,
        agent_id=agent_id,
    )
    all_summary_hits.extend(hits)
```

Where `ts_start`/`ts_end` come from the existing `time_range` dict (already parsed as timestamps).

Delete the bucket variable assignments (lines 426-434).

Update the `time_filters` dict (lines 429, 435) used for direct L0 search. Replace `event_time_ts` with overlap pattern:

```python
# OLD:
time_filters.setdefault("event_time_ts", {})["$gte"] = ts_start
time_filters.setdefault("event_time_ts", {})["$lte"] = ts_end
# NEW:
if ts_start is not None:
    time_filters["event_time_end_ts"] = {"$gte": ts_start}
if ts_end is not None:
    time_filters["event_time_start_ts"] = {"$lte": ts_end}
```

Update `_format_result` (lines 352-375): rename `event_time` → `event_time_start`, add `event_time_end`, delete `time_bucket`.

```python
"event_time_start": props.event_time_start.isoformat() if props.event_time_start else None,
"event_time_end": props.event_time_end.isoformat() if props.event_time_end else None,
# DELETE: "time_bucket": props.time_bucket,
```

- [ ] **Step 2: Update base_context_retrieval_tool.py**

Line 31: `time_type: Optional[str] = "event_time_start_ts"`

Lines 240-241: Update enum and default:
```python
"enum": ["create_time_ts", "update_time_ts", "event_time_start_ts"],
"default": "event_time_start_ts",
```

Lines 170-172: Update output format:
```python
"event_time_start": props.event_time_start.isoformat() if props.event_time_start else None,
"event_time_end": props.event_time_end.isoformat() if props.event_time_end else None,
# DELETE: "time_bucket": props.time_bucket,
```

- [ ] **Step 3: Update knowledge_retrieval_tool.py**

Line 79: `"- Time range filtering (by event_time_start, create_time, or update_time)\n"`

- [ ] **Step 4: Verify**

```bash
python -m py_compile opencontext/tools/retrieval_tools/hierarchical_event_tool.py
python -m py_compile opencontext/tools/retrieval_tools/base_context_retrieval_tool.py
python -m py_compile opencontext/tools/retrieval_tools/knowledge_retrieval_tool.py
```

- [ ] **Step 5: Commit**

```bash
git add opencontext/tools/retrieval_tools/
git commit -m "refactor: retrieval tools — numeric time filtering, remove time_bucket"
```

---

### Task 12: Update cache

**Files:**
- Modify: `opencontext/server/cache/models.py:30,46,53,61,71`
- Modify: `opencontext/server/cache/memory_cache_manager.py:410,435,445,519,584-602`

- [ ] **Step 1: Update cache models**

```python
# models.py line 30 — rename:
event_time_start: Optional[str] = None

# models.py line 46 — rename:
event_time_start: Optional[str] = None

# models.py line 53 — rename (DailySummaryItem):
event_time_start: str  # e.g. "2026-02-21"

# models.py line 61 — rename (SimpleDailySummary):
event_time_start: str

# models.py line 71 — rename (SimpleTodayEvent):
event_time_start: Optional[str] = None
```

- [ ] **Step 2: Update memory_cache_manager.py**

```python
# Line 410 — rename:
"event_time_start_ts": {"$gte": today_start_ts},

# Lines 435, 445 — replace created_at_ts:
filter={"create_time_ts": {"$gte": week_start_ts}},

# Line 519 — replace time_bucket:
"event_time_start": ctx.properties.event_time_start.strftime("%Y-%m-%d") if ctx.properties.event_time_start else "",

# Lines 584-591 — rename in _ctx_to_recent_item:
event_time_start = None
if props.event_time_start:
    if hasattr(props.event_time_start, "isoformat"):
        dt = props.event_time_start
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        event_time_start = dt.isoformat()
    else:
        event_time_start = str(props.event_time_start)

# Line 602 — rename in result dict:
"event_time_start": event_time_start,
```

Also update any sorting by `time_bucket` to sort by `event_time_start`.

Update the `search_hierarchy` call (around lines 417-426) to use new parameter names:

```python
# OLD:
time_bucket_start=period_start,
time_bucket_end=yesterday,
# NEW — convert date strings to timestamps:
time_start=datetime.datetime.combine(
    datetime.date.fromisoformat(period_start), datetime.time.min, tzinfo=datetime.timezone.utc
).timestamp(),
time_end=datetime.datetime.combine(
    datetime.date.fromisoformat(yesterday), datetime.time(23, 59, 59), tzinfo=datetime.timezone.utc
).timestamp(),
```

- [ ] **Step 3: Verify**

```bash
python -m py_compile opencontext/server/cache/models.py
python -m py_compile opencontext/server/cache/memory_cache_manager.py
```

- [ ] **Step 4: Commit**

```bash
git add opencontext/server/cache/
git commit -m "refactor: cache — event_time_start, remove time_bucket/created_at_ts"
```

---

## Chunk 4: Prompts, Monitoring, Templates, Docs

### Task 13: Update LLM prompts

**Files:**
- Modify: `config/prompts_en.yaml:489,565,579,590,635,643`
- Modify: `config/prompts_zh.yaml:489,565,579,590,634,642`

- [ ] **Step 1: Update prompts_en.yaml extraction prompts**

Remove `event_time` from the user_memory extraction prompt:
- Line 489: Remove the sentence about putting timestamps in `event_time` field
- Line 565: Remove field 8 (`event_time`) from the numbered list
- Line 579: Remove `"event_time": "2026-03-08T14:30:00"` from example
- Line 590: Remove `"event_time": null` from example

Remove from agent_memory extraction prompt:
- Line 635: Remove `"event_time": "YYYY-MM-DDTHH:MM:SS"` from schema
- Line 643: Remove `For event_time, use the time the conversation took place.`

**Do NOT touch merge prompts** (lines 671, 703).

- [ ] **Step 2: Update prompts_zh.yaml**

Same changes as above at corresponding lines.

- [ ] **Step 3: Commit**

```bash
git add config/prompts_en.yaml config/prompts_zh.yaml
git commit -m "refactor: remove event_time from LLM extraction prompts"
```

---

### Task 14: Update monitoring route

**Files:**
- Modify: `opencontext/server/routes/monitoring.py:361`

- [ ] **Step 1: Replace time_bucket reference**

```python
# Line 361 — replace:
"event_time_start": result.properties.event_time_start.isoformat() if result.properties.event_time_start else None,
"event_time_end": result.properties.event_time_end.isoformat() if result.properties.event_time_end else None,
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile opencontext/server/routes/monitoring.py`

- [ ] **Step 3: Commit**

```bash
git add opencontext/server/routes/monitoring.py
git commit -m "refactor: monitoring — event_time_start/end, remove time_bucket"
```

---

### Task 15: Update web templates

**Files:**
- Modify: `opencontext/web/templates/agents.html:357,389`
- Modify: `opencontext/web/templates/context_detail.html:39,139-142,328`
- Modify: `opencontext/web/templates/memory_cache.html:250-252,276`
- Modify: `opencontext/web/templates/vector_search.html:576-577,636`

- [ ] **Step 1: Update all templates**

Global find-and-replace in each file:
- `event_time` → `event_time_start` (in JS property access and display)
- `time_bucket` → `event_time_start` (or remove if only used as badge label)
- Delete any `time_bucket`-only display blocks (like context_detail.html lines 139-142)

- [ ] **Step 2: Commit**

```bash
git add opencontext/web/templates/
git commit -m "refactor: web templates — event_time_start, remove time_bucket"
```

---

### Task 16: Update documentation

**Files:**
- Modify: `opencontext/models/MODULE.md`
- Modify: `opencontext/server/MODULE.md`
- Modify: `opencontext/tools/MODULE.md`
- Modify: `docs/api_reference.md`
- Modify: `docs/curls.sh`

- [ ] **Step 1: Update MODULE.md files**

Replace all `event_time` → `event_time_start`, `time_bucket` → removed, document `event_time_end`.

- [ ] **Step 2: Update api_reference.md and curls.sh**

Update response schemas and examples to reflect new field names.

- [ ] **Step 3: Commit**

```bash
git add opencontext/models/MODULE.md opencontext/server/MODULE.md opencontext/tools/MODULE.md docs/api_reference.md docs/curls.sh
git commit -m "docs: update MODULE.md and API docs for time field changes"
```

---

### Task 17: Final verification

- [ ] **Step 1: Full compile check**

```bash
find opencontext -name "*.py" -exec python -m py_compile {} \;
```

- [ ] **Step 2: Search for any remaining references**

```bash
grep -rn "time_bucket" opencontext/ --include="*.py" | grep -v "__pycache__"
grep -rn "\.event_time[^_]" opencontext/ --include="*.py" | grep -v "__pycache__" | grep -v "event_time_start" | grep -v "event_time_end"
grep -rn "created_at_ts" opencontext/ --include="*.py" | grep -v "__pycache__"
grep -rn "FIELD_CREATED_AT" opencontext/ --include="*.py" | grep -v "__pycache__"
grep -rn "FIELD_TIME_BUCKET" opencontext/ --include="*.py" | grep -v "__pycache__"
grep -rn "FIELD_EVENT_TIME_TS\b" opencontext/ --include="*.py" | grep -v "__pycache__"
grep -rn "event_time_ts" opencontext/ --include="*.py" | grep -v "__pycache__" | grep -v "event_time_start_ts" | grep -v "event_time_end_ts"
```

Fix any remaining references found.

- [ ] **Step 3: Final commit if any fixes**

```bash
git add -A
git commit -m "fix: clean up remaining old time field references"
```
