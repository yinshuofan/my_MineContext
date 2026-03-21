# Global Timezone Unification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify all datetime handling to use a single globally-configured timezone, eliminating inconsistent naive/UTC datetime usage across the codebase.

**Architecture:** Create a `time_utils` module that reads a `timezone` setting from `config.yaml` and exposes `now()` / `utc_now()` / `today_start()` / `get_timezone()`. All code replaces direct `datetime.now()` or `datetime.now(tz=timezone.utc)` calls with `time_utils.now()`. AWS S3 auth and similar protocol-mandated UTC scenarios use `utc_now()`. Memory cache stops stripping timezone info. The configured timezone flows through storage, processing, capture, and API layers.

**Tech Stack:** Python 3.10+, `zoneinfo` (stdlib), existing `GlobalConfig` singleton

**Important invariant:** `time_utils.now()` must NOT be called at module-import time or in class-level expressions. Only use it inside function bodies, method bodies, or `default_factory` lambdas — because the timezone is initialized at startup, after module imports.

**Old data:** This migration does NOT provide backward compatibility with pre-migration naive datetimes. All existing data should be regenerated after deployment.

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `opencontext/utils/time_utils.py` | Global timezone config + `now()`, `utc_now()`, `today_start()`, `get_timezone()` |
| Modify | `config/config.yaml` | Add `timezone` top-level config section |
| Modify | `.env.example` | Add `TIMEZONE` env var |
| Modify | `opencontext/cli.py` | Initialize timezone on startup |
| Modify | `opencontext/config/global_config.py` | Call `init_timezone()` in `_auto_initialize()` for multi-worker subprocesses |
| Modify | `opencontext/models/context.py` | `ProfileData` field defaults use `time_utils.now()` |
| Modify | `opencontext/context_capture/base.py` | Replace `datetime.now()` |
| Modify | `opencontext/context_capture/text_chat.py` | Replace `datetime.now()` |
| Modify | `opencontext/context_capture/folder_monitor.py` | Replace `datetime.now()` |
| Modify | `opencontext/context_capture/vault_document_monitor.py` | Replace `datetime.now()` |
| Modify | `opencontext/context_capture/web_link_capture.py` | Replace `datetime.now()` |
| Modify | `opencontext/context_processing/processor/text_chat_processor.py` | Replace `datetime.datetime.now()` |
| Modify | `opencontext/context_processing/processor/document_processor.py` | Replace `datetime.datetime.now()` |
| Modify | `opencontext/context_processing/processor/agent_memory_processor.py` | Replace `datetime.datetime.now(tz=...)` |
| Modify | `opencontext/context_processing/merger/context_merger.py` | Replace `datetime.datetime.now()` + fix `fromisoformat` timezone |
| Modify | `opencontext/context_processing/merger/merge_strategies.py` | Replace `datetime.now()` |
| Modify | `opencontext/storage/backends/mysql_backend.py` | Replace `datetime.now()` (~19 locations) |
| Modify | `opencontext/storage/backends/sqlite_backend.py` | Replace `datetime.now()` (~22 locations) |
| Modify | `opencontext/storage/backends/vikingdb_backend.py` | Replace `datetime.datetime.now(tz=...)` |
| Modify | `opencontext/storage/redis_cache.py` | Replace `datetime.now()` (4 locations) |
| Modify | `opencontext/storage/object_storage/s3_auth.py` | Replace `datetime.now(tz=timezone.utc)` with `utc_now()` (must stay UTC for AWS Signature V4) |
| Modify | `opencontext/server/context_operations.py` | Replace `datetime.datetime.now()` |
| Modify | `opencontext/server/routes/agents.py` | Replace `datetime.datetime.now(tz=...)` |
| Modify | `opencontext/server/routes/push.py` | Replace `datetime.datetime.now(tz=...)` |
| Modify | `opencontext/server/routes/completions.py` | Replace `datetime.now()` |
| Modify | `opencontext/server/routes/vaults.py` | Replace `datetime.now()` |
| Modify | `opencontext/server/routes/web.py` | Replace `.replace(tzinfo=...)` pattern (3 locations: lines 83, 93, 140) |
| Modify | `opencontext/server/cache/memory_cache_manager.py` | Use `time_utils.now()` / `today_start()`, stop stripping tzinfo |
| Modify | `opencontext/periodic_task/hierarchy_summary.py` | Replace `datetime.date.today()`, `datetime.combine(..., tzinfo=UTC)` (10 locations), and `datetime.datetime.now(tz=UTC)` |
| Modify | `opencontext/context_consumption/completion/completion_service.py` | Replace `datetime.now()` |
| Modify | `opencontext/context_consumption/completion/completion_cache.py` | Replace `datetime.now()` |
| Modify | `opencontext/context_consumption/context_agent/core/state.py` | Replace `datetime.now()` + `default_factory=datetime.now` at lines 34-35 |
| Modify | `opencontext/context_consumption/context_agent/core/llm_context_strategy.py` | Replace `datetime.now()` |
| Modify | `opencontext/context_consumption/context_agent/nodes/base.py` | Replace `datetime.now()` |
| Modify | `opencontext/context_consumption/context_agent/nodes/executor.py` | Replace `datetime.now()` |
| Modify | `opencontext/context_consumption/context_agent/models/events.py` | Replace `default_factory=datetime.now` at line 25 |
| Modify | `opencontext/context_consumption/context_agent/models/schemas.py` | Replace `default_factory=datetime.now` at line 70 |
| Modify | `opencontext/scheduler/redis_scheduler.py` | Replace `datetime.now(tz=timezone.utc)` |
| Modify | `opencontext/monitoring/monitor.py` | Replace `datetime.now(tz=timezone.utc)` + sentinel `.replace(tzinfo=...)` at line 110 |
| Modify | `opencontext/server/routes/monitoring.py` | Replace `datetime.now(tz=timezone.utc)` |
| Modify | `docs/api_reference.md` | Note timezone info now included in API datetime fields |
| Modify | `CLAUDE.md` | Add `time_utils.now()` guidance to pitfalls |

---

## Chunk 1: Core Infrastructure

### Task 1: Create `time_utils` module

**Files:**
- Create: `opencontext/utils/time_utils.py`

- [ ] **Step 1: Create the time_utils module**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Global timezone utilities.

Provides a single configured timezone for the entire application.
All code should use ``now()`` from this module instead of
``datetime.now()`` or ``datetime.now(tz=timezone.utc)``.

Initialize once at startup via ``init_timezone(tz_name)``.

**Important:** Do NOT call ``now()`` at module-import time or in
class-level expressions. The timezone is initialized at startup,
after module imports. Use only inside function/method bodies or
``default_factory`` lambdas.
"""

import datetime
from zoneinfo import ZoneInfo

from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

_configured_tz: datetime.tzinfo = datetime.timezone.utc


def init_timezone(tz_name: str | None = None) -> None:
    """Initialize the global timezone from config.

    Args:
        tz_name: IANA timezone name (e.g. ``"Asia/Shanghai"``, ``"UTC"``).
                 Defaults to ``"UTC"`` if *None* or empty.
    """
    global _configured_tz
    name = tz_name or "UTC"
    try:
        _configured_tz = ZoneInfo(name)
        logger.info(f"Global timezone set to: {name}")
    except KeyError:
        logger.error(f"Unknown timezone '{name}', falling back to UTC")
        _configured_tz = datetime.timezone.utc


def get_timezone() -> datetime.tzinfo:
    """Return the configured timezone object."""
    return _configured_tz


def now() -> datetime.datetime:
    """Return the current time in the configured timezone.

    This is the **primary** function for getting "now" throughout
    the codebase.
    """
    return datetime.datetime.now(tz=_configured_tz)


def utc_now() -> datetime.datetime:
    """Return the current time in UTC.

    Use this **only** when a protocol mandates UTC
    (e.g. AWS Signature V4, HTTP Date headers).
    For all other cases, use ``now()``.
    """
    return datetime.datetime.now(tz=datetime.timezone.utc)


def today_start() -> datetime.datetime:
    """Return midnight of today in the configured timezone.

    Useful for "today events" boundary calculations.
    """
    n = now()
    return n.replace(hour=0, minute=0, second=0, microsecond=0)
```

- [ ] **Step 2: Verify the module compiles**

Run: `python -m py_compile opencontext/utils/time_utils.py`
Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add opencontext/utils/time_utils.py
git commit -m "feat: add global time_utils module for unified timezone handling"
```

---

### Task 2: Add timezone config and initialize on startup

**Files:**
- Modify: `config/config.yaml` (after the `prompts` section, before `tools`)
- Modify: `.env.example`
- Modify: `opencontext/cli.py:434` (after `GlobalConfig.initialize()`)
- Modify: `opencontext/config/global_config.py:74-90` (`_auto_initialize()`)

- [ ] **Step 1: Add timezone section to config.yaml**

Insert after `prompts` section (after line 283) and before `tools` section:

```yaml
# ============================================================
# 时区配置
# 控制所有时间的生成和显示时区
# 使用 IANA 时区名称，如 "Asia/Shanghai"、"UTC"、"America/New_York"
# ============================================================
timezone: "${TIMEZONE:UTC}"
```

- [ ] **Step 2: Add TIMEZONE to .env.example**

Add near other config env vars:

```
# Timezone (IANA name, e.g. Asia/Shanghai, UTC, America/New_York)
TIMEZONE=UTC
```

- [ ] **Step 3: Initialize timezone in cli.py startup**

In `_setup_logging()` (around line 434), after `GlobalConfig.get_instance().initialize(config_path)`, add:

```python
    from opencontext.utils.time_utils import init_timezone
    tz_name = GlobalConfig.get_instance().get_config("timezone")
    init_timezone(tz_name)
```

- [ ] **Step 4: Initialize timezone in _auto_initialize() for multi-worker subprocesses**

In `opencontext/config/global_config.py`, inside `_auto_initialize()` (around line 87, after `self._auto_initialized = True`), add:

```python
            # Initialize timezone for subprocess workers
            try:
                from opencontext.utils.time_utils import init_timezone
                tz_name = self._config_manager.get_config().get("timezone") if self._config_manager else None
                init_timezone(tz_name)
            except Exception as e:
                logger.warning(f"Failed to init timezone in auto-initialize: {e}")
```

This ensures multi-worker subprocesses (spawned by uvicorn) that don't go through `cli.py:main()` still get the timezone initialized.

- [ ] **Step 5: Verify**

```bash
python -m py_compile opencontext/cli.py
python -m py_compile opencontext/config/global_config.py
```

- [ ] **Step 6: Commit**

```bash
git add config/config.yaml .env.example opencontext/cli.py opencontext/config/global_config.py
git commit -m "feat: add timezone config and initialize on startup (incl. multi-worker)"
```

---

## Chunk 2: Core Data Path — Models, Processing, Capture

### Task 3: Update model defaults in `context.py`

**Files:**
- Modify: `opencontext/models/context.py:420-424`

- [ ] **Step 1: Replace ProfileData datetime defaults**

Add import at the top of the file (no circular import risk — `time_utils` only depends on `logging_utils` and stdlib):
```python
from opencontext.utils.time_utils import now as _tz_now
```

Change the `ProfileData` field defaults from:

```python
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc)
    )
    updated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc)
    )
```

To:

```python
    created_at: datetime.datetime = Field(default_factory=_tz_now)
    updated_at: datetime.datetime = Field(default_factory=_tz_now)
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile opencontext/models/context.py`

- [ ] **Step 3: Commit**

```bash
git add opencontext/models/context.py
git commit -m "refactor: use time_utils.now() for ProfileData defaults"
```

---

### Task 4: Update context processing — processors

**Files:**
- Modify: `opencontext/context_processing/processor/text_chat_processor.py:81,183,354`
- Modify: `opencontext/context_processing/processor/document_processor.py:231`
- Modify: `opencontext/context_processing/processor/agent_memory_processor.py:137,300,317,318`

- [ ] **Step 1: Fix text_chat_processor.py**

Add import at the top:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace these 3 occurrences:
- Line 81: `current_time=datetime.datetime.now().isoformat()` → `current_time=tz_now().isoformat()`
- Line 183: `current_time=datetime.datetime.now().isoformat()` → `current_time=tz_now().isoformat()`
- Line 354: `update_time=datetime.datetime.now()` → `update_time=tz_now()`

- [ ] **Step 2: Fix document_processor.py**

Add import at the top:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace:
- Line 231: `now = datetime.datetime.now()` → `now = tz_now()`

- [ ] **Step 3: Fix agent_memory_processor.py**

Add import at the top:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace all `datetime.datetime.now(tz=datetime.timezone.utc)` with `tz_now()`:
- Line 137
- Line 300
- Lines 317-318

- [ ] **Step 4: Verify all three compile**

Run:
```bash
python -m py_compile opencontext/context_processing/processor/text_chat_processor.py
python -m py_compile opencontext/context_processing/processor/document_processor.py
python -m py_compile opencontext/context_processing/processor/agent_memory_processor.py
```

- [ ] **Step 5: Commit**

```bash
git add opencontext/context_processing/processor/text_chat_processor.py \
        opencontext/context_processing/processor/document_processor.py \
        opencontext/context_processing/processor/agent_memory_processor.py
git commit -m "refactor: use time_utils.now() in all context processors"
```

---

### Task 5: Update context processing — merger

**Files:**
- Modify: `opencontext/context_processing/merger/context_merger.py:343,364,366-372,419,424,511,516`
- Modify: `opencontext/context_processing/merger/merge_strategies.py:72,90,241`

- [ ] **Step 1: Fix context_merger.py**

Add import at the top:
```python
from opencontext.utils.time_utils import now as tz_now, get_timezone
```

Replace all `datetime.datetime.now()` with `tz_now()`:
- Line 343: `now = datetime.datetime.now()` → `now = tz_now()`
- Line 364: `today = datetime.date.today().isoformat()` → `today = tz_now().date().isoformat()`
- Line 419: `datetime.datetime.now()` → `tz_now()` (inside timestamp calc)
- Line 424: `datetime.datetime.now()` → `tz_now()` (inside timestamp calc)
- Line 511: `datetime.datetime.now()` → `tz_now()` (inside timestamp calc)
- Line 516: `datetime.datetime.now()` → `tz_now()` (inside timestamp calc)

Fix `fromisoformat` results at lines 366-370 — the constructed string `f"{today}T{time_str}"` has no timezone, producing a naive datetime. Attach configured timezone via `.replace()`:

```python
# Before (lines 366-368):
properties.event_time_start = datetime.datetime.fromisoformat(
    full_event_time_str
)
# After:
properties.event_time_start = datetime.datetime.fromisoformat(
    full_event_time_str
).replace(tzinfo=get_timezone())
```

Same for lines 370-372 (the `else` branch):
```python
properties.event_time_start = datetime.datetime.fromisoformat(
    event_time_str
).replace(tzinfo=get_timezone())
```

- [ ] **Step 2: Fix merge_strategies.py**

Add import at the top:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace all `datetime.now()` with `tz_now()`:
- Line 72: `(datetime.now() - context.properties.update_time).days` → `(tz_now() - context.properties.update_time).days`
- Line 90: same pattern
- Line 241: `update_time=datetime.now()` → `update_time=tz_now()`

- [ ] **Step 3: Verify**

```bash
python -m py_compile opencontext/context_processing/merger/context_merger.py
python -m py_compile opencontext/context_processing/merger/merge_strategies.py
```

- [ ] **Step 4: Commit**

```bash
git add opencontext/context_processing/merger/context_merger.py \
        opencontext/context_processing/merger/merge_strategies.py
git commit -m "refactor: use time_utils.now() in merger components"
```

---

### Task 6: Update context capture

**Files:**
- Modify: `opencontext/context_capture/base.py:193,356`
- Modify: `opencontext/context_capture/text_chat.py:52`
- Modify: `opencontext/context_capture/folder_monitor.py:77,172`
- Modify: `opencontext/context_capture/vault_document_monitor.py:84,185,201`
- Modify: `opencontext/context_capture/web_link_capture.py:234,242`

- [ ] **Step 1: Fix each file**

For each file, add `from opencontext.utils.time_utils import now as tz_now` and replace every `datetime.now()` / `datetime.datetime.now()` with `tz_now()`.

**base.py** (uses `from datetime import datetime`):
- Line 193: `self._last_capture_time = datetime.now()` → `self._last_capture_time = tz_now()`
- Line 356: `(datetime.now() - self._last_capture_time)` → `(tz_now() - self._last_capture_time)`

**text_chat.py** (uses `import datetime`):
- Line 52: `now = datetime.datetime.now()` → `now = tz_now()`

**folder_monitor.py** (uses `from datetime import datetime`):
- Line 77: `self._last_scan_time = datetime.now()` → `self._last_scan_time = tz_now()`
- Line 172: `current_time = datetime.now()` → `current_time = tz_now()`

**vault_document_monitor.py** (uses `from datetime import datetime`):
- Line 84: `self._last_scan_time = datetime.now()` → `self._last_scan_time = tz_now()`
- Line 185: `"timestamp": datetime.now()` → `"timestamp": tz_now()`
- Line 201: `current_time = datetime.now()` → `current_time = tz_now()`

**web_link_capture.py** (uses `from datetime import datetime`):
- Line 234: `create_time=datetime.now()` → `create_time=tz_now()`
- Line 242: `self._last_activity_time = datetime.now()` → `self._last_activity_time = tz_now()`

- [ ] **Step 2: Verify all compile**

```bash
python -m py_compile opencontext/context_capture/base.py
python -m py_compile opencontext/context_capture/text_chat.py
python -m py_compile opencontext/context_capture/folder_monitor.py
python -m py_compile opencontext/context_capture/vault_document_monitor.py
python -m py_compile opencontext/context_capture/web_link_capture.py
```

- [ ] **Step 3: Commit**

```bash
git add opencontext/context_capture/base.py \
        opencontext/context_capture/text_chat.py \
        opencontext/context_capture/folder_monitor.py \
        opencontext/context_capture/vault_document_monitor.py \
        opencontext/context_capture/web_link_capture.py
git commit -m "refactor: use time_utils.now() in all context capture modules"
```

---

## Chunk 3: Storage Layer

### Task 7: Update MySQL backend

**Files:**
- Modify: `opencontext/storage/backends/mysql_backend.py` (~19 locations)

- [ ] **Step 1: Add import and replace all occurrences**

Add import at the top:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace every `datetime.now()` with `tz_now()`. The file uses `from datetime import datetime, timedelta` so all calls are `datetime.now()`.

Locations (line numbers are approximate — find by searching `datetime.now()`):
437, 576, 636, 658, 727, 838, 866, 904, 925, 947, 979, 1027, 1050, 1074, 1217, 1255, 1294, 1323, 1345

- [ ] **Step 2: Verify**

Run: `python -m py_compile opencontext/storage/backends/mysql_backend.py`

- [ ] **Step 3: Commit**

```bash
git add opencontext/storage/backends/mysql_backend.py
git commit -m "refactor: use time_utils.now() in MySQL backend"
```

---

### Task 8: Update SQLite backend

**Files:**
- Modify: `opencontext/storage/backends/sqlite_backend.py` (~22 locations)

- [ ] **Step 1: Add import and replace all occurrences**

Add import at the top:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace every `datetime.now()` with `tz_now()`. Same pattern as MySQL backend.

Locations: 480, 481, 690, 696, 761, 792, 868, 995, 1045, 1093, 1123, 1156, 1195, 1271, 1310, 1359, 1500, 1592, 1669, 1727, 1772, 1806

- [ ] **Step 2: Verify**

Run: `python -m py_compile opencontext/storage/backends/sqlite_backend.py`

- [ ] **Step 3: Commit**

```bash
git add opencontext/storage/backends/sqlite_backend.py
git commit -m "refactor: use time_utils.now() in SQLite backend"
```

---

### Task 9: Update VikingDB backend, Redis cache, and S3 auth

**Files:**
- Modify: `opencontext/storage/backends/vikingdb_backend.py:186`
- Modify: `opencontext/storage/redis_cache.py:1020,1049,1087,1094`
- Modify: `opencontext/storage/object_storage/s3_auth.py:48`

- [ ] **Step 1: Fix vikingdb_backend.py**

Add import:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace:
- Line 186: `datetime.datetime.now(tz=datetime.timezone.utc)` → `tz_now()`

- [ ] **Step 2: Fix redis_cache.py**

Add import:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace all `datetime.now()` with `tz_now()`:
- Line 1020: `if datetime.now() > self._expiry[key]`
- Line 1049: `self._expiry[key] = datetime.now() + timedelta(seconds=ttl)`
- Line 1087: `self._expiry[key] = datetime.now() + timedelta(seconds=ttl)`
- Line 1094: `remaining = (self._expiry[key] - datetime.now())`

- [ ] **Step 3: Fix s3_auth.py — must stay UTC**

Add import:
```python
from opencontext.utils.time_utils import utc_now
```

Replace:
- Line 48: `now = datetime.now(tz=timezone.utc)` → `now = utc_now()`

**Note:** AWS Signature V4 mandates UTC. This file intentionally uses `utc_now()` instead of `now()`.

- [ ] **Step 4: Verify**

```bash
python -m py_compile opencontext/storage/backends/vikingdb_backend.py
python -m py_compile opencontext/storage/redis_cache.py
python -m py_compile opencontext/storage/object_storage/s3_auth.py
```

- [ ] **Step 5: Commit**

```bash
git add opencontext/storage/backends/vikingdb_backend.py \
        opencontext/storage/redis_cache.py \
        opencontext/storage/object_storage/s3_auth.py
git commit -m "refactor: use time_utils in VikingDB backend, Redis cache, and S3 auth"
```

---

## Chunk 4: Server Layer

### Task 10: Update server routes and context operations

**Files:**
- Modify: `opencontext/server/context_operations.py:91`
- Modify: `opencontext/server/routes/agents.py:198,200-202`
- Modify: `opencontext/server/routes/push.py:460`
- Modify: `opencontext/server/routes/completions.py:67,100,115,120,146,229,256,274`
- Modify: `opencontext/server/routes/vaults.py:335`
- Modify: `opencontext/server/routes/web.py:83,93,140`

- [ ] **Step 1: Fix context_operations.py**

Add import, replace:
- Line 91: `create_time=datetime.datetime.now()` → `create_time=tz_now()`

- [ ] **Step 2: Fix agents.py**

Add imports:
```python
from opencontext.utils.time_utils import now as tz_now, get_timezone
```

Replace:
- Line 198: `now = datetime.datetime.now(tz=datetime.timezone.utc)` → `now = tz_now()`

Fix the `fromisoformat` timezone handling (lines 200-202):
```python
# Before:
event_time = datetime.datetime.fromisoformat(event.event_time)
if event_time.tzinfo is None:
    event_time = event_time.replace(tzinfo=datetime.timezone.utc)

# After:
event_time = datetime.datetime.fromisoformat(event.event_time)
if event_time.tzinfo is None:
    event_time = event_time.replace(tzinfo=get_timezone())
```

- [ ] **Step 3: Fix push.py**

Replace:
- Line 460: `datetime.datetime.now(tz=datetime.timezone.utc)` → `tz_now()`

- [ ] **Step 4: Fix completions.py**

Add import, replace all `datetime.now()` with `tz_now()`:
- Lines 67, 100, 115, 120, 146, 229, 256, 274

- [ ] **Step 5: Fix vaults.py**

Replace:
- Line 335: `create_time=datetime.now()` → `create_time=tz_now()`

- [ ] **Step 6: Fix web.py (3 locations)**

Add import:
```python
from opencontext.utils.time_utils import get_timezone
```

Replace the `.replace(tzinfo=datetime.timezone.utc)` pattern with configured timezone at all 3 locations:

Lines 82-83 (start_date filter):
```python
start_dt = datetime.datetime.strptime(start_date, fmt).replace(
    tzinfo=get_timezone()
)
```

Lines 92-93 (end_date filter):
```python
end_dt = datetime.datetime.strptime(end_date, fmt).replace(
    tzinfo=get_timezone()
)
```

Line 140 (sort key helper):
```python
# Before:
dt = dt.replace(tzinfo=datetime.timezone.utc)
# After:
dt = dt.replace(tzinfo=get_timezone())
```

- [ ] **Step 7: Verify all compile**

```bash
python -m py_compile opencontext/server/context_operations.py
python -m py_compile opencontext/server/routes/agents.py
python -m py_compile opencontext/server/routes/push.py
python -m py_compile opencontext/server/routes/completions.py
python -m py_compile opencontext/server/routes/vaults.py
python -m py_compile opencontext/server/routes/web.py
```

- [ ] **Step 8: Commit**

```bash
git add opencontext/server/context_operations.py \
        opencontext/server/routes/agents.py \
        opencontext/server/routes/push.py \
        opencontext/server/routes/completions.py \
        opencontext/server/routes/vaults.py \
        opencontext/server/routes/web.py
git commit -m "refactor: use time_utils in server routes and operations"
```

---

### Task 11: Fix memory cache manager — stop stripping timezone + use today_start()

**Files:**
- Modify: `opencontext/server/cache/memory_cache_manager.py:388-396,586-601`

This is the most impactful change — fixes the "today events" timezone issue.

- [ ] **Step 1: Add import at the top of the file**

```python
from opencontext.utils.time_utils import now as tz_now, today_start as tz_today_start
```

- [ ] **Step 2: Fix time boundary calculation (lines 388-396)**

Replace:
```python
        now = datetime.now(tz=timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_ts = int(today_start.timestamp())
```

With:
```python
        now = tz_now()
        today_start_val = tz_today_start()
        today_start_ts = int(today_start_val.timestamp())
```

Also update the derived variables that reference `today_start` → `today_start_val`:
```python
        yesterday_start = today_start_val - timedelta(days=1)
        yesterday_end_ts = float(today_start_ts - 1)
        week_start = today_start_val - timedelta(days=days)
        ...
        period_start_dt = today_start_val - timedelta(days=days - 1)
```

- [ ] **Step 3: Stop stripping timezone in _ctx_to_recent_item (lines 586-601)**

Replace the timezone-stripping pattern:
```python
        create_time = None
        if props.create_time:
            if hasattr(props.create_time, "isoformat"):
                dt = props.create_time
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                create_time = dt.isoformat()
            else:
                create_time = str(props.create_time)

        event_time_start = None
        if props.event_time_start:
            if hasattr(props.event_time_start, "isoformat"):
                dt = props.event_time_start
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                event_time_start = dt.isoformat()
            else:
                event_time_start = str(props.event_time_start)
```

With (keep timezone info):
```python
        create_time = None
        if props.create_time:
            if hasattr(props.create_time, "isoformat"):
                create_time = props.create_time.isoformat()
            else:
                create_time = str(props.create_time)

        event_time_start = None
        if props.event_time_start:
            if hasattr(props.event_time_start, "isoformat"):
                event_time_start = props.event_time_start.isoformat()
            else:
                event_time_start = str(props.event_time_start)
```

- [ ] **Step 4: Replace any remaining `datetime.now(tz=timezone.utc)` in the file with `tz_now()`**

- [ ] **Step 5: Verify**

Run: `python -m py_compile opencontext/server/cache/memory_cache_manager.py`

- [ ] **Step 6: Commit**

```bash
git add opencontext/server/cache/memory_cache_manager.py
git commit -m "fix: memory cache uses configured timezone for today boundary, stops stripping tzinfo"
```

---

## Chunk 5: Periodic Tasks, Scheduler, Monitoring, Consumption

### Task 12: Update hierarchy summary (10 `datetime.combine` + 1 `date.today` + 1 `now`)

**Files:**
- Modify: `opencontext/periodic_task/hierarchy_summary.py`

- [ ] **Step 1: Add imports**

```python
from opencontext.utils.time_utils import now as tz_now, get_timezone
```

- [ ] **Step 2: Replace `datetime.date.today()` (line 216)**

```python
# Before:
today = datetime.date.today()
# After:
today = tz_now().date()
```

- [ ] **Step 3: Replace `datetime.datetime.now(tz=datetime.timezone.utc)` (line 1463)**

```python
# Before:
now = datetime.datetime.now(tz=datetime.timezone.utc)
# After:
now = tz_now()
```

- [ ] **Step 4: Replace all 10 `datetime.combine(..., tzinfo=datetime.timezone.utc)` calls**

Each must change `tzinfo=datetime.timezone.utc` → `tzinfo=get_timezone()`:

| Line | Context |
|------|---------|
| 767 | `backfill_start_ts = datetime.datetime.combine(..., tzinfo=datetime.timezone.utc)` |
| 770 | `backfill_end_ts = datetime.datetime.combine(..., tzinfo=datetime.timezone.utc)` |
| 867 | `day_start = datetime.datetime.combine(..., tzinfo=datetime.timezone.utc)` |
| 872 | `day_end = datetime.datetime.combine(..., tzinfo=datetime.timezone.utc)` |
| 988 | `week_start = datetime.datetime.combine(..., tzinfo=datetime.timezone.utc)` |
| 991 | `week_end = datetime.datetime.combine(..., tzinfo=datetime.timezone.utc)` |
| 1119 | `month_start = datetime.datetime.combine(..., tzinfo=datetime.timezone.utc)` |
| 1122 | `month_end = datetime.datetime.combine(..., tzinfo=datetime.timezone.utc)` |
| 1176 | `wk_start_dt = datetime.datetime.combine(..., tzinfo=datetime.timezone.utc)` |
| 1179 | `wk_end_dt = datetime.datetime.combine(..., tzinfo=datetime.timezone.utc)` |

For all 10 locations, the change is:
```python
# Before:
datetime.datetime.combine(some_date, datetime.time.min, tzinfo=datetime.timezone.utc)
# After:
datetime.datetime.combine(some_date, datetime.time.min, tzinfo=get_timezone())
```

- [ ] **Step 5: Verify**

Run: `python -m py_compile opencontext/periodic_task/hierarchy_summary.py`

- [ ] **Step 6: Commit**

```bash
git add opencontext/periodic_task/hierarchy_summary.py
git commit -m "refactor: use time_utils in hierarchy summary generation"
```

---

### Task 13: Update scheduler and monitoring

**Files:**
- Modify: `opencontext/scheduler/redis_scheduler.py:434,441`
- Modify: `opencontext/monitoring/monitor.py` (~15 `datetime.now(tz=timezone.utc)` + 1 sentinel at line 110)
- Modify: `opencontext/server/routes/monitoring.py:190`

- [ ] **Step 1: Fix redis_scheduler.py**

Add import:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace:
- Line 434: `datetime.now(tz=timezone.utc)` → `tz_now()`
- Line 441: `datetime.now(tz=timezone.utc)` → `tz_now()`

- [ ] **Step 2: Fix monitor.py**

Add import:
```python
from opencontext.utils.time_utils import now as tz_now, get_timezone
```

Replace all `datetime.now(tz=timezone.utc)` with `tz_now()` — this includes two categories:

**6 dataclass `default_factory` lambdas** (lines 32, 44, 55, 64, 74, 87):
```python
# Before:
timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
# After:
timestamp: datetime = field(default_factory=tz_now)
```

**~9 in-function calls** throughout the file — search for `datetime.now(tz=timezone.utc)` and replace with `tz_now()`.

Also fix the sentinel at line 110:
```python
# Before:
self._last_stats_update = datetime.min.replace(tzinfo=timezone.utc)
# After:
self._last_stats_update = datetime.min.replace(tzinfo=get_timezone())
```

- [ ] **Step 3: Fix monitoring.py route**

Add import, replace:
- Line 190: `datetime.now(tz=timezone.utc)` → `tz_now()`

- [ ] **Step 4: Verify**

```bash
python -m py_compile opencontext/scheduler/redis_scheduler.py
python -m py_compile opencontext/monitoring/monitor.py
python -m py_compile opencontext/server/routes/monitoring.py
```

- [ ] **Step 5: Commit**

```bash
git add opencontext/scheduler/redis_scheduler.py \
        opencontext/monitoring/monitor.py \
        opencontext/server/routes/monitoring.py
git commit -m "refactor: use time_utils.now() in scheduler and monitoring"
```

---

### Task 14: Update context consumption modules

**Files:**
- Modify: `opencontext/context_consumption/completion/completion_service.py:42`
- Modify: `opencontext/context_consumption/completion/completion_cache.py:184,196,230,241,279,320,492,535,565`
- Modify: `opencontext/context_consumption/context_agent/core/state.py:34,35,77,81,83,119,248`
- Modify: `opencontext/context_consumption/context_agent/core/llm_context_strategy.py:64,65,313`
- Modify: `opencontext/context_consumption/context_agent/nodes/base.py:35,38`
- Modify: `opencontext/context_consumption/context_agent/nodes/executor.py:72,79,110`
- Modify: `opencontext/context_consumption/context_agent/models/events.py:25`
- Modify: `opencontext/context_consumption/context_agent/models/schemas.py:70`

- [ ] **Step 1: Fix completion_service.py**

Add import:
```python
from opencontext.utils.time_utils import now as tz_now
```

- Line 42: `self.timestamp = datetime.now()` → `self.timestamp = tz_now()`

- [ ] **Step 2: Fix completion_cache.py**

Add import:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace all `datetime.now()` with `tz_now()`:
- Lines 184, 196, 230, 241, 279, 320, 492, 535, 565: `datetime.now()` → `tz_now()`
- Lines with `.isoformat()`: `datetime.now().isoformat()` → `tz_now().isoformat()`

- [ ] **Step 3: Fix state.py (including class-level `default_factory`)**

Add import:
```python
from opencontext.utils.time_utils import now as tz_now
```

Replace class-level `default_factory=datetime.now` (no parens — stores function reference):
- Line 34: `created_at: datetime = field(default_factory=datetime.now)` → `created_at: datetime = field(default_factory=tz_now)`
- Line 35: `updated_at: datetime = field(default_factory=datetime.now)` → `updated_at: datetime = field(default_factory=tz_now)`

Replace method-body calls:
- Lines 77, 81, 83, 119, 248: `datetime.now()` → `tz_now()`
- `.isoformat()` variants: `datetime.now().isoformat()` → `tz_now().isoformat()`

- [ ] **Step 4: Fix events.py and schemas.py (`default_factory=datetime.now`)**

**events.py** — add import, replace:
```python
from opencontext.utils.time_utils import now as tz_now
```
- Line 25: `timestamp: datetime = field(default_factory=datetime.now)` → `timestamp: datetime = field(default_factory=tz_now)`

**schemas.py** — add import, replace:
```python
from opencontext.utils.time_utils import now as tz_now
```
- Line 70: `timestamp: datetime = field(default_factory=datetime.now)` → `timestamp: datetime = field(default_factory=tz_now)`

- [ ] **Step 5: Fix llm_context_strategy.py**

Add import:
```python
from opencontext.utils.time_utils import now as tz_now
```

- Line 64: `current_date=datetime.now().strftime(...)` → `current_date=tz_now().strftime(...)`
- Line 65: `current_timestamp=int(datetime.now().timestamp())` → `current_timestamp=int(tz_now().timestamp())`
- Line 313: `timestamp=datetime.now()` → `timestamp=tz_now()`

- [ ] **Step 6: Fix base.py**

Add import:
```python
from opencontext.utils.time_utils import now as tz_now
```

- Line 35: `start_time = datetime.now()` → `start_time = tz_now()`
- Line 38: `duration = (datetime.now() - start_time)` → `duration = (tz_now() - start_time)`

- [ ] **Step 7: Fix executor.py**

Add import:
```python
from opencontext.utils.time_utils import now as tz_now
```

- Line 72: `step.end_time = datetime.now()` → `step.end_time = tz_now()`
- Line 79: `(datetime.now() - plan.steps[0].start_time)` → `(tz_now() - plan.steps[0].start_time)`
- Line 110: `step.start_time = datetime.now()` → `step.start_time = tz_now()`

- [ ] **Step 8: Verify all compile**

```bash
python -m py_compile opencontext/context_consumption/completion/completion_service.py
python -m py_compile opencontext/context_consumption/completion/completion_cache.py
python -m py_compile opencontext/context_consumption/context_agent/core/state.py
python -m py_compile opencontext/context_consumption/context_agent/core/llm_context_strategy.py
python -m py_compile opencontext/context_consumption/context_agent/nodes/base.py
python -m py_compile opencontext/context_consumption/context_agent/nodes/executor.py
python -m py_compile opencontext/context_consumption/context_agent/models/events.py
python -m py_compile opencontext/context_consumption/context_agent/models/schemas.py
```

- [ ] **Step 9: Commit**

```bash
git add opencontext/context_consumption/completion/completion_service.py \
        opencontext/context_consumption/completion/completion_cache.py \
        opencontext/context_consumption/context_agent/core/state.py \
        opencontext/context_consumption/context_agent/core/llm_context_strategy.py \
        opencontext/context_consumption/context_agent/nodes/base.py \
        opencontext/context_consumption/context_agent/nodes/executor.py \
        opencontext/context_consumption/context_agent/models/events.py \
        opencontext/context_consumption/context_agent/models/schemas.py
git commit -m "refactor: use time_utils.now() in all context consumption modules"
```

---

## Chunk 6: Cleanup, Docs, and Verification

### Task 15: Final sweep — verify no remaining `datetime.now()` calls

- [ ] **Step 1: Search for any remaining bare datetime.now() calls**

Run:
```bash
grep -rn "datetime\.now()" opencontext/ --include="*.py" | grep -v "time_utils.py" | grep -v "__pycache__"
```

Expected: no results.

Also check for `datetime.datetime.now()`:
```bash
grep -rn "datetime\.datetime\.now()" opencontext/ --include="*.py" | grep -v "time_utils.py" | grep -v "__pycache__"
```

Expected: no results.

Also check for `datetime.date.today()`:
```bash
grep -rn "datetime\.date\.today()" opencontext/ --include="*.py" | grep -v "__pycache__"
```

Expected: no results.

Also check for `default_factory=datetime.now` (no parens — function reference, not a call):
```bash
grep -rn "default_factory=datetime.now" opencontext/ --include="*.py" | grep -v "time_utils" | grep -v "__pycache__"
```

Expected: no results.

Also check for `default_factory=lambda.*datetime.now` (lambda wrapping datetime.now):
```bash
grep -rn "default_factory=lambda.*datetime.now" opencontext/ --include="*.py" | grep -v "time_utils" | grep -v "__pycache__"
```

Expected: no results.

Also check for `.replace(tzinfo=` to find any remaining hardcoded timezone assignments:
```bash
grep -rn "\.replace(tzinfo=" opencontext/ --include="*.py" | grep -v "time_utils.py" | grep -v "__pycache__"
```

Expected: no results outside `time_utils.py`.

- [ ] **Step 2: If any remaining, fix them following the same pattern**

- [ ] **Step 3: Run full compile check on all modified files**

```bash
find opencontext -name "*.py" -exec python -m py_compile {} \;
```

Expected: no errors.

- [ ] **Step 4: Commit any remaining fixes**

```bash
git add -u
git commit -m "refactor: final sweep — remove all remaining bare datetime.now() calls"
```

---

### Task 16: Update documentation

**Files:**
- Modify: `CLAUDE.md` (Pitfalls section)
- Modify: `docs/api_reference.md`

- [ ] **Step 1: Add timezone guidance to CLAUDE.md Pitfalls section**

Replace the existing "Use timezone-aware datetime everywhere" entry with an updated version:

```markdown
### Use `time_utils.now()` instead of `datetime.now()`
All datetime generation must go through `opencontext.utils.time_utils`. Import as `from opencontext.utils.time_utils import now as tz_now`. This ensures all datetimes use the globally-configured timezone from `config.yaml` (`timezone` key, IANA name like `"Asia/Shanghai"`). Never use `datetime.now()`, `datetime.now(tz=timezone.utc)`, or `datetime.utcnow()` directly. For protocol-mandated UTC (e.g. AWS S3 auth), use `utc_now()`. **Important:** Do not call `now()` at module-import time — the timezone is initialized at startup, after imports.
```

- [ ] **Step 2: Add timezone format note to docs/api_reference.md**

Add a note in the general section:

```markdown
> **时区说明**: 所有 API 返回的时间字段现在包含时区信息（如 `2026-03-21T08:30:00+08:00`），而非之前的无时区格式（`2026-03-21T00:30:00`）。时区由服务端 `config.yaml` 中的 `timezone` 配置项决定。
```

**Note:** This format change comes from `opencontext/utils/json_encoder.py` which calls `obj.isoformat()` on datetime objects. With timezone-aware datetimes, `isoformat()` automatically includes the offset. No code change needed in `json_encoder.py` — this is the desired behavior.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/api_reference.md
git commit -m "docs: add time_utils guidance and timezone format note"
```
