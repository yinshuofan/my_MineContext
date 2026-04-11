# Scheduler Fixed Trigger Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `trigger_mode` a code-declared contract on `register_handler()` instead of a user-configurable YAML field, preventing users from silently breaking handler call contracts.

**Architecture:** Introduce a two-phase pending state inside `RedisTaskScheduler`: `_collect_task_types()` stores raw config dicts into `_pending_raw_configs`; `register_handler(name, handler, *, trigger_mode)` then builds the final `TaskConfig` by combining the raw dict with the caller-supplied `TriggerMode`. YAML `trigger_mode` fields become deprecated — detected, logged, and ignored. The four existing `register_handler` call sites in `component_initializer.py` hardcode each task's trigger mode.

**Tech Stack:** Python 3.11+, `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`), `unittest.mock.AsyncMock`, `loguru` (for capturing warnings in tests).

**Spec:** `docs/superpowers/specs/2026-04-11-scheduler-fixed-trigger-mode.md`

---

## File Structure

**Files to modify:**
- `opencontext/scheduler/base.py` — update `ITaskScheduler.register_handler` abstract signature to add `*, trigger_mode: TriggerMode` kwarg
- `opencontext/scheduler/redis_scheduler.py` — main refactor:
  - Change `_pending_task_configs` from `list` to `dict[str, TaskConfig]`
  - Add `_pending_raw_configs: dict[str, dict]`
  - Rewrite `_collect_task_types()` to populate raw configs, warn on residual `trigger_mode`
  - Rewrite `register_handler()` with required `trigger_mode` kwarg, type/name validation, TaskConfig construction
  - Update `init_task_types()` to iterate dict values and warn on configured-but-unregistered
- `opencontext/server/component_initializer.py` — update 4 `scheduler.register_handler(...)` calls at lines 233, 256, 262, 270 to pass `trigger_mode=TriggerMode.USER_ACTIVITY` or `TriggerMode.PERIODIC`
- `config/config.yaml` — delete `trigger_mode` lines from 4 tasks (lines 330, 337, 346, 356)
- `config/config-docker.yaml` — delete `trigger_mode` lines from 3 tasks (lines 157, 161, 167)
- `opencontext/scheduler/MODULE.md` — update `register_handler` signature description, add "code-determined trigger_mode" conventions entry
- `CLAUDE.md` — add pitfall entry in `Scheduler pitfalls` section; update `Extending the System > New scheduler task` flow

**Files to create:**
- `tests/scheduler/__init__.py` — empty package marker
- `tests/scheduler/conftest.py` — pytest fixtures (`fake_redis`, `base_scheduler_config`, `loguru_capture`)
- `tests/scheduler/test_redis_scheduler.py` — 10 unit tests (T1–T10)
- `tests/server/__init__.py` — empty package marker (only if not already present)
- `tests/server/test_component_initializer.py` — 1 integration-style test (T11)

---

## Task 1: Test Scaffolding

**Files:**
- Create: `tests/scheduler/__init__.py`
- Create: `tests/scheduler/conftest.py`
- Create: `tests/server/__init__.py` (if not exists)

- [ ] **Step 1: Check whether `tests/server/__init__.py` already exists**

Run: `ls tests/server/ 2>/dev/null || echo "directory does not exist"`

If the directory does not exist, both `tests/server/` and `tests/server/__init__.py` need to be created.

- [ ] **Step 2: Create `tests/scheduler/__init__.py`**

Create an empty file at `tests/scheduler/__init__.py`:

```python
```

(Zero bytes; pytest discovers tests under this package.)

- [ ] **Step 3: Create `tests/server/__init__.py` if missing**

Only if Step 1 showed the directory/file missing, create `tests/server/__init__.py`:

```python
```

- [ ] **Step 4: Create `tests/scheduler/conftest.py` with fixtures**

Create `tests/scheduler/conftest.py` with the full content below. This file defines three reusable fixtures for all scheduler tests.

```python
"""Shared fixtures for scheduler unit tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger


@pytest.fixture
def fake_redis():
    """In-memory stand-in for RedisCache.

    Every coroutine method used by init_task_types / register_task_type /
    _sync_disabled_task_types is stubbed as AsyncMock so the scheduler can
    run its startup path without a real Redis.
    """
    r = AsyncMock()
    r.hmset = AsyncMock(return_value=True)
    r.hset = AsyncMock(return_value=True)
    r.hget = AsyncMock(return_value=None)
    r.hgetall = AsyncMock(return_value={})
    r.exists = AsyncMock(return_value=False)
    r.expire = AsyncMock(return_value=True)
    r.delete = AsyncMock(return_value=True)

    # pipeline() is used by _write_heartbeat; return a MagicMock so async with works.
    pipe = MagicMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=None)
    pipe.hset = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock(return_value=[])
    r.pipeline = MagicMock(return_value=pipe)

    return r


@pytest.fixture
def base_scheduler_config() -> dict[str, Any]:
    """Minimal scheduler config with two tasks (one user_activity, one periodic).

    Tests mutate copies of this dict to model different scenarios.
    """
    return {
        "user_key_config": {
            "use_user_id": True,
            "use_device_id": True,
            "use_agent_id": True,
            "default_device_id": "default",
            "default_agent_id": "default",
        },
        "executor": {
            "check_interval": 10,
            "max_concurrent": 5,
        },
        "tasks": {
            "memory_compression": {
                "enabled": True,
                "interval": 30,
                "timeout": 300,
                "task_ttl": 7200,
            },
            "data_cleanup": {
                "enabled": True,
                "interval": 60,
                "timeout": 3600,
                "task_ttl": 86400,
                "retention_days": 30,
            },
        },
    }


@pytest.fixture
def loguru_capture():
    """Capture WARNING-and-above loguru messages into a list.

    Usage:
        def test_something(loguru_capture):
            do_thing_that_warns()
            assert any("deprecated" in m for m in loguru_capture)
    """
    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(str(msg)),
        level="WARNING",
        format="{message}",
    )
    try:
        yield messages
    finally:
        logger.remove(sink_id)
```

- [ ] **Step 5: Run existing tests to confirm nothing broke**

Run: `uv run pytest tests/ -v`
Expected: all existing tests pass; the new `tests/scheduler/` dir is discovered but collects zero tests (which is fine).

- [ ] **Step 6: Commit**

```bash
git add tests/scheduler/__init__.py tests/scheduler/conftest.py tests/server/__init__.py
git commit -m "$(cat <<'EOF'
test: add scheduler test scaffolding and fixtures

Adds tests/scheduler/{__init__.py,conftest.py} with fake_redis,
base_scheduler_config, and loguru_capture fixtures. Scaffolding only —
no test cases yet.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

(If `tests/server/__init__.py` was newly created in Step 3, include it in the `git add` above.)

---

## Task 2: Refactor scheduler to make trigger_mode code-driven (T1–T5)

**Files:**
- Modify: `opencontext/scheduler/base.py` (ITaskScheduler.register_handler abstract signature)
- Modify: `opencontext/scheduler/redis_scheduler.py` (core refactor)
- Modify: `opencontext/server/component_initializer.py` (4 call sites)
- Create: `tests/scheduler/test_redis_scheduler.py` (tests T1–T5)

This is the core atomic change. We write unit tests first (TDD), then make the implementation changes that satisfy them, then update the callers so the whole app still builds.

- [ ] **Step 1: Write T1 — register_handler requires trigger_mode**

Create `tests/scheduler/test_redis_scheduler.py` with the initial test:

```python
"""Unit tests for opencontext.scheduler.redis_scheduler."""

from __future__ import annotations

import pytest

from opencontext.scheduler.base import TaskConfig, TriggerMode
from opencontext.scheduler.redis_scheduler import RedisTaskScheduler


async def _noop_handler(user_id, device_id, agent_id):  # noqa: ARG001
    return True


@pytest.mark.unit
class TestRegisterHandler:
    """Tests for RedisTaskScheduler.register_handler signature and validation."""

    def test_register_handler_requires_trigger_mode(self, fake_redis, base_scheduler_config):
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        with pytest.raises(TypeError):
            scheduler.register_handler("memory_compression", _noop_handler)  # type: ignore[call-arg]
```

- [ ] **Step 2: Run T1 to verify it fails**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestRegisterHandler::test_register_handler_requires_trigger_mode -v`
Expected: **FAIL** — current `register_handler(self, task_type, handler)` accepts the call without raising.

- [ ] **Step 3: Update `ITaskScheduler.register_handler` abstract signature in `base.py`**

In `opencontext/scheduler/base.py`, replace the `register_handler` abstract method (around lines 199–216) with the new signature:

```python
    @abc.abstractmethod
    def register_handler(
        self,
        task_type: str,
        handler: Callable[[str, str | None, str | None], Awaitable[bool]],
        *,
        trigger_mode: "TriggerMode",
    ) -> bool:
        """
        Register an async handler function for a task type.

        Args:
            task_type: Name of the task type (must be present in the
                scheduler's config under scheduler.tasks)
            handler: Async function that takes (user_id, device_id,
                agent_id) and returns a success bool. For periodic tasks,
                all three arguments will be None.
            trigger_mode: Code-declared trigger mode for this handler.
                This is a handler implementation contract and must be
                provided at registration time; it is NOT read from YAML.

        Returns:
            True if registration successful

        Raises:
            TypeError: If trigger_mode is not a TriggerMode enum value
            ValueError: If task_type is not declared in config['tasks']
        """
        pass
```

- [ ] **Step 4: Rewrite scheduler internals in `redis_scheduler.py`**

In `opencontext/scheduler/redis_scheduler.py`, make the following changes:

**4a. Change `_pending_task_configs` field type in `__init__` (around line 109) from `list` to `dict` and add the new `_pending_raw_configs` field:**

Replace:

```python
        # Store pending task type configs for async initialization
        self._pending_task_configs: list = []
        self._collect_task_types()
```

with:

```python
        # Two-phase pending state for async initialization:
        #   raw configs are collected from YAML at __init__ time;
        #   full TaskConfigs are built later when register_handler() is
        #   called with a code-declared trigger_mode.
        self._pending_raw_configs: dict[str, dict[str, Any]] = {}
        self._pending_task_configs: dict[str, TaskConfig] = {}
        self._collect_task_types()
```

**4b. Rewrite `_collect_task_types` (lines 112–127):**

Replace the entire method body:

```python
    def _collect_task_types(self) -> None:
        """Collect task types from configuration for async initialization.

        Stores raw config dicts keyed by task name. trigger_mode is NOT
        read here — it must be supplied via register_handler() at
        registration time. If YAML still contains a trigger_mode field
        it is logged as deprecated and ignored.
        """
        tasks_config = self._config.get("tasks", {})
        for task_name, task_config in tasks_config.items():
            if not task_config.get("enabled", False):
                continue
            if "trigger_mode" in task_config:
                logger.warning(
                    f"YAML field 'scheduler.tasks.{task_name}.trigger_mode' is "
                    f"deprecated and ignored; trigger_mode is now defined in code "
                    f"at register_handler() time. Please remove this field from "
                    f"your config."
                )
            # Shallow copy, stripping the deprecated field to keep raw_config clean.
            raw = {k: v for k, v in task_config.items() if k != "trigger_mode"}
            self._pending_raw_configs[task_name] = raw
```

**4c. Rewrite `register_handler` (lines 173–177):**

Replace:

```python
    def register_handler(self, task_type: str, handler: TaskHandler) -> bool:
        """Register a handler function for a task type"""
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
        return True
```

with:

```python
    def register_handler(
        self,
        task_type: str,
        handler: TaskHandler,
        *,
        trigger_mode: TriggerMode,
    ) -> bool:
        """Register an async handler function for a task type.

        trigger_mode is a required code-declared contract: user_activity
        handlers receive (user_id, device_id, agent_id) from a user push;
        periodic handlers receive (None, None, None) from the periodic
        worker. Mixing these up will make the handler crash or silently
        no-op, so we force callers to state which contract this handler
        fulfils.

        Raises:
            TypeError: If trigger_mode is not a TriggerMode enum value
            ValueError: If task_type is not declared in config['tasks']
        """
        if not isinstance(trigger_mode, TriggerMode):
            raise TypeError(
                f"trigger_mode must be a TriggerMode enum value, "
                f"got {type(trigger_mode).__name__}: {trigger_mode!r}"
            )
        if task_type not in self._pending_raw_configs:
            raise ValueError(
                f"Handler registered for unknown task type: {task_type!r}. "
                f"Make sure it is declared (and enabled) under "
                f"scheduler.tasks in config.yaml. Known task types: "
                f"{sorted(self._pending_raw_configs.keys())}"
            )

        raw = self._pending_raw_configs[task_type]
        task_config = TaskConfig(
            name=task_type,
            enabled=True,
            trigger_mode=trigger_mode,
            interval=int(raw.get("interval", 1800)),
            timeout=int(raw.get("timeout", 300)),
            task_ttl=int(raw.get("task_ttl", 7200)),
            max_retries=int(raw.get("max_retries", 3)),
            description=raw.get("description", ""),
        )
        self._pending_task_configs[task_type] = task_config
        self._task_handlers[task_type] = handler
        logger.info(
            f"Registered handler for task type: {task_type} "
            f"(trigger_mode={trigger_mode.value})"
        )
        return True
```

**4d. Update `init_task_types` (lines 129–138) to iterate dict values and warn on configured-but-unregistered:**

Replace:

```python
    async def init_task_types(self) -> None:
        """Initialize task types in Redis (async)"""
        enabled_names: set[str] = set()
        for config in self._pending_task_configs:
            await self.register_task_type(config)
            enabled_names.add(config.name)
        self._pending_task_configs.clear()

        # Mark disabled task types in Redis so other workers' runtime guards take effect
        await self._sync_disabled_task_types(enabled_names)
```

with:

```python
    async def init_task_types(self) -> None:
        """Initialize task types in Redis (async).

        Writes one Redis hash per registered task. Any task that is
        declared in config but has no handler registered is logged as a
        warning and skipped — the task type will not be usable at
        runtime, but startup does not fail.
        """
        unregistered = set(self._pending_raw_configs.keys()) - set(
            self._pending_task_configs.keys()
        )
        for name in sorted(unregistered):
            logger.warning(
                f"Task '{name}' is configured in scheduler.tasks but no handler "
                f"was registered via register_handler(); it will not run. This is "
                f"usually a code bug in component_initializer."
            )

        enabled_names: set[str] = set()
        for config in self._pending_task_configs.values():
            await self.register_task_type(config)
            enabled_names.add(config.name)

        self._pending_raw_configs.clear()
        self._pending_task_configs.clear()

        # Mark disabled task types in Redis so other workers' runtime guards take effect
        await self._sync_disabled_task_types(enabled_names)
```

- [ ] **Step 5: Update the four `register_handler` calls in `component_initializer.py`**

In `opencontext/server/component_initializer.py`:

**5a.** Add the `TriggerMode` import to the imports inside `initialize_task_scheduler()` (around line 202):

Replace:

```python
            from opencontext.scheduler import init_scheduler
```

with:

```python
            from opencontext.scheduler import init_scheduler
            from opencontext.scheduler.base import TriggerMode
```

**5b.** Update line 233 (memory_compression):

Replace:

```python
                scheduler.register_handler("memory_compression", compression_handler)
```

with:

```python
                scheduler.register_handler(
                    "memory_compression",
                    compression_handler,
                    trigger_mode=TriggerMode.USER_ACTIVITY,
                )
```

**5c.** Update line 256 (data_cleanup):

Replace:

```python
                scheduler.register_handler("data_cleanup", cleanup_handler)
```

with:

```python
                scheduler.register_handler(
                    "data_cleanup",
                    cleanup_handler,
                    trigger_mode=TriggerMode.PERIODIC,
                )
```

**5d.** Update line 262 (hierarchy_summary):

Replace:

```python
                scheduler.register_handler("hierarchy_summary", hierarchy_handler)
```

with:

```python
                scheduler.register_handler(
                    "hierarchy_summary",
                    hierarchy_handler,
                    trigger_mode=TriggerMode.USER_ACTIVITY,
                )
```

**5e.** Update line 270 (agent_profile_update):

Replace:

```python
                scheduler.register_handler("agent_profile_update", agent_profile_handler)
```

with:

```python
                scheduler.register_handler(
                    "agent_profile_update",
                    agent_profile_handler,
                    trigger_mode=TriggerMode.USER_ACTIVITY,
                )
```

- [ ] **Step 6: Run T1 again to confirm it now passes**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestRegisterHandler::test_register_handler_requires_trigger_mode -v`
Expected: **PASS**.

- [ ] **Step 7: Add T2 — accepts USER_ACTIVITY and persists to TaskConfig**

Append to `tests/scheduler/test_redis_scheduler.py` inside the `TestRegisterHandler` class:

```python
    def test_register_handler_with_user_activity(self, fake_redis, base_scheduler_config):
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        result = scheduler.register_handler(
            "memory_compression", _noop_handler, trigger_mode=TriggerMode.USER_ACTIVITY
        )
        assert result is True
        assert "memory_compression" in scheduler._pending_task_configs
        assert (
            scheduler._pending_task_configs["memory_compression"].trigger_mode
            == TriggerMode.USER_ACTIVITY
        )
        assert scheduler._task_handlers["memory_compression"] is _noop_handler
```

- [ ] **Step 8: Run T2 to verify it passes**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestRegisterHandler::test_register_handler_with_user_activity -v`
Expected: **PASS**.

- [ ] **Step 9: Add T3 — accepts PERIODIC**

Append to `TestRegisterHandler`:

```python
    def test_register_handler_with_periodic(self, fake_redis, base_scheduler_config):
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        scheduler.register_handler(
            "data_cleanup", _noop_handler, trigger_mode=TriggerMode.PERIODIC
        )
        assert (
            scheduler._pending_task_configs["data_cleanup"].trigger_mode
            == TriggerMode.PERIODIC
        )
```

- [ ] **Step 10: Run T3 to verify it passes**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestRegisterHandler::test_register_handler_with_periodic -v`
Expected: **PASS**.

- [ ] **Step 11: Add T4 — unknown task name raises ValueError**

Append to `TestRegisterHandler`:

```python
    def test_register_handler_unknown_task_name_raises(
        self, fake_redis, base_scheduler_config
    ):
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        with pytest.raises(ValueError, match="unknown task type"):
            scheduler.register_handler(
                "nonexistent_task",
                _noop_handler,
                trigger_mode=TriggerMode.USER_ACTIVITY,
            )
```

- [ ] **Step 12: Run T4 to verify it passes**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestRegisterHandler::test_register_handler_unknown_task_name_raises -v`
Expected: **PASS**.

- [ ] **Step 13: Add T5 — rejects non-enum trigger_mode value**

Append to `TestRegisterHandler`:

```python
    def test_register_handler_rejects_non_enum_value(
        self, fake_redis, base_scheduler_config
    ):
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        with pytest.raises(TypeError, match="TriggerMode enum"):
            scheduler.register_handler(
                "memory_compression",
                _noop_handler,
                trigger_mode="user_activity",  # type: ignore[arg-type]
            )
```

- [ ] **Step 14: Run T5 to verify it passes**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestRegisterHandler::test_register_handler_rejects_non_enum_value -v`
Expected: **PASS**.

- [ ] **Step 15: Run the full scheduler test module to confirm all 5 tests pass**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py -v`
Expected: 5 passed, 0 failed.

- [ ] **Step 16: Run the full unit test suite to ensure no regressions**

Run: `uv run pytest -m unit -v`
Expected: all previously-passing tests still pass; new 5 tests pass.

- [ ] **Step 17: Commit**

```bash
git add opencontext/scheduler/base.py opencontext/scheduler/redis_scheduler.py opencontext/server/component_initializer.py tests/scheduler/test_redis_scheduler.py
git commit -m "$(cat <<'EOF'
refactor(scheduler): make trigger_mode a required register_handler kwarg

trigger_mode is the handler's call contract (user_activity handlers
receive a real user_id, periodic handlers receive None), so it belongs
in code next to the handler registration, not in YAML where users can
silently break it.

RedisTaskScheduler now stores raw task configs collected from YAML in
_pending_raw_configs and builds the final TaskConfig inside
register_handler() using the caller-supplied TriggerMode. Updates the
four registration sites in component_initializer with the correct
mode per handler, and adds 5 unit tests (T1-T5) covering the new
validation behavior.

YAML `trigger_mode` fields are still tolerated for backwards
compatibility (a deprecation warning is added in a follow-up commit).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Deprecation warning behavior tests (T6, T7)

The deprecation warning logic was already implemented in Task 2 Step 4b (the `if "trigger_mode" in task_config:` branch in `_collect_task_types`). This task adds tests that lock that behavior in.

**Files:**
- Modify: `tests/scheduler/test_redis_scheduler.py`

- [ ] **Step 1: Add T6 — YAML trigger_mode field fires deprecation warning**

Append a new test class to `tests/scheduler/test_redis_scheduler.py`:

```python
@pytest.mark.unit
class TestCollectTaskTypesDeprecation:
    """Tests for the YAML trigger_mode deprecation path."""

    def test_yaml_trigger_mode_fires_warning(
        self, fake_redis, base_scheduler_config, loguru_capture
    ):
        # Inject a trigger_mode field into the YAML-style config dict.
        base_scheduler_config["tasks"]["memory_compression"]["trigger_mode"] = "periodic"

        scheduler = RedisTaskScheduler(
            redis_cache=fake_redis, config=base_scheduler_config
        )

        assert any(
            "memory_compression.trigger_mode" in m and "deprecated" in m
            for m in loguru_capture
        ), f"Expected deprecation warning, got: {loguru_capture}"
        # raw config should have the field stripped so it cannot leak through
        assert "trigger_mode" not in scheduler._pending_raw_configs["memory_compression"]
```

- [ ] **Step 2: Run T6 to verify it passes**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestCollectTaskTypesDeprecation::test_yaml_trigger_mode_fires_warning -v`
Expected: **PASS**.

- [ ] **Step 3: Add T7 — code-declared trigger_mode wins over YAML**

Append to `TestCollectTaskTypesDeprecation`:

```python
    def test_code_trigger_mode_wins_over_yaml(
        self, fake_redis, base_scheduler_config
    ):
        # YAML says periodic, code registers user_activity — code wins.
        base_scheduler_config["tasks"]["memory_compression"]["trigger_mode"] = "periodic"

        scheduler = RedisTaskScheduler(
            redis_cache=fake_redis, config=base_scheduler_config
        )
        scheduler.register_handler(
            "memory_compression", _noop_handler, trigger_mode=TriggerMode.USER_ACTIVITY
        )

        assert (
            scheduler._pending_task_configs["memory_compression"].trigger_mode
            == TriggerMode.USER_ACTIVITY
        )
```

- [ ] **Step 4: Run T7 to verify it passes**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestCollectTaskTypesDeprecation::test_code_trigger_mode_wins_over_yaml -v`
Expected: **PASS**.

- [ ] **Step 5: Run the full scheduler test module**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/scheduler/test_redis_scheduler.py
git commit -m "$(cat <<'EOF'
test(scheduler): lock in YAML trigger_mode deprecation behavior

Adds T6 (deprecation warning fires and the field is stripped from the
raw config) and T7 (code-declared trigger_mode overrides any residual
YAML value).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Configured-but-unregistered warning test (T8)

The unregistered-task warning was already added in Task 2 Step 4d. This task adds a test.

**Files:**
- Modify: `tests/scheduler/test_redis_scheduler.py`

- [ ] **Step 1: Add T8**

Append a new test class to `tests/scheduler/test_redis_scheduler.py`:

```python
@pytest.mark.unit
class TestInitTaskTypes:
    """Tests for init_task_types warn-and-skip behavior."""

    async def test_warns_for_configured_but_unregistered_task(
        self, fake_redis, base_scheduler_config, loguru_capture
    ):
        # base_scheduler_config has 2 enabled tasks (memory_compression, data_cleanup).
        # Only register one handler — the other should trigger a warning.
        scheduler = RedisTaskScheduler(
            redis_cache=fake_redis, config=base_scheduler_config
        )
        scheduler.register_handler(
            "memory_compression", _noop_handler, trigger_mode=TriggerMode.USER_ACTIVITY
        )

        await scheduler.init_task_types()

        assert any(
            "data_cleanup" in m and "no handler was registered" in m
            for m in loguru_capture
        ), f"Expected unregistered-task warning, got: {loguru_capture}"

        # Only memory_compression should have been written to Redis — verify
        # the Redis mock was called exactly once for register_task_type's hmset.
        hmset_calls = [
            c for c in fake_redis.hmset.call_args_list
            if "scheduler:task_type:memory_compression" in str(c)
        ]
        assert len(hmset_calls) == 1
        data_cleanup_calls = [
            c for c in fake_redis.hmset.call_args_list
            if "scheduler:task_type:data_cleanup" in str(c)
        ]
        assert len(data_cleanup_calls) == 0
```

- [ ] **Step 2: Run T8 to verify it passes**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestInitTaskTypes::test_warns_for_configured_but_unregistered_task -v`
Expected: **PASS**.

- [ ] **Step 3: Commit**

```bash
git add tests/scheduler/test_redis_scheduler.py
git commit -m "$(cat <<'EOF'
test(scheduler): verify init_task_types warns on unregistered tasks

When a task is declared in scheduler.tasks but no handler is registered,
init_task_types should log a warning and skip it (not crash). T8 locks
this in.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Redis state + TaskConfig roundtrip tests (T9, T10)

**Files:**
- Modify: `tests/scheduler/test_redis_scheduler.py`

- [ ] **Step 1: Add T9 — init_task_types writes correct trigger_mode to Redis**

Append to `TestInitTaskTypes`:

```python
    async def test_writes_correct_trigger_mode_to_redis(
        self, fake_redis, base_scheduler_config
    ):
        scheduler = RedisTaskScheduler(
            redis_cache=fake_redis, config=base_scheduler_config
        )
        scheduler.register_handler(
            "memory_compression", _noop_handler, trigger_mode=TriggerMode.USER_ACTIVITY
        )
        scheduler.register_handler(
            "data_cleanup", _noop_handler, trigger_mode=TriggerMode.PERIODIC
        )

        await scheduler.init_task_types()

        # hmset is called as hmset(key, mapping_dict). Extract the mapping for
        # each task type and verify the trigger_mode field.
        calls_by_key: dict[str, dict] = {}
        for call in fake_redis.hmset.call_args_list:
            args, kwargs = call
            key = args[0] if args else kwargs.get("name")
            mapping = args[1] if len(args) > 1 else kwargs.get("mapping")
            calls_by_key[key] = mapping

        mc_mapping = calls_by_key["scheduler:task_type:memory_compression"]
        dc_mapping = calls_by_key["scheduler:task_type:data_cleanup"]
        assert mc_mapping["trigger_mode"] == "user_activity"
        assert dc_mapping["trigger_mode"] == "periodic"
```

- [ ] **Step 2: Run T9 to verify it passes**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestInitTaskTypes::test_writes_correct_trigger_mode_to_redis -v`
Expected: **PASS**.

If T9 fails because `fake_redis.hmset` is called positionally vs keyword — adjust the extraction: `register_task_type` calls `await self._redis.hmset(key, config.to_dict())`, so args are `(key, mapping)`. The test extracts args[0] and args[1], which should work.

- [ ] **Step 3: Add T10 — TaskConfig dataclass roundtrip preserves trigger_mode**

Append a new test class:

```python
@pytest.mark.unit
class TestTaskConfigRoundtrip:
    """Tests for TaskConfig serialization to/from Redis hash dict."""

    def test_roundtrip_preserves_user_activity(self):
        original = TaskConfig(
            name="memory_compression",
            enabled=True,
            trigger_mode=TriggerMode.USER_ACTIVITY,
            interval=30,
            timeout=300,
            task_ttl=7200,
            max_retries=3,
        )
        restored = TaskConfig.from_dict(original.to_dict())
        assert restored.trigger_mode == TriggerMode.USER_ACTIVITY
        assert restored.name == "memory_compression"
        assert restored.interval == 30

    def test_roundtrip_preserves_periodic(self):
        original = TaskConfig(
            name="data_cleanup",
            enabled=True,
            trigger_mode=TriggerMode.PERIODIC,
            interval=60,
            timeout=3600,
            task_ttl=86400,
            max_retries=3,
        )
        restored = TaskConfig.from_dict(original.to_dict())
        assert restored.trigger_mode == TriggerMode.PERIODIC
```

- [ ] **Step 4: Run T10 to verify it passes**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py::TestTaskConfigRoundtrip -v`
Expected: **PASS** (2 tests).

- [ ] **Step 5: Run the full scheduler test module**

Run: `uv run pytest tests/scheduler/test_redis_scheduler.py -v`
Expected: 11 passed (T1–T10, with T10 contributing 2 test methods).

- [ ] **Step 6: Commit**

```bash
git add tests/scheduler/test_redis_scheduler.py
git commit -m "$(cat <<'EOF'
test(scheduler): verify Redis write + TaskConfig roundtrip preserves trigger_mode

T9 inspects the fake Redis hmset calls to confirm each registered task
is written with the code-declared trigger_mode value. T10 exercises
TaskConfig.to_dict / from_dict to guard against future serialization
regressions.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Integration test for component_initializer (T11)

**Files:**
- Create: `tests/server/test_component_initializer.py`

This test monkeypatches the external dependencies (`peek_redis_cache`, `init_scheduler`, periodic_task factories, `ContextMerger`) so we can run `ComponentInitializer.initialize_task_scheduler()` in isolation and verify that each of the four tasks registers with the correct trigger mode.

- [ ] **Step 1: Create `tests/server/test_component_initializer.py`**

Create the file with this content:

```python
"""Integration test: ComponentInitializer.initialize_task_scheduler() must
register all four built-in tasks with the correct code-declared trigger modes."""

from __future__ import annotations

import pytest

from opencontext.scheduler.base import TriggerMode


@pytest.mark.unit
class TestInitializeTaskScheduler:
    def test_registers_four_tasks_with_correct_trigger_modes(self, monkeypatch):
        # --- Arrange: capture register_handler calls ---
        captured_calls: list[tuple[str, TriggerMode]] = []

        class FakeScheduler:
            def register_handler(self, task_type, handler, *, trigger_mode):
                captured_calls.append((task_type, trigger_mode))
                return True

        fake_scheduler = FakeScheduler()

        # Patch init_scheduler so ComponentInitializer receives our FakeScheduler.
        # The import inside initialize_task_scheduler is
        #   `from opencontext.scheduler import init_scheduler`
        # which resolves to opencontext.scheduler.init_scheduler — patch there.
        monkeypatch.setattr(
            "opencontext.scheduler.init_scheduler",
            lambda redis, config: fake_scheduler,
        )

        # Patch peek_redis_cache to return a truthy placeholder.
        monkeypatch.setattr(
            "opencontext.storage.redis_cache.peek_redis_cache",
            lambda: object(),
        )

        # Patch ContextMerger so we don't spin up real LLM clients / storage.
        class FakeMerger:
            pass

        monkeypatch.setattr(
            "opencontext.context_processing.merger.context_merger.ContextMerger",
            lambda *args, **kwargs: FakeMerger(),
        )

        # Patch the four periodic_task factories to return a trivial noop handler.
        async def noop_handler(user_id, device_id, agent_id):  # noqa: ARG001
            return True

        for factory_path in (
            "opencontext.periodic_task.create_compression_handler",
            "opencontext.periodic_task.create_cleanup_handler",
            "opencontext.periodic_task.create_hierarchy_handler",
            "opencontext.periodic_task.create_agent_profile_update_handler",
        ):
            monkeypatch.setattr(factory_path, lambda *args, **kwargs: noop_handler)

        # --- Build a ComponentInitializer without running __init__ ---
        # __init__ pulls from GlobalConfig which we don't want to hit here.
        from opencontext.server.component_initializer import ComponentInitializer

        initializer = ComponentInitializer.__new__(ComponentInitializer)
        initializer.global_config = None
        initializer.config_manager = None
        initializer.config = {
            "scheduler": {
                "enabled": True,
                "user_key_config": {
                    "use_user_id": True,
                    "use_device_id": True,
                    "use_agent_id": True,
                    "default_device_id": "default",
                    "default_agent_id": "default",
                },
                "executor": {"check_interval": 10, "max_concurrent": 5},
                "tasks": {
                    "memory_compression": {
                        "enabled": True,
                        "interval": 30,
                        "timeout": 300,
                        "task_ttl": 7200,
                    },
                    "data_cleanup": {
                        "enabled": True,
                        "interval": 60,
                        "timeout": 3600,
                        "task_ttl": 86400,
                        "retention_days": 30,
                    },
                    "hierarchy_summary": {
                        "enabled": True,
                        "interval": 86400,
                        "timeout": 600,
                        "task_ttl": 172800,
                        "backfill_days": 7,
                    },
                    "agent_profile_update": {
                        "enabled": True,
                        "interval": 86400,
                        "timeout": 300,
                        "task_ttl": 14400,
                    },
                },
            },
        }

        # --- Act ---
        initializer.initialize_task_scheduler(processor_manager=None)

        # --- Assert ---
        assert len(captured_calls) == 4, f"expected 4 registrations, got {captured_calls}"
        calls_map = dict(captured_calls)
        assert calls_map["memory_compression"] == TriggerMode.USER_ACTIVITY
        assert calls_map["data_cleanup"] == TriggerMode.PERIODIC
        assert calls_map["hierarchy_summary"] == TriggerMode.USER_ACTIVITY
        assert calls_map["agent_profile_update"] == TriggerMode.USER_ACTIVITY
```

- [ ] **Step 2: Run T11 to verify it passes**

Run: `uv run pytest tests/server/test_component_initializer.py -v`
Expected: **PASS** (1 test).

If the test fails with a `ModuleNotFoundError` or `AttributeError` on a monkeypatch target, that means the import structure of one of the modules differs from the assumption — inspect the actual import path with `uv run python -c "import opencontext.periodic_task; print(dir(opencontext.periodic_task))"` and fix the `monkeypatch.setattr` target accordingly.

- [ ] **Step 3: Run all new tests together**

Run: `uv run pytest tests/scheduler/ tests/server/test_component_initializer.py -v`
Expected: 12 tests passed.

- [ ] **Step 4: Commit**

```bash
git add tests/server/test_component_initializer.py
git commit -m "$(cat <<'EOF'
test(server): verify ComponentInitializer registers tasks with correct trigger modes

T11 pins the hardcoded trigger_mode per built-in task
(memory_compression=USER_ACTIVITY, data_cleanup=PERIODIC,
hierarchy_summary=USER_ACTIVITY, agent_profile_update=USER_ACTIVITY).
If someone edits component_initializer and flips one of these, this
test catches it.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Clean YAML configs

**Files:**
- Modify: `config/config.yaml`
- Modify: `config/config-docker.yaml`

- [ ] **Step 1: Remove `trigger_mode` line and its comment from `config/config.yaml`**

In `config/config.yaml`, make four deletions:

**1a.** Line 330 — memory_compression. Replace:

```yaml
    memory_compression:
      enabled: "${MEMORY_COMPRESSION_TASK_ENABLED:true}"
      trigger_mode: "user_activity" # 触发模式: user_activity（用户推送后延迟执行）, periodic（定时执行）
      interval: 30         # 用户活动后的延迟（秒）
```

with:

```yaml
    memory_compression:
      enabled: "${MEMORY_COMPRESSION_TASK_ENABLED:true}"
      interval: 30         # 用户活动后的延迟（秒）
```

**1b.** Line 337 — data_cleanup. Replace:

```yaml
    data_cleanup:
      enabled: false
      trigger_mode: "periodic"
      interval: 60         # 执行间隔（秒）
```

with:

```yaml
    data_cleanup:
      enabled: false
      interval: 60         # 执行间隔（秒）
```

**1c.** Line 346 — hierarchy_summary. Replace:

```yaml
    hierarchy_summary:
      enabled: "${HIERARCHY_SUMMARY_ENABLED:true}"
      trigger_mode: "user_activity"
      interval: 86400      # 用户活动后延迟 24h
```

with:

```yaml
    hierarchy_summary:
      enabled: "${HIERARCHY_SUMMARY_ENABLED:true}"
      interval: 86400      # 用户活动后延迟 24h
```

**1d.** Line 356 — agent_profile_update. Replace:

```yaml
    agent_profile_update:
      enabled: "${AGENT_PROFILE_UPDATE_ENABLED:true}"
      trigger_mode: "user_activity"
      interval: 86400       # 用户活动后延迟 24h
```

with:

```yaml
    agent_profile_update:
      enabled: "${AGENT_PROFILE_UPDATE_ENABLED:true}"
      interval: 86400       # 用户活动后延迟 24h
```

**1e.** Update the top-of-section comment at line 306. Replace:

```yaml
# ============================================================
# 任务调度器配置
# 基于 Redis 的分布式任务调度，支持 user_activity 和 periodic 两种触发模式
# ============================================================
```

with:

```yaml
# ============================================================
# 任务调度器配置
# 基于 Redis 的分布式任务调度。触发模式（user_activity / periodic）由代码在
# register_handler() 时声明，不在此处配置。
# ============================================================
```

- [ ] **Step 2: Remove `trigger_mode` lines from `config/config-docker.yaml`**

In `config/config-docker.yaml`, make three deletions:

**2a.** Line 157 — memory_compression. Replace:

```yaml
    memory_compression:
      enabled: true
      trigger_mode: "user_activity"
      interval: 1800
```

with:

```yaml
    memory_compression:
      enabled: true
      interval: 1800
```

**2b.** Line 161 — data_cleanup. Replace:

```yaml
    data_cleanup:
      enabled: true
      trigger_mode: "periodic"
      interval: 86400
```

with:

```yaml
    data_cleanup:
      enabled: true
      interval: 86400
```

**2c.** Line 167 — hierarchy_summary. Replace:

```yaml
    hierarchy_summary:
      enabled: "${HIERARCHY_SUMMARY_ENABLED:true}"
      trigger_mode: "user_activity"
      interval: 86400
```

with:

```yaml
    hierarchy_summary:
      enabled: "${HIERARCHY_SUMMARY_ENABLED:true}"
      interval: 86400
```

- [ ] **Step 3: Run the full test suite to make sure nothing broke**

Run: `uv run pytest -m unit -v`
Expected: all tests pass (same count as after Task 6).

- [ ] **Step 4: Commit**

```bash
git add config/config.yaml config/config-docker.yaml
git commit -m "$(cat <<'EOF'
config: remove trigger_mode from scheduler.tasks in yaml

trigger_mode is now declared in code at register_handler() time; YAML
values are ignored with a deprecation warning. Also updates the section
comment in config.yaml to reflect the new semantics.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Documentation updates

**Files:**
- Modify: `opencontext/scheduler/MODULE.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Locate the `register_handler` signature description in `MODULE.md`**

Run: `grep -n "register_handler" opencontext/scheduler/MODULE.md`

This returns the lines that need updating. The file is long (22k bytes); expect 2–4 hits.

- [ ] **Step 2: Update each `register_handler` occurrence in `MODULE.md`**

Where the signature is described, replace the old signature with:

```
register_handler(task_type, handler, *, trigger_mode: TriggerMode) -> bool
```

And where `register_handler` is described in prose, make sure the description mentions:

> `trigger_mode` is a keyword-only, required parameter. It is a code-declared contract (not user config) that determines the handler's calling convention: `USER_ACTIVITY` handlers receive `(user_id, device_id, agent_id)`; `PERIODIC` handlers receive `(None, None, None)`.

- [ ] **Step 3: Add a conventions entry in `MODULE.md`**

Find the "Conventions & Constraints" section and add a new bullet:

```markdown
- **`trigger_mode` is code-declared, not user-configurable**: Every `register_handler` call must pass a `TriggerMode` enum value as a keyword argument. YAML fields named `scheduler.tasks.<name>.trigger_mode` are deprecated — they are detected during `_collect_task_types`, logged as a deprecation warning, and stripped from the internal raw config. If you need to change a task's trigger mode you must edit `component_initializer.initialize_task_scheduler` and also update the handler implementation to match the new calling convention.
```

- [ ] **Step 4: Update the example in `MODULE.md` for adding a new task**

Find the "Extension Points" / "Adding a new task" section (if present). Update the example `register_handler` call to use the new signature:

```python
scheduler.register_handler(
    "my_new_task",
    create_my_new_task_handler(...),
    trigger_mode=TriggerMode.USER_ACTIVITY,  # or TriggerMode.PERIODIC
)
```

And remove any YAML example line that shows `trigger_mode: "..."`.

- [ ] **Step 5: Add a pitfall entry in `CLAUDE.md`**

In `CLAUDE.md`, find the `Scheduler pitfalls` subsection under `Pitfalls and Lessons Learned`. Append a new entry at the end of the subsection:

```markdown
### `trigger_mode` is code-determined, not user-configurable
`trigger_mode` is declared via `scheduler.register_handler(..., trigger_mode=TriggerMode.XXX)` at handler registration time in `component_initializer.py`, NOT in YAML. YAML fields named `scheduler.tasks.<name>.trigger_mode` are deprecated: `_collect_task_types` detects them, logs a deprecation warning, and strips them from the internal raw config. The reason is that `trigger_mode` determines the handler's call contract — `user_activity` handlers receive `(user_id, device_id, agent_id)` from user push flows, `periodic` handlers receive `(None, None, None)` from the periodic worker. Letting users flip this via config silently breaks the handler.
```

- [ ] **Step 6: Update the "New scheduler task" extension entry in `CLAUDE.md`**

In `CLAUDE.md`, find the `Extending the System` section. Update the "New scheduler task" bullet (or equivalent) to the new flow. Replace whatever the existing bullet says about scheduler tasks with:

```markdown
- **New scheduler task**:
  1. Implement `BasePeriodicTask` subclass in `opencontext/periodic_task/<name>.py`
  2. Add a `create_<name>_handler(...)` factory in the same file that returns the async handler
  3. Add the task's tunable fields under `scheduler.tasks.<name>` in `config/config.yaml` (NOT `trigger_mode` — that is set in code)
  4. Register the handler in `ComponentInitializer.initialize_task_scheduler()` with `scheduler.register_handler(name, handler, trigger_mode=TriggerMode.USER_ACTIVITY)` (or `TriggerMode.PERIODIC`)
  5. For `user_activity` tasks: call `scheduler.schedule_user_task(name, user_id, device_id, agent_id)` from the push endpoint that should trigger it (see `opencontext/server/routes/push.py`)
  6. Add a corresponding branch to `tests/server/test_component_initializer.py::test_registers_four_tasks_with_correct_trigger_modes` (rename if needed) so the hardcoded mode is locked under test
```

(If the existing bullet already follows a different format, preserve that format — just make sure the new content is there and `trigger_mode` is called out as code-declared.)

- [ ] **Step 7: Verify docs compile (no broken markdown)**

Run: `uv run python -c "import pathlib; print(pathlib.Path('CLAUDE.md').read_text(encoding='utf-8')[:200])"`
Expected: prints the first 200 chars of CLAUDE.md without error. (Also a visual skim of the edited sections to confirm markdown formatting is intact.)

- [ ] **Step 8: Commit**

```bash
git add opencontext/scheduler/MODULE.md CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: document trigger_mode as a code-declared contract

Updates scheduler/MODULE.md with the new register_handler signature
and a Conventions entry. Adds a Scheduler pitfall in CLAUDE.md and
updates the "New scheduler task" extension steps so future contributors
know trigger_mode is set in code, not YAML.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Manual smoke verification

This task has no commits — it is a list of manual checks to run before declaring the work done.

- [ ] **Step 1: Start the server and watch the startup log**

Run (in a separate terminal):
```bash
uv run opencontext start
```
Expected:
- No `ERROR` log lines
- No `TypeError` or `ValueError` about `trigger_mode`
- Four `Registered handler for task type: <name> (trigger_mode=<mode>)` info lines
- No deprecation warnings (because Task 7 removed YAML `trigger_mode` fields)

Stop the server with Ctrl+C after startup completes.

- [ ] **Step 2: Inspect Redis state for the 4 task types**

Run:
```bash
redis-cli HGETALL scheduler:task_type:memory_compression
redis-cli HGETALL scheduler:task_type:data_cleanup
redis-cli HGETALL scheduler:task_type:hierarchy_summary
redis-cli HGETALL scheduler:task_type:agent_profile_update
```
Expected: each hash has a `trigger_mode` field with the hardcoded value (`user_activity` for all except `data_cleanup` which is `periodic`).

- [ ] **Step 3: Test deprecation warning with a re-added YAML field**

Temporarily add the line `      trigger_mode: "user_activity"` back to `config/config.yaml` under `scheduler.tasks.memory_compression`, restart the server, and verify the log contains:

```
... WARNING ... YAML field 'scheduler.tasks.memory_compression.trigger_mode' is deprecated and ignored ...
```

**Important**: revert the YAML edit after this check — do NOT commit it.

- [ ] **Step 4: Test missing trigger_mode produces TypeError**

Temporarily comment out `trigger_mode=TriggerMode.USER_ACTIVITY,` in one `register_handler` call in `component_initializer.py` (e.g. the memory_compression one). Restart the server. Expected: startup crashes with a clear `TypeError: register_handler() missing 1 required keyword-only argument: 'trigger_mode'`.

**Important**: revert the edit after this check.

- [ ] **Step 5: Run the full test suite one final time**

Run: `uv run pytest -v`
Expected: all tests pass (including the 12 new ones from Tasks 1–6).

- [ ] **Step 6: Run the Python linter / formatter**

Run:
```bash
uv run ruff check opencontext tests
uv run ruff format opencontext tests
```
Expected: no lint errors and no format changes (or only whitespace fixups on the new files). If `ruff format` makes changes, commit them:

```bash
git add -u
git commit -m "$(cat <<'EOF'
style: ruff formatting on scheduler trigger_mode changes

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task(s) |
|---|---|
| `register_handler` gains required `trigger_mode` kwarg | Task 2 (steps 3–6) |
| `_collect_task_types` ignores YAML `trigger_mode` with WARN | Task 2 step 4b + Task 3 (T6, T7) |
| `init_task_types` warns on configured-but-unregistered | Task 2 step 4d + Task 4 (T8) |
| `component_initializer` updated with hardcoded modes | Task 2 step 5 |
| YAML cleanup in both config files | Task 7 |
| 10 unit tests (T1–T10) | Tasks 2, 3, 4, 5 |
| 1 integration test T11 (mandatory) | Task 6 |
| Backwards compat: YAML `trigger_mode` does not break startup | Task 2 step 4b (just warns), verified in Task 9 step 3 |
| Redis state auto-migrates (hmset overwrites) | No explicit task — natural behavior verified in Task 9 step 2 |
| MODULE.md + CLAUDE.md docs updated | Task 8 |
| Test scaffolding (`fake_redis`, `loguru_capture`, fixtures) | Task 1 |
| Manual smoke verification | Task 9 |

All spec requirements have a corresponding task.

### Type / signature consistency

- `TriggerMode` is the canonical enum, imported from `opencontext.scheduler.base` throughout.
- `register_handler` keyword-only signature is identical in `base.py` (abstract), `redis_scheduler.py` (implementation), `component_initializer.py` (4 call sites), and all tests.
- `_pending_raw_configs: dict[str, dict[str, Any]]` and `_pending_task_configs: dict[str, TaskConfig]` are used consistently (note: the old `_pending_task_configs: list` is fully replaced).
- Test fixture field names (`fake_redis`, `base_scheduler_config`, `loguru_capture`) match across all tests.

### Placeholder scan

- No `TBD`, `TODO`, `fill in details`, `similar to above`, or hand-waved error handling.
- Every code step contains the exact code to write or the exact replacement block with before/after.
- Every `Run:` step has an explicit command and expected outcome.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-11-scheduler-fixed-trigger-mode.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
