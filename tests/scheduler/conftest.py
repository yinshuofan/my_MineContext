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
