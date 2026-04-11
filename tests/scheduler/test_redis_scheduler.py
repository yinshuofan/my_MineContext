"""Unit tests for opencontext.scheduler.redis_scheduler."""

from __future__ import annotations

import pytest

from opencontext.scheduler.base import TriggerMode
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

    def test_register_handler_with_periodic(self, fake_redis, base_scheduler_config):
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        scheduler.register_handler("data_cleanup", _noop_handler, trigger_mode=TriggerMode.PERIODIC)
        assert scheduler._pending_task_configs["data_cleanup"].trigger_mode == TriggerMode.PERIODIC

    def test_register_handler_unknown_task_name_raises(self, fake_redis, base_scheduler_config):
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        with pytest.raises(ValueError, match="unknown task type"):
            scheduler.register_handler(
                "nonexistent_task",
                _noop_handler,
                trigger_mode=TriggerMode.USER_ACTIVITY,
            )

    def test_register_handler_rejects_non_enum_value(self, fake_redis, base_scheduler_config):
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        with pytest.raises(TypeError, match="TriggerMode enum"):
            scheduler.register_handler(
                "memory_compression",
                _noop_handler,
                trigger_mode="user_activity",  # type: ignore[arg-type]
            )
