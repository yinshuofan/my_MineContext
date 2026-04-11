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


@pytest.mark.unit
class TestCollectTaskTypesDeprecation:
    """Tests for the YAML trigger_mode deprecation path."""

    def test_yaml_trigger_mode_fires_warning(
        self, fake_redis, base_scheduler_config, loguru_capture
    ):
        # Inject a trigger_mode field into the YAML-style config dict.
        base_scheduler_config["tasks"]["memory_compression"]["trigger_mode"] = "periodic"

        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)

        assert any(
            "memory_compression.trigger_mode" in m and "deprecated" in m for m in loguru_capture
        ), f"Expected deprecation warning, got: {loguru_capture}"
        # raw config should have the field stripped so it cannot leak through
        assert "trigger_mode" not in scheduler._pending_raw_configs["memory_compression"]

    def test_code_trigger_mode_wins_over_yaml(self, fake_redis, base_scheduler_config):
        # YAML says periodic, code registers user_activity — code wins.
        base_scheduler_config["tasks"]["memory_compression"]["trigger_mode"] = "periodic"

        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        scheduler.register_handler(
            "memory_compression", _noop_handler, trigger_mode=TriggerMode.USER_ACTIVITY
        )

        assert (
            scheduler._pending_task_configs["memory_compression"].trigger_mode
            == TriggerMode.USER_ACTIVITY
        )


@pytest.mark.unit
class TestInitTaskTypes:
    """Tests for init_task_types warn-and-skip behavior."""

    async def test_warns_for_configured_but_unregistered_task(
        self, fake_redis, base_scheduler_config, loguru_capture
    ):
        # base_scheduler_config has 2 enabled tasks (memory_compression, data_cleanup).
        # Only register one handler — the other should trigger a warning.
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        scheduler.register_handler(
            "memory_compression", _noop_handler, trigger_mode=TriggerMode.USER_ACTIVITY
        )

        await scheduler.init_task_types()

        assert any(
            "data_cleanup" in m and "no handler was registered" in m for m in loguru_capture
        ), f"Expected unregistered-task warning, got: {loguru_capture}"

        # Only memory_compression should have been written to Redis — verify
        # the Redis mock was called exactly once for register_task_type's hmset.
        hmset_calls = [
            c
            for c in fake_redis.hmset.call_args_list
            if "scheduler:task_type:memory_compression" in str(c)
        ]
        assert len(hmset_calls) == 1
        data_cleanup_calls = [
            c
            for c in fake_redis.hmset.call_args_list
            if "scheduler:task_type:data_cleanup" in str(c)
        ]
        assert len(data_cleanup_calls) == 0
