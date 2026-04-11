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

    async def test_writes_correct_trigger_mode_to_redis(self, fake_redis, base_scheduler_config):
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        scheduler.register_handler(
            "memory_compression", _noop_handler, trigger_mode=TriggerMode.USER_ACTIVITY
        )
        scheduler.register_handler("data_cleanup", _noop_handler, trigger_mode=TriggerMode.PERIODIC)

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


@pytest.mark.unit
class TestListTaskTriggerModes:
    """Tests for RedisTaskScheduler.list_task_trigger_modes (read-only view for UI)."""

    async def test_empty_before_init(self, fake_redis, base_scheduler_config):
        # Nothing registered, cache is empty — returns empty dict.
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        assert scheduler.list_task_trigger_modes() == {}

    async def test_returns_registered_modes_after_init(self, fake_redis, base_scheduler_config):
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        scheduler.register_handler(
            "memory_compression", _noop_handler, trigger_mode=TriggerMode.USER_ACTIVITY
        )
        scheduler.register_handler("data_cleanup", _noop_handler, trigger_mode=TriggerMode.PERIODIC)

        # _task_config_cache is populated by register_task_type, which runs
        # inside init_task_types — we must flush the pending dicts through
        # to the cache before the read-only snapshot has anything to show.
        await scheduler.init_task_types()

        modes = scheduler.list_task_trigger_modes()
        assert modes == {
            "memory_compression": "user_activity",
            "data_cleanup": "periodic",
        }

    async def test_returned_values_are_strings_not_enums(self, fake_redis, base_scheduler_config):
        # The snapshot must be JSON-serializable for the settings API, so
        # the values should be the raw StrEnum .value strings, not the
        # enum members themselves.
        scheduler = RedisTaskScheduler(redis_cache=fake_redis, config=base_scheduler_config)
        scheduler.register_handler(
            "memory_compression", _noop_handler, trigger_mode=TriggerMode.USER_ACTIVITY
        )
        await scheduler.init_task_types()

        modes = scheduler.list_task_trigger_modes()
        assert isinstance(modes["memory_compression"], str)
        assert modes["memory_compression"] == "user_activity"
