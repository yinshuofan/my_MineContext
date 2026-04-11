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
