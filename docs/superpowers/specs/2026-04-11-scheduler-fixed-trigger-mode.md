# Spec: 把 scheduler 的 `trigger_mode` 从用户配置升级为代码契约

**Date**: 2026-04-11
**Status**: Draft (pending review)
**Owner**: yinshuofan
**Type**: Refactor / Config hardening

## 背景

`RedisTaskScheduler` 目前从 `config.yaml` 的 `scheduler.tasks.<name>.trigger_mode` 读取每个任务的触发模式（`user_activity` 或 `periodic`）。这个字段可以被用户在 YAML 里修改，也可以通过 `/api/settings` 热更新覆盖。

但 `trigger_mode` 实际上是 handler 的**接口契约**，不是运维旋钮：

- `user_activity` 模式下，handler 被调用时收到真实的 `(user_id, device_id, agent_id)`。
- `periodic` 模式下，handler 被调用时收到 `(None, None, None)`（由 `_process_periodic_tasks` 硬编码）。

任何 handler 的实现都是围绕这两种入参形态之一写的：

- `memory_compression`、`hierarchy_summary`、`agent_profile_update` 依赖 `user_id` 做用户隔离查询，必须是 `user_activity`。
- `data_cleanup` 做全局清理，天然是 `periodic`。

如果用户把 `data_cleanup.trigger_mode` 改成 `user_activity`，scheduler 会接受这个配置、把它排入 user-keyed 队列，但没人会调 `schedule_user_task("data_cleanup", ...)`，任务永远不会执行。反过来，把 `memory_compression.trigger_mode` 改成 `periodic`，`_process_periodic_tasks` 会以 `(None, None, None)` 调用 handler，handler 会因拿不到 `user_id` 立即崩溃或静默失效。

这是一个"配置项伪装成运维旋钮，实际是代码决定"的典型案例。Spec 的目标是把这个字段从 YAML 剔除，让它在代码中被权威地声明。

## 目标 / 非目标

### 目标

- `trigger_mode` 不再作为 YAML 用户配置的一部分被读取。
- `trigger_mode` 成为 `register_handler()` 的必填参数，由调用方在代码里显式传入 `TriggerMode` 枚举值。
- 未来新增 task 时，开发者在 `component_initializer.py` 的 `register_handler` 调用处必须提供 trigger_mode，缺失会在 Python 调用参数绑定时立即抛 `TypeError`（keyword-only 必填参数）。
- 给现存部署留软迁移空间：YAML 里残留 `trigger_mode` 字段不会导致启动失败，只打 `WARN` 日志提示清理。

### 非目标

- **不**改动其他 task 字段（`enabled` / `interval` / `timeout` / `task_ttl` / `max_retries` / `retention_days` / `backfill_days` 保持现状，继续通过 YAML + `/api/settings` 热更新管理）。
- **不**新增新的触发模式或新的 task。
- **不**重构 `BasePeriodicTask` 的类层级。
- **不**清理 Redis 里旧的 `trigger_mode` 字段——`init_task_types()` 每次启动的 `hmset` 会自然覆盖，无需迁移脚本。

## 方案选型

评估了三种放置"权威 trigger_mode"的方案（详见 brainstorm 对话）：

| 方案 | 描述 | 结论 |
|---|---|---|
| **A. `register_handler()` 必填参数** | `scheduler.register_handler(name, handler, trigger_mode=TriggerMode.XXX)` | ✅ **采用** |
| B. 类属性 + initializer 引用 | `MemoryCompressionTask.TRIGGER_MODE` + 在 initializer 里 import 并引用 | 否 |
| C. 集中注册表 | 新建 `TASK_TRIGGER_MODES: dict[str, TriggerMode]` 常量 | 否 |

**选 A 的理由**：

1. 改动最小 —— 只动 `register_handler` 签名、`_collect_task_types`、4 处调用点，加 YAML 清理。
2. 类型安全 —— `TriggerMode` 是 StrEnum，非枚举值/拼错会在启动时直接 `TypeError`。
3. 方案 A 让 `register_handler` 的调用位置同时承担"handler 登记"和"契约声明"两个职责，声明和使用零距离。
4. 方案 B 把声明放在 task 类里，但 task 类本身从不读自己的 `trigger_mode`（读它的是 scheduler）。这种"写了不用"的属性反而容易被忽视。
5. 方案 C 的字符串 key 注册表会带来 typo 风险，且在当前只有 4 个 task 的规模下引入集中注册表是过度抽象。

## 架构变更

### 涉及的文件

| 文件 | 变更类型 |
|---|---|
| `opencontext/scheduler/base.py` | `ITaskScheduler.register_handler` 抽象签名加 `trigger_mode` 必填参数 |
| `opencontext/scheduler/redis_scheduler.py` | `register_handler` 实现同步更新；`_collect_task_types` 不再读 `trigger_mode` 并对残留字段打 WARN；`init_task_types` 对"configured but unregistered"的 task 打 WARN；新增"handler 注册时把 trigger_mode 写回对应的 `TaskConfig`"逻辑 |
| `opencontext/server/component_initializer.py` | 4 处 `scheduler.register_handler(...)` 调用传入 `trigger_mode=TriggerMode.USER_ACTIVITY/PERIODIC` |
| `config/config.yaml` | 删除 4 个 task 下的 `trigger_mode` 字段 |
| `config/config-docker.yaml` | 删除 3 个 task 下的 `trigger_mode` 字段（该文件没有 `agent_profile_update` 块，跳过） |
| `opencontext/scheduler/MODULE.md` | 更新：说明 `trigger_mode` 现在是代码契约，更新"扩展指南"章节 |
| `CLAUDE.md` | `Pitfalls` 章节 `Scheduler` 相关段落更新；`Extending the System` 的 "New scheduler task" 流程更新 |
| `tests/scheduler/__init__.py` | 新建 |
| `tests/scheduler/conftest.py` | 新建：`fake_redis`、`base_scheduler_config`、`loguru_capture` fixtures |
| `tests/scheduler/test_redis_scheduler.py` | 新建：10 个单元测试 |
| `tests/server/__init__.py` | 新建（如不存在） |
| `tests/server/test_component_initializer.py` | 新建：1 个集成风格测试，验证 4 个 task 的 trigger mode |

### 新的启动数据流

```
1. RedisTaskScheduler.__init__(redis_cache, config)
   └─ _collect_task_types()
      对 config["tasks"] 里每个 enabled=True 的 task：
        - 读取 { enabled, interval, timeout, task_ttl, max_retries, description }
        - 存入 self._pending_raw_configs: dict[str, dict]
        - 如果 raw_config 里包含 "trigger_mode" 键：
            WARN("YAML field 'scheduler.tasks.{name}.trigger_mode' is deprecated
                  and ignored; trigger_mode is now defined in code at registration time.
                  Please remove this field from your config.")
      (不再构造 TaskConfig —— TaskConfig 会在 register_handler 里构造)

2. component_initializer.initialize_task_scheduler()
   依次调用：
     scheduler.register_handler("memory_compression", handler,
                                 trigger_mode=TriggerMode.USER_ACTIVITY)
     scheduler.register_handler("data_cleanup", handler,
                                 trigger_mode=TriggerMode.PERIODIC)
     scheduler.register_handler("hierarchy_summary", handler,
                                 trigger_mode=TriggerMode.USER_ACTIVITY)
     scheduler.register_handler("agent_profile_update", handler,
                                 trigger_mode=TriggerMode.USER_ACTIVITY)

   register_handler(name, handler, *, trigger_mode) 的实现：
     - 类型校验：isinstance(trigger_mode, TriggerMode) 否则 TypeError
     - 检查 name 是否在 self._pending_raw_configs 中：
         不在 → ValueError(f"Handler registered for unknown task type: {name}. "
                          f"Make sure it's declared in config['tasks'].")
     - 从 raw_config 构造完整的 TaskConfig（trigger_mode 用传入值）
     - 存入 self._pending_task_configs[name]
     - self._task_handlers[name] = handler

3. scheduler.start() → init_task_types()
   遍历 self._pending_raw_configs 的所有 name：
     - 如果在 self._pending_task_configs 中 → hmset 到 Redis
     - 不在（配置了但没注册 handler） → WARN("Task '{name}' is configured but no
                                            handler was registered; it will not run.")
   清空 self._pending_raw_configs 和 self._pending_task_configs。
   最后调用 _sync_disabled_task_types() 同步 disabled 标记（逻辑不变）。
```

### 签名变更

```python
# 旧：
def register_handler(self, task_type: str, handler: TaskHandler) -> None: ...

# 新：
def register_handler(
    self,
    task_type: str,
    handler: TaskHandler,
    *,
    trigger_mode: TriggerMode,
) -> None: ...
```

`trigger_mode` 是 keyword-only，强制调用方显式命名参数、避免和位置参数顺序混淆，也让未来扩展 `register_handler` 签名更安全。

### TaskConfig 的处理

`TaskConfig` 的字段不变（`trigger_mode` 仍然是它的一部分，因为 scheduler 内部运行时逻辑——`_executor_loop` 区分 type worker vs periodic worker、`_process_periodic_tasks` 的筛选、`schedule_user_task` 对 periodic 的拒绝——都依赖这个字段）。

变化的只是 `TaskConfig` 的**构造时机**：从"`_collect_task_types` 在 `__init__` 里构造"迁移到"`register_handler` 在注册时构造"。

为此引入一个新的内部字段：

```python
class RedisTaskScheduler:
    def __init__(self, ...):
        ...
        self._pending_raw_configs: Dict[str, Dict[str, Any]] = {}  # 新增
        self._pending_task_configs: Dict[str, TaskConfig] = {}     # 保留
```

## 错误处理与边界情况

| 场景 | 行为 |
|---|---|
| YAML 仍有 `trigger_mode` 字段 | `_collect_task_types` 打 `WARN`，字段被忽略 |
| `register_handler` 漏传 `trigger_mode` | Python 在参数绑定阶段 `TypeError`（keyword-only 必填参数）|
| `register_handler` 传入字符串 `"user_activity"` 而不是 `TriggerMode` 枚举 | 主动 `isinstance` 检查，`TypeError` + 清晰错误消息 |
| `register_handler` 的 name 在 config 里不存在 | `ValueError`，错误消息指出应该去 config 里声明 |
| config 里声明了 task 但没注册 handler | `init_task_types` 打 `WARN` 并跳过，保留现有"配了但没启用"的宽容语义 |
| 热更新：用户通过 `/api/settings` 给 scheduler 塞 `trigger_mode` | `reload_components` 重启 scheduler → 新的 `_collect_task_types` WARN + 忽略。config 层的 `deep_merge` 会保留 DB 里的字段，但 scheduler 读不到 |
| Redis 里旧 hash 有 `trigger_mode` | 下次 `init_task_types` 的 `hmset` 会覆盖为代码声明的值 |
| 回滚 | `git revert`；Redis 里的 `trigger_mode` 字段会在下次启动被自然覆盖，无需手动清理 |

## 测试计划

### 新增文件

```
tests/
├── scheduler/
│   ├── __init__.py
│   ├── conftest.py                  # fixtures
│   └── test_redis_scheduler.py      # 10 个单元测试（T1–T10）
└── server/
    ├── __init__.py
    └── test_component_initializer.py # 1 个集成测试（T11，必须）
```

### 测试用例

所有测试都用 `@pytest.mark.unit` 或 `@pytest.mark.integration` 标记；`asyncio_mode = "auto"` 自动处理 `async def test_*`。

| # | 文件 | 用例 | 断言 |
|---|---|---|---|
| T1 | test_redis_scheduler.py | `test_register_handler_requires_trigger_mode` | 不传 `trigger_mode` → `TypeError` |
| T2 | test_redis_scheduler.py | `test_register_handler_with_user_activity` | 传 `USER_ACTIVITY` → `_pending_task_configs[name].trigger_mode == USER_ACTIVITY` |
| T3 | test_redis_scheduler.py | `test_register_handler_with_periodic` | 传 `PERIODIC` → `_pending_task_configs[name].trigger_mode == PERIODIC` |
| T4 | test_redis_scheduler.py | `test_register_handler_unknown_task_name_raises` | name 不在 config → `ValueError`, 消息含 `"unknown task type"` |
| T5 | test_redis_scheduler.py | `test_register_handler_rejects_non_enum_value` | 传 `"user_activity"` 字符串 → `TypeError` |
| T6 | test_redis_scheduler.py | `test_collect_task_types_warns_on_yaml_trigger_mode` | config 里残留字段 → `loguru_capture` 有 "deprecated" 关键字 WARN；初始化不崩 |
| T7 | test_redis_scheduler.py | `test_code_trigger_mode_wins_over_yaml` | YAML 写 `"periodic"`，代码注册 `USER_ACTIVITY` → 最终 `TaskConfig.trigger_mode == USER_ACTIVITY` |
| T8 | test_redis_scheduler.py | `test_init_task_types_warns_for_configured_but_unregistered_task` | 有 2 个 task 配置，只注册 1 个 handler → `await init_task_types()` 日志有 WARN，`fake_redis.hmset` 只被调用 1 次 |
| T9 | test_redis_scheduler.py | `test_init_task_types_writes_correct_trigger_mode_to_redis` | 验证 `fake_redis.hmset` 调用参数里每个 task 的 `trigger_mode` 字段和代码声明一致 |
| T10 | test_redis_scheduler.py | `test_task_config_roundtrip_preserves_trigger_mode` | `TaskConfig.from_dict(TaskConfig(...).to_dict()).trigger_mode` 往返无损 |
| T11 | test_component_initializer.py | `test_initialize_task_scheduler_registers_four_tasks_with_correct_modes` | monkeypatch `init_scheduler` / `peek_redis_cache` / 相关 getter，调用 `ComponentInitializer.initialize_task_scheduler(fake_processor_manager)`，捕获 `register_handler` 的所有调用，断言：`("memory_compression", _, USER_ACTIVITY)`、`("data_cleanup", _, PERIODIC)`、`("hierarchy_summary", _, USER_ACTIVITY)`、`("agent_profile_update", _, USER_ACTIVITY)` 四次调用都存在且参数正确 |

### Fixtures（`tests/scheduler/conftest.py`）

- **`fake_redis`**：`unittest.mock.AsyncMock`，打桩 `hmset` / `hget` / `hgetall` / `expire` / `pipeline`。
- **`base_scheduler_config`**：最小可用 scheduler config dict，包含 `user_key_config`、`executor`、`tasks`（2 个示例任务）。
- **`loguru_capture`**：通过 `logger.add(lambda msg: messages.append(str(msg)), level="WARNING")` 捕获 loguru 日志；yield `messages` list；teardown 时 `logger.remove(handler_id)`。

### 运行命令

```bash
# 必须全部通过
uv run pytest tests/scheduler/ tests/server/test_component_initializer.py -v

# 覆盖率（新增代码应覆盖）
uv run pytest tests/scheduler/ --cov=opencontext.scheduler --cov-report=term-missing

# 全部 unit 测试（确保现有测试不回归）
uv run pytest -m unit
```

### 手动 smoke 验证（补充）

1. `uv run opencontext start` 启动无 WARN/ERROR
2. `redis-cli HGETALL scheduler:task_type:memory_compression` → `trigger_mode=user_activity`
3. `redis-cli HGETALL scheduler:task_type:data_cleanup` → `trigger_mode=periodic`
4. 往 `config.yaml` 的 `memory_compression` 加回 `trigger_mode: "periodic"` 重启 → 日志有 deprecation WARN，Redis 里值仍然是 `user_activity`
5. 临时把 `component_initializer.py` 里某个 `register_handler` 的 `trigger_mode=` 删掉启动 → 立即 `TypeError` 报错，错误位置清晰

## 文档更新

### `CLAUDE.md`

1. `Pitfalls and Lessons Learned > Scheduler pitfalls` 下新增一条：
   > **`trigger_mode` is code-determined, not user-configurable**: `trigger_mode` is declared via `scheduler.register_handler(..., trigger_mode=TriggerMode.XXX)` at handler registration time, not in YAML. YAML fields named `trigger_mode` are deprecated and ignored with a warning. This is because `trigger_mode` determines the handler's call contract (`user_activity` handlers receive `(user_id, device_id, agent_id)`; `periodic` handlers receive `(None, None, None)`), which is coupled to handler implementation.

2. `Extending the System > New scheduler task` 步骤更新：
   > - Implement `BasePeriodicTask` subclass
   > - Add factory `create_<name>_handler(...)` in the task file
   > - Add YAML config in `config/config.yaml` under `scheduler.tasks.<name>` (NO `trigger_mode` field — it's set in code)
   > - Call `scheduler.register_handler(name, handler, trigger_mode=TriggerMode.XXX)` in `component_initializer.initialize_task_scheduler`
   > - For `user_activity` tasks: call `scheduler.schedule_user_task` from the relevant push endpoint

### `opencontext/scheduler/MODULE.md`

1. 更新 `register_handler` 签名描述。
2. 在 "Conventions & Constraints" 部分新增一条说明 trigger_mode 是代码契约。
3. 更新任何列出 YAML 字段的示例，删掉 `trigger_mode`。

## 往后兼容 & 回滚

- **前向兼容**：老部署的 YAML 保留 `trigger_mode` 字段能正常启动，只多一条 WARN 日志。不需要迁移脚本。
- **回滚**：纯代码改动，`git revert` 即可。Redis 状态在下次启动会被 `init_task_types` 的 `hmset` 自然覆盖。
- **热更新路径**：通过 `/api/settings` 修改 `scheduler.tasks.<name>.trigger_mode` 仍能被 DB deep_merge 保存（因为 `SAVEABLE_KEYS` 白名单仍然包括 `scheduler`），但 scheduler 不读这个字段；下次 reload 时会打 WARN。

## 未采纳的替代

**更严厉的 deprecation 策略**：启动时直接对残留 `trigger_mode` 字段 `ValueError` 而不是 WARN。否决理由：会让老部署起不来，违背"最小 blast radius"原则。内部工具用 WARN 足够引起注意。

**把 `TaskConfig.trigger_mode` 改为 Optional**：让 `_collect_task_types` 构造 TaskConfig 时留空，`register_handler` 后补。否决理由：运行时代码（`_executor_loop`、`_process_periodic_tasks`）假设 `trigger_mode` 永远非空，Optional 化只会让这些地方的 `.trigger_mode` 访问变成 `assert` 或 `Optional` 分支，得不偿失。改为"两段式 pending（raw dict → TaskConfig）"更干净。

**集中注册表方案 C**：见"方案选型"章节。
