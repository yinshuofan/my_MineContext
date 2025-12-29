# Merger 初始化逻辑修改方案

## 当前状态分析

### 1. 当前代码结构

```
OpenContext.initialize()
    │
    ├── component_initializer.initialize_processors()
    │       │
    │       ├── 创建各种 Processor (document, screenshot, text_chat)
    │       │
    │       └── 如果 context_merger.enabled:
    │               ├── 创建 ContextMerger
    │               ├── processor_manager.set_merger(merger)
    │               └── processor_manager.start_periodic_compression()  ← 启动本地定时器
    │
    └── (其他初始化...)
```

### 2. processor_manager.py 中的定时逻辑

```python
class ContextProcessorManager:
    def __init__(self):
        self._compression_timer: Optional[Timer] = None  # 本地 Timer
        self._compression_interval: int = 1800
    
    def start_periodic_compression(self):
        """使用 threading.Timer 启动本地定时任务"""
        self._compression_timer = Timer(1, self._run_periodic_compression)
        self._compression_timer.daemon = True
        self._compression_timer.start()
    
    def _run_periodic_compression(self):
        """执行压缩并重新调度"""
        self._merger.periodic_memory_compression(self._compression_interval)
        # 重新调度
        self._compression_timer = Timer(self._compression_interval, self._run_periodic_compression)
        self._compression_timer.start()
```

### 3. 问题

| 问题 | 说明 |
|------|------|
| **本地定时器** | `threading.Timer` 是本地状态，多实例部署时每个实例都会执行 |
| **全局压缩** | 当前压缩是全局的，不区分用户 |
| **与新 scheduler 重复** | 新的 scheduler 已经实现了用户级压缩调度 |

---

## 修改方案

### 目标

1. **移除 processor_manager.py 中的本地定时器逻辑**
2. **将定时任务启动改为启动 scheduler**
3. **保持 merger 的 set_merger 逻辑不变**（merger 仍然需要被设置，供其他地方调用）

### 修改内容

#### 1. processor_manager.py

**删除以下内容：**
- `_compression_timer` 成员变量
- `_compression_interval` 成员变量
- `start_periodic_compression()` 方法
- `_run_periodic_compression()` 方法
- `stop_periodic_compression()` 方法
- `shutdown()` 中对 `stop_periodic_compression()` 的调用

**保留以下内容：**
- `_merger` 成员变量
- `set_merger()` 方法（merger 仍需被设置，供 Push API 触发的压缩使用）

#### 2. component_initializer.py

**修改 `initialize_processors()` 方法：**

```python
# 原代码
if processing_config.get("context_merger", {}).get("enabled", False):
    merger = ContextMerger()
    processor_manager.set_merger(merger)
    processor_manager.start_periodic_compression()  # ← 删除这行
    logger.info("Periodic memory compression started")

# 新代码
if processing_config.get("context_merger", {}).get("enabled", False):
    merger = ContextMerger()
    processor_manager.set_merger(merger)
    logger.info("Context merger initialized")
    # 注意：定时压缩任务现在由 scheduler 管理，不再在这里启动
```

**修改 `initialize_task_scheduler()` 方法：**

确保 scheduler 的初始化在 processor 初始化之后，这样可以获取到 merger 实例：

```python
def initialize_task_scheduler(self, processor_manager: ContextProcessorManager = None) -> None:
    """
    Initialize the task scheduler for periodic tasks.
    
    Args:
        processor_manager: Optional processor manager to get merger instance
    """
    # ... 现有逻辑 ...
    
    # Memory compression handler - 使用 processor_manager 中的 merger
    if tasks_config.get("memory_compression", {}).get("enabled", False):
        merger = None
        if processor_manager:
            merger = processor_manager._merger
        if not merger:
            from opencontext.context_processing.merger.context_merger import ContextMerger
            merger = ContextMerger()
        
        compression_handler = create_compression_handler(merger)
        scheduler.register_handler("memory_compression", compression_handler)
```

#### 3. opencontext.py

**修改 `initialize()` 方法，添加 scheduler 初始化：**

```python
def initialize(self) -> None:
    """Initialize all components in proper order."""
    # ... 现有初始化 ...
    
    self.component_initializer.initialize_processors(
        self.processor_manager, self._handle_processed_context
    )
    
    # 在 processor 初始化之后，初始化 task scheduler
    self.component_initializer.initialize_task_scheduler(self.processor_manager)
    
    # ... 其他初始化 ...
```

**修改 `shutdown()` 方法，添加 scheduler 停止：**

```python
def shutdown(self, graceful: bool = False) -> None:
    """Shutdown all components."""
    # ... 现有逻辑 ...
    
    # 停止 task scheduler
    self.component_initializer.stop_task_scheduler()
    
    # ... 其他清理 ...
```

---

## 代码修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `processor_manager.py` | **修改** | 删除本地定时器相关代码 |
| `component_initializer.py` | **修改** | 移除 `start_periodic_compression()` 调用，修改 `initialize_task_scheduler()` |
| `opencontext.py` | **修改** | 添加 scheduler 初始化和停止调用 |

---

## 修改后的初始化流程

```
OpenContext.initialize()
    │
    ├── component_initializer.initialize_processors()
    │       │
    │       ├── 创建各种 Processor
    │       │
    │       └── 如果 context_merger.enabled:
    │               ├── 创建 ContextMerger
    │               └── processor_manager.set_merger(merger)  # 只设置 merger，不启动定时器
    │
    ├── component_initializer.initialize_task_scheduler(processor_manager)
    │       │
    │       ├── 创建 RedisTaskScheduler
    │       ├── 注册 memory_compression handler (使用 processor_manager._merger)
    │       ├── 注册 data_cleanup handler
    │       └── 启动 scheduler 后台执行器
    │
    └── (其他初始化...)
```

---

## 任务触发流程

### 用户活动触发（user_activity 模式）

```
用户调用 Push API (chat/screenshot/document)
    │
    ▼
push.py: _schedule_user_compression()
    │
    ▼
scheduler.schedule_user_task("memory_compression", user_id, device_id, agent_id)
    │
    ▼
Redis: 创建/更新任务状态，加入待执行队列
    │
    ▼
(等待 interval 时间后)
    │
    ▼
scheduler 后台执行器检测到任务到期
    │
    ▼
获取分布式锁，执行 memory_compression handler
    │
    ▼
handler 调用 merger.compress_user_memory(user_id, device_id, agent_id)
```

### 周期性触发（periodic 模式，如 data_cleanup）

```
scheduler 后台执行器定期检查
    │
    ▼
检测到 periodic 任务到期
    │
    ▼
获取分布式锁，执行 data_cleanup handler
    │
    ▼
handler 调用 storage.cleanup_old_data(retention_days)
```

---

## 兼容性说明

1. **merger 仍然可用**：`processor_manager._merger` 仍然保留，其他代码可以直接调用 merger 的方法
2. **配置兼容**：`context_merger.enabled` 配置仍然有效，控制是否创建 merger
3. **新旧模式切换**：如果 `scheduler.enabled` 为 false，则不启动 scheduler，系统仍可正常运行（只是没有自动压缩）

---

## 测试要点

1. **单实例测试**：验证 scheduler 正常启动和执行任务
2. **多实例测试**：验证分布式锁防止重复执行
3. **用户级压缩**：验证不同用户的压缩任务独立执行
4. **优雅关闭**：验证 shutdown 时 scheduler 正确停止
