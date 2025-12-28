# 通用定时任务调度器方案

## 1. 方案概述

### 1.1 设计目标

1. **通用调度器**：独立的定时任务调度服务，支持多种任务类型（压缩、清理等）
2. **用户级任务**：支持 `user_id`, `device_id`, `agent_id` 三个维度，可配置使用哪些维度
3. **无状态化**：所有状态存储在 Redis，支持多实例部署
4. **非阻塞**：异步执行，不阻塞主服务

### 1.2 架构设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           多实例服务                                     │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┤
│  实例 1     │  实例 2     │  实例 3     │    ...      │  实例 N         │
│  (无状态)   │  (无状态)   │  (无状态)   │             │  (无状态)       │
│             │             │             │             │                 │
│ ┌─────────┐ │ ┌─────────┐ │ ┌─────────┐ │             │ ┌─────────────┐ │
│ │Scheduler│ │ │Scheduler│ │ │Scheduler│ │             │ │  Scheduler  │ │
│ │Executor │ │ │Executor │ │ │Executor │ │             │ │  Executor   │ │
│ └─────────┘ │ └─────────┘ │ └─────────┘ │             │ └─────────────┘ │
└──────┬──────┴──────┬──────┴──────┬──────┴─────────────┴────────┬────────┘
       │             │             │                              │
       └─────────────┴──────┬──────┴──────────────────────────────┘
                            │
                     ┌──────┴──────┐
                     │    Redis    │
                     ├─────────────┤
                     │ 任务定义    │
                     │ 任务队列    │
                     │ 执行状态    │
                     │ 分布式锁    │
                     └─────────────┘
```

## 2. 核心设计

### 2.1 用户标识 Key 设计

支持 3 个维度的用户标识，可通过配置控制使用哪些维度：

```python
# 完整 3 key 模式
user_key = f"{user_id}:{device_id}:{agent_id}"

# 2 key 模式（配置 use_agent_id: false）
user_key = f"{user_id}:{device_id}"

# 1 key 模式（配置 use_device_id: false, use_agent_id: false）
user_key = f"{user_id}"
```

**配置示例**：

```yaml
scheduler:
  user_key_config:
    use_user_id: true      # 必须为 true
    use_device_id: true    # 是否使用 device_id
    use_agent_id: true     # 是否使用 agent_id
    default_device_id: "default"
    default_agent_id: "default"
```

### 2.2 任务类型设计

调度器支持注册多种任务类型：

| 任务类型 | 说明 | 触发方式 |
|----------|------|----------|
| `memory_compression` | 记忆压缩 | 用户活动触发 |
| `data_cleanup` | 数据清理 | 定时触发 |
| `embedding_refresh` | 向量刷新 | 用户活动触发 |
| `statistics_update` | 统计更新 | 定时触发 |
| ... | 可扩展 | ... |

### 2.3 任务触发模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `user_activity` | 用户活动触发，延迟执行 | 压缩、刷新 |
| `periodic` | 固定周期执行 | 清理、统计 |
| `cron` | Cron 表达式调度 | 复杂调度需求 |

## 3. 数据结构设计

### 3.1 Redis Key 设计

```
# 任务类型定义 (Hash)
scheduler:task_type:{task_type}
{
    "name": "memory_compression",
    "description": "Memory compression task",
    "trigger_mode": "user_activity",  # user_activity | periodic | cron
    "interval": 1800,                  # 执行间隔（秒）
    "timeout": 300,                    # 执行超时（秒）
    "enabled": true
}

# 用户任务状态 (Hash)
scheduler:task:{task_type}:{user_key}
{
    "status": "pending|running|completed|failed",
    "created_at": 1735200000,
    "scheduled_at": 1735201800,
    "last_activity": 1735200500,
    "user_id": "user_123",
    "device_id": "device_456",
    "agent_id": "agent_789",
    "retry_count": 0
}
TTL: 根据任务类型配置

# 待执行任务队列 (Sorted Set)
scheduler:queue:{task_type}
Score: scheduled_at (Unix timestamp)
Member: "{user_key}"

# 上次执行时间 (String)
scheduler:last_exec:{task_type}:{user_key}
Value: Unix timestamp
TTL: 86400 (24小时)

# 分布式锁 (String)
scheduler:lock:{task_type}:{user_key}
Value: lock_token
TTL: 根据任务 timeout 配置

# 全局周期任务状态 (Hash) - 用于 periodic 模式
scheduler:periodic:{task_type}
{
    "last_run": 1735200000,
    "next_run": 1735203600,
    "status": "idle|running"
}
```

### 3.2 配置结构

```yaml
# config/config_mysql.yaml

scheduler:
  enabled: true
  check_interval: 10          # 任务检查间隔（秒）
  
  # 用户 Key 配置
  user_key_config:
    use_user_id: true         # 必须为 true
    use_device_id: true       # 是否使用 device_id
    use_agent_id: true        # 是否使用 agent_id
    default_device_id: "default"
    default_agent_id: "default"
  
  # 任务类型配置
  tasks:
    memory_compression:
      enabled: true
      trigger_mode: "user_activity"  # 用户活动触发
      interval: 1800                  # 30分钟后执行
      timeout: 300                    # 5分钟超时
      task_ttl: 7200                  # 任务状态保留2小时
      max_retries: 3
      
    data_cleanup:
      enabled: true
      trigger_mode: "periodic"        # 固定周期执行
      interval: 86400                 # 每24小时执行一次
      timeout: 600                    # 10分钟超时
      
    embedding_refresh:
      enabled: false
      trigger_mode: "user_activity"
      interval: 3600                  # 1小时后执行
      timeout: 120
```

## 4. 核心组件设计

### 4.1 UserKeyBuilder（用户 Key 构建器）

```python
class UserKeyBuilder:
    """
    用户 Key 构建器
    
    根据配置决定使用哪些维度构建用户标识 Key
    """
    
    def __init__(self, config: dict):
        self._use_user_id = config.get("use_user_id", True)
        self._use_device_id = config.get("use_device_id", True)
        self._use_agent_id = config.get("use_agent_id", True)
        self._default_device_id = config.get("default_device_id", "default")
        self._default_agent_id = config.get("default_agent_id", "default")
    
    def build_key(
        self,
        user_id: str,
        device_id: str = None,
        agent_id: str = None
    ) -> str:
        """
        构建用户 Key
        
        根据配置决定使用哪些维度：
        - 3 key: user_id:device_id:agent_id
        - 2 key: user_id:device_id
        - 1 key: user_id
        """
        parts = [user_id]
        
        if self._use_device_id:
            parts.append(device_id or self._default_device_id)
        
        if self._use_agent_id:
            parts.append(agent_id or self._default_agent_id)
        
        return ":".join(parts)
    
    def parse_key(self, user_key: str) -> dict:
        """
        解析用户 Key 为各个维度
        """
        parts = user_key.split(":")
        result = {"user_id": parts[0]}
        
        idx = 1
        if self._use_device_id and idx < len(parts):
            result["device_id"] = parts[idx]
            idx += 1
        
        if self._use_agent_id and idx < len(parts):
            result["agent_id"] = parts[idx]
        
        return result
    
    def get_key_dimensions(self) -> list:
        """返回当前使用的维度列表"""
        dims = ["user_id"]
        if self._use_device_id:
            dims.append("device_id")
        if self._use_agent_id:
            dims.append("agent_id")
        return dims
```

### 4.2 TaskScheduler（任务调度器）

```python
class TaskScheduler:
    """
    通用定时任务调度器（无状态）
    
    支持多种任务类型，所有状态存储在 Redis 中。
    """
    
    # Redis Key 前缀
    TASK_TYPE_PREFIX = "scheduler:task_type:"
    TASK_PREFIX = "scheduler:task:"
    QUEUE_PREFIX = "scheduler:queue:"
    LAST_EXEC_PREFIX = "scheduler:last_exec:"
    LOCK_PREFIX = "scheduler:lock:"
    PERIODIC_PREFIX = "scheduler:periodic:"
    
    def __init__(self, redis_cache: RedisCache, config: dict):
        self._redis = redis_cache
        self._config = config
        self._check_interval = config.get("check_interval", 10)
        self._user_key_builder = UserKeyBuilder(config.get("user_key_config", {}))
        self._task_handlers: Dict[str, Callable] = {}
        self._running = False
        
        # 初始化任务类型配置
        self._init_task_types()
    
    def _init_task_types(self):
        """从配置初始化任务类型"""
        tasks_config = self._config.get("tasks", {})
        for task_type, task_config in tasks_config.items():
            if task_config.get("enabled", False):
                self._register_task_type(task_type, task_config)
    
    def _register_task_type(self, task_type: str, config: dict):
        """注册任务类型到 Redis"""
        key = f"{self.TASK_TYPE_PREFIX}{task_type}"
        self._redis.hset_all(key, {
            "name": task_type,
            "trigger_mode": config.get("trigger_mode", "user_activity"),
            "interval": str(config.get("interval", 1800)),
            "timeout": str(config.get("timeout", 300)),
            "task_ttl": str(config.get("task_ttl", 7200)),
            "max_retries": str(config.get("max_retries", 3)),
            "enabled": "true",
        })
    
    def register_handler(self, task_type: str, handler: Callable):
        """
        注册任务处理器
        
        Args:
            task_type: 任务类型名称
            handler: 处理函数，签名为 (user_id, device_id, agent_id, **kwargs) -> bool
        """
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def schedule_user_task(
        self,
        task_type: str,
        user_id: str,
        device_id: str = None,
        agent_id: str = None
    ) -> bool:
        """
        为用户调度任务（用于 user_activity 触发模式）
        
        如果用户已有待执行任务，则不重复创建，只更新活动时间。
        
        Returns:
            True 如果创建了新任务，False 如果任务已存在
        """
        # 获取任务类型配置
        task_config = self._get_task_config(task_type)
        if not task_config or task_config.get("trigger_mode") != "user_activity":
            logger.warning(f"Task type {task_type} not found or not user_activity mode")
            return False
        
        user_key = self._user_key_builder.build_key(user_id, device_id, agent_id)
        task_key = f"{self.TASK_PREFIX}{task_type}:{user_key}"
        queue_key = f"{self.QUEUE_PREFIX}{task_type}"
        last_exec_key = f"{self.LAST_EXEC_PREFIX}{task_type}:{user_key}"
        
        interval = int(task_config.get("interval", 1800))
        task_ttl = int(task_config.get("task_ttl", 7200))
        
        # 检查是否已有任务
        existing = self._redis.hgetall(task_key)
        if existing and existing.get("status") in ("pending", "running"):
            # 任务已存在，更新活动时间并刷新 TTL
            now = int(time.time())
            self._redis.hset(task_key, "last_activity", str(now))
            self._redis.expire(task_key, task_ttl)
            logger.debug(f"Task {task_type} already exists for {user_key}")
            return False
        
        # 检查上次执行时间
        last_exec = self._redis.get(last_exec_key)
        if last_exec:
            last_exec_time = int(last_exec)
            if time.time() - last_exec_time < interval:
                logger.debug(
                    f"Skipping {task_type} for {user_key}, "
                    f"last executed {int(time.time() - last_exec_time)}s ago"
                )
                return False
        
        # 创建新任务
        now = int(time.time())
        scheduled_at = now + interval
        
        # 解析 user_key 获取各维度值
        key_parts = self._user_key_builder.parse_key(user_key)
        
        task_data = {
            "status": "pending",
            "created_at": str(now),
            "scheduled_at": str(scheduled_at),
            "last_activity": str(now),
            "user_id": key_parts.get("user_id", user_id),
            "device_id": key_parts.get("device_id", device_id or "default"),
            "agent_id": key_parts.get("agent_id", agent_id or "default"),
            "retry_count": "0",
        }
        
        # 写入任务状态
        self._redis.hset_all(task_key, task_data)
        self._redis.expire(task_key, task_ttl)
        
        # 加入任务队列
        self._redis.zadd(queue_key, {user_key: scheduled_at})
        
        logger.info(
            f"Scheduled {task_type} task for {user_key}, "
            f"will execute at {scheduled_at}"
        )
        return True
    
    def get_pending_task(self, task_type: str) -> Optional[Dict[str, Any]]:
        """
        获取一个待执行的任务（带分布式锁）
        
        Returns:
            任务信息字典，如果没有待执行任务则返回 None
        """
        task_config = self._get_task_config(task_type)
        if not task_config:
            return None
        
        queue_key = f"{self.QUEUE_PREFIX}{task_type}"
        timeout = int(task_config.get("timeout", 300))
        now = int(time.time())
        
        # 获取所有到期的任务
        tasks = self._redis.zrangebyscore(queue_key, 0, now)
        
        for user_key in tasks:
            # 尝试获取分布式锁
            lock_key = f"{self.LOCK_PREFIX}{task_type}:{user_key}"
            lock_token = self._redis.acquire_lock(
                lock_key,
                timeout=timeout,
                blocking=False
            )
            
            if not lock_token:
                # 其他实例正在处理，跳过
                continue
            
            # 更新任务状态
            task_key = f"{self.TASK_PREFIX}{task_type}:{user_key}"
            self._redis.hset(task_key, "status", "running")
            
            # 从队列中移除
            self._redis.zrem(queue_key, user_key)
            
            # 解析 user_key
            key_parts = self._user_key_builder.parse_key(user_key)
            
            return {
                "task_type": task_type,
                "user_key": user_key,
                "user_id": key_parts.get("user_id"),
                "device_id": key_parts.get("device_id"),
                "agent_id": key_parts.get("agent_id"),
                "lock_token": lock_token,
            }
        
        return None
    
    def complete_task(
        self,
        task_type: str,
        user_key: str,
        lock_token: str,
        success: bool = True
    ):
        """标记任务完成并释放锁"""
        task_key = f"{self.TASK_PREFIX}{task_type}:{user_key}"
        lock_key = f"{self.LOCK_PREFIX}{task_type}:{user_key}"
        last_exec_key = f"{self.LAST_EXEC_PREFIX}{task_type}:{user_key}"
        
        # 更新任务状态
        self._redis.hset(task_key, "status", "completed" if success else "failed")
        
        # 记录执行时间
        if success:
            self._redis.set(last_exec_key, str(int(time.time())))
            self._redis.expire(last_exec_key, 86400)  # 24小时
        
        # 释放锁
        self._redis.release_lock(lock_key, lock_token)
        
        logger.info(
            f"Task {task_type} for {user_key} "
            f"{'completed' if success else 'failed'}"
        )
    
    def _get_task_config(self, task_type: str) -> Optional[dict]:
        """获取任务类型配置"""
        key = f"{self.TASK_TYPE_PREFIX}{task_type}"
        return self._redis.hgetall(key)
    
    async def start(self):
        """启动调度器后台执行器"""
        self._running = True
        logger.info(f"Starting task scheduler, check interval: {self._check_interval}s")
        
        while self._running:
            try:
                # 遍历所有任务类型
                for task_type in self._task_handlers.keys():
                    await self._process_task_type(task_type)
                
                # 处理周期性任务
                await self._process_periodic_tasks()
                
            except Exception as e:
                logger.exception(f"Error in task scheduler: {e}")
            
            await asyncio.sleep(self._check_interval)
    
    async def _process_task_type(self, task_type: str):
        """处理指定类型的任务"""
        task = self.get_pending_task(task_type)
        if not task:
            return
        
        handler = self._task_handlers.get(task_type)
        if not handler:
            logger.warning(f"No handler for task type: {task_type}")
            return
        
        try:
            logger.info(f"Executing {task_type} for {task['user_key']}")
            
            # 在线程池中执行（避免阻塞事件循环）
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                handler,
                task["user_id"],
                task.get("device_id"),
                task.get("agent_id")
            )
            
            self.complete_task(
                task_type,
                task["user_key"],
                task["lock_token"],
                success
            )
        except Exception as e:
            logger.exception(f"Task {task_type} failed for {task['user_key']}: {e}")
            self.complete_task(
                task_type,
                task["user_key"],
                task["lock_token"],
                False
            )
    
    async def _process_periodic_tasks(self):
        """处理周期性任务（全局任务，不区分用户）"""
        tasks_config = self._config.get("tasks", {})
        
        for task_type, task_config in tasks_config.items():
            if not task_config.get("enabled", False):
                continue
            if task_config.get("trigger_mode") != "periodic":
                continue
            
            periodic_key = f"{self.PERIODIC_PREFIX}{task_type}"
            lock_key = f"{self.LOCK_PREFIX}{task_type}:global"
            
            # 检查是否到执行时间
            periodic_state = self._redis.hgetall(periodic_key)
            now = int(time.time())
            
            next_run = int(periodic_state.get("next_run", 0)) if periodic_state else 0
            if now < next_run:
                continue
            
            # 尝试获取锁
            timeout = int(task_config.get("timeout", 300))
            lock_token = self._redis.acquire_lock(lock_key, timeout=timeout, blocking=False)
            if not lock_token:
                continue
            
            try:
                # 更新状态
                interval = int(task_config.get("interval", 3600))
                self._redis.hset_all(periodic_key, {
                    "last_run": str(now),
                    "next_run": str(now + interval),
                    "status": "running"
                })
                
                # 执行任务
                handler = self._task_handlers.get(task_type)
                if handler:
                    logger.info(f"Executing periodic task: {task_type}")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler, None, None, None)
                
                self._redis.hset(periodic_key, "status", "idle")
            finally:
                self._redis.release_lock(lock_key, lock_token)
    
    def stop(self):
        """停止调度器"""
        self._running = False
        logger.info("Task scheduler stopped")
```

### 4.3 任务处理器注册

```python
# opencontext/managers/task_handlers.py

from opencontext.context_processing.merger.context_merger import ContextMerger


def create_compression_handler(merger: ContextMerger):
    """创建记忆压缩任务处理器"""
    def handler(user_id: str, device_id: str, agent_id: str) -> bool:
        try:
            merger.periodic_memory_compression_for_user(
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
                interval_seconds=1800
            )
            return True
        except Exception as e:
            logger.exception(f"Compression failed: {e}")
            return False
    return handler


def create_cleanup_handler(storage):
    """创建数据清理任务处理器"""
    def handler(user_id: str, device_id: str, agent_id: str) -> bool:
        try:
            # 全局清理任务，user_id 等参数为 None
            storage.cleanup_expired_data()
            return True
        except Exception as e:
            logger.exception(f"Cleanup failed: {e}")
            return False
    return handler
```

## 5. 代码修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `opencontext/managers/task_scheduler.py` | **新增** | 通用任务调度器 |
| `opencontext/managers/user_key_builder.py` | **新增** | 用户 Key 构建器 |
| `opencontext/managers/task_handlers.py` | **新增** | 任务处理器集合 |
| `opencontext/managers/__init__.py` | **修改** | 导出新模块 |
| `opencontext/managers/processor_manager.py` | **修改** | 移除本地定时器相关代码 |
| `opencontext/context_processing/merger/context_merger.py` | **修改** | 添加用户级压缩方法 |
| `opencontext/server/routes/push.py` | **修改** | 在捕获接口中触发任务调度 |
| `opencontext/server/component_initializer.py` | **修改** | 初始化调度器和注册处理器 |
| `opencontext/server/opencontext.py` | **修改** | 启动调度器后台任务 |
| `config/config_mysql.yaml` | **修改** | 添加调度器配置 |

## 6. 使用示例

### 6.1 在捕获接口中触发任务

```python
# opencontext/server/routes/push.py

from opencontext.managers.task_scheduler import get_scheduler

@router.post("/api/push/chat/message")
async def push_chat_message(request: ChatMessageRequest):
    # ... 现有逻辑 ...
    
    # 触发压缩任务调度
    scheduler = get_scheduler()
    if scheduler:
        scheduler.schedule_user_task(
            task_type="memory_compression",
            user_id=request.user_id or "default",
            device_id=request.device_id or "default",
            agent_id=request.agent_id or "default"
        )
    
    return {"status": "success", ...}
```

### 6.2 注册自定义任务处理器

```python
# 在服务初始化时
scheduler = get_scheduler()

# 注册压缩处理器
scheduler.register_handler(
    "memory_compression",
    create_compression_handler(merger)
)

# 注册清理处理器
scheduler.register_handler(
    "data_cleanup",
    create_cleanup_handler(storage)
)

# 启动调度器
asyncio.create_task(scheduler.start())
```

### 6.3 切换 Key 维度

```yaml
# 使用 3 个 key（默认）
scheduler:
  user_key_config:
    use_user_id: true
    use_device_id: true
    use_agent_id: true

# 切换到 2 个 key
scheduler:
  user_key_config:
    use_user_id: true
    use_device_id: true
    use_agent_id: false  # 禁用 agent_id
```

## 7. 实现优先级

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | `user_key_builder.py` | 用户 Key 构建器 |
| **P0** | `task_scheduler.py` | 通用任务调度器 |
| **P0** | `task_handlers.py` | 任务处理器 |
| **P1** | 修改 `context_merger.py` | 添加用户级压缩方法 |
| **P1** | 修改 `push.py` | 在捕获接口中触发调度 |
| **P1** | 修改 `component_initializer.py` | 初始化和注册 |
| **P2** | 修改 `processor_manager.py` | 移除旧代码 |
| **P2** | 更新配置文件 | 添加调度器配置 |

## 8. 扩展性

### 8.1 添加新任务类型

1. 在配置文件中添加任务类型配置
2. 创建任务处理器函数
3. 在服务初始化时注册处理器

```python
# 1. 配置
tasks:
  my_new_task:
    enabled: true
    trigger_mode: "user_activity"
    interval: 600

# 2. 处理器
def my_task_handler(user_id, device_id, agent_id):
    # 执行任务逻辑
    return True

# 3. 注册
scheduler.register_handler("my_new_task", my_task_handler)
```

### 8.2 未来扩展

- **Cron 表达式支持**：添加 `cron` 触发模式
- **任务优先级**：支持任务优先级队列
- **任务依赖**：支持任务间依赖关系
- **任务重试策略**：可配置的重试策略
- **任务监控**：任务执行统计和告警
