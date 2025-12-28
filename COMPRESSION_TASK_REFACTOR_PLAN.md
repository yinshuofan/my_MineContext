# 周期性压缩记忆功能无状态化修改方案

## 1. 当前实现分析

### 1.1 现有架构

```
┌─────────────────────────────────────────────────────────┐
│                  processor_manager.py                    │
├─────────────────────────────────────────────────────────┤
│  _compression_timer: Timer      # 本地定时器            │
│  _compression_interval: int     # 压缩间隔              │
│                                                         │
│  start_periodic_compression()   # 启动定时器            │
│  _run_periodic_compression()    # 执行压缩 + 重新调度   │
│  stop_periodic_compression()    # 停止定时器            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  context_merger.py                       │
├─────────────────────────────────────────────────────────┤
│  periodic_memory_compression(interval_seconds)          │
│  - 获取时间窗口内未压缩的上下文                          │
│  - 按相似度分组                                          │
│  - 合并同组内的上下文                                    │
└─────────────────────────────────────────────────────────┘
```

### 1.2 现有问题

| 问题 | 说明 |
|------|------|
| **本地定时器** | `threading.Timer` 是本地状态，多实例部署时每个实例都会执行压缩 |
| **全局压缩** | 当前压缩是全局的，不区分用户 |
| **无触发机制** | 压缩任务固定周期执行，不考虑用户活动 |
| **阻塞线程** | `periodic_memory_compression` 是同步方法，可能阻塞 |
| **重复执行** | 多实例部署时，同一批数据可能被多个实例重复处理 |

## 2. 需求分析

### 2.1 核心需求

1. **配置化周期**：在配置文件中定义压缩任务的时间周期
2. **用户级任务**：每个用户独立的压缩任务
3. **触发机制**：用户调用信息捕获接口时，为该用户创建压缩任务
4. **去重机制**：如果用户已有任务存在，不重复创建
5. **非阻塞**：不阻塞主线程
6. **无状态**：支持多实例部署

### 2.2 设计约束

- 必须使用 Redis 作为状态存储
- 必须支持多实例部署，任务不重复执行
- 必须支持用户级别的任务隔离
- 必须非阻塞，不影响 API 响应

## 3. 修改方案

### 3.1 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                         多实例服务                                   │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│  实例 1     │  实例 2     │  实例 3     │    ...      │  实例 N     │
│  (无状态)   │  (无状态)   │  (无状态)   │             │  (无状态)   │
└──────┬──────┴──────┬──────┴──────┬──────┴─────────────┴──────┬──────┘
       │             │             │                           │
       └─────────────┴──────┬──────┴───────────────────────────┘
                            │
                     ┌──────┴──────┐
                     │    Redis    │
                     ├─────────────┤
                     │ 任务调度状态 │
                     │ 用户任务队列 │
                     │ 分布式锁     │
                     └─────────────┘
```

### 3.2 核心组件

#### 3.2.1 CompressionTaskScheduler（新增）

负责管理用户级压缩任务的调度：

```python
class CompressionTaskScheduler:
    """
    用户级压缩任务调度器（无状态）
    
    所有状态存储在 Redis 中：
    - compression:task:{user_id}:{device_id} - 用户任务状态
    - compression:queue - 待执行任务队列
    - compression:lock:{user_id}:{device_id} - 分布式锁
    """
    
    # Redis Key 前缀
    TASK_KEY_PREFIX = "compression:task:"      # 任务状态
    QUEUE_KEY = "compression:queue"            # 任务队列
    LOCK_KEY_PREFIX = "compression:lock:"      # 分布式锁
    LAST_EXEC_KEY_PREFIX = "compression:last:" # 上次执行时间
    
    def __init__(self, redis_cache: RedisCache, config: dict):
        self._redis = redis_cache
        self._interval = config.get("interval", 1800)  # 默认30分钟
        self._task_ttl = config.get("task_ttl", 7200)  # 任务状态 TTL
        
    def schedule_user_compression(
        self,
        user_id: str,
        device_id: str = "default"
    ) -> bool:
        """
        为用户调度压缩任务
        
        逻辑：
        1. 检查用户是否已有待执行任务
        2. 如果没有，创建新任务并加入队列
        3. 如果有，检查是否需要更新（延长 TTL）
        
        Returns:
            True 如果创建了新任务，False 如果任务已存在
        """
        
    def get_pending_task(self) -> Optional[dict]:
        """
        获取一个待执行的任务（带分布式锁）
        
        逻辑：
        1. 从队列中获取到期的任务
        2. 尝试获取该任务的分布式锁
        3. 如果获取成功，返回任务信息
        4. 如果失败（被其他实例处理），继续获取下一个
        """
        
    def complete_task(self, user_id: str, device_id: str, success: bool):
        """
        标记任务完成
        
        逻辑：
        1. 释放分布式锁
        2. 更新任务状态
        3. 记录上次执行时间
        """
        
    def should_execute(self, user_id: str, device_id: str) -> bool:
        """
        检查用户任务是否应该执行
        
        逻辑：
        1. 检查上次执行时间
        2. 如果距离上次执行超过 interval，返回 True
        """
```

#### 3.2.2 Redis 数据结构

```
# 用户任务状态 (Hash)
compression:task:{user_id}:{device_id}
{
    "status": "pending|running|completed",
    "created_at": "2025-01-01T00:00:00",
    "scheduled_at": "2025-01-01T00:30:00",  # 计划执行时间
    "last_activity": "2025-01-01T00:25:00"  # 最后活动时间
}
TTL: 7200 (2小时)

# 待执行任务队列 (Sorted Set)
compression:queue
Score: scheduled_at (Unix timestamp)
Member: "{user_id}:{device_id}"

# 上次执行时间 (String)
compression:last:{user_id}:{device_id}
Value: Unix timestamp
TTL: 86400 (24小时)

# 分布式锁 (String)
compression:lock:{user_id}:{device_id}
Value: lock_token
TTL: 300 (5分钟，防止死锁)
```

#### 3.2.3 任务执行流程

```
用户调用信息捕获接口
        │
        ▼
┌───────────────────────────────────────┐
│ schedule_user_compression(user_id)    │
│                                       │
│ 1. 检查 compression:task:{user_id}    │
│ 2. 如果不存在或已过期：               │
│    - 创建任务状态                     │
│    - 加入 compression:queue           │
│    - scheduled_at = now + interval    │
│ 3. 如果存在：                         │
│    - 更新 last_activity               │
│    - 刷新 TTL                         │
└───────────────────────────────────────┘
        │
        ▼
    返回（非阻塞）


后台任务执行器（每个实例都运行）
        │
        ▼
┌───────────────────────────────────────┐
│ 定期检查 compression:queue            │
│ (每 10 秒检查一次)                    │
│                                       │
│ 1. ZRANGEBYSCORE 获取到期任务         │
│ 2. 尝试获取分布式锁                   │
│ 3. 如果获取成功：                     │
│    - 执行 periodic_memory_compression │
│    - 标记任务完成                     │
│    - 释放锁                           │
│ 4. 如果获取失败：                     │
│    - 跳过（其他实例在处理）           │
└───────────────────────────────────────┘
```

### 3.3 配置文件修改

```yaml
# config/config_mysql.yaml

processing:
  # ... 其他配置 ...
  
  # 记忆压缩配置
  memory_compression:
    enabled: true
    interval: 1800          # 压缩间隔（秒），默认30分钟
    task_ttl: 7200          # 任务状态 TTL（秒），默认2小时
    check_interval: 10      # 任务检查间隔（秒），默认10秒
    lock_timeout: 300       # 分布式锁超时（秒），默认5分钟
    batch_size: 100         # 每次压缩处理的上下文数量
```

### 3.4 代码修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `opencontext/managers/compression_scheduler.py` | **新增** | 压缩任务调度器 |
| `opencontext/managers/processor_manager.py` | **修改** | 移除本地定时器，集成调度器 |
| `opencontext/context_processing/merger/context_merger.py` | **修改** | 添加用户级压缩方法 |
| `opencontext/server/routes/push.py` | **修改** | 在捕获接口中触发任务调度 |
| `opencontext/server/component_initializer.py` | **修改** | 初始化调度器 |
| `config/config_mysql.yaml` | **修改** | 添加压缩配置 |

### 3.5 详细修改

#### 3.5.1 新增 compression_scheduler.py

```python
# opencontext/managers/compression_scheduler.py

import asyncio
import time
from typing import Optional, Dict, Any
from opencontext.storage.redis_cache import RedisCache
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CompressionTaskScheduler:
    """
    用户级压缩任务调度器（无状态版本）
    
    所有状态存储在 Redis 中，支持多实例部署。
    """
    
    TASK_KEY_PREFIX = "compression:task:"
    QUEUE_KEY = "compression:queue"
    LOCK_KEY_PREFIX = "compression:lock:"
    LAST_EXEC_KEY_PREFIX = "compression:last:"
    
    def __init__(self, redis_cache: RedisCache, config: Dict[str, Any] = None):
        self._redis = redis_cache
        config = config or {}
        self._interval = config.get("interval", 1800)
        self._task_ttl = config.get("task_ttl", 7200)
        self._check_interval = config.get("check_interval", 10)
        self._lock_timeout = config.get("lock_timeout", 300)
        self._running = False
        self._check_task = None
        
    def _make_task_key(self, user_id: str, device_id: str) -> str:
        return f"{self.TASK_KEY_PREFIX}{user_id}:{device_id}"
    
    def _make_lock_key(self, user_id: str, device_id: str) -> str:
        return f"{self.LOCK_KEY_PREFIX}{user_id}:{device_id}"
    
    def _make_last_exec_key(self, user_id: str, device_id: str) -> str:
        return f"{self.LAST_EXEC_KEY_PREFIX}{user_id}:{device_id}"
    
    def _make_queue_member(self, user_id: str, device_id: str) -> str:
        return f"{user_id}:{device_id}"
    
    def schedule_user_compression(
        self,
        user_id: str,
        device_id: str = "default"
    ) -> bool:
        """
        为用户调度压缩任务。
        
        如果用户已有待执行任务，则不重复创建，只更新活动时间。
        
        Returns:
            True 如果创建了新任务，False 如果任务已存在
        """
        task_key = self._make_task_key(user_id, device_id)
        queue_member = self._make_queue_member(user_id, device_id)
        
        # 检查是否已有任务
        existing = self._redis.hgetall(task_key)
        if existing and existing.get("status") in ("pending", "running"):
            # 任务已存在，更新活动时间并刷新 TTL
            self._redis.hset(task_key, "last_activity", str(int(time.time())))
            self._redis.expire(task_key, self._task_ttl)
            logger.debug(f"Compression task already exists for {user_id}:{device_id}")
            return False
        
        # 检查上次执行时间
        last_exec_key = self._make_last_exec_key(user_id, device_id)
        last_exec = self._redis.get(last_exec_key)
        if last_exec:
            last_exec_time = int(last_exec)
            if time.time() - last_exec_time < self._interval:
                # 距离上次执行不足一个周期，不创建任务
                logger.debug(
                    f"Skipping compression for {user_id}:{device_id}, "
                    f"last executed {int(time.time() - last_exec_time)}s ago"
                )
                return False
        
        # 创建新任务
        now = int(time.time())
        scheduled_at = now + self._interval
        
        task_data = {
            "status": "pending",
            "created_at": str(now),
            "scheduled_at": str(scheduled_at),
            "last_activity": str(now),
            "user_id": user_id,
            "device_id": device_id,
        }
        
        # 写入任务状态
        self._redis.hset_all(task_key, task_data)
        self._redis.expire(task_key, self._task_ttl)
        
        # 加入任务队列
        self._redis.zadd(self.QUEUE_KEY, {queue_member: scheduled_at})
        
        logger.info(
            f"Scheduled compression task for {user_id}:{device_id}, "
            f"will execute at {scheduled_at}"
        )
        return True
    
    def get_pending_task(self) -> Optional[Dict[str, Any]]:
        """
        获取一个待执行的任务（带分布式锁）。
        
        Returns:
            任务信息字典，如果没有待执行任务则返回 None
        """
        now = int(time.time())
        
        # 获取所有到期的任务
        tasks = self._redis.zrangebyscore(self.QUEUE_KEY, 0, now)
        
        for queue_member in tasks:
            parts = queue_member.split(":", 1)
            if len(parts) != 2:
                continue
            user_id, device_id = parts
            
            # 尝试获取分布式锁
            lock_key = self._make_lock_key(user_id, device_id)
            lock_token = self._redis.acquire_lock(
                lock_key,
                timeout=self._lock_timeout,
                blocking=False
            )
            
            if not lock_token:
                # 其他实例正在处理，跳过
                continue
            
            # 更新任务状态
            task_key = self._make_task_key(user_id, device_id)
            self._redis.hset(task_key, "status", "running")
            
            # 从队列中移除
            self._redis.zrem(self.QUEUE_KEY, queue_member)
            
            return {
                "user_id": user_id,
                "device_id": device_id,
                "lock_token": lock_token,
            }
        
        return None
    
    def complete_task(
        self,
        user_id: str,
        device_id: str,
        lock_token: str,
        success: bool = True
    ):
        """标记任务完成并释放锁。"""
        task_key = self._make_task_key(user_id, device_id)
        lock_key = self._make_lock_key(user_id, device_id)
        last_exec_key = self._make_last_exec_key(user_id, device_id)
        
        # 更新任务状态
        self._redis.hset(task_key, "status", "completed" if success else "failed")
        
        # 记录执行时间
        if success:
            self._redis.set(last_exec_key, str(int(time.time())))
            self._redis.expire(last_exec_key, 86400)  # 24小时
        
        # 释放锁
        self._redis.release_lock(lock_key, lock_token)
        
        logger.info(
            f"Compression task for {user_id}:{device_id} "
            f"{'completed' if success else 'failed'}"
        )
    
    async def start_background_executor(self, merger):
        """
        启动后台任务执行器。
        
        Args:
            merger: ContextMerger 实例，用于执行实际的压缩操作
        """
        self._running = True
        logger.info(
            f"Starting compression task executor, "
            f"check interval: {self._check_interval}s"
        )
        
        while self._running:
            try:
                task = self.get_pending_task()
                if task:
                    user_id = task["user_id"]
                    device_id = task["device_id"]
                    lock_token = task["lock_token"]
                    
                    try:
                        logger.info(
                            f"Executing compression for {user_id}:{device_id}"
                        )
                        # 执行压缩（在线程池中运行以避免阻塞）
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            merger.periodic_memory_compression_for_user,
                            user_id,
                            device_id,
                            self._interval
                        )
                        self.complete_task(user_id, device_id, lock_token, True)
                    except Exception as e:
                        logger.exception(
                            f"Compression failed for {user_id}:{device_id}: {e}"
                        )
                        self.complete_task(user_id, device_id, lock_token, False)
                
            except Exception as e:
                logger.exception(f"Error in compression executor: {e}")
            
            await asyncio.sleep(self._check_interval)
    
    def stop(self):
        """停止后台执行器。"""
        self._running = False
        logger.info("Compression task executor stopped")
```

#### 3.5.2 修改 context_merger.py

添加用户级压缩方法：

```python
def periodic_memory_compression_for_user(
    self,
    user_id: str,
    device_id: str,
    interval_seconds: int
):
    """
    对指定用户的上下文进行记忆压缩。
    
    与全局压缩类似，但只处理指定用户的数据。
    """
    if interval_seconds <= 0:
        logger.warning("interval_seconds must be greater than 0.")
        return
    
    logger.info(f"Starting memory compression for user {user_id}:{device_id}...")
    
    try:
        filter = {
            "user_id": user_id,
            "device_id": device_id,
            "update_time_ts": {
                "$gte": int(
                    (
                        datetime.datetime.now()
                        - timedelta(seconds=interval_seconds)
                        - timedelta(minutes=5)
                    ).timestamp()
                ),
                "$lte": int((datetime.datetime.now() - timedelta(minutes=5)).timestamp()),
            },
            "has_compression": False,
            "enable_merge": True,
        }
        
        # ... 其余逻辑与 periodic_memory_compression 相同 ...
        
    except Exception as e:
        logger.exception(f"Error during user memory compression: {e}")
```

#### 3.5.3 修改 push.py

在信息捕获接口中触发任务调度：

```python
# 在 push_chat_message, push_screenshot 等接口中添加

from opencontext.managers.compression_scheduler import get_compression_scheduler

@router.post("/api/push/chat/message")
async def push_chat_message(request: ChatMessageRequest):
    # ... 现有逻辑 ...
    
    # 触发压缩任务调度
    scheduler = get_compression_scheduler()
    if scheduler:
        scheduler.schedule_user_compression(
            user_id=request.user_id or "default",
            device_id=request.device_id or "default"
        )
    
    return {"status": "success", ...}
```

#### 3.5.4 修改 processor_manager.py

移除本地定时器相关代码：

```python
class ContextProcessorManager:
    def __init__(self, max_workers: int = 5):
        # 移除以下行：
        # self._compression_timer: Optional[Timer] = None
        # self._compression_interval: int = 1800
        
        # ... 其余初始化代码 ...
    
    # 移除以下方法：
    # def start_periodic_compression(self)
    # def _run_periodic_compression(self)
    # def stop_periodic_compression(self)
```

## 4. 实现优先级

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | 新增 `compression_scheduler.py` | 核心调度器实现 |
| **P0** | 修改 `context_merger.py` | 添加用户级压缩方法 |
| **P0** | 修改 `push.py` | 在捕获接口中触发调度 |
| **P1** | 修改 `processor_manager.py` | 移除本地定时器 |
| **P1** | 修改 `component_initializer.py` | 初始化调度器 |
| **P1** | 更新配置文件 | 添加压缩配置 |

## 5. 测试要点

1. **单实例测试**
   - 用户调用捕获接口后，任务是否正确创建
   - 任务是否在指定时间后执行
   - 重复调用是否不会创建重复任务

2. **多实例测试**
   - 多个实例是否只有一个执行任务
   - 分布式锁是否正常工作
   - 任务完成后锁是否正确释放

3. **异常测试**
   - 压缩失败后任务状态是否正确
   - 锁超时后是否自动释放
   - Redis 连接断开后的行为

## 6. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Redis 不可用 | 任务调度失败 | 添加健康检查，降级到本地模式（可选） |
| 锁死锁 | 任务永不执行 | 设置锁超时，定期清理过期锁 |
| 任务积压 | 内存压力 | 限制队列大小，添加监控告警 |
| 压缩耗时过长 | 阻塞其他任务 | 设置压缩超时，分批处理 |
