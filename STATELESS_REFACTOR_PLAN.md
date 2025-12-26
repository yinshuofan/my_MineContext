# 状态外置重构方案

## 目标

将 `context_capture` 和 `context_processing` 模块中的所有本地状态外置到 Redis/MySQL，使服务完全无状态化，支持容器化部署和水平扩展。

---

## 一、context_capture 模块状态分析

### 1.1 BaseCaptureComponent (base.py)

| 状态变量 | 类型 | 用途 | 处理方案 |
|----------|------|------|----------|
| `_running` | bool | 组件运行状态 | **删除** - 服务模式下不需要 |
| `_capture_thread` | Thread | 自动捕获线程 | **删除** - 服务模式下禁用自动捕获 |
| `_stop_event` | Event | 停止信号 | **删除** - 同上 |
| `_callback` | Callable | 数据回调 | **保留** - 无状态，每次请求设置 |
| `_capture_interval` | float | 捕获间隔 | **删除** - 服务模式下不需要 |
| `_last_capture_time` | datetime | 最后捕获时间 | **删除** - 统计信息不需要持久化 |
| `_capture_count` | int | 捕获计数 | **可选** - 如需持久化可存 Redis |
| `_error_count` | int | 错误计数 | **可选** - 同上 |
| `_last_error` | str | 最后错误 | **可选** - 同上 |
| `_lock` | RLock | 线程锁 | **删除** - 无状态后不需要本地锁 |

**建议**：服务模式下，`BaseCaptureComponent` 的自动捕获功能应完全禁用，仅保留 `push_*` 方法供 API 调用。

---

### 1.2 TextChatCapture (text_chat.py)

| 状态变量 | 类型 | 用途 | 当前状态 | 处理方案 |
|----------|------|------|----------|----------|
| `_redis_cache` | RedisCache | Redis 缓存客户端 | ✅ 已外置 | 保留 |
| `_local_buffers` | Dict | 本地缓冲区（降级用） | ⚠️ 本地状态 | **删除或改为仅日志** |
| `_buffer_size` | int | 缓冲区大小 | 配置项 | 保留 |
| `_buffer_ttl` | int | 缓冲区 TTL | 配置项 | 保留 |
| `_flush_interval` | int | 刷新间隔 | 配置项 | 保留 |
| `_storage` | Storage | 存储后端 | 无状态 | 保留 |

**建议**：
- `_local_buffers` 在 Redis 不可用时作为降级方案，但会导致状态不一致
- 建议：Redis 不可用时直接报错，不使用本地降级

---

### 1.3 ScreenshotCapture (screenshot.py)

| 状态变量 | 类型 | 用途 | 处理方案 |
|----------|------|------|----------|
| `_screenshot_tool` | ScreenshotTool | 截图工具 | **删除** - 服务模式下不本地截图 |
| `_last_hash` | str | 上次截图 hash | **删除** - 去重逻辑移到 Processor |
| `_consecutive_similar` | int | 连续相似计数 | **删除** - 同上 |
| `_storage` | Storage | 存储后端 | 无状态，保留 |

**建议**：服务模式下完全禁用 `ScreenshotCapture`，截图通过 Push API 接收。

---

### 1.4 FolderMonitorCapture (folder_monitor.py)

| 状态变量 | 类型 | 用途 | 处理方案 |
|----------|------|------|----------|
| `_file_info_cache` | Dict | 文件信息缓存 | **外置到 Redis** 或 **删除** |
| `_document_events` | List | 文档事件队列 | **删除** - 服务模式下不监控本地文件 |
| `_monitor_thread` | Thread | 监控线程 | **删除** |
| `_stop_event` | Event | 停止信号 | **删除** |
| `_processed_vault_ids` | Set | 已处理 ID | **外置到 Redis** 或 **删除** |

**建议**：服务模式下完全禁用 `FolderMonitorCapture`，文档通过 Push API 接收。

---

### 1.5 VaultDocumentMonitor (vault_document_monitor.py)

| 状态变量 | 类型 | 用途 | 处理方案 |
|----------|------|------|----------|
| `_processed_vault_ids` | Set | 已处理的 Vault ID | **外置到 Redis** |
| `_document_events` | List | 文档事件队列 | **删除** |
| `_monitor_thread` | Thread | 监控线程 | **删除** |
| `_last_scan_time` | datetime | 最后扫描时间 | **外置到 Redis** 或 **删除** |

**建议**：如果需要保留 Vault 监控功能，`_processed_vault_ids` 需要外置到 Redis。

---

### 1.6 WebLinkCapture (web_link_capture.py)

| 状态变量 | 类型 | 用途 | 处理方案 |
|----------|------|------|----------|
| `_output_dir` | Path | 输出目录 | 配置项，保留 |
| `_mode` | str | 捕获模式 | 配置项，保留 |
| `_timeout` | int | 超时时间 | 配置项，保留 |
| `_browser` | Browser | 浏览器实例 | **无状态** - 每次请求创建 |

**建议**：基本无状态，可保留。

---

## 二、context_processing 模块状态分析

### 2.1 BaseContextProcessor (base_processor.py)

| 状态变量 | 类型 | 用途 | 处理方案 |
|----------|------|------|----------|
| `_is_initialized` | bool | 初始化状态 | **保留** - 实例级状态 |
| `_callback` | Callable | 处理回调 | **保留** - 无状态 |
| `_processing_stats` | Dict | 处理统计 | **可选外置** - 如需跨实例统计 |

**建议**：基本无状态，可保留。

---

### 2.2 ScreenshotProcessor (screenshot_processor.py) ⚠️ 重点

| 状态变量 | 类型 | 用途 | 处理方案 |
|----------|------|------|----------|
| `_processed_cache` | Dict | 已处理上下文缓存（用于合并） | **外置到 Redis Hash** |
| `_current_screenshot` | deque | 最近截图 phash（用于去重） | **外置到 Redis List** |
| `_input_queue` | Queue | 本地处理队列 | **删除** - 改为同步处理 |
| `_processing_task` | Thread | 后台处理线程 | **删除** - 改为同步处理 |
| `_stop_event` | Event | 停止信号 | **删除** |

**详细方案**：

```python
# Redis Key 设计
SCREENSHOT_PHASH_KEY = "screenshot:phash:{user_id}:{device_id}"  # List, 存储最近的 phash
PROCESSED_CACHE_KEY = "processed_cache:{context_type}:{user_id}:{device_id}"  # Hash, 存储已处理上下文

# 去重检查
def _check_duplicate_redis(self, phash: str, user_id: str, device_id: str) -> bool:
    key = f"screenshot:phash:{user_id}:{device_id}"
    recent_hashes = self._redis.lrange(key, 0, 100)
    for h in recent_hashes:
        if hamming_distance(phash, h) <= self._similarity_hash_threshold:
            return True
    # 添加新 hash
    self._redis.lpush(key, phash)
    self._redis.ltrim(key, 0, 100)
    self._redis.expire(key, 3600)  # 1小时过期
    return False

# 获取缓存的上下文
def _get_cached_contexts_redis(self, context_type: str, user_id: str, device_id: str) -> Dict:
    key = f"processed_cache:{context_type}:{user_id}:{device_id}"
    return self._redis.hgetall_json(key)

# 设置缓存的上下文
def _set_cached_contexts_redis(self, context_type: str, user_id: str, device_id: str, contexts: Dict):
    key = f"processed_cache:{context_type}:{user_id}:{device_id}"
    self._redis.hset_json(key, contexts)
    self._redis.expire(key, 3600)
```

---

### 2.3 DocumentProcessor (document_processor.py) ⚠️ 重点

| 状态变量 | 类型 | 用途 | 处理方案 |
|----------|------|------|----------|
| `_input_queue` | Queue | 本地处理队列 | **删除** - 改为同步处理 |
| `_processing_task` | Thread | 后台处理线程 | **删除** - 改为同步处理 |
| `_stop_event` | Event | 停止信号 | **删除** |
| `_document_converter` | DocumentConverter | 文档转换器 | **保留** - 无状态工具类 |
| `_structured_chunker` | Chunker | 分块器 | **保留** - 无状态工具类 |

**建议**：移除本地队列和后台线程，改为同步处理。

---

### 2.4 TextChatProcessor (text_chat_processor.py)

| 状态变量 | 类型 | 用途 | 处理方案 |
|----------|------|------|----------|
| `_callback` | Callable | 处理回调 | **保留** - 无状态 |

**建议**：已基本无状态，无需修改。

---

### 2.5 EntityProcessor (entity_processor.py)

无本地状态，已是无状态设计。

---

## 三、修改清单

### 3.1 需要修改的文件

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `context_processing/processor/screenshot_processor.py` | 状态外置到 Redis，移除本地队列 | **P0** |
| `context_processing/processor/document_processor.py` | 移除本地队列，改为同步处理 | **P0** |
| `context_capture/text_chat.py` | 移除 `_local_buffers` 降级逻辑 | **P1** |
| `context_capture/base.py` | 添加服务模式开关，禁用自动捕获 | **P1** |
| `context_capture/vault_document_monitor.py` | `_processed_vault_ids` 外置到 Redis | **P2** |
| `storage/redis_cache.py` | 添加 Hash 和 List 操作方法 | **P0** |

### 3.2 新增文件

| 文件 | 内容 |
|------|------|
| `config/config_stateless.yaml` | 无状态服务模式配置模板 |

---

## 四、Redis Key 设计规范

```
# 截图去重 (List)
screenshot:phash:{user_id}:{device_id}
  - 存储最近 100 个 phash
  - TTL: 1 小时

# 已处理上下文缓存 (Hash)
processed_cache:{context_type}:{user_id}:{device_id}
  - field: context_id
  - value: ProcessedContext JSON
  - TTL: 1 小时

# 已处理 Vault ID (Set)
vault:processed:{user_id}
  - 存储已处理的 vault_id
  - TTL: 24 小时

# 分布式锁
lock:screenshot:{user_id}:{device_id}
lock:chat:{user_id}:{device_id}
```

---

## 五、处理模式变更

### 5.1 当前模式（有状态）

```
API Request → 入队列 → 后台线程消费 → 处理 → 存储
                ↓
           本地缓存
```

### 5.2 目标模式（无状态）

```
API Request → 同步处理 → 存储
                ↓
           Redis 缓存
```

### 5.3 处理流程对比

**ScreenshotProcessor 当前流程**：
1. `process()` 将任务放入 `_input_queue`
2. `_processing_task` 线程从队列取出任务
3. 检查 `_current_screenshot` 去重
4. 调用 VLM 分析
5. 与 `_processed_cache` 合并
6. 存储结果

**ScreenshotProcessor 目标流程**：
1. `process()` 同步执行
2. 从 Redis 获取最近 phash 列表，检查去重
3. 调用 VLM 分析
4. 从 Redis 获取已处理缓存，合并
5. 更新 Redis 缓存
6. 存储结果

---

## 六、实现步骤

### Phase 1: 基础设施准备
1. 扩展 `redis_cache.py`，添加 Hash/List/Set 操作
2. 添加 Redis Key 前缀和 TTL 配置

### Phase 2: ScreenshotProcessor 无状态化
1. 移除 `_input_queue` 和 `_processing_task`
2. `_current_screenshot` → Redis List
3. `_processed_cache` → Redis Hash
4. 添加分布式锁保护并发处理

### Phase 3: DocumentProcessor 无状态化
1. 移除 `_input_queue` 和 `_processing_task`
2. 改为同步处理

### Phase 4: context_capture 清理
1. 添加服务模式配置开关
2. 服务模式下禁用自动捕获组件
3. 清理 `TextChatCapture` 的本地降级逻辑

### Phase 5: 测试和文档
1. 单元测试
2. 集成测试
3. 更新部署文档

---

## 七、配置示例

```yaml
# config/config_stateless.yaml
server:
  host: "0.0.0.0"
  port: 1733
  mode: "stateless"  # 无状态服务模式

redis:
  host: "${REDIS_HOST:localhost}"
  port: ${REDIS_PORT:6379}
  db: 0
  password: "${REDIS_PASSWORD:}"
  key_prefix: "minecontext:"
  default_ttl: 3600

# 禁用所有自动捕获
capture:
  screenshot:
    enabled: false
  folder_monitor:
    enabled: false
  vault_monitor:
    enabled: false
  text_chat:
    enabled: true
    use_redis: true
    local_fallback: false  # 禁用本地降级

# 处理器配置
processing:
  screenshot_processor:
    use_redis_cache: true
    sync_mode: true  # 同步处理模式
  document_processor:
    sync_mode: true
```

---

## 八、风险和注意事项

1. **Redis 可用性**：Redis 成为单点依赖，需要配置高可用（Sentinel/Cluster）
2. **性能影响**：Redis 网络调用会增加延迟，但可接受
3. **数据一致性**：使用分布式锁保护关键操作
4. **迁移兼容**：需要处理旧数据迁移

---

## 九、预期收益

1. **水平扩展**：任意数量的服务实例
2. **容器化友好**：无本地状态，随时启停
3. **高可用**：单实例故障不影响服务
4. **运维简化**：无需关心实例亲和性
