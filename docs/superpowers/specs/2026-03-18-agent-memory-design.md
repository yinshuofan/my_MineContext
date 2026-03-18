# Agent Memory — Design Spec

## Overview

为 MineContext 新增 Agent Memory 功能，让 AI agent（如小说角色）拥有独立的记忆体系。Agent 记忆分为两层：**基础记忆**（角色设定/背景事件，通过 API 直推）和**交互记忆**（与用户对话产生的记忆，通过 processor 自动提取）。

本次 scope 仅覆盖 **agent memory 核心服务**（push/search/cache）。小说自动提取子系统为独立后续项目。

> **文件路径约定**：本文档中所有文件路径均相对于 `opencontext/` 包根目录，如 `models/enums.py` 实际为 `opencontext/models/enums.py`。

## Design Decisions

以下为 brainstorming 过程中确认的核心决策：

1. **两层记忆模型**：base（角色设定+背景事件）+ interaction（对话产生的记忆）
2. **用 ContextType 区分**：新增 `AGENT_EVENT` 等类型，不在 vector DB 加额外字段
3. **统一 API + 参数化分发**：不为 agent memory 新建一套接口，复用现有 push/search/cache 端点
4. **Push 并行处理**：多个 processor 并行执行，共同引用 `batch_id`，无顺序依赖
5. **删除 buffer 模式**：每次 push = 一次直接处理，批量由调用方控制。**这是 breaking API change**（`process_mode` 和 `flush_immediately` 参数废弃）
6. **持久化原始聊天记录**：新增 `chat_batches` 表，processor 通过 `raw_id=batch_id` 引用
7. **refs 统一引用模型**：新增 `refs` 字段替代 `parent_id`/`children_ids`，key 为 `ContextType.value`
8. **集成现有模块**：不建独立 `agent_memory/` 模块，各组件放到架构中的自然位置
9. **Agent profile**：单一 profile，base 为初始值，交互中 LLM merge 演化（OVERWRITE 策略）
10. **Agent hierarchy**：ContextType 预留，生成逻辑暂不实现

## Scope

### In Scope

- 前置重构：删除 buffer、持久化 chat_batch、ContextType 扩展、refs 统一、profiles 表 owner_type
- Agent 注册管理（CRUD）
- Agent 基础记忆管理（profile + events 直推）
- Agent memory processor（agent 视角的 LLM 提取）
- Push 多 processor 并行分发机制
- Search context_type 参数化
- Memory cache memory_owner 参数化
- Prompt 模板（agent 视角提取，双语）

### Out of Scope

- 小说自动提取子系统
- Agent event hierarchy 生成逻辑
- Knowledge / document 类型的 agent 记忆
- 已有数据迁移（手动脚本处理）

### Breaking Changes

- `process_mode` 参数废弃（buffer 模式删除）
- `flush_immediately` 参数废弃
- `EventNode` API 响应中 `parent_id` 改为 `refs`
- Cache key 格式变更（新增 `memory_owner` 前缀，旧 key 通过 TTL 自然过期）

---

## 1. 前置重构

### 1.1 删除 Buffer 模式

**变更文件**：`context_capture/text_chat.py`、`server/routes/push.py`、`context_capture/MODULE.md`、`server/MODULE.md`

- 废弃 `process_mode` 参数（`"buffer"` / `"direct"`）和 `flush_immediately` 参数
- 删除 `TextChatCapture` 的 Redis buffer 逻辑（push_message、flush_buffer、buffer key、flush lock、`_RPUSH_EXPIRE_LLEN_LUA` 脚本）
- 每次 push 请求 = 一次直接处理
- `TextChatCapture` 简化为无状态的消息格式化 + callback 分发
- 原有的 graceful shutdown `_flush_all_buffers()` 不再需要：消息在处理前已持久化到 `chat_batches`，即使进程中断也不会丢失

**PushChatRequest 模型变更**：移除 `process_mode` 和 `flush_immediately` 字段。新增 `processors` 字段（见 Section 3.1）。

### 1.2 持久化原始聊天记录

**变更文件**：`storage/backends/mysql_backend.py`、`storage/backends/sqlite_backend.py`

新增 `chat_batches` 表：

```sql
CREATE TABLE chat_batches (
    batch_id VARCHAR(36) PRIMARY KEY,
    messages JSON NOT NULL,
    user_id VARCHAR(255),
    device_id VARCHAR(100) DEFAULT 'default',
    agent_id VARCHAR(100) DEFAULT 'default',
    message_count INT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_chat_batches_created_at (created_at)
);
```

> **注意**：`batch_id` 为 VARCHAR PK（UUID），`lastrowid` 返回 0。插入后应 SELECT 确认，或信任应用层生成的 UUID。参见 CLAUDE.md MySQL pitfalls。

**清理策略**：`chat_batches` 保留最近 90 天数据。通过 periodic task 定期清理过期记录（复用现有 scheduler 的 `periodic` 触发模式）。

Push 流程变为：
1. 持久化 messages → 得到 `batch_id`
2. 将 `batch_id` 存入 `RawContextProperties.additional_info["batch_id"]`，传给所有 processor
3. 每个 processor 在输出的 `ProcessedContext` 上设置 `properties.raw_type="chat_batch"`, `properties.raw_id=batch_id`（这两个字段在 `ContextProperties` 上，不在 `RawContextProperties` 上）

### 1.3 ContextType 扩展

**变更文件**：`models/enums.py`、`config/prompts_en.yaml`、`config/prompts_zh.yaml`

新增类型：

```python
class ContextType(str, Enum):
    PROFILE = "profile"
    DOCUMENT = "document"
    EVENT = "event"                                # 用户 L0
    KNOWLEDGE = "knowledge"
    DAILY_SUMMARY = "daily_summary"                # 用户 L1
    WEEKLY_SUMMARY = "weekly_summary"              # 用户 L2
    MONTHLY_SUMMARY = "monthly_summary"            # 用户 L3
    AGENT_EVENT = "agent_event"                    # Agent L0
    AGENT_DAILY_SUMMARY = "agent_daily_summary"    # Agent L1（预留）
    AGENT_WEEKLY_SUMMARY = "agent_weekly_summary"  # Agent L2（预留）
    AGENT_MONTHLY_SUMMARY = "agent_monthly_summary" # Agent L3（预留）
```

**与现有 `hierarchy_level` 的关系**：

现有系统用 `ContextType.EVENT` + `hierarchy_level`（0/1/2/3）来区分层级。新的 ContextType（`DAILY_SUMMARY` 等）将**替代** `hierarchy_level` 作为层级判断的主要依据：

- **写入端**（`hierarchy_summary.py` `_store_summary()`）：生成 L1 摘要时设 `context_type=DAILY_SUMMARY`（不再是 `EVENT`），L2 设 `WEEKLY_SUMMARY`，L3 设 `MONTHLY_SUMMARY`。`hierarchy_level` 字段保留但降级为辅助信息。摘要的幂等性保证仍来自 `_generate_daily_summary` 等方法中的 dedup check（`search_hierarchy`），与 UpdateStrategy 无关。
- **查询端**（`search_hierarchy()`）：改用 `context_type` 过滤而非 `hierarchy_level`。例如查 L1 摘要：`context_types=[DAILY_SUMMARY]`。
- **`hierarchy_levels` 请求参数兼容**：`EventSearchRequest` 的 `hierarchy_levels: List[int]` 参数保留，内部通过 `MEMORY_OWNER_TYPES` 映射为对应的 ContextType 列表。例如 `hierarchy_levels=[0, 1]` + `memory_owner="user"` → `context_types=[EVENT, DAILY_SUMMARY]`。
- **drill-up 遍历**（`_collect_ancestors()`）：通过 `refs` 字段找到父节点 ID → `get_contexts_by_ids()` 获取父节点 → 父节点的 `context_type` 自然标识其层级。不再需要按 `hierarchy_level` 过滤。
- **drill-down 遍历**（`hierarchical_event_tool.py`）：通过 `refs` 字段获取子节点 ID 列表 → `get_contexts_by_ids()`。

**部署顺序要求**：ContextType 扩展（本节）必须在 agent memory processor（Section 3.4）之前部署，否则 `get_context_type_for_analysis()` 会将 `"agent_event"` 回退为 `KNOWLEDGE`。

同步更新：

| 映射 | DAILY/WEEKLY/MONTHLY_SUMMARY | AGENT_EVENT | AGENT_DAILY/WEEKLY/MONTHLY |
|------|------------------------------|-------------|----------------------------|
| `CONTEXT_UPDATE_STRATEGIES` | APPEND | APPEND | APPEND |
| `CONTEXT_STORAGE_BACKENDS` | vector_db | vector_db | vector_db |
| `ContextDescriptions` | 新增描述 | 新增描述 | 新增描述 |
| Prompt 文件 | 更新分类描述 | 更新分类描述 | 更新分类描述 |

### 1.4 refs 统一引用模型

**变更文件**：`models/context.py`、`models/enums.py`

`ContextProperties` 新增：

```python
refs: Dict[str, List[str]] = {}
```

- key = `ContextType.value`（如 `"event"`、`"daily_summary"`、`"agent_event"`）
- value = 关联的 context ID 列表

废弃字段：`parent_id`、`children_ids`（从 `ContextProperties` 移除）

#### refs 具体示例

**L1 daily summary**（包含 3 个 L0 events，属于 1 个 L2 weekly summary）：
```json
{
  "event": ["evt_001", "evt_002", "evt_003"],
  "weekly_summary": ["ws_001"]
}
```
- `refs["event"]`：该摘要包含的子事件（原 `children_ids`）
- `refs["weekly_summary"]`：该摘要的父级（原 `parent_id`）

**L0 event**（属于 1 个 L1 daily summary）：
```json
{
  "daily_summary": ["ds_001"]
}
```
- `refs["daily_summary"]`：该事件的父级摘要

**L2 weekly summary**：
```json
{
  "daily_summary": ["ds_001", "ds_002", "ds_003", "ds_004", "ds_005"],
  "monthly_summary": ["ms_001"]
}
```

**规则**：
- 子节点的 refs key = 父节点的 ContextType.value（指向父级）
- 父节点的 refs key = 子节点的 ContextType.value（指向子级）
- 关系是双向的，父子各自持有对方的引用

#### Vector DB 存储表示

`refs` 在 vector DB payload 中存为 **JSON 字符串**字段：

```python
# 写入
data["refs"] = json.dumps(context.properties.refs)

# 读取
refs = json.loads(doc.get("refs", "{}"))
```

替代现有的 `parent_id`（scalar string）和 `children_ids`（JSON string of list）两个独立字段。

#### batch_set_parent_id 替代方案

现有 `storage.batch_set_parent_id(children_ids, parent_id)` 用于在 hierarchy 生成后回填子节点的 parent。替代为：

```python
async def batch_update_refs(self, context_ids: List[str], ref_key: str, ref_value: str):
    """为一批 context 的 refs 添加一个引用。"""
    # 对每个 context：读取现有 refs JSON → 添加 ref_key: [ref_value] → 写回
```

各 vector DB backend 实现此方法。VikingDB/Qdrant 通过 upsert payload 实现。

#### ProcessedContextModel / EventNode API 变更

**Breaking change**：`EventNode` 和 `ProcessedContextModel` 的 `parent_id` 字段替换为 `refs: Dict[str, List[str]]`。`children_ids` 同样移除。

**连带变更**：
- `ProcessedContextModel`：新增 `refs` 字段，移除 `parent_id`/`children_ids`，更新 `from_processed_context()`
- `EventNode`：移除 `parent_id`，新增 `refs`（可选，search hit 返回完整 refs）
- `periodic_task/hierarchy_summary.py`：`_store_summary()` 写 refs 替代直接写字段；调用 `batch_update_refs()` 替代 `batch_set_parent_id()`
- `tools/` 和 `server/search/`：通过 refs 遍历 hierarchy
- Vector DB backends：payload 中用 `refs` JSON 字段替代 `parent_id`/`children_ids`

### 1.5 Profiles 表 owner_type

**变更文件**：`storage/backends/mysql_backend.py`、`storage/backends/sqlite_backend.py`

```sql
ALTER TABLE profiles ADD COLUMN owner_type VARCHAR(50) DEFAULT 'user';
CREATE INDEX idx_profiles_owner_type ON profiles(owner_type);
```

- `owner_type="user"`：现有用户 profile（默认值，向后兼容）
- `owner_type="agent"`：agent profile
- Base profile：`owner_type="agent"`, `user_id="__base__"`, `agent_id=X`
- Interaction profile：`owner_type="agent"`, `user_id=Y`, `agent_id=X`

#### `__base__` 哨兵值处理规则

- **Base profile 写入**：通过 `/api/agents/{id}/base/profile` 端点直接写入，**不走 `refresh_profile()` 的 LLM merge**。是纯 OVERWRITE（调用方提供完整 profile 内容）。
- **Interaction profile 写入**：通过 push 流程中 agent_memory_processor 触发 `refresh_profile()`，执行 LLM merge。**首次写入时**，`get_profile(owner_type="agent", user_id=Y, agent_id=X)` 返回空，此时 **fallback 读取 base profile**（`user_id="__base__"`）作为初始值，再 LLM merge 新提取的内容，写入 interaction profile。后续更新正常读取已有的 interaction profile → LLM merge → 覆盖写入。
- **用户侧查询隔离**：所有面向用户的 profile 查询默认带 `owner_type="user"` 过滤，不会返回 agent profile。现有 `get_profile()` 方法签名不变，但内部默认加 `owner_type="user"` 条件，新增可选参数 `owner_type` 供 agent 场景使用。

---

## 2. Agent 注册管理

### 2.1 数据模型

**变更文件**：`storage/backends/mysql_backend.py`、`storage/backends/sqlite_backend.py`

新增 `agent_registry` 表：

```sql
CREATE TABLE agent_registry (
    agent_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

- 使用**软删除**（`is_deleted`）。`DELETE /api/agents/{id}` 仅标记 `is_deleted=TRUE`，不立即清除关联数据。
- 后续可提供硬删除接口或定时清理任务。硬删除级联顺序：vector DB agent events → profiles（`owner_type="agent"` 相关行）→ agent_registry。

### 2.2 API 端点

**新增文件**：`server/routes/agents.py`

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/agents` | POST | 注册新 agent（name, description） |
| `/api/agents` | GET | 列出所有 agents（排除已软删除） |
| `/api/agents/{agent_id}` | GET | 获取 agent 详情 |
| `/api/agents/{agent_id}` | PUT | 更新 agent 信息 |
| `/api/agents/{agent_id}` | DELETE | 软删除 agent |
| `/api/agents/{agent_id}/base/profile` | POST | 设置/更新 agent 基础 profile |
| `/api/agents/{agent_id}/base/profile` | GET | 获取 agent 基础 profile |
| `/api/agents/{agent_id}/base/events` | POST | 批量推送基础 events |
| `/api/agents/{agent_id}/base/events` | GET | 列出基础 events（分页） |
| `/api/agents/{agent_id}/base/events/{event_id}` | PUT | 编辑单条基础 event |
| `/api/agents/{agent_id}/base/events/{event_id}` | DELETE | 删除单条基础 event |

在 `server/api.py` 中注册 router。所有端点需 `dependencies=[Depends(auth_dependency)]`。

### 2.3 基础记忆推送

基础记忆是调用方已整理好的结构化数据，**不走 LLM 提取**：

**Base profile**：直接写入 profiles 表（`owner_type="agent"`, `user_id="__base__"`），纯 OVERWRITE，不走 LLM merge。

**Base events**：请求体：
```json
{
  "events": [
    {
      "title": "初入贾府",
      "summary": "林黛玉初次踏入贾府...",
      "event_time": "...",
      "keywords": ["贾府", "初见"],
      "entities": ["贾宝玉", "贾母"],
      "importance": 9
    }
  ]
}
```

处理：生成 embedding → 写入 vector DB，`context_type=AGENT_EVENT`。

---

## 3. Push 多 Processor 并行分发

### 3.1 请求格式

**变更文件**：`server/routes/push.py`

```json
POST /api/push/chat
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "user_id": "user_123",
  "agent_id": "lindaiyu_001",
  "device_id": "default",
  "processors": ["user_memory", "agent_memory"]
}
```

- `processors` 默认 `["user_memory"]`（向后兼容）
- 向后兼容：不传 `processors` 等同于 `["user_memory"]`

### 3.2 处理流程

```
POST /api/push/chat
  → 持久化 messages → chat_batches 表 → batch_id
  → 构建 RawContextProperties（source=CHAT_LOG, content_text=messages, additional_info={"batch_id": batch_id}）
  → ProcessorManager.process_batch(raw_context, processor_names=["user_memory", "agent_memory"])
    → 按 processor_names 查找已注册的 processor 实例
    → asyncio.gather(
        user_memory_processor.process(raw_context),
        agent_memory_processor.process(raw_context),
        return_exceptions=True  # 单个 processor 失败不影响其他
      )
    → 过滤掉异常结果，合并成功的 List[ProcessedContext]
    → 通过统一 callback 路由存储
  → _handle_processed_context(all_contexts)
    → 按 ContextType 路由存储
  → 失效相关 cache（user cache + agent cache）
```

**错误处理**：使用 `return_exceptions=True`，单个 processor 失败时记录错误日志，其余 processor 的输出照常存储。

### 3.3 ProcessorManager 变更

**变更文件**：`managers/processor_manager.py`

新增 `process_batch()` 方法：

```python
async def process_batch(
    self, raw_context: RawContextProperties, processor_names: List[str]
) -> List[ProcessedContext]:
    """并行调用指定 processors 处理同一输入，合并输出。"""
```

- 接收 `RawContextProperties`（与现有 `process()` 的输入类型一致）
- 按 `processor_names` 从 processor 名称映射表中查找实例
- 每个 processor 仍通过 `can_process()` → `process()` 标准接口调用
- 合并所有 processor 的输出后通过 callback 路由存储

**Processor 名称映射**：`process_batch` 使用独立的名称映射表（非 `ProcessorFactory` 的 type 注册表，也非 routing table）：

```python
BATCH_PROCESSOR_MAP = {
    "user_memory": "text_chat_processor",      # 已有 processor 的别名
    "agent_memory": "agent_memory_processor",   # 新增 processor
}
```

`processors` 请求参数中的名称通过此映射表解析为已注册的 processor 实例。

现有的 `process()` 方法保留用于非 chat 场景（document processor 等），其行为不变（按 routing table 选择单个 processor）。

### 3.4 Agent Memory Processor

**新增文件**：`context_processing/processor/agent_memory_processor.py`

- 继承 `BaseContextProcessor`
- `can_process(raw_context)`：检查 `source == ContextSource.CHAT_LOG`
- `process(raw_context) -> List[ProcessedContext]`：
  1. 从 `raw_context.content_text` 解析 messages
  2. 加载 agent 信息（从 agent_registry 查 agent_id 对应的 name/description）
  3. 加载 agent 视角 prompt（`processing.extraction.agent_memory_analyze`）
  4. 调用 LLM 从 agent 视角提取：
     - Agent profile 更新（对用户的认知变化）→ `context_type=PROFILE`（通过 `_store_profile` 走 agent profile merge）
     - Agent events（agent 的感受/评价/反应）→ `context_type=AGENT_EVENT`
  5. 每个输出的 `ProcessedContext` 设置 `properties.raw_type="chat_batch"`, `properties.raw_id=batch_id`（从 `raw_context.additional_info["batch_id"]` 获取）
  6. Profile 类型输出通过 `_store_profile()` → `refresh_profile()` 存储。`refresh_profile()` 需新增 `owner_type` 参数，agent profile 调用时传 `owner_type="agent"`，确保 `get_profile()` 和 `upsert_profile()` 操作正确的 profile 行
  7. 返回 `List[ProcessedContext]`

在 `ProcessorFactory._register_built_in_processors()` 中注册为 `"agent_memory"`。

**Agent 自主思考**也走 `/api/push/chat`：
```json
{
  "messages": [{"role": "assistant", "content": "三日未见..."}],
  "agent_id": "lindaiyu_001",
  "user_id": "user_123",
  "processors": ["agent_memory"]
}
```

Processor 根据消息内容（无 user role 消息）自行判断为自主思考场景，调整提取 prompt。

---

## 4. Search 参数化

### 4.1 请求格式变更

**变更文件**：`server/routes/search.py`、`server/search/models.py`

```json
POST /api/search
{
  "query": [{"type": "text", "text": "贾宝玉"}],
  "agent_id": "lindaiyu_001",
  "user_id": "user_123",
  "memory_owner": "agent",
  "top_k": 10,
  "drill_up": true
}
```

- `memory_owner` 默认 `"user"`
- `memory_owner="user"` → 搜索 `EVENT` + `DAILY_SUMMARY` + `WEEKLY_SUMMARY` + `MONTHLY_SUMMARY`
- `memory_owner="agent"` → 搜索 `AGENT_EVENT`（+ `AGENT_DAILY_SUMMARY` 等，待 hierarchy 实现后启用）

### 4.2 内部变更

- 移除硬编码 `EVENT_TYPE = ContextType.EVENT.value`
- 新增 `MEMORY_OWNER_TYPES`：

```python
MEMORY_OWNER_TYPES = {
    "user": [ContextType.EVENT, ContextType.DAILY_SUMMARY, ContextType.WEEKLY_SUMMARY, ContextType.MONTHLY_SUMMARY],
    "agent": [ContextType.AGENT_EVENT, ContextType.AGENT_DAILY_SUMMARY, ContextType.AGENT_WEEKLY_SUMMARY, ContextType.AGENT_MONTHLY_SUMMARY],
}
# index 0=L0, 1=L1, 2=L2, 3=L3
```

### 4.3 Hierarchy drill-up 端到端示例

以 `memory_owner="agent"` + `drill_up=true` 为例：

```
1. 语义搜索：context_types=[AGENT_EVENT] → 命中 agent_evt_042（L0）
2. 读取 agent_evt_042.refs → {"agent_daily_summary": ["ads_007"]}
3. get_contexts_by_ids(["ads_007"]) → 获取 L1 摘要
4. 读取 ads_007.refs → {"agent_weekly_summary": ["aws_002"]}
5. get_contexts_by_ids(["aws_002"]) → 获取 L2 摘要
6. 读取 aws_002.refs → {"agent_monthly_summary": ["ams_001"]}（如有）
7. 构建 tree：ams_001 → aws_002 → ads_007 → agent_evt_042（hit）
```

关键：**refs 字段的 key 就是父/子节点的 ContextType.value**，遍历时自然限定在同一类型体系内（user hierarchy 的 refs key 都是 user 类型，agent hierarchy 的 refs key 都是 agent 类型），无需额外过滤。

---

## 5. Memory Cache 参数化

### 5.1 请求格式变更

**变更文件**：`server/routes/memory_cache.py`、`server/cache/memory_cache_manager.py`、`server/cache/models.py`

```
GET /api/memory-cache?agent_id=X&user_id=Y&memory_owner=agent
```

- `memory_owner` 默认 `"user"`

### 5.2 Cache Manager 重命名

`UserMemoryCacheManager` → `MemoryCacheManager`

### 5.3 Snapshot 配置差异

| memory_owner | profile 查询 | events 查询 | daily summaries 查询 | docs/knowledge |
|---|---|---|---|---|
| `"user"` | `owner_type="user"` | `context_type=EVENT` | `context_type=DAILY_SUMMARY` | 查询 |
| `"agent"` | `owner_type="agent"`, `user_id=Y` | `context_type=AGENT_EVENT` | `context_type=AGENT_DAILY_SUMMARY` | 不查询 |

### 5.4 Cache Key

```
memory_cache:snapshot:{memory_owner}:{user_id}:{device_id}:{agent_id}
```

> **部署说明**：key 格式变更后，旧格式的 cached snapshot 成为孤儿 key，通过 TTL 自然过期，无需手动清理。

### 5.5 Cache 失效/刷新

复用现有模式（proactive refresh + 分布式锁 + double-check）。Push 完成后根据涉及的 `memory_owner` 类型分别失效对应 cache。

---

## 6. Prompt 模板

**变更文件**：`config/prompts_en.yaml`、`config/prompts_zh.yaml`

新增 prompt 组 `processing.extraction.agent_memory_analyze`：

```yaml
agent_memory_analyze:
  system: |
    You are {agent_name}. Based on the following conversation with a user,
    extract from YOUR perspective:
    1. profile: What you learned or how your perception of this user changed
    2. events: What happened from your point of view, your feelings and reactions

    Output JSON format:
    {
      "memories": [
        {
          "type": "agent_event" | "profile",
          "title": "...",
          "summary": "...",
          "event_time": "YYYY-MM-DDTHH:MM:SS",
          "keywords": [...],
          "entities": [...],
          "importance": 0-10
        }
      ]
    }

    For "event_time", use the time the conversation took place.
    For events: describe what happened from your subjective point of view.
    For profile: describe what you now know or feel about the user.
```

双语文件同步更新。

---

## 7. 集成位置总结

| 组件 | 文件位置（相对于 `opencontext/`） | 操作 |
|------|----------------------------------|------|
| ContextType 扩展 | `models/enums.py` | 修改 |
| refs 字段 | `models/context.py` | 修改 |
| ProcessedContextModel | `models/context.py` | 修改 |
| chat_batches 表 | `storage/backends/mysql_backend.py`、`sqlite_backend.py` | 修改 |
| agent_registry 表 | `storage/backends/mysql_backend.py`、`sqlite_backend.py` | 修改 |
| profiles owner_type | `storage/backends/mysql_backend.py`、`sqlite_backend.py` | 修改 |
| Vector DB refs 字段 | `storage/backends/qdrant_backend.py` 等 | 修改 |
| batch_update_refs 方法 | `storage/backends/*.py` | 新增 |
| Agent memory processor | `context_processing/processor/agent_memory_processor.py` | 新增 |
| Processor factory 注册 | `context_processing/processor/processor_factory.py` | 修改 |
| ProcessorManager 多分发 | `managers/processor_manager.py` | 修改 |
| Push 端点 | `server/routes/push.py` | 修改 |
| Search 端点 | `server/routes/search.py`、`server/search/models.py` | 修改 |
| Cache manager | `server/cache/memory_cache_manager.py` | 修改（重命名+参数化） |
| Cache models | `server/cache/models.py` | 修改 |
| Cache 端点 | `server/routes/memory_cache.py` | 修改 |
| Agent routes | `server/routes/agents.py` | 新增 |
| API router 注册 | `server/api.py` | 修改 |
| Prompt 模板 | `config/prompts_en.yaml`、`config/prompts_zh.yaml` | 修改 |
| Hierarchy 生成 | `periodic_task/hierarchy_summary.py` | 修改（refs + 新 ContextType） |
| Hierarchy 检索 | `tools/retrieval_tools/hierarchical_event_tool.py` | 修改（refs） |
| Search hierarchy | `server/routes/search.py` | 修改（refs + type map） |
| TextChatCapture | `context_capture/text_chat.py` | 修改（删 buffer） |
| refresh_profile | `context_processing/processor/profile_processor.py` | 修改（新增 owner_type 参数） |
| chat_batches 清理任务 | `periodic_task/` | 新增（90 天过期清理） |
| MODULE.md 更新 | `context_capture/MODULE.md`、`server/MODULE.md`、`models/MODULE.md` 等 | 修改 |

---

## 8. 不做的事

- 小说自动提取子系统（独立后续项目）
- Agent event hierarchy 生成逻辑（ContextType 预留，逻辑暂不实现）
- Knowledge / document 类型的 agent 记忆
- 已有数据迁移（用户手动脚本）
- 通用 entity 抽象
- 独立的 agent memory 端点（push/search/cache）
- Vector DB 额外的 `memory_owner` 字段
- Buffer 模式（删除）
