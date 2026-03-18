# Agent Memory — Design Spec

## Overview

为 MineContext 新增 Agent Memory 功能，让 AI agent（如小说角色）拥有独立的记忆体系。Agent 记忆分为两层：**基础记忆**（角色设定/背景事件，通过 API 直推）和**交互记忆**（与用户对话产生的记忆，通过 processor 自动提取）。

本次 scope 仅覆盖 **agent memory 核心服务**（push/search/cache）。小说自动提取子系统为独立后续项目。

## Design Decisions

以下为 brainstorming 过程中确认的核心决策：

1. **两层记忆模型**：base（角色设定+背景事件）+ interaction（对话产生的记忆）
2. **用 ContextType 区分**：新增 `AGENT_EVENT` 类型，不在 vector DB 加 `memory_owner` 字段
3. **统一 API + 参数化分发**：不为 agent memory 新建一套接口，复用现有 push/search/cache 端点
4. **Push 并行处理**：多个 processor 并行执行，共同引用 `batch_id`，无顺序依赖
5. **删除 buffer 模式**：每次 push = 一次直接处理，批量由调用方控制
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

---

## 1. 前置重构

### 1.1 删除 Buffer 模式

**变更文件**：`context_capture/text_chat.py`、`server/routes/push.py`

- 废弃 `process_mode` 参数（`"buffer"` / `"direct"`）
- 删除 `TextChatCapture` 的 Redis buffer 逻辑（push_message、flush_buffer、buffer key、flush lock、Lua 脚本）
- 每次 push 请求 = 一次直接处理
- `TextChatCapture` 简化为无状态的消息格式化 + callback 分发

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
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

Push 流程变为：
1. 持久化 messages → 得到 `batch_id`
2. 将 `batch_id` 传给所有 processor
3. 每个 processor 产出的 `ProcessedContext` 设置 `raw_type="chat_batch"`, `raw_id=batch_id`

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

同步更新：

| 映射 | DAILY_SUMMARY / WEEKLY / MONTHLY | AGENT_EVENT | AGENT_DAILY / WEEKLY / MONTHLY |
|------|----------------------------------|-------------|-------------------------------|
| `CONTEXT_UPDATE_STRATEGIES` | OVERWRITE | APPEND | OVERWRITE |
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

迁移映射：
- `parent_id` → `refs[ContextType.DAILY_SUMMARY.value]`（或对应的 summary 类型）
- `children_ids` → `refs[ContextType.EVENT.value]`（或对应的子类型）

**连带变更**：
- `ProcessedContextModel`：新增 `refs` 字段，移除 `parent_id`/`children_ids`，更新 `from_processed_context()`
- `periodic_task/hierarchy_summary.py`：写 refs 替代直接写字段
- `tools/` 和 `server/search/`：读 refs 替代读字段
- Vector DB backends：payload 中用 `refs` JSON 字段替代 `parent_id`/`children_ids`

### 1.5 Profiles 表 owner_type

**变更文件**：`storage/backends/mysql_backend.py`、`storage/backends/sqlite_backend.py`

```sql
ALTER TABLE profiles ADD COLUMN owner_type VARCHAR(50) DEFAULT 'user';
```

- `owner_type="user"`：现有用户 profile（默认值，向后兼容）
- `owner_type="agent"`：agent profile
- Base profile：`owner_type="agent"`, `user_id="__base__"`, `agent_id=X`
- Interaction profile：`owner_type="agent"`, `user_id=Y`, `agent_id=X`

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
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### 2.2 API 端点

**新增文件**：`server/routes/agents.py`

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/agents` | POST | 注册新 agent（name, description） |
| `/api/agents` | GET | 列出所有 agents |
| `/api/agents/{agent_id}` | GET | 获取 agent 详情 |
| `/api/agents/{agent_id}` | PUT | 更新 agent 信息 |
| `/api/agents/{agent_id}` | DELETE | 删除 agent 及其所有记忆 |
| `/api/agents/{agent_id}/base/profile` | POST | 设置/更新 agent 基础 profile |
| `/api/agents/{agent_id}/base/profile` | GET | 获取 agent 基础 profile |
| `/api/agents/{agent_id}/base/events` | POST | 批量推送基础 events |
| `/api/agents/{agent_id}/base/events` | GET | 列出基础 events（分页） |
| `/api/agents/{agent_id}/base/events/{event_id}` | PUT | 编辑单条基础 event |
| `/api/agents/{agent_id}/base/events/{event_id}` | DELETE | 删除单条基础 event |

在 `server/api.py` 中注册 router。

### 2.3 基础记忆推送

基础记忆是调用方已整理好的结构化数据，**不走 LLM 提取**：

**Base profile**：直接写入 profiles 表（`owner_type="agent"`, `user_id="__base__"`）

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
  → ProcessorManager.process_batch(batch_id, messages, processors)
    → asyncio.gather(
        user_memory_processor.process(batch_id, messages),
        agent_memory_processor.process(batch_id, messages),
      )
    → 合并所有 List[ProcessedContext]
    → 每个 ProcessedContext: raw_type="chat_batch", raw_id=batch_id
  → _handle_processed_context(all_contexts)
    → 按 ContextType 路由存储
  → 失效相关 cache
```

### 3.3 ProcessorManager 变更

**变更文件**：`managers/processor_manager.py`

新增 `process_batch()` 方法：
- 接收 `processors: List[str]`，并行调用所有指定 processor
- 合并所有 processor 的输出
- 通过统一 callback 路由存储

现有的 `process()` 方法保留用于非 chat 场景（document processor 等）。

### 3.4 Agent Memory Processor

**新增文件**：`context_processing/processor/agent_memory_processor.py`

- 继承 `BaseContextProcessor`
- `can_process()`：检查 source 和 processor 注册
- `process()`：
  1. 加载 agent 视角 prompt（`processing.extraction.agent_memory_analyze`）
  2. 调用 LLM 从 agent 视角提取：
     - Agent profile 更新（对用户的认知变化）
     - Agent events（agent 的感受/评价/反应）
  3. 返回 `List[ProcessedContext]`，`context_type=AGENT_EVENT` 或 profile 更新

在 `ProcessorFactory._register_built_in_processors()` 中注册。

**Agent 自主思考**也走 `/api/push/chat`：
```json
{
  "messages": [{"role": "assistant", "content": "三日未见..."}],
  "agent_id": "lindaiyu_001",
  "user_id": "user_123",
  "processors": ["agent_memory"]
}
```

Processor 根据消息内容（无 user role 消息）自行判断为自主思考场景。

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
  "top_k": 10
}
```

- `memory_owner` 默认 `"user"`
- `memory_owner="user"` → 搜索 `EVENT` + `DAILY_SUMMARY` + `WEEKLY_SUMMARY` + `MONTHLY_SUMMARY`
- `memory_owner="agent"` → 搜索 `AGENT_EVENT` + `AGENT_DAILY_SUMMARY` + ...

### 4.2 内部变更

- 移除硬编码 `EVENT_TYPE = ContextType.EVENT.value`
- 根据 `memory_owner` 映射到对应的 context types 列表
- Hierarchy drill-up：通过 `refs` 字段遍历，自动限定在同类型体系内
- Storage 查询传入正确的 `context_types` 列表

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
          "summary": "...",  // Your subjective perspective
          "keywords": [...],
          "entities": [...],
          "importance": 0-10
        }
      ]
    }
```

双语文件同步更新。

---

## 7. 集成位置总结

| 组件 | 文件位置 | 操作 |
|------|---------|------|
| ContextType 扩展 | `models/enums.py` | 修改 |
| refs 字段 | `models/context.py` | 修改 |
| ProcessedContextModel | `models/context.py` | 修改 |
| chat_batches 表 | `storage/backends/mysql_backend.py`、`sqlite_backend.py` | 修改 |
| agent_registry 表 | `storage/backends/mysql_backend.py`、`sqlite_backend.py` | 修改 |
| profiles owner_type | `storage/backends/mysql_backend.py`、`sqlite_backend.py` | 修改 |
| Agent memory processor | `context_processing/processor/agent_memory_processor.py` | 新增 |
| Processor factory 注册 | `context_processing/processor/processor_factory.py` | 修改 |
| ProcessorManager 多分发 | `managers/processor_manager.py` | 修改 |
| Push 端点 | `server/routes/push.py` | 修改 |
| Search 端点 | `server/routes/search.py`、`server/search/models.py` | 修改 |
| Cache manager | `server/cache/memory_cache_manager.py` | 修改 |
| Cache models | `server/cache/models.py` | 修改 |
| Cache 端点 | `server/routes/memory_cache.py` | 修改 |
| Agent routes | `server/routes/agents.py` | 新增 |
| API router 注册 | `server/api.py` | 修改 |
| Prompt 模板 | `config/prompts_en.yaml`、`config/prompts_zh.yaml` | 修改 |
| Hierarchy 生成 | `periodic_task/hierarchy_summary.py` | 修改（refs 迁移） |
| Hierarchy 检索 | `tools/`、`server/search/` | 修改（refs 迁移） |
| Vector DB backends | `storage/backends/qdrant_backend.py` 等 | 修改（refs 字段） |
| TextChatCapture | `context_capture/text_chat.py` | 修改（删 buffer） |

---

## 8. 不做的事

- 小说自动提取子系统（独立后续项目）
- Agent event hierarchy 生成逻辑（ContextType 预留，逻辑暂不实现）
- Knowledge / document 类型的 agent 记忆
- 已有数据迁移（用户手动脚本）
- 通用 entity 抽象
- `/api/agent-memory/push` 独立端点
- `/api/agent-memory/search` 独立端点
- `/api/agent-memory/cache` 独立端点
- Vector DB `memory_owner` 字段
- Buffer 模式（删除）
