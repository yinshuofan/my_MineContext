# API Reference

Base URL: `http://{host}:{port}` (默认 `http://localhost:1733`)

认证：通过 `config.yaml` 中 `api_auth.enabled` 控制。启用时需在请求头携带 `Authorization: Bearer {api_key}`。

用户标识：所有接口通过 `(user_id, device_id, agent_id)` 三元组标识用户，`device_id` 和 `agent_id` 默认为 `"default"`。

> **时区说明**: 所有 API 返回的时间字段现在包含时区信息（如 `2026-03-21T08:30:00+08:00`），而非之前的无时区格式（`2026-03-21T00:30:00`）。时区由服务端 `config.yaml` 中的 `timezone` 配置项决定。

---

## 1. Push Chat — 推送聊天消息

`POST /api/push/chat`

将对话消息推送到系统进行处理，提取 profile / event / knowledge 等记忆。消息会持久化到 `chat_batches` 表，然后在后台分发给指定的处理器。

> **Breaking Change**: `process_mode` 和 `flush_immediately` 参数已移除。Buffer 模式不再存在。所有消息均直接处理。新增 `processors` 参数控制运行哪些处理器。

### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `messages` | `List[Object]` | **是** | — | 聊天消息列表（OpenAI 格式），1-100 条 |
| `messages[].role` | `string` | **是** | — | 角色：`"user"` / `"assistant"` / `"system"` |
| `messages[].content` | `string \| List` | **是** | — | 消息内容，支持纯文本或多模态内容数组 |
| `user_id` | `string` | 否 | `"default"` | 用户标识 |
| `device_id` | `string` | 否 | `"default"` | 设备标识 |
| `agent_id` | `string` | 否 | `"default"` | Agent 标识 |
| `processors` | `List[string]` | 否 | `["user_memory"]` | 要运行的处理器列表：`"user_memory"` 等 |

多模态 content 格式（OpenAI content parts）：
```json
[
  {"type": "text", "text": "这张照片是今天拍的"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
]
```

### 请求示例

**基本用法（默认处理器）**
```bash
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Remind me to buy groceries tomorrow"}
    ],
    "user_id": "test_user"
  }'
```

**多条消息**
```bash
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I had a meeting with the product team today. We discussed the Q2 roadmap."},
      {"role": "assistant", "content": "Got it! The Q2 roadmap discussion is noted."}
    ],
    "user_id": "test_user"
  }'
```

**指定处理器**
```bash
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tomorrow I need to prepare the quarterly report for the finance team."}
    ],
    "user_id": "test_user",
    "processors": ["user_memory"]
  }'
```

### 响应示例

**成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "Chat messages submitted for processing",
  "data": {
    "batch_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "message_count": 2
  }
}
```

**超时**
```json
{
  "code": 504,
  "status": 504,
  "message": "Request timed out"
}
```

**Agent 未注册（400）**
```json
{
  "detail": "Agent 'my_agent' is not registered. Please register the agent via POST /api/agents before using agent_memory processor."
}
```
> 当 `processors` 包含 `"agent_memory"` 且 `agent_id` 指定了未注册的 agent 时返回 400。

---

## 2. Push Document — 推送文档

`POST /api/push/document`

推送文件文档到系统，支持本地文件路径或 Base64 编码数据。

### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file_path` | `string` | 二选一 | — | 服务器上的本地文件路径 |
| `base64_data` | `string` | 二选一 | — | Base64 编码的文件内容 |
| `filename` | `string` | 否 | — | 文件名（使用 base64_data 时建议提供） |
| `content_type` | `string` | 否 | — | MIME 类型（如 `"text/plain"`、`"application/pdf"`） |
| `user_id` | `string` | 否 | `"default"` | 用户标识 |
| `device_id` | `string` | 否 | `"default"` | 设备标识 |
| `metadata` | `Object` | 否 | — | 附加元数据 |

> `file_path` 和 `base64_data` 必须提供其中一个。

### 请求示例

**通过文件路径推送**
```bash
curl -X POST http://localhost:1733/api/push/document \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/data/docs/report.pdf",
    "user_id": "test_user",
    "metadata": {"source": "upload", "tags": ["report"]}
  }'
```

**通过 Base64 数据推送**
```bash
curl -X POST http://localhost:1733/api/push/document \
  -H "Content-Type: application/json" \
  -d '{
    "base64_data": "IyBUZXN0IERvY3VtZW50CgpUaGlzIGlzIGEgdGVzdC4=",
    "filename": "test_doc.md",
    "content_type": "text/markdown",
    "user_id": "test_user"
  }'
```

### 响应示例

**成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "Document pushed successfully",
  "data": {
    "path": "/data/docs/report.pdf"
  }
}
```

**缺少文件参数**
```json
{
  "code": 400,
  "status": 400,
  "message": "Either 'file_path' or 'base64_data' must be provided"
}
```

**文件不存在**
```json
{
  "code": 400,
  "status": 400,
  "message": "Document path /data/docs/nonexistent.pdf does not exist"
}
```

---

## 3. Search — 事件搜索

`POST /api/search`

语义搜索事件记忆，支持多模态查询、精确 ID 查找、时间范围过滤，以及层级摘要钻取。

### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | `List[Object]` | 至少一项 | — | 多模态搜索查询（OpenAI content parts 格式） |
| `event_ids` | `List[string]` | 至少一项 | — | 精确事件 ID 列表 |
| `time_range` | `Object` | 至少一项 | — | 时间范围过滤 |
| `time_range.start` | `int` | 否 | — | 起始时间（Unix 秒） |
| `time_range.end` | `int` | 否 | — | 结束时间（Unix 秒） |
| `hierarchy_levels` | `List[int]` | 至少一项 | — | 层级过滤：`0`=原始事件, `1`=日摘要, `2`=周摘要, `3`=月摘要 |
| `drill_up` | `bool` | 否 | `true` | 是否递归获取祖先摘要，构建树形结构 |
| `top_k` | `int` | 否 | `10` | 最大返回数量（1-100） |
| `score_threshold` | `float` | 否 | 无阈值 | 最低相似度分数（0-1），低于此分数的结果被过滤 |
| `user_id` | `string` | 否 | — | 用户标识 |
| `device_id` | `string` | 否 | — | 设备标识 |
| `agent_id` | `string` | 否 | — | Agent 标识 |

> `query`、`event_ids`、`time_range`、`hierarchy_levels` 必须至少提供一项。

### 请求示例

**语义搜索**
```bash
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [{"type": "text", "text": "meeting"}],
    "user_id": "test_user",
    "top_k": 5
  }'
```

**精确 ID 查找**
```bash
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "event_ids": ["0f9ae964-bd55-48bc-83a0-8cc8258db85d"],
    "user_id": "test_user"
  }'
```

**时间范围 + 层级过滤**
```bash
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "time_range": {"start": 1741340000, "end": 1741490000},
    "hierarchy_levels": [0],
    "user_id": "test_user",
    "top_k": 3
  }'
```

**关闭 drill_up + 设置分数阈值**
```bash
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [{"type": "text", "text": "roadmap planning"}],
    "user_id": "test_user",
    "top_k": 3,
    "drill_up": false,
    "score_threshold": 0.1
  }'
```

### 响应示例

**语义搜索成功**
```json
{
  "success": true,
  "events": [
    {
      "id": "0f9ae964-bd55-48bc-83a0-8cc8258db85d",
      "hierarchy_level": 0,
      "refs": {},
      "title": "Discussion on birthday gift for girlfriend",
      "summary": "Discussed choosing a birthday gift and decided to add matching earring.\n- Selected item priced at 580 yuan for girlfriend's next-week birthday\n- Confirmed the item fits girlfriend's style preference\n- Decided to pair with a matching earring after AI suggestion",
      "event_time_start": "2026-03-08T21:50:46.393574+00:00",
      "event_time_end": "2026-03-08T21:50:46.393574+00:00",
      "create_time": "2026-03-08T21:50:46.378637",
      "is_search_hit": true,
      "children": [],
      "media_refs": [
        {"type": "image", "url": "https://example.com/media/image1.jpg", "media_index": 0},
        {"type": "video", "url": "https://example.com/media/video1.mp4", "media_index": 1}
      ],
      "keywords": ["birthday gift", "earring", "shopping"],
      "entities": ["earring"],
      "agent_commentary": null,
      "score": 0.108
    },
    {
      "id": "0f0c5487-561b-4950-ba7c-c244b5261adf",
      "hierarchy_level": 0,
      "refs": {},
      "title": "User Shared Photo and Received Feedback",
      "summary": "User shared a photo and got positive feedback on its quality.\n- Photo features a traditional decorated item\n- AI praised the photo's lighting and composition.",
      "event_time_start": "2026-03-08T22:09:04+00:00",
      "event_time_end": "2026-03-08T22:09:04+00:00",
      "create_time": "2026-03-08T22:09:03.996846",
      "is_search_hit": true,
      "children": [],
      "media_refs": [
        {"type": "image", "url": "https://example.com/media/image2.jpg"}
      ],
      "keywords": ["photo feedback", "historical site"],
      "entities": [],
      "agent_commentary": "User seems enthusiastic about photography and traditional architecture. This aligns with their stated interests.",
      "score": 0.122
    }
  ],
  "metadata": {
    "query": "meeting",
    "total_results": 2,
    "search_time_ms": 1424.6
  }
}
```

**精确 ID 查找成功**（score 固定为 1.0）
```json
{
  "success": true,
  "events": [
    {
      "id": "0f9ae964-bd55-48bc-83a0-8cc8258db85d",
      "hierarchy_level": 0,
      "refs": {},
      "title": "Discussion on birthday gift for girlfriend",
      "summary": "...",
      "event_time_start": "2026-03-08T21:50:46.393574+00:00",
      "event_time_end": "2026-03-08T21:50:46.393574+00:00",
      "create_time": "2026-03-08T21:50:46.378637",
      "is_search_hit": true,
      "children": [],
      "media_refs": [],
      "keywords": ["birthday gift"],
      "entities": [],
      "score": 1.0
    }
  ],
  "metadata": {
    "query": null,
    "total_results": 1,
    "search_time_ms": 404.58
  }
}
```

**无结果**
```json
{
  "success": true,
  "events": [],
  "metadata": {
    "query": null,
    "total_results": 0,
    "search_time_ms": 113.38
  }
}
```

**缺少搜索条件（422）**
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, At least one search criterion required: query, event_ids, time_range, or hierarchy_levels",
      "input": {"user_id": "test_user"},
      "ctx": {"error": {}}
    }
  ]
}
```

---

## 4. Memory Cache — 用户记忆快照

`GET /api/memory-cache`

获取用户的记忆快照，包含 profile（用户画像）、agent_prompt（Agent 画像/提示词）、近期事件摘要、今日事件、最近访问记录。支持 Redis 缓存，可按 section 过滤返回内容。

### 查询参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user_id` | `string` | **是** | — | 用户标识 |
| `device_id` | `string` | 否 | `"default"` | 设备标识 |
| `agent_id` | `string` | 否 | `"default"` | Agent 标识 |
| `include` | `string` | 否 | `"profile,agent_prompt,events,accessed"` | 逗号分隔的返回 section：`profile` / `agent_prompt` / `events` / `accessed` / `all` |
| `recent_days` | `int` | 否 | 配置值（默认 3） | 近期记忆时间窗口（天） |
| `max_recent_events_today` | `int` | 否 | 配置值（默认 5） | 今日最大 L0 事件数 |
| `max_accessed` | `int` | 否 | `5`（1-100） | 最近访问记录最大数量 |
| `force_refresh` | `bool` | 否 | `false` | 是否强制重建缓存（跳过 Redis） |

**`include` 参数说明：**
- `profile`：返回用户画像（`profile` 字段）
- `agent_prompt`：返回 Agent 的画像/提示词（`agent_prompt` 字段，独立于 `profile` 的新 section）
- `events`：返回每日摘要和今日事件（`daily_summaries` + `today_events` 字段）
- `accessed`：返回最近访问记录（`recently_accessed` 字段）
- `all`：返回所有 section
- 未包含的 section 对应字段为 `null`，已包含但无数据的为 `[]`

**响应字段说明（补充）：**
- `profile` (`SimpleProfile | null`) — 用户自身画像。仅在 `include` 包含 `profile` 时返回。
- `agent_prompt` (`SimpleProfile | null`) — The agent's own profile / prompt when `agent_id != "default"`. Falls back to `agent_base_profile` (same `device_id` / `agent_id`, `user_id="__base__"`) if an agent-specific entry is not yet stored. Populated only when `include` contains `agent_prompt` (a new section, independent of `profile`). `null` when `agent_id == "default"`, when no entry exists, or when the section is not requested.

### 请求示例

**默认请求（返回全部 section）**
```bash
curl "http://localhost:1733/api/memory-cache?user_id=test_user"
```

**仅返回 profile**
```bash
curl "http://localhost:1733/api/memory-cache?user_id=test_user&include=profile"
```

**仅返回事件信息**
```bash
curl "http://localhost:1733/api/memory-cache?user_id=test_user&include=events"
```

**仅返回 Agent 画像/提示词**
```bash
curl "http://localhost:1733/api/memory-cache?user_id=test_user&agent_id=kiki&include=agent_prompt"
```

**返回用户画像 + Agent 画像**
```bash
curl "http://localhost:1733/api/memory-cache?user_id=test_user&agent_id=kiki&include=profile,agent_prompt"
```

**全部参数**
```bash
curl "http://localhost:1733/api/memory-cache?user_id=test_user&device_id=iphone&agent_id=chatbot&include=profile,agent_prompt,events&recent_days=7&max_accessed=3&force_refresh=true"
```

### 响应示例

**有数据的完整响应**
```json
{
  "success": true,
  "user_id": "test_user",
  "device_id": "default",
  "agent_id": "kiki",
  "profile": {
    "factual_profile": "用户是一名25岁的男性，喜欢摄影和传统文化。",
    "behavioral_profile": "沟通风格简洁直接，偏好中文交流。",
    "metadata": {}
  },
  "agent_prompt": {
    "factual_profile": "Kiki 是一位专注于摄影和传统文化话题的助手。",
    "behavioral_profile": "语气友好亲切，主动分享拍摄建议。",
    "metadata": {}
  },
  "recently_accessed": [
    {
      "id": "0f9ae964-bd55-48bc-83a0-8cc8258db85d",
      "title": "Discussion on birthday gift for girlfriend",
      "summary": "Discussed choosing a birthday gift and decided to add matching earring.",
      "context_type": "event",
      "keywords": ["birthday gift", "earring"],
      "accessed_ts": 1773220687.724764,
      "score": 1.0,
      "event_time_start": "2026-03-08T21:50:46.393574+00:00",
      "create_time": "2026-03-08T21:50:46.378637",
      "media_refs": []
    }
  ],
  "daily_summaries": [
    {
      "event_time_start": "2026-03-08T00:00:00+00:00",
      "title": "Daily Summary",
      "summary": "Discussed birthday gift selection and shared photos from the Forbidden City."
    }
  ],
  "today_events": [
    {
      "title": "Morning standup meeting",
      "summary": "Discussed sprint progress and blockers.",
      "event_time_start": "2026-03-11T09:00:00+00:00"
    }
  ]
}
```

**仅 profile section（其他字段为 null）**
```json
{
  "success": true,
  "user_id": "test_user",
  "device_id": "default",
  "agent_id": "default",
  "profile": null,
  "agent_prompt": null,
  "recently_accessed": null,
  "daily_summaries": null,
  "today_events": null
}
```

**仅 events section**
```json
{
  "success": true,
  "user_id": "test_user",
  "device_id": "default",
  "agent_id": "default",
  "profile": null,
  "agent_prompt": null,
  "recently_accessed": null,
  "daily_summaries": [],
  "today_events": []
}
```

**仅 agent_prompt section（agent_id=kiki）**
```json
{
  "success": true,
  "user_id": "test_user",
  "device_id": "default",
  "agent_id": "kiki",
  "profile": null,
  "agent_prompt": {
    "factual_profile": "Kiki 是一位专注于摄影和传统文化话题的助手。",
    "behavioral_profile": "语气友好亲切，主动分享拍摄建议。",
    "metadata": {}
  },
  "recently_accessed": null,
  "daily_summaries": null,
  "today_events": null
}
```

**新用户（无数据）**
```json
{
  "success": true,
  "user_id": "new_user",
  "device_id": "default",
  "agent_id": "default",
  "profile": null,
  "agent_prompt": null,
  "recently_accessed": [],
  "daily_summaries": [],
  "today_events": []
}
```

**缺少 user_id（422）**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["query", "user_id"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```

---

## 5. Agents — Agent 注册与管理

Agent CRUD 和基础记忆管理。Agent 注册后可通过 push/chat 的 `processors: ["agent_memory"]` 从对话中提取 agent 视角的记忆。

### 5.1 创建 Agent

`POST /api/agents`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `agent_id` | `string` | **是** | Agent 唯一标识（1-100 字符，不可为 `"__base__"`） |
| `name` | `string` | **是** | Agent 名称（1-255 字符） |
| `description` | `string` | 否 | Agent 描述 |

```bash
curl -X POST http://localhost:1733/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "assistant_01",
    "name": "Personal Assistant",
    "description": "A helpful personal assistant that remembers user preferences"
  }'
```

**成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "Agent created",
  "data": {"agent_id": "assistant_01"}
}
```

**ID 已存在（400）**
```json
{
  "detail": "Agent creation failed (ID may already exist)"
}
```

### 5.2 列出所有 Agent

`GET /api/agents`

```bash
curl -X GET http://localhost:1733/api/agents
```

**成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "success",
  "data": {
    "agents": [
      {
        "agent_id": "assistant_01",
        "name": "Personal Assistant",
        "description": "A helpful personal assistant",
        "created_at": "2026-03-18T10:00:00",
        "updated_at": "2026-03-18T10:00:00"
      }
    ]
  }
}
```

### 5.3 获取单个 Agent

`GET /api/agents/{agent_id}`

```bash
curl -X GET http://localhost:1733/api/agents/assistant_01
```

**成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "success",
  "data": {
    "agent": {
      "agent_id": "assistant_01",
      "name": "Personal Assistant",
      "description": "A helpful personal assistant",
      "created_at": "2026-03-18T10:00:00",
      "updated_at": "2026-03-18T10:00:00"
    }
  }
}
```

**不存在（404）**
```json
{
  "detail": "Agent not found"
}
```

### 5.4 更新 Agent

`PUT /api/agents/{agent_id}`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | `string` | 否 | 新名称（最长 255 字符） |
| `description` | `string` | 否 | 新描述 |

```bash
curl -X PUT http://localhost:1733/api/agents/assistant_01 \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Smart Assistant",
    "description": "Updated description"
  }'
```

**成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "Agent updated"
}
```

### 5.5 删除 Agent（软删除）

`DELETE /api/agents/{agent_id}`

```bash
curl -X DELETE http://localhost:1733/api/agents/assistant_01
```

**成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "Agent deleted"
}
```

---

## 6. Agent Base Memory — Agent 基础记忆

Agent 的基础记忆（base memory）是预先配置的 profile 和事件，与对话提取的记忆分开管理。基础 profile 和基础事件均使用 `user_id="__base__"` 作为哨兵值，以区分对话中产生的 per-user 数据。

### 6.1 设置基础 Profile

`POST /api/agents/{agent_id}/base/profile`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `factual_profile` | `string` | **是** | Agent 的事实性 profile 描述 |
| `behavioral_profile` | `string` | 否 | 行为模式描述 |
| `entities` | `List[string]` | 否 | 相关实体列表 |
| `importance` | `int` | 否 | 重要性（默认 0） |

```bash
curl -X POST http://localhost:1733/api/agents/assistant_01/base/profile \
  -H "Content-Type: application/json" \
  -d '{
    "factual_profile": "I am a personal assistant specialized in scheduling and task management. I prefer concise communication.",
    "behavioral_profile": "Responds in a friendly, professional tone. Proactively suggests reminders.",
    "entities": ["calendar", "task management", "reminders"],
    "importance": 8
  }'
```

**成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "Base profile saved"
}
```

### 6.2 获取基础 Profile

`GET /api/agents/{agent_id}/base/profile`

```bash
curl -X GET http://localhost:1733/api/agents/assistant_01/base/profile
```

**成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "success",
  "data": {
    "profile": {
      "user_id": "__base__",
      "device_id": "default",
      "agent_id": "assistant_01",
      "context_type": "agent_base_profile",
      "factual_profile": "I am a personal assistant specialized in scheduling and task management.",
      "behavioral_profile": "Responds in a friendly, professional tone.",
      "keywords": [],
      "entities": ["calendar", "task management", "reminders"],
      "importance": 8,
      "metadata": {}
    }
  }
}
```

**不存在（404）**
```json
{
  "detail": "Base profile not found"
}
```

### 6.3 推送基础事件

`POST /api/agents/{agent_id}/base/events`

> **⚠️ Breaking change (2026-04-17):** This endpoint now uses **replace semantics**. Each request treats the submitted tree as the agent's complete base-event state. Any existing `AGENT_BASE_*` context not present in the new tree will be deleted. Clients that previously relied on additive/incremental uploads must now merge locally before posting.

推送结构化的基础事件，不经过 LLM 提取。系统会为每个事件生成 embedding，并根据 `hierarchy_level` 存储为对应的 context type：L0 → `agent_base_event`，L1 → `agent_base_l1_summary`，L2 → `agent_base_l2_summary`，L3 → `agent_base_l3_summary`。

支持嵌套层级结构（L3→L2→L1→L0），父节点时间范围必须覆盖所有子节点。

**请求体字段：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `events` | `List[Object]` | **是** | 顶层事件列表（至少 1 项，总数含子节点不超过 500） |
| `events[].title` | `string` | **是** | 事件标题 |
| `events[].summary` | `string` | **是** | 事件摘要 |
| `events[].event_time_start` | `string` | 否 | ISO 8601 开始时间（默认当前时间） |
| `events[].event_time_end` | `string` | 条件必填 | ISO 8601 结束时间；`hierarchy_level > 0` 时必填 |
| `events[].hierarchy_level` | `int` | 否 | 层级深度 0/1/2/3（默认 0）；0 = 原始事件，1 = 日摘要，2 = 周摘要，3 = 月摘要 |
| `events[].children` | `List[Object]` | 条件必填 | 嵌套子事件；`hierarchy_level > 0` 时必填，`hierarchy_level == 0` 时不可填 |
| `events[].keywords` | `List[string]` | 否 | 关键词列表 |
| `events[].entities` | `List[string]` | 否 | 实体列表 |
| `events[].importance` | `int` | 否 | 重要性 0-10（默认 5） |

**验证规则：**

- `hierarchy_level > 0` 必须同时提供 `event_time_end` 和 `children`
- `hierarchy_level == 0` 不可包含 `children`
- 子节点的 `hierarchy_level` 必须等于父节点 `hierarchy_level - 1`
- 父节点时间范围必须覆盖所有子节点（`event_time_start` ≤ 子节点最小开始时间，`event_time_end` ≥ 子节点最大结束时间）
- 单次请求总事件数（含所有层级子节点）不超过 500
- `hierarchy_level` 最大为 3（支持 L3→L2→L1→L0 四级树）

**扁平事件示例：**
```bash
curl -X POST http://localhost:1733/api/agents/assistant_01/base/events \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {
        "title": "Product launch v2.0",
        "summary": "The product team launched version 2.0 with new scheduling features and improved UI.",
        "event_time_start": "2026-03-15T09:00:00+08:00",
        "keywords": ["product launch", "v2.0", "scheduling"],
        "entities": ["product team"],
        "importance": 8
      },
      {
        "title": "Company policy update",
        "summary": "New remote work policy allows 3 days WFH per week starting April.",
        "keywords": ["policy", "remote work"],
        "importance": 6
      }
    ]
  }'
```

**嵌套层级结构示例：**
```bash
curl -X POST http://localhost:1733/api/agents/assistant_01/base/events \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {
        "title": "Standalone event",
        "summary": "A simple L0 event",
        "event_time_start": "2026-03-15T10:00:00+08:00",
        "importance": 5
      },
      {
        "title": "Daily Summary",
        "summary": "Summary of the day...",
        "event_time_start": "2026-03-15T00:00:00+08:00",
        "event_time_end": "2026-03-15T23:59:59+08:00",
        "hierarchy_level": 1,
        "children": [
          {
            "title": "Morning standup",
            "summary": "Discussed sprint progress",
            "event_time_start": "2026-03-15T09:00:00+08:00",
            "keywords": ["standup", "sprint"],
            "importance": 6
          },
          {
            "title": "Code review",
            "summary": "Reviewed auth module PR",
            "event_time_start": "2026-03-15T14:00:00+08:00",
            "keywords": ["code review", "auth"],
            "importance": 5
          }
        ]
      }
    ]
  }'
```

**成功 (`200 OK`)**
```json
{
  "code": 0,
  "status": 200,
  "message": "Base events replaced",
  "data": {
    "upserted": 12,
    "deleted": 4,
    "stragglers": 0,
    "ids": ["<new_id_1>", "..."]
  }
}
```

- `upserted` — count of contexts written
- `deleted` — count of previously-existing contexts removed
- `stragglers` — count of contexts that should have been deleted but whose deletion failed (non-fatal; cleaned on next POST)
- `ids` — ids of upserted contexts

**另一编辑进行中（503）**
```json
{
  "detail": "another edit is in progress for this agent"
}
```

> 同一 agent 在同一时刻只能有一个 replace 请求进行中；并发请求会收到 `503 Service Unavailable`。客户端应重试。

### 6.4 列出基础事件

`GET /api/agents/{agent_id}/base/events`

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `limit` | `int` | 否 | `50` | 每页数量 |
| `offset` | `int` | 否 | `0` | 偏移量 |
| `hierarchy_level` | `int` | 否 | — | 按层级过滤：`0`=原始事件（`agent_base_event`），`1`=日摘要，`2`=周摘要，`3`=月摘要；不传则返回所有层级 |

```bash
# 返回所有层级
curl -X GET "http://localhost:1733/api/agents/assistant_01/base/events?limit=10&offset=0"

# 仅返回 L0 原始事件
curl "http://localhost:1733/api/agents/assistant_01/base/events?hierarchy_level=0"

# 仅返回 L1 日摘要
curl "http://localhost:1733/api/agents/assistant_01/base/events?hierarchy_level=1"
```

**成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "success",
  "data": {
    "events": [
      {
        "id": "a1b2c3d4-...",
        "context_type": "agent_base_event",
        "hierarchy_level": 0,
        "title": "Product launch v2.0",
        "summary": "The product team launched version 2.0...",
        "keywords": ["product launch", "v2.0"],
        "entities": ["product team"],
        "importance": 8,
        "event_time_start": "2026-03-15T09:00:00+08:00",
        "event_time_end": "2026-03-15T09:00:00+08:00",
        "create_time": "2026-03-18T10:30:00+00:00"
      }
    ]
  }
}
```

> `context_type` 依 `hierarchy_level` 不同而变化：`agent_base_event`（L0）、`agent_base_l1_summary`（L1）、`agent_base_l2_summary`（L2）、`agent_base_l3_summary`（L3）。

### 6.5 删除基础事件

`DELETE /api/agents/{agent_id}/base/events/{event_id}`

Deletes an event and its entire subtree (all descendants reachable via
downward refs). The surviving parent's ref list is scrubbed of the
deleted root id. Empty parent summaries (whose only child was just
deleted) are preserved — to remove them, POST a new tree without them.

```bash
curl -X DELETE http://localhost:1733/api/agents/assistant_01/base/events/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**成功 (`200 OK`)**
```json
{
  "code": 0,
  "status": 200,
  "message": "Event deleted",
  "data": {
    "deleted_ids": ["<root_id>", "<child_id_1>", "..."],
    "updated_parent_id": "<parent_id_or_null>",
    "stragglers": 0
  }
}
```

- `deleted_ids` — 已删除的 context id 列表（root + 所有后代）
- `updated_parent_id` — 其 ref 列表被清理的父节点 id；若被删根节点无父节点，则为 `null`
- `stragglers` — 应被删除但删除失败的 context 数量（non-fatal；下次 POST 时清理）

**不存在（404）**
```json
{
  "detail": "Event not found"
}
```

**另一编辑进行中（503）**
```json
{
  "detail": "another edit is in progress for this agent"
}
```

> 同一 agent 在同一时刻只能有一个 edit 请求进行中（replace 或 delete）；并发请求会收到 `503 Service Unavailable`。客户端应重试。

---

## 7. Agent Memory via Push Chat

通过 push/chat 接口的 `processors` 参数触发 agent 记忆提取。需要先注册 agent。

```bash
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to prepare the quarterly report by Friday"},
      {"role": "assistant", "content": "Got it! I will remind you about the quarterly report deadline on Friday."}
    ],
    "user_id": "test_user",
    "agent_id": "assistant_01",
    "processors": ["user_memory", "agent_memory"]
  }'
```

`"agent_memory"` 处理器会从 agent 视角分析对话，提取 `AGENT_EVENT` 和 `AGENT_PROFILE` 类型的记忆。提取的记忆通过标准存储路由：agent_profile -> 关系数据库（`context_type="agent_profile"`），agent_event -> 向量数据库。

---

## 8. Chat Batches (Debug)

调试接口，用于查看聊天批次及其产生的向量上下文。

### 8.1 列出聊天批次

`GET /api/chat-batches`

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user_id` | `string` | 否 | — | 用户标识 |
| `device_id` | `string` | 否 | — | 设备标识 |
| `agent_id` | `string` | 否 | — | Agent 标识 |
| `start_date` | `string` | 否 | — | 起始日期 |
| `end_date` | `string` | 否 | — | 结束日期 |
| `page` | `int` | 否 | `1` | 页码 |
| `limit` | `int` | 否 | `20` | 每页数量 |

```bash
curl "http://localhost:1733/api/chat-batches?user_id=test_user&page=1&limit=20"
```

**成功**
```json
{
  "data": {
    "batches": [...],
    "total": 42,
    "page": 1,
    "limit": 20,
    "total_pages": 3
  }
}
```

### 8.2 获取聊天批次详情

`GET /api/chat-batches/{batch_id}`

```bash
curl "http://localhost:1733/api/chat-batches/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

**成功**
```json
{
  "data": {
    "batch": {
      "batch_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "messages": [...],
      "user_id": "test_user",
      "device_id": "default",
      "agent_id": "default",
      "message_count": 2,
      "created_at": "2026-03-18T10:00:00"
    }
  }
}
```

### 8.3 获取批次关联的上下文

`GET /api/chat-batches/{batch_id}/contexts`

返回该批次产生的所有向量数据库上下文（通过 `raw_type="chat_batch"` + `raw_id=batch_id` 匹配）。

```bash
curl "http://localhost:1733/api/chat-batches/a1b2c3d4-e5f6-7890-abcd-ef1234567890/contexts"
```

**成功**
```json
{
  "data": {
    "contexts": [...]
  }
}
```

