# API Reference

Base URL: `http://{host}:{port}` (默认 `http://localhost:1733`)

认证：通过 `config.yaml` 中 `api_auth.enabled` 控制。启用时需在请求头携带 `Authorization: Bearer {api_key}`。

用户标识：所有接口通过 `(user_id, device_id, agent_id)` 三元组标识用户，`device_id` 和 `agent_id` 默认为 `"default"`。

---

## 1. Push Chat — 推送聊天消息

`POST /api/push/chat`

将对话消息推送到系统进行处理，提取 profile / event / knowledge 等记忆。

### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `messages` | `List[Object]` | **是** | — | 聊天消息列表（OpenAI 格式），1-100 条 |
| `messages[].role` | `string` | **是** | — | 角色：`"user"` / `"assistant"` / `"system"` |
| `messages[].content` | `string \| List` | **是** | — | 消息内容，支持纯文本或多模态内容数组 |
| `user_id` | `string` | 否 | `"default"` | 用户标识 |
| `device_id` | `string` | 否 | `"default"` | 设备标识 |
| `agent_id` | `string` | 否 | `"default"` | Agent 标识 |
| `process_mode` | `string` | 否 | `"buffer"` | 处理模式：`"buffer"`（缓冲后批量处理）/ `"direct"`（立即处理） |
| `flush_immediately` | `bool` | 否 | `false` | 仅 buffer 模式有效，是否立即刷新缓冲区 |

多模态 content 格式（OpenAI content parts）：
```json
[
  {"type": "text", "text": "这张照片是今天拍的"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
]
```

### 请求示例

**Buffer 模式（默认）**
```bash
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Remind me to buy groceries tomorrow"}
    ],
    "user_id": "test_user",
    "process_mode": "buffer"
  }'
```

**Direct 模式**
```bash
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I had a meeting with the product team today. We discussed the Q2 roadmap."},
      {"role": "assistant", "content": "Got it! The Q2 roadmap discussion is noted."}
    ],
    "user_id": "test_user",
    "process_mode": "direct"
  }'
```

**Buffer + 立即刷新**
```bash
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tomorrow I need to prepare the quarterly report for the finance team."}
    ],
    "user_id": "test_user",
    "process_mode": "buffer",
    "flush_immediately": true
  }'
```

### 响应示例

**Buffer 模式成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "Chat messages pushed successfully",
  "data": {
    "count": 1
  }
}
```

**Direct 模式成功**
```json
{
  "code": 0,
  "status": 200,
  "message": "Chat messages submitted for processing",
  "data": {
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
      "time_bucket": "2026-03-08T21:50:46",
      "parent_id": "default",
      "title": "Discussion on birthday gift for girlfriend",
      "summary": "Discussed choosing a birthday gift and decided to add matching earring.\n- Selected item priced at 580 yuan for girlfriend's next-week birthday\n- Confirmed the item fits girlfriend's style preference\n- Decided to pair with a matching earring after AI suggestion",
      "event_time": "2026-03-08T21:50:46.393574+00:00",
      "create_time": "2026-03-08T21:50:46.378637",
      "is_search_hit": true,
      "children": [],
      "media_refs": [
        {"type": "image", "url": "https://example.com/media/image1.jpg", "media_index": 0},
        {"type": "video", "url": "https://example.com/media/video1.mp4", "media_index": 1}
      ],
      "keywords": ["birthday gift", "earring", "shopping"],
      "entities": ["earring"],
      "score": 0.108
    },
    {
      "id": "0f0c5487-561b-4950-ba7c-c244b5261adf",
      "hierarchy_level": 0,
      "time_bucket": "2026-03-08T22:09:04",
      "parent_id": "default",
      "title": "User Shared Photo and Received Feedback",
      "summary": "User shared a photo and got positive feedback on its quality.\n- Photo features a traditional decorated item\n- AI praised the photo's lighting and composition.",
      "event_time": "2026-03-08T22:09:04+00:00",
      "create_time": "2026-03-08T22:09:03.996846",
      "is_search_hit": true,
      "children": [],
      "media_refs": [
        {"type": "image", "url": "https://example.com/media/image2.jpg"}
      ],
      "keywords": ["photo feedback", "historical site"],
      "entities": [],
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
      "time_bucket": "2026-03-08T21:50:46",
      "parent_id": "default",
      "title": "Discussion on birthday gift for girlfriend",
      "summary": "...",
      "event_time": "2026-03-08T21:50:46.393574+00:00",
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

获取用户的记忆快照，包含 profile（用户画像）、近期事件摘要、今日事件、最近访问记录。支持 Redis 缓存，可按 section 过滤返回内容。

### 查询参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user_id` | `string` | **是** | — | 用户标识 |
| `device_id` | `string` | 否 | `"default"` | 设备标识 |
| `agent_id` | `string` | 否 | `"default"` | Agent 标识 |
| `include` | `string` | 否 | `"profile,events,accessed"` | 逗号分隔的返回 section：`profile` / `events` / `accessed` / `all` |
| `recent_days` | `int` | 否 | 配置值（默认 3） | 近期记忆时间窗口（天） |
| `max_recent_events_today` | `int` | 否 | 配置值（默认 5） | 今日最大 L0 事件数 |
| `max_accessed` | `int` | 否 | `5`（1-100） | 最近访问记录最大数量 |
| `force_refresh` | `bool` | 否 | `false` | 是否强制重建缓存（跳过 Redis） |

**`include` 参数说明：**
- `profile`：返回用户画像（`profile` 字段）
- `events`：返回每日摘要和今日事件（`daily_summaries` + `today_events` 字段）
- `accessed`：返回最近访问记录（`recently_accessed` 字段）
- `all`：返回所有 section
- 未包含的 section 对应字段为 `null`，已包含但无数据的为 `[]`

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

**全部参数**
```bash
curl "http://localhost:1733/api/memory-cache?user_id=test_user&device_id=iphone&agent_id=chatbot&include=profile,events&recent_days=7&max_accessed=3&force_refresh=true"
```

### 响应示例

**有数据的完整响应**
```json
{
  "success": true,
  "user_id": "test_user",
  "device_id": "default",
  "agent_id": "default",
  "profile": {
    "factual_profile": "用户是一名25岁的男性，喜欢摄影和传统文化。",
    "behavioral_profile": "沟通风格简洁直接，偏好中文交流。",
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
      "event_time": "2026-03-08T21:50:46.393574+00:00",
      "create_time": "2026-03-08T21:50:46.378637",
      "media_refs": []
    }
  ],
  "daily_summaries": [
    {
      "time_bucket": "2026-03-08",
      "title": "Daily Summary",
      "summary": "Discussed birthday gift selection and shared photos from the Forbidden City."
    }
  ],
  "today_events": [
    {
      "title": "Morning standup meeting",
      "summary": "Discussed sprint progress and blockers.",
      "event_time": "2026-03-11T09:00:00+00:00"
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
  "recently_accessed": null,
  "daily_summaries": [],
  "today_events": []
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
