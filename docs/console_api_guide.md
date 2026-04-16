# Console 页面 API 速查表

前端开发参考文档。包含 Agent 调试台和记忆浏览器两个页面用到的全部 API 接口。

参考实现代码位于 `scripts/devtools/`。

---

## 通用约定

### 响应格式

大多数接口使用统一 wrapper：

```json
{
  "code": 0,
  "status": 200,
  "message": "success",
  "data": { ... }
}
```

- `code=0` 表示成功，非 0 表示错误
- `data` 字段仅在有数据时存在
- 例外接口（不使用此 wrapper）：`/api/memory-cache`、`/api/search`、`/api/media/upload`、`/contexts/detail`、`/contexts/delete`

### 认证

所有 `/api/*` 接口支持可选的 API Key 认证。启用时通过 `X-API-Key` header 传递。

---

## 页面 1: Agent 调试台

### 页面结构

```
左侧: Agent 列表 (190px)
右侧: [基本信息] [消息追踪] [关联 Contexts]
```

### 左侧 — Agent 列表

#### `GET /api/agents` — 获取 Agent 列表

```
GET /api/agents
```

**响应:**

```json
{
  "code": 0, "status": 200, "message": "success",
  "data": {
    "agents": [
      {
        "agent_id": "customer_service",
        "name": "客服助手",
        "description": "处理客户咨询",
        "created_at": "2026-04-10 14:30:00",
        "updated_at": "2026-04-15 09:00:00"
      }
    ]
  }
}
```

#### `POST /api/agents` — 创建 Agent

```json
POST /api/agents
{
  "agent_id": "my_agent",
  "name": "我的助手",
  "description": "可选描述"
}
```

**响应:** `{ "code": 0, "data": { "agent_id": "my_agent" } }`

#### `DELETE /api/agents/{agent_id}` — 删除 Agent

```
DELETE /api/agents/my_agent
```

**响应:** `{ "code": 0, "message": "Agent deleted" }`

---

### Tab 1: 基本信息

#### `GET /api/agents/{agent_id}` — 获取 Agent 详情

```
GET /api/agents/customer_service
```

**响应:**

```json
{
  "code": 0,
  "data": {
    "agent": {
      "agent_id": "customer_service",
      "name": "客服助手",
      "description": "处理客户咨询",
      "created_at": "2026-04-10 14:30:00",
      "updated_at": "2026-04-15 09:00:00"
    }
  }
}
```

#### `PUT /api/agents/{agent_id}` — 更新 Agent 名称/描述

```json
PUT /api/agents/customer_service
{
  "name": "新名称",
  "description": "新描述"
}
```

#### `GET /api/agents/{agent_id}/base/profile` — 获取 Base Profile

```
GET /api/agents/customer_service/base/profile
```

**响应:**

```json
{
  "code": 0,
  "data": {
    "profile": {
      "factual_profile": "你是一个专业的客服助手...",
      "behavioral_profile": null,
      "entities": [],
      "importance": 0,
      "created_at": "2026-04-10 14:30:00",
      "updated_at": "2026-04-15 09:00:00"
    }
  }
}
```

#### `POST /api/agents/{agent_id}/base/profile` — 保存 Base Profile

```json
POST /api/agents/customer_service/base/profile
{
  "factual_profile": "你是一个专业的客服助手，擅长处理用户投诉和产品咨询"
}
```

#### `GET /api/agents/{agent_id}/base/events` — 获取 Base Events

```
GET /api/agents/customer_service/base/events?limit=50&offset=0&hierarchy_level=0
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `limit` | int | 50 | 最大条数 |
| `offset` | int | 0 | 跳过条数 |
| `hierarchy_level` | int\|null | null | 0=事件, 1=日摘要, 2=周摘要, 3=月摘要; null=全部 |

**响应:**

```json
{
  "code": 0,
  "data": {
    "events": [
      {
        "id": "uuid",
        "title": "产品知识库更新",
        "summary": "更新了退款政策相关内容",
        "keywords": ["退款", "政策"],
        "entities": [],
        "context_type": "agent_base_event",
        "importance": 5,
        "hierarchy_level": 0,
        "refs": { "agent_base_l1_summary": ["parent_id"] },
        "event_time_start": "2026-04-10 14:30:00",
        "event_time_end": "2026-04-10 14:30:00",
        "create_time": "2026-04-10 14:30:00",
        "user_id": "__base__",
        "agent_id": "customer_service"
      }
    ]
  }
}
```

**层级树结构:** 事件通过 `refs` 字段建立父子关系。parent 的 refs 包含 child 类型和 ID 列表（向下引用），child 的 refs 包含 parent 类型和 ID（向上引用）。

#### `POST /api/agents/{agent_id}/base/events` — 添加 Base Events

```json
POST /api/agents/customer_service/base/events
{
  "events": [
    {
      "title": "产品知识库更新",
      "summary": "更新了退款政策相关内容",
      "keywords": ["退款", "政策"],
      "importance": 5,
      "hierarchy_level": 0,
      "event_time_start": "2026-04-10T14:30:00"
    }
  ]
}
```

支持嵌套 `children` 字段来批量导入层级结构。总数上限 500 条。

#### `DELETE /api/agents/{agent_id}/base/events/{event_id}` — 删除单个 Event

```
DELETE /api/agents/customer_service/base/events/uuid-here
```

---

### Tab 2: 消息追踪

#### `GET /api/chat-batches` — 获取消息批次列表

```
GET /api/chat-batches?agent_id=customer_service&page=1&limit=20
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `user_id` | string\|null | null | 按用户过滤 |
| `device_id` | string\|null | null | 按设备过滤 |
| `agent_id` | string\|null | null | 按 agent 过滤 |
| `start_date` | string\|null | null | 开始日期 |
| `end_date` | string\|null | null | 结束日期 |
| `page` | int | 1 | 页码 |
| `limit` | int | 20 | 每页条数 (1-100) |

**响应:**

```json
{
  "code": 0,
  "data": {
    "batches": [
      {
        "batch_id": "a1b2c3d4-e5f6-7890",
        "user_id": "user_001",
        "device_id": "default",
        "agent_id": "customer_service",
        "message_count": 12,
        "created_at": "2026-04-16 14:30:00"
      }
    ],
    "total": 42,
    "page": 1,
    "limit": 20,
    "total_pages": 3
  }
}
```

#### `GET /api/chat-batches/{batch_id}` — 获取批次消息详情

```
GET /api/chat-batches/a1b2c3d4-e5f6-7890
```

**响应:**

```json
{
  "code": 0,
  "data": {
    "batch": {
      "batch_id": "a1b2c3d4-e5f6-7890",
      "messages": [
        { "role": "user", "content": "你好" },
        { "role": "assistant", "content": "你好！有什么可以帮助你的？" },
        { "role": "user", "content": [
          { "type": "text", "text": "看看这张图" },
          { "type": "image_url", "image_url": { "url": "https://..." } }
        ]}
      ],
      "user_id": "user_001",
      "message_count": 12,
      "created_at": "2026-04-16 14:30:00"
    }
  }
}
```

`messages` 中每条消息的 `content` 可以是字符串（纯文本）或数组（多模态，包含 text/image_url/video_url 部分）。

#### `GET /api/chat-batches/{batch_id}/contexts` — 获取批次关联的 Contexts

```
GET /api/chat-batches/a1b2c3d4-e5f6-7890/contexts
```

**响应:** `{ "code": 0, "data": { "contexts": [ProcessedContext, ...] } }`

返回该批次消息处理后产生的 context 数据（事件、知识等）。

---

### Tab 3: 关联 Contexts

#### `GET /api/contexts` — 分页获取 Context 列表

```
GET /api/contexts?agent_id=customer_service&type=event&page=1&limit=15
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | string\|null | null | 按类型过滤 (event, knowledge, document, agent_base_event, ...) |
| `user_id` | string\|null | null | 按用户过滤 |
| `device_id` | string\|null | null | 按设备过滤 |
| `agent_id` | string\|null | null | 按 agent 过滤 |
| `hierarchy_level` | int\|null | null | 按层级过滤 (0-3) |
| `start_date` | string\|null | null | 开始日期 (`YYYY-MM-DD` 或 `YYYY-MM-DDThh:mm`) |
| `end_date` | string\|null | null | 结束日期 |
| `page` | int | 1 | 页码 |
| `limit` | int | 15 | 每页条数 (1-100) |

**响应:**

```json
{
  "code": 0,
  "data": {
    "contexts": [
      {
        "id": "uuid",
        "title": "讨论了退款流程",
        "summary": "用户咨询了关于订单退款的具体步骤...",
        "keywords": ["退款", "订单"],
        "entities": ["退款系统"],
        "context_type": "event",
        "importance": 7,
        "hierarchy_level": 0,
        "refs": {},
        "create_time": "2026-04-16 14:30:00",
        "event_time_start": "2026-04-16 14:30:00",
        "metadata": { "media_refs": [{ "type": "image", "url": "..." }] },
        "user_id": "user_001",
        "device_id": "default",
        "agent_id": "customer_service"
      }
    ],
    "page": 1,
    "limit": 15,
    "total": 142,
    "total_pages": 10,
    "context_types": ["event", "knowledge", "document", "daily_summary", "agent_base_event", "..."]
  }
}
```

**注意:** 响应不包含 `embedding` 和 `raw_contexts` 字段（轻量版）。自动排除 profile/agent_profile/agent_base_profile 类型。

---

## 页面 2: 记忆浏览器

### 页面结构

```
左侧: 用户列表 (190px) — 按 user_id 分组
右侧: 顶部 device/agent 选择器
       [记忆快照] [语义搜索] [Contexts 列表]
```

### 左侧 — 用户列表

#### `GET /api/users` — 获取用户三元组列表

> **注意:** 当前为 demo 实现，仅返回已生成画像的用户。生产环境应对接平台自身的用户系统。

```
GET /api/users
```

**响应:**

```json
{
  "code": 0,
  "data": {
    "users": [
      { "user_id": "user_001", "device_id": "default", "agent_id": "default" },
      { "user_id": "user_001", "device_id": "mobile", "agent_id": "customer_service" },
      { "user_id": "user_002", "device_id": "default", "agent_id": "default" }
    ],
    "total": 3
  }
}
```

前端按 `user_id` 去重展示用户列表。选中用户后，从该用户的组合中提取 `device_id` 和 `agent_id` 作为下拉选项。

---

### Tab 1: 记忆快照

#### `GET /api/memory-cache` — 获取用户记忆快照

> **注意:** 此接口不使用统一 wrapper，直接返回 Pydantic model。

```
GET /api/memory-cache?user_id=user_001&device_id=default&agent_id=default
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user_id` | string | **是** | — | 用户 ID |
| `device_id` | string | 否 | "default" | 设备 ID |
| `agent_id` | string | 否 | "default" | Agent ID |
| `include` | string\|null | 否 | null (全部) | 逗号分隔: `profile,agent_prompt,events,accessed` |
| `recent_days` | int | 否 | 配置默认值 | 近 N 天 (1-90) |
| `max_recent_events_today` | int | 否 | 配置默认值 | 今日事件上限 |
| `max_accessed` | int | 否 | 5 | 最近访问上限 (1-100) |
| `force_refresh` | bool | 否 | false | 强制刷新缓存 |

**响应:**

```json
{
  "success": true,
  "user_id": "user_001",
  "device_id": "default",
  "agent_id": "default",
  "profile": {
    "factual_profile": "偏好简洁的沟通风格，关注数据分析领域...",
    "behavioral_profile": null,
    "metadata": {}
  },
  "agent_prompt": {
    "factual_profile": "Based on user preferences: use concise language...",
    "behavioral_profile": "Respond in a professional yet approachable tone...",
    "metadata": {}
  },
  "recently_accessed": [
    {
      "id": "uuid",
      "title": "退款政策要点",
      "summary": "退款需在 7 日内...",
      "context_type": "knowledge",
      "keywords": ["退款"],
      "accessed_ts": 1712345678.9,
      "score": 0.95,
      "event_time_start": "2026-04-16T14:30:00",
      "media_refs": [{ "type": "image", "url": "..." }]
    }
  ],
  "today_events": [
    {
      "title": "讨论了 Q2 数据报表",
      "summary": "用户询问了报表格式要求...",
      "event_time_start": "2026-04-16T14:30:00"
    }
  ],
  "daily_summaries": [
    {
      "title": "2026-04-15 日摘要",
      "summary": "主要处理数据可视化任务...",
      "event_time_start": "2026-04-15T00:00:00"
    }
  ]
}
```

各 section 为 null 表示未请求或无数据。

---

### Tab 2: 语义搜索

#### `POST /api/search` — 语义搜索

> **注意:** 此接口不使用统一 wrapper，直接返回 Pydantic model。

```json
POST /api/search
{
  "query": [
    { "type": "text", "text": "退款流程" },
    { "type": "image_url", "image_url": { "url": "https://..." } }
  ],
  "top_k": 10,
  "drill": "up",
  "hierarchy_levels": [0, 1],
  "user_id": "user_001",
  "device_id": "default",
  "agent_id": "default",
  "time_range": { "start": 1712345678, "end": 1712432078 }
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | array\|null | 否* | null | 多模态查询 (text/image_url/video_url 部分) |
| `event_ids` | string[]\|null | 否* | null | 精确 ID 查找 |
| `time_range` | object\|null | 否* | null | `{start: unix_ts, end: unix_ts}` |
| `hierarchy_levels` | int[]\|null | 否* | null | 层级过滤 [0,1,2,3] |
| `drill` | string | 否 | "up" | 钻取方向: none/up/down/both |
| `top_k` | int | 否 | 10 | 最大结果数 (1-100) |
| `user_id` | string\|null | 否 | null | 用户过滤 |
| `device_id` | string\|null | 否 | null | 设备过滤 |
| `agent_id` | string\|null | 否 | null | Agent 过滤 |

*至少提供 query/event_ids/time_range/hierarchy_levels 之一。

**响应:**

```json
{
  "success": true,
  "events": [
    {
      "id": "uuid",
      "title": "用户咨询退款",
      "summary": "用户询问了订单退款的具体步骤...",
      "context_type": "event",
      "hierarchy_level": 0,
      "refs": { "daily_summary": ["parent_id"] },
      "is_search_hit": true,
      "score": 0.95,
      "keywords": ["退款", "订单"],
      "entities": ["退款系统"],
      "agent_commentary": "用户情绪较为焦虑",
      "media_refs": [{ "type": "image", "url": "..." }],
      "event_time_start": "2026-04-16T14:30:00",
      "event_time_end": "2026-04-16T15:00:00",
      "children": [
        {
          "id": "child_uuid",
          "title": "日摘要",
          "hierarchy_level": 1,
          "is_search_hit": false,
          "score": null,
          "children": []
        }
      ]
    }
  ],
  "metadata": {
    "query": "退款流程",
    "total_results": 10,
    "search_time_ms": 123.45
  }
}
```

结果为树形结构。`is_search_hit=true` 的节点是搜索命中项（有 score/keywords），`is_search_hit=false` 的节点是通过 drill 展开的祖先/后代（关联节点）。

#### `POST /api/media/upload` — 上传媒体文件

> **注意:** 此接口不使用统一 wrapper。

```
POST /api/media/upload
Content-Type: multipart/form-data

file: (binary)
```

支持格式: jpg/jpeg/png/gif/webp/bmp (图片, <10MB), mp4/avi/mov/webm (视频, <50MB)

**响应:**

```json
{
  "url": "https://... 或 /uploads/local/path",
  "type": "image",
  "filename": "photo.jpg"
}
```

---

### Tab 3: Contexts 列表

使用与 Agent 调试台 Tab 3 相同的 `GET /api/contexts` 接口，区别是按用户三元组过滤而非 agent_id。

```
GET /api/contexts?user_id=user_001&device_id=default&agent_id=default&page=1&limit=15
```

---

## Context 类型说明

| 类型 | 说明 | 存储位置 |
|------|------|----------|
| `event` | 用户活动事件 | 向量库 |
| `knowledge` | 可复用的知识/概念 | 向量库 |
| `document` | 上传的文档/链接 | 向量库 |
| `daily_summary` | 日摘要 (L1) | 向量库 |
| `weekly_summary` | 周摘要 (L2) | 向量库 |
| `monthly_summary` | 月摘要 (L3) | 向量库 |
| `agent_base_event` | Agent 预置事件 (L0) | 向量库 |
| `agent_base_l1_summary` | Agent 预置日摘要 | 向量库 |
| `agent_base_l2_summary` | Agent 预置周摘要 | 向量库 |
| `agent_base_l3_summary` | Agent 预置月摘要 | 向量库 |

---

## 跨页跳转

| 场景 | URL |
|------|-----|
| 从 Agent 页跳转到记忆浏览器查看某用户 | `/console/memory?user_id=xxx&device_id=yyy&agent_id=zzz` |
| 从记忆浏览器跳转到 Agent 页查看某 Agent | `/console/agents?agent_id=xxx` |

两个页面都支持 URL 参数深度链接，加载时自动选中对应项。
