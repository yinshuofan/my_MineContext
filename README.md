# MineContext

**开源的个人记忆后端服务** — 捕获、处理、存储和检索来自多种来源的上下文记忆，为 AI 应用提供持久化的用户记忆能力。

MineContext 将聊天记录、文档、网页链接等信息通过 LLM 分析和向量化，组织为 5 种类型的结构化记忆，支持语义搜索和层级化的时间索引，让 AI 应用能够真正"记住"用户。

## 核心特性

- **多源数据采集** — 聊天记录、文档（PDF/Word/Excel）、网页链接、活动记录等
- **5 种记忆类型** — Profile（用户画像）、Entity（实体）、Document（文档）、Event（事件）、Knowledge（知识），各有独立的更新策略和存储路由
- **双数据库架构** — 关系型数据库（MySQL/SQLite）存储 Profile 和 Entity；向量数据库（VikingDB/Qdrant/ChromaDB）存储 Document、Event、Knowledge
- **层级化事件索引** — 4 级时间层次（原始事件 → 日摘要 → 周摘要 → 月摘要），高效检索历史记忆
- **统一搜索 API** — 快速策略（零 LLM 调用）和智能策略（LLM 驱动）两种搜索模式
- **用户记忆缓存** — 单次调用获取用户完整记忆状态，Redis 快照缓存，适合下游服务快速接入
- **多用户支持** — 通过 `user_id`、`agent_id`、`device_id` 实现多租户隔离
- **Push API** — RESTful 推送接口，无需客户端 SDK，HTTP 调用即可接入

## 快速开始

### 环境要求

- Python >= 3.10
- MySQL（或 SQLite 用于本地开发）
- Redis
- LLM API（OpenAI 兼容格式，默认使用火山引擎豆包模型）

### 安装

推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
git clone https://github.com/volcengine/MineContext.git
cd MineContext
uv sync
```

也可使用传统 pip：

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### 配置

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 编辑 `.env`，配置必要的环境变量：

```bash
# LLM 配置（必填）
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_MODEL=doubao-seed-1-6-251015

# Embedding 配置（必填）
EMBEDDING_API_KEY=your-api-key
EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
EMBEDDING_MODEL=doubao-embedding-large-text-240915

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=opencontext

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

配置文件 `config/config.yaml` 支持 `${ENV_VAR:default}` 语法引用环境变量，大部分配置项都有合理的默认值。

### 启动服务

```bash
# 默认启动（端口 1733）
uv run opencontext start

# 指定配置文件和端口
uv run opencontext start --config config/config.yaml --port 1733

# 多进程模式
uv run opencontext start --workers 4
```

验证服务是否正常运行：

```bash
curl http://localhost:1733/api/health
```

### Docker 部署

```bash
# 单独启动
docker build -t minecontext .
docker run -p 1733:1733 --env-file .env minecontext

# 完整编排（包含 MySQL、Redis、Nginx）
docker-compose up
```

## API 概览

所有接口默认无需鉴权（可通过配置启用 API Key 鉴权）。

### 数据推送

将数据推送到 MineContext 进行处理和存储。

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/push/chat` | POST | 推送聊天消息（`process_mode: "buffer"` 或 `"direct"`） |
| `/api/push/document` | POST | 推送文档（本地路径或 Base64） |
| `/api/push/document/upload` | POST | 上传文档文件 |
| `/api/push/activity` | POST | 推送活动记录 |

**示例：推送聊天记录**

```bash
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "明天下午3点和张伟开会讨论项目进度"}]},
      {"role": "assistant", "content": [{"type": "text", "text": "好的，我已记录明天下午3点与张伟的项目进度会议。"}]}
    ],
    "user_id": "user_001",
    "agent_id": "default"
  }'
```

### 搜索

```bash
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "和张伟相关的会议",
    "strategy": "fast",
    "user_id": "user_001",
    "top_k": 10
  }'
```

| 参数 | 说明 |
|------|------|
| `query` | 搜索查询文本 |
| `strategy` | `fast`（零 LLM 调用，推荐）或 `intelligent`（LLM 驱动，更精准但较慢） |
| `user_id` | 用户标识 |
| `context_types` | 可选，限定搜索的记忆类型 |
| `top_k` | 每种类型的最大返回数 |

返回按类型分组的结果：`profile`、`entities`、`documents`、`events`、`knowledge`。

### 用户记忆缓存

一次调用获取用户完整的记忆状态快照，适合下游服务快速获取上下文。

```bash
curl "http://localhost:1733/api/memory-cache?user_id=user_001"
```

| 参数 | 说明 |
|------|------|
| `user_id` | 用户标识（必填） |
| `agent_id` | Agent 标识，默认 `default` |
| `recent_days` | 近期记忆天数，默认 7 |
| `max_accessed` | 最近访问记录数，默认 20 |
| `force_refresh` | 强制重建缓存，默认 false |

返回内容包含：

- **profile** — 用户画像
- **entities** — 已知实体（人物、项目、团队等）
- **recently_accessed** — 最近通过搜索访问的记忆（实时）
- **recent_memories** — 近期记忆（当日 L0 事件 + 历史日摘要 + 近期文档/知识）

快照缓存 5 分钟 TTL，数据写入时自动失效；最近访问记录独立存储，始终实时返回。

### 其他接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查（含组件状态） |
| `/api/context_types` | GET | 获取所有记忆类型定义 |
| `/api/agent/chat/stream` | POST | Agent 对话（流式 SSE） |
| `/api/agent/chat` | POST | Agent 对话（非流式） |
| `/api/agent/chat/conversations/list` | GET | 会话列表 |
| `/api/vector_search` | POST | 直接向量搜索 |
| `/api/memory-cache` | DELETE | 手动清除记忆缓存 |

## 架构

```
数据输入 → 处理器 → ProcessedContext → 路由（按类型分发） → 存储
                                            │
                 ┌──────────────────────────┤
                 │                          │
           关系型数据库                  向量数据库
        (Profile, Entity)      (Document, Event, Knowledge)
                 │                          │
                 └──────────┬───────────────┘
                            │
                     统一检索层
                   ┌────────┼────────┐
                 搜索API      记忆缓存     Agent对话
```

### 5 种记忆类型

| 类型 | 更新策略 | 存储 | 说明 |
|------|---------|------|------|
| `profile` | 覆写 | 关系型 DB | 用户偏好、习惯、沟通风格 |
| `entity` | 覆写 | 关系型 DB | 人物、项目、团队、组织 |
| `document` | 覆写 | 向量 DB | 上传的文件和网页链接 |
| `event` | 追加 | 向量 DB | 不可变的活动记录，支持 4 级时间层次 |
| `knowledge` | 追加合并 | 向量 DB | 可复用的概念、流程、模式，相似条目自动合并 |

### 事件层级索引

事件支持 4 级时间层次，通过定时任务自动生成摘要：

- **L0** — 原始事件
- **L1** — 日摘要（如 `2026-02-21`）
- **L2** — 周摘要（如 `2026-W08`）
- **L3** — 月摘要（如 `2026-02`）

搜索时自顶向下检索：先查摘要定位时间范围 → 钻取到原始事件 → 与直接语义搜索结果合并去重。

### 存储后端

**关系型数据库**（二选一）：
- MySQL — 生产推荐
- SQLite — 本地开发，零配置

**向量数据库**（三选一）：
- VikingDB — 默认，火山引擎托管
- Qdrant — 开源自建
- ChromaDB — 本地开发，零配置

通过 `config/config.yaml` 中的 `storage.vector_db.provider` 和 `storage.document_db.provider` 切换。

## 配置说明

主要配置项在 `config/config.yaml` 中，支持环境变量覆盖：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `llm.api_key` | LLM API Key | `${LLM_API_KEY}` |
| `llm.base_url` | LLM API 地址 | 火山引擎 |
| `llm.model` | LLM 模型名 | doubao-seed-1-6-251015 |
| `embedding.model` | 嵌入模型名 | doubao-embedding-large |
| `storage.vector_db.provider` | 向量数据库类型 | vikingdb |
| `storage.document_db.provider` | 关系型数据库类型 | mysql |
| `api_auth.enabled` | 是否启用 API 鉴权 | false |
| `api_auth.api_key` | API Key | `${CONTEXT_API_KEY}` |
| `memory_cache.snapshot_ttl` | 记忆缓存快照 TTL（秒） | 300 |
| `web.port` | 服务端口 | 1733 |

启用 API 鉴权后，请求需携带 `X-API-Key` 请求头或 `api_key` 查询参数。

## 典型接入流程

下游服务（如聊天机器人、AI 助手）接入 MineContext 的推荐流程：

```
1. 推送数据   →  POST /api/push/chat（每轮对话后推送）
2. 搜索记忆   →  POST /api/search（对话时检索相关记忆）
3. 获取缓存   →  GET /api/memory-cache（获取用户完整记忆状态）
```

搜索结果会自动记录到"最近访问"，下次通过记忆缓存即可获取。数据推送后缓存自动失效，确保下次读取到最新数据。

## 开发

```bash
# 安装依赖
uv sync

# 代码格式化
black opencontext --line-length 100
isort opencontext

# 安装 pre-commit 钩子（提交时自动格式化）
pre-commit install

# 编译检查（暂无测试套件）
python -m py_compile opencontext/path/to/file.py
```

详见 [CONTRIBUTING.md](CONTRIBUTING.md) 了解完整的贡献指南。

## 社区

- **GitHub Issues**: [提交问题或建议](https://github.com/volcengine/MineContext/issues)
- **Discord**: [加入社区](https://discord.gg/tGj7RQ3nUR)
- **飞书/微信群**: [加入群组](https://bytedance.larkoffice.com/wiki/Hg6VwrxnTiXtWUkgHexcFTqrnpg)

## 许可证

[Apache License 2.0](LICENSE)
