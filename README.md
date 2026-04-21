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
- **Web 管理界面** — 内置 Web UI，提供记忆浏览、事件搜索、记忆缓存可视化、系统监控、设置管理等功能
- **监控面板** — 实时监控处理指标、Token 用量、调度器状态、上下文统计等
- **定时任务调度** — 基于 Redis 的分布式任务调度，支持事件层级摘要生成、记忆压缩、数据清理等
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

启动后访问 `http://localhost:1733` 进入 Web 管理界面。

### Docker 部署

Docker Compose 以「HTTPS 反代 + 应用 + 可选内置数据库」三层编排：

| 服务 | 说明 | 默认状态 |
|------|------|---------|
| `caddy` | HTTPS 反向代理，自动申请/续期 Let's Encrypt 证书 | 启动 |
| `server` | MineContext 主服务（API + Web UI，内嵌定时调度器） | 启动（1 容器 × N workers） |
| `mysql` | 内置 MySQL 8.0 | 仅 `test` profile |
| `redis` | 内置 Redis | 仅 `test` profile |

> `mysql`、`redis` 属于 `test` profile，**默认不启动**——生产部署使用外部数据库，本地测试时加 `--profile test` 启动内置实例。

#### 部署步骤

**1. 配置环境变量**

```bash
cp .env.example .env
# 编辑 .env，填入 LLM / Embedding / MySQL / Redis / CONTEXT_API_KEY 等
```

**2. 配置 Caddy 域名**

编辑项目根目录的 `Caddyfile`，把示例域名替换成你自己的：

```caddyfile
your.domain.com {
    reverse_proxy server:1733 {
        flush_interval -1
    }
    request_body { max_size 50MB }
    encode gzip zstd
}
```

**3. 前置准备**

- 域名 A 记录指向服务器公网 IP（Let's Encrypt HTTP-01 签证书必需）
- 服务器防火墙/安全组放通 **80** 和 **443**（80 用于证书签发）
- 宿主机 80 端口未被其他进程占用

**4. 启动**

```bash
# 生产模式（外部 MySQL/Redis）
docker compose up -d --build

# 本地测试（内置 MySQL/Redis）
docker compose --profile test up -d --build
```

**5. 验证**

```bash
# 观察证书签发过程
docker compose logs -f caddy
# 出现 "certificate obtained successfully" 即成功

# 健康检查（把 DOMAIN 换成你在 Caddyfile 中配置的域名）
DOMAIN=your.domain.com
curl https://$DOMAIN/health
curl -H "X-API-Key: $CONTEXT_API_KEY" https://$DOMAIN/api/health
```

#### 仅 HTTP（开发 / 内网部署）

没有域名、只想 HTTP 直连的场景：

1. 在 `docker-compose.yml` 里注释掉 `caddy` 服务
2. 把 `server` 的 `expose: ["1733"]` 改回 `ports: ["1733:1733"]`
3. `docker compose up -d --build`
4. 直接访问 `http://<server-ip>:1733`

#### 扩缩容

通过 `SERVER_WORKERS` 控制单容器内 Uvicorn worker 数（默认 2）：

```bash
SERVER_WORKERS=4 docker compose up -d
```

> 当前 compose 中 `deploy.replicas: 1` 固定为单容器。需要多容器横向扩展时，须手动调整 compose 文件并在 Caddy 中配置多个上游（`reverse_proxy server1:1733 server2:1733 ...`）。

#### 热加载 Caddy 配置

改完 `Caddyfile` 不用重建容器：

```bash
docker compose exec caddy caddy reload --config /etc/caddy/Caddyfile
```

#### 证书持久化

证书存储在 Docker 卷 `caddy_data` 中，Caddy 自动在到期前 30 天续签。**不要删除这个卷**——删了下次启动会重新申请，可能撞上 Let's Encrypt 速率限制。备份时把 `caddy_data` 一并备上。

### 部署架构

#### 单机开发模式

直接运行 `uv run opencontext start`，适合本地开发：

```
客户端 → MineContext (端口 1733)
              ├── MySQL / SQLite
              └── Redis
```

#### Docker Compose 生产模式

Caddy 终止 TLS，反代到内网 server 容器：

```
                     HTTPS (443)
客户端 ─────────────────────────────> Caddy
                                        │ HTTP (内网)
                                        ▼
                                   MineContext
                                 (1 容器 × N workers)
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 │                      │                      │
              MySQL                  Redis              Vector DB
         (Profile, Entity)     (调度器 / 缓存)   (Document/Event/Knowledge)
```

- Caddy 自动申请并续期 Let's Encrypt 证书
- `server` 容器 `ports` 已换成 `expose`，**不再对公网直接暴露 1733**，Caddy 是唯一外部入口
- uvicorn 以 `proxy_headers=True, forwarded_allow_ips="*"` 启动，后端能拿到真实客户端 IP 和 `https` 协议（经 `X-Forwarded-For` / `X-Forwarded-Proto`）
- 定时调度器内嵌在 server 容器中（`SCHEDULER_ENABLED=true`），多 worker 场景下的竞态由 Redis 原子操作保证

## API 概览

所有接口默认无需鉴权（可通过配置启用 API Key 鉴权）。

### 数据推送

将数据推送到 MineContext 进行处理和存储。

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/push/chat` | POST | 推送聊天消息（`process_mode: "buffer"` 或 `"direct"`） |
| `/api/push/document` | POST | 推送文档（本地路径或 Base64） |
| `/api/push/document/upload` | POST | 上传文档文件 |

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
| `device_id` | 设备标识，默认 `default` |
| `agent_id` | Agent 标识，默认 `default` |
| `include` | 逗号分隔的 section：`profile,agent_prompt,events,accessed,all`，默认全部 |
| `recent_days` | 日摘要向前追溯的天数，默认 3 |
| `max_recent_events_today` | 今日事件上限，默认 5 |
| `max_accessed` | 最近访问记录数，默认 5 |
| `force_refresh` | 强制重建缓存，默认 false |

响应顶层字段（每个字段与 include section 一一对应，未请求时为 `null`）：

- **profile** — 用户自己的画像（`factual_profile` / `behavioral_profile` / `metadata`）。属 `profile` section。
- **agent_prompt** — Agent 的提示词 / 画像。`agent_id != "default"` 时查 `agent_profile`；缺失时自动 fallback 到 `agent_base_profile`（`user_id="__base__"`）。属 `agent_prompt` section。
- **today_events** — 今日 L0 事件列表。属 `events` section。
- **daily_summaries** — 历史日摘要列表。属 `events` section。
- **recently_accessed** — 最近通过搜索访问的记忆（实时、与 snapshot 去重）。属 `accessed` section。

快照缓存 TTL 默认 1 小时（可配置），上游写入新数据时主动失效；最近访问记录独立存储于 Redis Hash，始终实时返回。

### 其他接口

**健康检查与认证**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 基础健康检查 |
| `/api/health` | GET | 详细健康检查（含组件状态） |
| `/api/ready` | GET | 就绪探针 |
| `/api/auth/status` | GET | 认证状态 |

**Agent 对话**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/agent/chat` | POST | Agent 对话（非流式） |
| `/api/agent/chat/stream` | POST | Agent 对话（流式 SSE） |
| `/api/agent/resume/{workflow_id}` | POST | 恢复工作流 |
| `/api/agent/state/{workflow_id}` | GET | 获取工作流状态 |
| `/api/agent/cancel/{workflow_id}` | DELETE | 取消工作流 |

**会话管理**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/agent/chat/conversations` | POST | 创建会话 |
| `/api/agent/chat/conversations/list` | GET | 会话列表 |
| `/api/agent/chat/conversations/{id}` | GET | 会话详情 |
| `/api/agent/chat/conversations/{id}/update` | PATCH / DELETE | 更新/删除会话 |
| `/api/agent/chat/conversations/{id}/messages` | GET | 会话消息列表 |

**文档与 Vault**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/documents/upload` | POST | 上传文档（路径） |
| `/api/weblinks/upload` | POST | 上传网页链接 |
| `/api/vaults/list` | GET | Vault 文档列表 |
| `/api/vaults/create` | POST | 创建 Vault 文档 |
| `/api/vaults/{id}` | GET / POST / DELETE | 获取/更新/删除 Vault 文档 |
| `/api/vaults/{id}/context` | GET | 文档上下文状态 |

**监控**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/monitoring/overview` | GET | 系统概览 |
| `/api/monitoring/context-types` | GET | 上下文类型统计 |
| `/api/monitoring/token-usage` | GET | Token 用量 |
| `/api/monitoring/processing` | GET | 处理指标 |
| `/api/monitoring/stage-timing` | GET | 阶段耗时 |
| `/api/monitoring/data-stats` | GET | 数据统计 |
| `/api/monitoring/scheduler` | GET | 调度器执行摘要 |
| `/api/monitoring/scheduler/queues` | GET | 调度器队列深度 |
| `/api/monitoring/scheduler/failures` | GET | 调度器失败率 |
| `/api/monitoring/trigger-task` | POST | 手动触发任务（测试用） |

**设置管理**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/model_settings/get` | GET | 获取模型设置（LLM/VLM/Embedding） |
| `/api/model_settings/update` | POST | 更新模型设置 |
| `/api/model_settings/validate` | POST | 验证模型设置（不保存） |
| `/api/settings/general` | GET / POST | 获取/更新通用设置 |
| `/api/settings/prompts` | GET / POST | 获取/更新 Prompt 模板 |
| `/api/settings/prompts/language` | GET / POST | 获取/切换 Prompt 语言 |
| `/api/settings/prompts/import` | POST | 导入 Prompt（文件上传） |
| `/api/settings/prompts/export` | GET | 导出 Prompt（YAML 下载） |
| `/api/settings/reset` | POST | 重置所有设置为默认值 |

**其他**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/context_types` | GET | 获取所有记忆类型定义 |
| `/api/contexts/{id}` | GET | 获取上下文详情 |
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
- Qdrant — 开源自建，支持本地模式和远程服务器模式
- ChromaDB — 本地开发，零配置

通过 `config/config.yaml` 中的存储后端配置切换，取消对应后端的注释即可。

## 配置说明

### 主要配置项

配置文件 `config/config.yaml` 支持 `${ENV_VAR:default}` 语法引用环境变量：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `llm.config.api_key` | LLM API Key | `${LLM_API_KEY}` |
| `llm.config.base_url` | LLM API 地址 | 火山引擎 |
| `llm.config.model` | LLM 模型名 | doubao-seed-1-6-251015 |
| `embedding_model.model` | 嵌入模型名 | doubao-embedding-large-text-240915 |
| `embedding_model.output_dim` | 嵌入维度 | 2048 |
| `api_auth.enabled` | 是否启用 API 鉴权 | false |
| `api_auth.api_keys` | API Key 列表 | `${CONTEXT_API_KEY}` |
| `memory_cache.snapshot_ttl` | 记忆缓存快照 TTL（秒） | 3600 |
| `scheduler.enabled` | 是否启用任务调度器 | true |
| `web.port` | 服务端口 | 1733 |

启用 API 鉴权后，请求需携带 `X-API-Key` 请求头或 `api_key` 查询参数。

### Docker 环境变量

Docker Compose 部署时，以下环境变量可在 `.env` 中配置：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `LLM_API_KEY` | LLM API Key（必填） | — |
| `LLM_BASE_URL` | LLM API 地址（必填） | — |
| `LLM_MODEL` | LLM 模型名（必填） | — |
| `EMBEDDING_API_KEY` | Embedding API Key（必填） | — |
| `EMBEDDING_BASE_URL` | Embedding API 地址（必填） | — |
| `EMBEDDING_MODEL` | Embedding 模型名（必填） | — |
| `VLM_API_KEY` | VLM API Key（图片处理时必填） | — |
| `VLM_BASE_URL` | VLM API 地址 | — |
| `VLM_MODEL` | VLM 模型名 | — |
| `MYSQL_HOST` | MySQL 地址 | mysql |
| `MYSQL_PASSWORD` | MySQL 密码（必填） | — |
| `REDIS_HOST` | Redis 地址 | redis |
| `CONTEXT_API_KEY` | API 鉴权 Key | — |
| `SERVER_WORKERS` | 每容器 Uvicorn worker 数 | 2 |
| `MYSQL_PORT_PUBLISHED` | MySQL 外部映射端口（test profile） | 3307 |
| `REDIS_PORT_PUBLISHED` | Redis 外部映射端口（test profile） | 6380 |
| `SCHEDULER_EXECUTOR_MAX_CONCURRENT` | 调度器最大并发任务数 | 5 |

> - Caddy 的域名在 `Caddyfile` 中直接配置，不走环境变量。
> - `SCHEDULER_ENABLED` 在 `docker-compose.yml` 中对 server 硬编码为 `"true"`，`.env` 里的同名变量对 Docker 部署无效；需要关闭调度器须直接改 compose 文件。

### 存储后端选择指南

| 场景 | 关系型数据库 | 向量数据库 | 说明 |
|------|-------------|-----------|------|
| 本地开发/体验 | SQLite | ChromaDB | 零依赖，开箱即用 |
| 生产环境（火山引擎） | MySQL | VikingDB | 默认配置，全托管 |
| 生产环境（自建） | MySQL | Qdrant | 开源自建，本地或远程部署 |

切换方式：在 `config/config.yaml` 的 `storage.backends` 中注释/取消注释对应的后端配置块。注意向量数据库的 `dimension`/`vector_size` 须与 `embedding_model.output_dim` 一致。

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
uv run black opencontext --line-length 100
uv run isort opencontext

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
