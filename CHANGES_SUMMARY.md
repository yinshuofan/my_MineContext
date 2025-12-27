# 长期记忆后端服务改造说明

本次改造将 MineContext 项目转变为一个长期记忆后端服务，主要包含两个核心变更：

## 1. MySQL 后端支持

### 新增文件
- `opencontext/storage/backends/mysql_backend.py` - MySQL 后端完整实现

### 修改文件
- `opencontext/storage/backends/__init__.py` - 导出 MySQLBackend
- `opencontext/storage/unified_storage.py` - 添加 MySQL 后端工厂方法
- `pyproject.toml` - 添加 pymysql 依赖

### 功能特性
- 完整实现 `IDocumentStorageBackend` 接口
- 支持所有核心表：vaults, todo, activity, tips, conversations, messages, message_thinking
- 监控表：monitoring_token_usage, monitoring_stage_timing, monitoring_data_stats
- 自动创建数据库和表结构
- 连接自动重连机制

### 配置方式
在 `config.yaml` 中配置 MySQL 后端：

```yaml
storage:
  backends:
    - name: "document_store"
      storage_type: "document_db"
      backend: "mysql"
      default: true
      config:
        host: "${MYSQL_HOST:localhost}"
        port: ${MYSQL_PORT:3306}
        user: "${MYSQL_USER:root}"
        password: "${MYSQL_PASSWORD:}"
        database: "${MYSQL_DATABASE:opencontext}"
        charset: "utf8mb4"
```

## 2. Push API（外部推送模式）

### 新增文件
- `opencontext/server/routes/push.py` - Push API 路由
- `config/config_mysql.yaml` - MySQL + Push 模式配置示例

### 修改文件
- `opencontext/server/api.py` - 注册 push 路由

### API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/push/chat/message` | POST | 推送单条聊天消息 |
| `/api/push/chat/messages` | POST | 批量推送聊天消息 |
| `/api/push/chat/flush` | POST | 手动刷新聊天缓冲区 |
| `/api/push/screenshot` | POST | 推送截图（支持路径或 base64） |
| `/api/push/screenshots` | POST | 批量推送截图 |
| `/api/push/document` | POST | 推送文档 |
| `/api/push/document/upload` | POST | 上传文档文件（multipart） |
| `/api/push/activity` | POST | 推送活动记录 |
| `/api/push/todo` | POST | 推送待办事项 |
| `/api/push/tip` | POST | 推送提示 |
| `/api/push/context` | POST | 推送通用上下文 |
| `/api/push/batch` | POST | 批量推送多种类型数据 |

### 请求示例

#### 推送聊天消息
```json
POST /api/push/chat/message
{
  "role": "user",
  "content": "Hello, world!",
  "user_id": "user123",
  "device_id": "device456",
  "agent_id": "agent789"
}
```

#### 推送截图（base64）
```json
POST /api/push/screenshot
{
  "base64_data": "iVBORw0KGgoAAAANSUhEUgAA...",
  "filename": "screenshot.png",
  "window_title": "Chrome - Google",
  "app_name": "chrome",
  "user_id": "user123"
}
```

#### 推送活动记录
```json
POST /api/push/activity
{
  "title": "编写代码",
  "content": "完成了用户认证模块的开发",
  "start_time": "2025-01-01T09:00:00",
  "end_time": "2025-01-01T12:00:00",
  "resources": ["/path/to/file.py"],
  "user_id": "user123"
}
```

#### 批量推送
```json
POST /api/push/batch
{
  "user_id": "user123",
  "device_id": "device456",
  "items": [
    {
      "type": "chat",
      "data": {"role": "user", "content": "Hello"}
    },
    {
      "type": "activity",
      "data": {"title": "Meeting", "content": "Team standup"}
    },
    {
      "type": "todo",
      "data": {"content": "Review PR", "urgency": 2}
    }
  ]
}
```

## 3. 服务模式配置

使用 `config/config_mysql.yaml` 作为服务模式的配置模板：

```bash
# 启动服务
python -m opencontext.cli start --config config/config_mysql.yaml
```

### 关键配置变更

1. **禁用自动捕获组件**
   ```yaml
   capture:
     screenshot:
       enabled: false
     folder_monitor:
       enabled: false
     file_monitor:
       enabled: false
     vault_document_monitor:
       enabled: false
   ```

2. **启用 API 认证**
   ```yaml
   api_auth:
     enabled: true
     api_keys:
       - "${CONTEXT_API_KEY:your-api-key-here}"
   ```

3. **监听所有接口**
   ```yaml
   web:
     host: "0.0.0.0"
     port: 1733
   ```

## 4. 架构变更说明

### 原架构（Pull 模式）
```
[本地截图/文件] → [自动捕获组件] → [处理器] → [存储]
```

### 新架构（Push 模式）
```
[外部服务] → [Push API] → [处理器] → [存储]
                ↓
         [MySQL 后端]
```

### 优势
1. **解耦合** - 数据采集与存储服务分离
2. **可扩展** - 支持多个外部服务同时推送
3. **多用户** - 通过 user_id/device_id/agent_id 区分用户
4. **高可用** - MySQL 支持主从复制、集群部署

## 5. 环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `MYSQL_HOST` | MySQL 主机地址 | localhost |
| `MYSQL_PORT` | MySQL 端口 | 3306 |
| `MYSQL_USER` | MySQL 用户名 | root |
| `MYSQL_PASSWORD` | MySQL 密码 | (空) |
| `MYSQL_DATABASE` | 数据库名称 | opencontext |
| `CONTEXT_API_KEY` | API 认证密钥 | your-api-key-here |

## 6. 迁移指南

### 从 SQLite 迁移到 MySQL

1. 安装 MySQL 服务器
2. 创建数据库用户和权限
3. 修改配置文件使用 MySQL 后端
4. 启动服务（自动创建表结构）
5. 使用数据迁移脚本迁移历史数据（如需要）

### 从 Pull 模式迁移到 Push 模式

1. 修改外部服务，调用 Push API 推送数据
2. 禁用配置文件中的自动捕获组件
3. 启用 API 认证保护接口安全
