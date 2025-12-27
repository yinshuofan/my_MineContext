# Redis 缓存改造说明

## 概述

本次改造将项目中的缓存组件改为使用 Redis，以支持多服务实例部署时共享用户数据。

## 改造内容

### 1. 新增 Redis 缓存抽象层

**文件**: `opencontext/storage/redis_cache.py`

提供统一的 Redis 缓存接口，支持：

- **基础操作**: get/set/delete/exists/expire/ttl
- **JSON 操作**: get_json/set_json（自动序列化/反序列化）
- **列表操作**: lpush/rpush/lpop/rpop/lrange/llen/ltrim
- **哈希操作**: hget/hset/hmset/hgetall/hdel
- **集合操作**: sadd/srem/sismember/smembers
- **原子操作**: incr/decr/getset
- **分布式锁**: acquire_lock/release_lock

**降级机制**: 当 Redis 不可用时，自动使用 `InMemoryCache` 作为本地降级方案。

### 2. TextChatCapture 改造

**文件**: `opencontext/context_capture/text_chat.py`

改造内容：
- 聊天消息缓冲区存储在 Redis 中（key: `opencontext:chat:buffer:{user_id}:{device_id}:{agent_id}`）
- 使用分布式锁确保 flush 操作的原子性
- 支持缓冲区 TTL 配置（默认 24 小时）
- 自动降级到本地内存缓存

### 3. CompletionCache 改造

**文件**: `opencontext/context_consumption/completion/completion_cache.py`

改造内容：
- 补全结果缓存存储在 Redis 中
- 热点 key 和访问顺序在 Redis 中跟踪
- 预计算上下文存储在 Redis 中
- 支持 LRU/TTL/Hybrid 缓存策略
- 自动降级到本地缓存

## 配置说明

### Redis 配置

在 `config/config_mysql.yaml` 中添加：

```yaml
# Redis cache configuration (for multi-instance deployment)
redis:
  enabled: true
  host: "${REDIS_HOST:localhost}"
  port: ${REDIS_PORT:6379}
  password: "${REDIS_PASSWORD:}"
  db: ${REDIS_DB:0}
  key_prefix: "opencontext:"
  max_connections: 10
  socket_timeout: 5.0
  socket_connect_timeout: 5.0
  retry_on_timeout: true
```

### TextChatCapture Redis 配置

```yaml
capture:
  text_chat:
    enabled: true
    buffer_size: 10
    buffer_ttl: 86400  # Buffer TTL in seconds (24 hours)
    redis:
      enabled: true  # Use Redis for multi-instance support
```

### CompletionCache Redis 配置

```yaml
completion:
  enabled: true
  cache:
    max_size: 1000
    ttl_seconds: 300
    strategy: "hybrid"  # Options: lru, ttl, hybrid
    redis:
      enabled: true  # Use Redis for multi-instance cache sharing
```

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `REDIS_HOST` | localhost | Redis 服务器地址 |
| `REDIS_PORT` | 6379 | Redis 端口 |
| `REDIS_PASSWORD` | (空) | Redis 密码 |
| `REDIS_DB` | 0 | Redis 数据库编号 |

## 依赖

在 `pyproject.toml` 中添加：

```toml
dependencies = [
    ...
    "redis>=4.0.0",
]
```

## 使用方式

### 启动 Redis

```bash
# 使用 Docker 启动 Redis
docker run -d --name redis -p 6379:6379 redis:latest

# 或使用系统包管理器安装
sudo apt install redis-server
sudo systemctl start redis
```

### 启动服务

```bash
# 设置环境变量
export REDIS_HOST=localhost
export REDIS_PORT=6379

# 启动后端服务
python -m opencontext.cli start --config config/config_mysql.yaml
```

### 多实例部署

```bash
# 实例 1
python -m opencontext.cli start --config config/config_mysql.yaml --port 1733

# 实例 2
python -m opencontext.cli start --config config/config_mysql.yaml --port 1734

# 实例 3
python -m opencontext.cli start --config config/config_mysql.yaml --port 1735
```

所有实例共享同一个 Redis，用户数据可以在任意实例上处理。

## Redis Key 命名规范

| Key 模式 | 说明 |
|----------|------|
| `opencontext:chat:buffer:{user}:{device}:{agent}` | 聊天消息缓冲区 |
| `opencontext:completion:cache:{key}` | 补全结果缓存 |
| `opencontext:completion:access_order` | 补全缓存访问顺序 |
| `opencontext:completion:hot_keys` | 补全缓存热点 key |
| `opencontext:completion:stats:{stat}` | 补全缓存统计 |
| `opencontext:completion:precomputed:{doc_id}` | 预计算上下文 |
| `opencontext:lock:{name}` | 分布式锁 |

## 注意事项

1. **降级机制**: 当 Redis 不可用时，系统会自动降级到本地内存缓存，但此时多实例间无法共享数据。

2. **TTL 设置**: 建议根据业务需求合理设置缓存 TTL，避免 Redis 内存占用过高。

3. **分布式锁**: TextChatCapture 的 flush 操作使用分布式锁，确保同一用户的消息不会被多个实例同时处理。

4. **监控**: 建议监控 Redis 的内存使用和连接数，确保服务稳定运行。
