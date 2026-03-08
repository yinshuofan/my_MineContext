# 对象存储媒体管理设计

> Date: 2026-03-08
> Status: Draft
> Depends on: 2026-03-08-vikingdb-v2-multimodal-design.md

## 1. 问题

多模态消息中的 base64 图片/视频当前被保存为本地文件路径。远程 API（LLM、Ark Embedding）无法访问本地路径。多实例部署下文件不共享。

## 2. 方案

引入对象存储层。在 push.py 最早阶段将 base64 上传到对象存储，全管线使用 HTTPS URL。

```
Push API (base64 / HTTP URL)
  → push.py: 上传到对象存储，获得 HTTPS URL
  → Redis buffer / direct: 消息中只存 URL 字符串
  → LLM: 用 HTTPS URL
  → Ark Embedding: 用 HTTPS URL
  → VikingDB: 存 URL 作为 media_ref
```

## 3. 抽象层设计

### 接口

```python
class IObjectStorage(ABC):
    async def upload(self, data: bytes, key: str, content_type: str) -> str:
        """上传，返回 HTTPS URL"""

    def get_url(self, key: str) -> str:
        """获取对象的 HTTPS 访问 URL"""

    async def delete(self, key: str) -> bool:
        """删除对象"""

    async def close(self) -> None:
        """清理资源"""
```

### 实现

| 实现 | 场景 | URL 格式 |
|------|------|----------|
| `S3CompatibleBackend` | 生产环境（任意 S3 兼容服务） | `https://{bucket}.{endpoint}/{key}` |
| `LocalBackend` | 开发环境 | 存本地文件，API 调用时转 base64 data URI |

### S3 兼容覆盖范围

S3 协议是对象存储的事实标准。一个 `S3CompatibleBackend` 通过配置不同 endpoint 即可覆盖：

| 云厂商 | endpoint 配置 | 示例 URL |
|--------|--------------|----------|
| Volcengine TOS | `tos-cn-beijing.volces.com` | `https://bucket.tos-cn-beijing.volces.com/key` |
| 阿里云 OSS | `oss-cn-hangzhou.aliyuncs.com` | `https://bucket.oss-cn-hangzhou.aliyuncs.com/key` |
| 腾讯云 COS | `cos.ap-guangzhou.myqcloud.com` | `https://bucket.cos.ap-guangzhou.myqcloud.com/key` |
| AWS S3 | `s3.us-east-1.amazonaws.com` | `https://bucket.s3.us-east-1.amazonaws.com/key` |
| MinIO (自建) | `minio.internal:9000` | `https://bucket.minio.internal:9000/key` |

### S3 认证

使用 AWS Signature V4（S3 标准签名协议），所有 S3 兼容服务均支持。自行实现，不引入 boto3 依赖。

### S3 上传

```
PUT https://{bucket}.{endpoint}/{key}
Host: {bucket}.{endpoint}
Content-Type: image/jpeg
X-Amz-Content-Sha256: {sha256hex}
X-Amz-Date: {timestamp}
Authorization: AWS4-HMAC-SHA256 Credential=...

{binary data}
```

### Key 生成策略

```
media/{user_id}/{media_type}/{uuid}.{ext}
```

例如：`media/alice/image/a1b2c3d4.jpg`

### 配置

```yaml
object_storage:
  backend: "${OBJECT_STORAGE_BACKEND:local}"   # "s3" or "local"
  s3:
    endpoint: "${S3_ENDPOINT:tos-cn-beijing.volces.com}"
    access_key_id: "${S3_ACCESS_KEY_ID:}"
    secret_access_key: "${S3_SECRET_ACCESS_KEY:}"
    bucket: "${S3_BUCKET:opencontext-media}"
    region: "${S3_REGION:cn-beijing}"            # 用于签名
    use_https: true                              # 默认 HTTPS
  local:
    base_dir: "./uploads/media"
```

### 全局访问

```python
# opencontext/storage/object_storage/global_object_storage.py
def get_object_storage() -> Optional[IObjectStorage]:
    """获取对象存储单例，未配置时返回 None"""
```

## 4. 管线集成

### push.py 改造

```python
async def _process_multimodal_messages(messages, user_id):
    obj_storage = get_object_storage()
    for msg in messages:
        for part in content_parts:
            if is_base64_data_uri(url):
                data = base64.b64decode(extract_base64(url))
                key = f"media/{user_id}/{media_type}/{uuid4().hex}.{ext}"
                url = await obj_storage.upload(data, key, content_type)
                part["image_url"]["url"] = url   # 替换为 HTTPS URL
            elif is_http_url(url):
                pass  # HTTP URL 直接保留（后续可选 re-upload）
```

- `_process_multimodal_messages` 改为 async
- 对象存储不可用时回退到现有本地文件保存逻辑
- 不改变 buffer/direct mode 选择逻辑

### LocalBackend API 调用适配

LocalBackend 存文件返回本地路径，但远程 API 需要 base64。在 `Vectorize.build_ark_input()` 中处理：

```python
def build_ark_input(self) -> List[Dict[str, Any]]:
    for img in self.images or []:
        if _is_local_path(img):
            img = _file_to_data_uri(img)  # 读文件转 base64 data URI
        items.append({"type": "image_url", "image_url": {"url": img}})
```

同样处理 LLM 消息中的本地路径（TextChatProcessor._build_multimodal_llm_messages）。

## 5. 实施计划

### Phase 1: 基础设施（2 个 Agent 并行）

**Agent 1A: 对象存储接口 + LocalBackend**

| 任务 | 文件 |
|------|------|
| 创建 `IObjectStorage` 接口 | `opencontext/storage/object_storage/base.py` (新建) |
| 创建 `LocalBackend` | `opencontext/storage/object_storage/local_backend.py` (新建) |
| 创建全局单例 + 工厂 | `opencontext/storage/object_storage/global_object_storage.py` (新建) |
| 创建 `__init__.py` | `opencontext/storage/object_storage/__init__.py` (新建) |

**Agent 1B: S3CompatibleBackend 实现**

| 任务 | 文件 |
|------|------|
| 实现 AWS Signature V4 签名 | `opencontext/storage/object_storage/s3_auth.py` (新建) |
| 实现 `S3CompatibleBackend` | `opencontext/storage/object_storage/s3_backend.py` (新建) |
| 更新配置 | `config/config.yaml`, `.env.example` |
| 创建验证脚本 | `scripts/test_s3_upload.py` (新建) |

### Phase 2: 管线集成（1 个 Agent）

| 任务 | 文件 |
|------|------|
| `_process_multimodal_messages` 改为 async + 集成对象存储 | `opencontext/server/routes/push.py` |
| `build_ark_input` 本地路径转 base64 兼容 | `opencontext/models/context.py` |
| `_build_multimodal_llm_messages` 本地路径转 base64 兼容 | `opencontext/context_processing/processor/text_chat_processor.py` |
| 服务器生命周期关闭钩子 | `opencontext/cli.py` |

### Phase 3: 文档（1 个 Agent）

| 任务 | 文件 |
|------|------|
| MODULE.md | `opencontext/storage/object_storage/MODULE.md` (新建) |
| 更新 storage MODULE.md | `opencontext/storage/MODULE.md` |
| 更新 curls.sh | `docs/curls.sh` |

### Agent 总览

```
Phase 1: [Agent-1A] 接口 + LocalBackend               ─┐ 并行
         [Agent-1B] S3 Backend + 签名 + 验证脚本        ─┘
             ↓
Phase 2: [Agent-2]  管线集成 (push.py, context.py, processor, cli.py)
             ↓
Phase 3: [Agent-3]  文档
```

## 6. 注意事项

- AWS Signature V4 签名：`Authorization: AWS4-HMAC-SHA256 Credential={ak}/{date}/{region}/s3/aws4_request, SignedHeaders=..., Signature=...`
- PUT 请求需要 `X-Amz-Content-Sha256` header（二进制内容的 SHA256 hex）
- `_process_multimodal_messages` 改 async 后，push_chat 中的调用改为 await
- `object_storage` 未配置或 `get_object_storage()` 返回 None 时，回退到现有本地文件保存
- S3 `access_key_id` 为空时可自动回退到 `VIKINGDB_ACCESS_KEY_ID`（Volcengine TOS 的 S3 兼容接口使用相同凭证）
- 不同云厂商的 S3 兼容程度略有差异，但 PUT Object 和 DELETE Object 是全兼容的核心操作
