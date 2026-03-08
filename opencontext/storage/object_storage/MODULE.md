# object_storage/ -- Object storage for multimodal media files

## Purpose

Provides a thin abstraction over S3-compatible object storage for uploading/storing multimodal media (images, videos). Used by `push.py` to persist base64 media from push requests, replacing in-pipeline base64 strings with HTTPS URLs.

## File Overview

| File | Responsibility |
|------|---------------|
| `base.py` | `IObjectStorage` abstract interface (4 methods) |
| `local_backend.py` | `LocalBackend` -- saves to local filesystem (dev/testing) |
| `s3_backend.py` | `S3CompatibleBackend` -- uploads via S3 PUT Object (production) |
| `s3_auth.py` | `S3Auth` -- AWS Signature V4 signer for S3-compatible services |
| `global_object_storage.py` | `GlobalObjectStorage` singleton + `get_object_storage()` accessor |
| `__init__.py` | Re-exports: `IObjectStorage`, `GlobalObjectStorage`, `get_object_storage` |

## Interface

```python
class IObjectStorage(ABC):
    async def upload(data: bytes, key: str, content_type: str) -> str   # Returns URL/path
    def get_url(key: str) -> str                                        # URL for key
    async def delete(key: str) -> bool                                  # Delete object
    async def close() -> None                                           # Cleanup resources
```

## Backends

| Backend | `upload()` returns | When to use |
|---------|-------------------|-------------|
| `S3CompatibleBackend` | HTTPS URL (`https://{bucket}.{endpoint}/{key}`) | Production (TOS, OSS, COS, S3, MinIO) |
| `LocalBackend` | Absolute file path | Development, single-instance |

## S3 Compatibility

All S3-compatible services share the same protocol. `S3CompatibleBackend` works with any service by configuring `endpoint`:

| Service | endpoint |
|---------|----------|
| Volcengine TOS | `tos-cn-beijing.volces.com` |
| Alibaba OSS | `oss-cn-hangzhou.aliyuncs.com` |
| Tencent COS | `cos.ap-guangzhou.myqcloud.com` |
| AWS S3 | `s3.us-east-1.amazonaws.com` |
| MinIO | `minio.internal:9000` |

## Configuration

```yaml
# config/config.yaml
object_storage:
  backend: "${OBJECT_STORAGE_BACKEND:local}"   # "s3" or "local"
  s3:
    endpoint: "${S3_ENDPOINT:tos-cn-beijing.volces.com}"
    access_key_id: "${S3_ACCESS_KEY_ID:}"       # empty = fallback to VIKINGDB credentials
    secret_access_key: "${S3_SECRET_ACCESS_KEY:}"
    bucket: "${S3_BUCKET:opencontext-media}"
    region: "${S3_REGION:cn-beijing}"
    use_https: true
  local:
    base_dir: "./uploads/media"
```

## Global Access

```python
from opencontext.storage.object_storage import get_object_storage

obj_storage = get_object_storage()  # Returns IObjectStorage or None
if obj_storage:
    url = await obj_storage.upload(data, key, content_type)
```

Singleton is initialized lazily on first `get_object_storage()` call. S3 credentials fall back to VikingDB credentials (`VIKINGDB_ACCESS_KEY_ID`) when empty.

## Key Design Decisions

- **No boto3**: AWS Signature V4 is implemented directly in `s3_auth.py` to avoid heavy dependencies.
- **Lazy session**: `S3CompatibleBackend` creates `aiohttp.ClientSession` on first use with connection pooling.
- **Re-sign on retry**: Timestamps in S3 signatures expire, so each retry re-signs the request.
- **LocalBackend + data URI conversion**: When using `LocalBackend`, downstream consumers (`build_ark_input()`, `_build_multimodal_llm_messages()`) convert local paths to base64 data URIs before sending to remote APIs.

## Pipeline Integration

```
Push API (base64 / HTTP URL)
  → push.py: _process_multimodal_messages() uploads to object storage
  → Pipeline: messages contain HTTPS URLs (S3) or local paths (LocalBackend)
  → LLM: HTTPS URLs pass through; local paths → data URI in _build_multimodal_llm_messages()
  → Ark Embedding: HTTPS URLs pass through; local paths → data URI in build_ark_input()
  → VikingDB: URL stored as media_ref
```

## Lifecycle

Object storage is closed during server shutdown in `cli.py:lifespan()` to release HTTP sessions.
