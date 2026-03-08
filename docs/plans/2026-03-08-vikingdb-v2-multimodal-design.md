# VikingDB V2 升级与多模态支持设计

> Date: 2026-03-08
> Status: Draft

## 1. 目标

1. **VikingDB API V1 → V2 迁移**：更新 `vikingdb_backend.py` 全部 API 调用至 V2 格式
2. **多模态 Embedding**：使用 `doubao-embedding-vision-251215` 模型，支持 text/image/video 任意组合的向量化
3. **Push API 多模态**：接受 OpenAI 多模态消息格式，允许传入 text/image/video
4. **Chat Analyze 多模态**：LLM 直接分析多模态消息（要求配置支持 vision 的 LLM）

## 2. 架构概览

### 2.1 数据流

```
Push API (OpenAI multimodal message format)
  → TextChatCapture (保存媒体文件，构建 RawContext)
  → TextChatProcessor (多模态 LLM 直接分析，提取记忆)
  → ProcessedContext (text + images/videos 引用 + 模态信息)
  → GlobalEmbeddingClient (Ark API, doubao-embedding-vision-251215)
  → VikingDB V2 存储 (预计算向量 + 标量字段)
  → 检索: 多模态 query embedding → SearchByVector
```

### 2.2 方案选择：客户端侧多模态向量化（方案 B）

- 客户端调用 Ark 多模态 Embedding API 生成向量
- 上传预计算向量到 VikingDB V2（100 条/请求，无向量化批量限制）
- 搜索使用 `SearchByVector`，query 在客户端侧 embedding
- 架构与现有后端无关设计一致（ChromaDB/Qdrant 共用同一管线）

### 2.3 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 向量化位置 | 客户端侧 | 100条/批 vs 1条/批；架构一致性；模型灵活性 |
| Embedding API | Ark HTTP API (aiohttp) | 不使用 SDK；curl 风格调用；与 VikingDB auth 解耦 |
| Chat Analyze | 多模态 LLM 直接分析 | 无信息损失；更简单；要求 LLM 支持 vision |
| Instruction 策略 | 入库/检索分离 | 模型要求；直接影响检索精度 |
| 向量维度 | 2048（与现有一致） | doubao-embedding-vision-251215 默认 2048 |
| 稀疏向量 | 暂不启用 | 仅支持文本输入；可后续作为 VikingDB 特有优化添加 |

## 3. 详细设计

### 3.1 数据模型变更

#### Vectorize 模型扩展

```python
# opencontext/models/context.py

class VideoInput(BaseModel):
    """视频输入"""
    url: str            # HTTP URL, TOS path, 或 data:video/...;base64,...
    fps: float = 1.0    # 0.2-5.0, 帧提取率

class ContentFormat(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    MULTIMODAL = "multimodal"   # 新增：混合模态
    FILE = "file"

class Vectorize(BaseModel):
    content_format: ContentFormat = ContentFormat.TEXT
    text: Optional[str] = None
    images: Optional[List[str]] = None          # URL 或 base64, 替代原 image_path
    videos: Optional[List[VideoInput]] = None
    vector: Optional[List[float]] = None

    def get_modality_string(self) -> str:
        """根据实际内容推断模态字符串，用于 instruction 生成"""
        parts = []
        if self.text: parts.append("text")
        if self.images: parts.append("image")
        if self.videos: parts.append("video")
        return " and ".join(parts) if parts else "text"

    def build_ark_input(self) -> List[Dict]:
        """构建 Ark multimodal embedding API 的 input 数组"""
        items = []
        if self.text:
            items.append({"type": "text", "text": self.text})
        for img in (self.images or []):
            items.append({"type": "image_url", "image_url": {"url": img}})
        for vid in (self.videos or []):
            items.append({"type": "video_url", "video_url": {"url": vid.url, "fps": vid.fps}})
        return items
```

**向后兼容**：`image_path` 字段标记 deprecated，内部转换为 `images=[image_path]`。

#### ProcessedContext 媒体引用

媒体文件引用存储在 `metadata` 字典中，不改变 ProcessedContext 结构：

```python
context.metadata["media_refs"] = [
    {"type": "image", "url": "https://...", "local_path": "/uploads/media/xxx.jpg"},
    {"type": "video", "url": "https://...", "local_path": "/uploads/media/xxx.mp4"}
]
context.metadata["content_modalities"] = "text,image"  # 逗号分隔
```

### 3.2 多模态 Embedding 客户端

#### API 调用方式

使用 aiohttp 直接调用 Ark 多模态 Embedding API（不使用 SDK）：

```bash
# 入库侧（Corpus）
curl https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -d '{
    "model": "doubao-embedding-vision-251215",
    "encoding_format": "float",
    "dimensions": 2048,
    "instructions": "Instruction:Compress the text and image into one word.\nQuery:",
    "input": [
      {"type": "text", "text": "用户昨天去了西湖"},
      {"type": "image_url", "image_url": {"url": "https://example.com/westlake.jpg"}}
    ]
  }'

# 检索侧（Query）
curl https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -d '{
    "model": "doubao-embedding-vision-251215",
    "encoding_format": "float",
    "dimensions": 2048,
    "instructions": "Target_modality: text/image/video.\nInstruction:根据这个查询，找到最相关的记忆内容\nQuery:",
    "input": [
      {"type": "text", "text": "西湖的照片"}
    ]
  }'
```

#### Instruction 策略

| 角色 | 模板 | `{modality}` 来源 |
|------|------|-------------------|
| **入库（Corpus）** | `Instruction:Compress the {modality} into one word.\nQuery:` | `Vectorize.get_modality_string()` 自动推断 |
| **检索（Query）** | `Target_modality: {target}.\nInstruction:{task}\nQuery:` | `{target}` 从配置读取（默认 `text/image/video`）；`{task}` 按检索场景配置 |

**`{target}` 填写规则**：由语料库的模态组成决定。本项目语料库混合 text/image/video，故 target 为 `text/image/video`。

**`{task}` 配置（config.yaml）**：
```yaml
embedding_model:
  search_instruction: "根据这个查询，找到最相关的记忆内容"
  target_modality: "text/image/video"
```

#### GlobalEmbeddingClient 改造

```python
class GlobalEmbeddingClient:
    """扩展为支持多模态的 embedding 客户端"""

    async def do_vectorize(self, vectorize: Vectorize, role: str = "corpus") -> None:
        """单条向量化，role 为 'corpus'（入库）或 'query'（检索）"""
        if vectorize.vector is not None:
            return  # 已有向量，跳过
        instruction = self._build_instruction(vectorize, role)
        ark_input = vectorize.build_ark_input()
        embedding = await self._call_ark_api(ark_input, instruction)
        vectorize.vector = embedding

    async def do_vectorize_batch(self, vectorizes: Sequence[Vectorize], role: str = "corpus") -> None:
        """批量向量化，使用并发控制"""
        pending = [(i, v) for i, v in enumerate(vectorizes) if v.vector is None]
        semaphore = asyncio.Semaphore(self._max_concurrency)  # 默认 15

        async def _vectorize_one(idx, v):
            async with semaphore:
                await self.do_vectorize(v, role)

        await asyncio.gather(*[_vectorize_one(i, v) for i, v in pending])

    async def _call_ark_api(self, input_data: List[Dict], instruction: str) -> List[float]:
        """调用 Ark multimodal embedding API"""
        payload = {
            "model": self._model_name,           # doubao-embedding-vision-251215
            "encoding_format": "float",
            "dimensions": self._dimensions,       # 2048
            "instructions": instruction,
            "input": input_data,
        }
        async with self._session.post(
            self._ark_endpoint,                   # https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        ) as resp:
            result = await resp.json()
            return result["data"]["embedding"]
```

**关键约束**：
- Ark API 每请求仅返回 1 条 embedding（多模态内容作为一个整体）
- 因此批量向量化通过并发请求实现，非 API 级别 batch
- 并发度建议 15（基于图片 ~15/sec 的限制）
- TPM: 1,200k；RPM: 15k——并发 15 下不会触达

#### 图片/视频约束

| 约束 | 图片 | 视频 |
|------|------|------|
| 文件大小 | < 10 MB | < 50 MB（推荐 ≤ 30 MB） |
| 分辨率 | > 14px, < 36MP | — |
| 格式 | JPEG/PNG/GIF/WEBP/BMP/TIFF | MP4/AVI/MOV |
| Token 消耗 | `min(W*H/784, 1312)` | 10k-80k（取决于 fps 和时长） |
| 传输方式 | URL 或 `data:image/...;base64,...` | URL 或 `data:video/...;base64,...` |
| 压缩建议 | 缩放至 0.30-0.35× 原始尺寸 | fps 0.5-1.0 兼顾质量和成本 |

### 3.3 VikingDB V2 API 迁移

#### 端点变更映射

| 操作 | V1 端点/格式 | V2 端点/格式 |
|------|-------------|-------------|
| **Upsert** | `POST /api/vikingdb/data/upsert` `{collection_name, fields: [...]}` | `POST /api/vikingdb/data/upsert` `{collection_name, data: [...]}` |
| **Delete** | `POST /api/vikingdb/data/delete` `{collection_name, primary_keys: [...]}` | `POST /api/vikingdb/data/delete` `{collection_name, ids: [...]}` |
| **Fetch** | `POST /api/vikingdb/data/fetch_in_collection` `{collection_name, primary_keys: [...]}` | `POST /api/vikingdb/data/fetch_in_collection` `{collection_name, ids: [...]}` |
| **向量搜索** | `POST /api/vikingdb/data/search/vector` `{dense_vector: [...]}` | `POST /api/vikingdb/data/search/vector` `{dense_vector: [...]}` （格式不变） |
| **标量搜索** | `POST /api/vikingdb/data/search/scalar` | `POST /api/vikingdb/data/search/scalar` （格式不变） |
| **Update** | `POST /api/vikingdb/data/update` `{collection_name, fields: [...]}` | `POST /api/vikingdb/data/update` `{collection_name, data: [...]}` |
| **创建 Collection** | `Action=GetVikingdbCollection` + snake_case params | `Action=CreateVikingdbCollection` + **PascalCase** params |
| **创建 Index** | `Action=CreateVikingdbIndex` + snake_case params | `Action=CreateVikingdbIndex` + **PascalCase** params |

#### 控制面 API 参数映射（PascalCase）

```python
# V1
{"collection_name": "opencontext", "fields": [...], ...}

# V2
{"CollectionName": "opencontext", "Fields": [...], "ProjectName": "default", ...}
```

主要参数映射：
- `collection_name` → `CollectionName`
- `index_name` → `IndexName`
- `description` → `Description`
- `fields` → `Fields`（定义时）
- `field_name` → `FieldName`
- `field_type` → `FieldType`
- `is_primary_key` → `IsPrimaryKey`
- `vector_index` → `VectorIndex`（含 `IndexType`, `Distance`, `Quant`, `HnswM`, `HnswCef`, `HnswSef`）
- `scalar_index` → `ScalarIndex`

#### 数据面 API 响应变更

```python
# V1 响应: ID 在 fields 内部
{"result": {"data": [{"id": "xxx", "score": 0.95, "fields": {"id": "xxx", "title": "..."}}]}}

# V2 响应: ID 与 fields 分离
{"result": {"data": [{"id": "xxx", "score": 0.95, "fields": {"title": "..."}}]}}
```

#### 新增 VikingDB Collection 字段

| 字段 | 类型 | 用途 |
|------|------|------|
| `content_modalities` | `string` | 逗号分隔的模态列表，如 `"text"`, `"text,image"` |
| `media_refs` | `string` | JSON 数组，媒体文件引用 |

### 3.4 Push API 多模态支持

#### 消息格式（兼容 OpenAI multimodal format）

```python
# 纯文本（现有格式，完全兼容）
{"role": "user", "content": "hello"}

# 多模态（content 为列表）
{"role": "user", "content": [
    {"type": "text", "text": "看看这张图和这段视频"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
    {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
]}
```

#### PushChatRequest 无需改动

```python
class PushChatRequest(BaseModel):
    messages: List[Dict[str, Any]]  # 已经是 Any 类型，天然兼容多模态
    ...
```

#### 媒体文件处理

在 `TextChatCapture` 层处理：

1. 解析 `content`：如果是 `str` → 纯文本（现有逻辑）；如果是 `list` → 多模态
2. base64 图片/视频 → 保存到 `./uploads/media/{uuid}.{ext}`，替换 content 中的 base64 为本地路径
3. URL 媒体 → 直接保留
4. 约束校验：文件大小、格式、分辨率（参照 3.2 节约束表）
5. 构建 `RawContextProperties`，`content_text` 保持完整多模态 JSON

### 3.5 处理管线多模态

#### TextChatProcessor

直接将多模态消息传给 LLM（要求 LLM 支持 vision）：

```python
async def process(self, raw_context: RawContextProperties) -> List[ProcessedContext]:
    messages = json.loads(raw_context.content_text)
    # messages 已经是 OpenAI 多模态格式，直接传给 LLM
    # LLM 能看到图片/视频，做出更准确的记忆提取

    llm_messages = [{"role": "system", "content": CHAT_ANALYZE_PROMPT}]
    llm_messages.extend(messages)  # 包含多模态内容

    result = await self.llm_client.generate_with_messages(llm_messages)
    memories = parse_memories(result)

    contexts = []
    for memory in memories:
        ctx = self._build_processed_context(memory, raw_context)
        # 如果 memory 关联特定媒体，附带到 Vectorize
        if memory.get("related_media"):
            ctx.vectorize.images = memory["related_media"].get("images")
            ctx.vectorize.videos = [VideoInput(**v) for v in memory["related_media"].get("videos", [])]
            ctx.vectorize.content_format = ContentFormat.MULTIMODAL
        contexts.append(ctx)
    return contexts
```

**chat_analyze prompt 更新**：在 `prompts_en.yaml` / `prompts_zh.yaml` 中更新 `processing.extraction.chat_analyze`，说明输入可能包含图片/视频，LLM 应在提取的记忆中标注关联的媒体索引。

#### 记忆的 Embedding 模态策略

- 纯文本记忆 → `content_format=TEXT`，仅 `text` 字段 → 纯文本 embedding
- 关联媒体的记忆 → `content_format=MULTIMODAL`，`text` + `images`/`videos` → 多模态 embedding
- 由 LLM 在 chat_analyze 中判断记忆是否与特定媒体强关联

### 3.6 检索与搜索

#### 检索工具多模态 Query

工具 schema 添加可选参数：

```python
# tool_definitions.py 中各检索工具的 parameters 扩展
{
    "query": {"type": "string", "description": "文本查询"},
    "image_url": {"type": "string", "description": "图片 URL（可选）"},
    "video_url": {"type": "string", "description": "视频 URL（可选）"}
}
```

工具实现中构建 Vectorize：

```python
vectorize = Vectorize(
    text=query,
    images=[image_url] if image_url else None,
    videos=[VideoInput(url=video_url)] if video_url else None,
    content_format=ContentFormat.MULTIMODAL if (image_url or video_url) else ContentFormat.TEXT,
)
await do_vectorize(vectorize, role="query")  # 使用 query-side instruction
```

#### Search API 多模态

`POST /api/search` 请求体扩展：

```python
class SearchRequest(BaseModel):
    query: Optional[str] = None
    image_url: Optional[str] = None    # 新增
    video_url: Optional[str] = None    # 新增
    ...
```

### 3.7 配置变更

```yaml
# config/config.yaml 新增/修改

embedding_model:
  model: "doubao-embedding-vision-251215"
  api_key: "${ARK_API_KEY}"
  base_url: "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
  dimensions: 2048
  max_concurrency: 15                    # 并发请求数
  search_instruction: "根据这个查询，找到最相关的记忆内容"
  target_modality: "text/image/video"    # 语料库模态组成

media:
  upload_dir: "./uploads/media"
  max_image_size_mb: 10
  max_video_size_mb: 50
  image_formats: ["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"]
  video_formats: ["mp4", "avi", "mov"]
  default_video_fps: 1.0
```

## 4. 实施方案

### 4.1 阶段划分与依赖关系

```
Phase 0: 验证脚本（无依赖）
    ↓
Phase 1: 数据模型变更（无依赖）
    ↓
Phase 2A: VikingDB V2 迁移 ──────────┐
Phase 2B: 多模态 Embedding 客户端 ────┤ 并行，均依赖 Phase 1
Phase 2C: Push API + 处理管线 ────────┘
    ↓
Phase 3: 检索与搜索（依赖 2A + 2B）
    ↓
Phase 4: 集成验证 + 文档（依赖全部）
```

### 4.2 Phase 0：验证脚本（测试先行）

**目标**：在写任何业务代码前，先用独立脚本验证外部 API 可用性。

**任务（1 个 Agent）**：

| 脚本 | 验证内容 |
|------|----------|
| `scripts/test_ark_embedding.py` | 调用 Ark multimodal embedding API，验证 text/image/video/混合输入的请求格式和响应解析；验证 instruction 格式 |
| `scripts/test_vikingdb_v2.py` | 用 V2 格式调用 VikingDB 控制面（创建 collection + index）和数据面（upsert + search + fetch + delete），验证参数格式和响应解析 |

这些脚本是可执行的独立文件，使用 `aiohttp` + 环境变量，不依赖项目代码。后续实现可直接参考其中验证通过的请求格式。

### 4.3 Phase 1：数据模型变更

**任务（1 个 Agent）**：

| 文件 | 变更 |
|------|------|
| `opencontext/models/context.py` | 新增 `VideoInput`；扩展 `Vectorize`（`images`, `videos`, `build_ark_input()`, `get_modality_string()`）；扩展 `ContentFormat`（`VIDEO`, `MULTIMODAL`）；`image_path` 标记 deprecated |
| `opencontext/models/enums.py` | 如果 `ContentFormat` 在此定义，同步更新 |

**验收标准**：`python -m py_compile opencontext/models/context.py` 通过。

### 4.4 Phase 2：核心实现（3 个 Agent 并行）

#### Agent 2A：VikingDB V2 API 迁移

**文件**：`opencontext/storage/backends/vikingdb_backend.py`（1847 行）

**该文件较大，但改动是机械性的（参数重命名 + 端点调整），由 1 个 Agent 顺序处理**：

| 改动区域 | 行数范围（约） | 内容 |
|----------|----------------|------|
| 控制面 | `initialize()` | `_ensure_collection()`, `_ensure_index()` 的参数改 PascalCase |
| Upsert | `upsert_processed_context()`, `batch_upsert_processed_context()` | `fields` → `data`；`_context_to_doc_format()` 新增 `content_modalities`, `media_refs` 字段 |
| Delete | `delete_processed_context()`, `delete_contexts()`, `delete_by_source_file()` | `primary_keys` → `ids` |
| Fetch | `get_processed_context()`, `get_by_ids()` | `primary_keys` → `ids`；响应解析：ID 与 fields 分离 |
| Search | `search()`, `search_by_hierarchy()` | 端点不变，响应解析更新 |
| Scalar Search | `get_all_processed_contexts()`, `get_processed_context_count()` | 响应解析更新 |
| Update | `batch_set_parent_id()` | `fields` → `data` |
| Doc→Context | `_doc_to_processed_context()` | 解析新字段 `content_modalities`, `media_refs` |

**验收标准**：Phase 0 的 `test_vikingdb_v2.py` 脚本中的请求格式与实现一致；`py_compile` 通过。

#### Agent 2B：多模态 Embedding 客户端

| 文件 | 变更 |
|------|------|
| `opencontext/llm/global_embedding_client.py` | 重构核心：`do_vectorize()` / `do_vectorize_batch()` 改为调用 Ark API；添加 `role` 参数（corpus/query）；添加 instruction 生成逻辑；添加并发控制（Semaphore） |
| `opencontext/llm/llm_client.py` | 新增 `generate_multimodal_embedding()` 方法，使用 aiohttp 调用 Ark API（不使用 AsyncOpenAI SDK） |
| `config/config.yaml` | 更新 `embedding_model` 配置节 |

**关键实现细节**：

```python
# llm_client.py 新增方法
async def generate_multimodal_embedding(
    self,
    input_data: List[Dict],
    instruction: str,
    dimensions: int = 2048,
) -> List[float]:
    """调用 Ark multimodal embedding API（HTTP 直调，非 SDK）"""
    payload = {
        "model": self.model_name,
        "encoding_format": "float",
        "dimensions": dimensions,
        "instructions": instruction,
        "input": input_data,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            self.base_url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ) as resp:
            result = await resp.json()
            return result["data"]["embedding"]
```

**aiohttp session 管理**：复用 VikingDB 后端已有的 session 管理模式（TCP 连接池），不要每次创建新 session。

**验收标准**：Phase 0 的 `test_ark_embedding.py` 验证通过的 API 格式与实现一致。

#### Agent 2C：Push API + 处理管线

| 文件 | 变更 |
|------|------|
| `opencontext/server/routes/push.py` | 媒体文件处理：解析多模态 content，base64 保存为文件，约束校验 |
| `opencontext/context_capture/text_chat.py` | `push_message()` 支持 list 类型 content；序列化保留多模态结构 |
| `opencontext/context_processing/processor/text_chat_processor.py` | `process()` 直接将多模态消息传给 LLM；`_build_processed_context()` 附带媒体引用 |
| `config/prompts_en.yaml` | 更新 `processing.extraction.chat_analyze` prompt |
| `config/prompts_zh.yaml` | 同步更新 |

**chat_analyze prompt 更新要点**：
- 说明输入消息可能包含图片和视频
- 对于与特定图片/视频强相关的记忆，在 JSON 输出中增加 `related_media` 字段
- `related_media` 格式：`{"images": [索引], "videos": [索引]}`，索引指向消息中媒体的位置

### 4.5 Phase 3：检索与搜索

**任务（1-2 个 Agent）**：

| 文件 | 变更 |
|------|------|
| `opencontext/tools/tool_definitions.py` | 工具 schema 添加 `image_url`, `video_url` 参数 |
| `opencontext/tools/retrieval_tools/*.py` | 构建 Vectorize 时支持多模态 query；`do_vectorize(v, role="query")` |
| `opencontext/tools/tools_executor.py` | 传递新参数 |
| `opencontext/server/routes/search.py` | SearchRequest 添加 `image_url`, `video_url` |
| `opencontext/server/search/*.py` | 搜索策略支持多模态 query embedding |
| `opencontext/server/cache/memory_cache.py` | 缓存数据包含 media_refs |

### 4.6 Phase 4：集成验证 + 文档

**任务（1 个 Agent）**：

| 任务 | 内容 |
|------|------|
| 端到端验证 | 推送多模态聊天 → 检查存储 → 执行搜索 → 验证结果 |
| Config | 确认所有配置项完整，环境变量文档更新 |
| `docs/curls.sh` | 更新 API curl 示例（含多模态请求） |
| MODULE.md | 更新 `llm/`, `storage/`, `server/`, `models/` 的 MODULE.md |
| `.env.example` | 添加 `ARK_API_KEY` |

### 4.7 Agent 任务总览

```
Phase 0:  [Agent-V]  验证脚本 (test_ark_embedding.py + test_vikingdb_v2.py)
              ↓
Phase 1:  [Agent-M]  数据模型变更 (context.py, enums.py)
              ↓
Phase 2:  [Agent-A]  VikingDB V2 迁移 (vikingdb_backend.py)          ─┐
          [Agent-B]  多模态 Embedding 客户端 (embedding + llm_client) ─┤ 并行
          [Agent-C]  Push API + 处理管线 (routes, capture, processor) ─┘
              ↓
Phase 3:  [Agent-R]  检索与搜索 (tools, search, cache)
              ↓
Phase 4:  [Agent-I]  集成验证 + 文档
```

**共计 7 个 Agent 任务，最多 3 个并行。**

### 4.8 验证策略

| 阶段 | 验证方式 |
|------|----------|
| Phase 0 | 独立脚本直接运行，验证外部 API |
| Phase 1 | `py_compile` 编译检查 |
| Phase 2A | `py_compile` + 对照 Phase 0 脚本的请求格式 |
| Phase 2B | `py_compile` + 对照 Phase 0 脚本的请求格式 |
| Phase 2C | `py_compile` + 检查 prompt 文件 YAML 格式 |
| Phase 3 | `py_compile` + 工具 schema JSON 校验 |
| Phase 4 | 端到端 curl 测试（`docs/curls.sh` 中的多模态请求） |

## 5. 风险与关注点

| 风险 | 应对 |
|------|------|
| Ark embedding API 每请求 1 条数据，批量处理可能慢 | 并发控制（Semaphore=15），TPM 1.2M 足够 |
| 图片 base64 传输体积大 | 入库前压缩（0.30-0.35× 原始尺寸），请求体 < 64MB |
| 视频 token 消耗高（10k-80k/条） | 默认 fps=1.0，长视频建议 fps=0.5 |
| chat_analyze prompt 变更影响提取质量 | 增量调整 prompt，保持现有文本提取能力 |
| V2 API 控制面/数据面命名风格不一致（PascalCase vs snake_case） | 在代码中用常量映射，不硬编码 |
