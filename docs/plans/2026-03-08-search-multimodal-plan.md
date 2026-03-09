# Search API 多模态适配 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 Search API 完整支持多模态：响应携带 media_refs，查询格式统一为 OpenAI content parts，Vectorize 模型消除冗余字段。

**Architecture:** 重构 Vectorize 模型（content parts list 替代分离的 text/images/videos），修复搜索响应的 media_refs 数据丢失，统一搜索请求格式为 OpenAI content parts。

**Tech Stack:** Python 3.10+, Pydantic V2, FastAPI, aiohttp, VikingDB, Qdrant

**Design doc:** `docs/plans/2026-03-08-search-multimodal-design.md`

**No test suite** — 使用 `python -m py_compile` 编译检查验证。

---

### Task 1: Vectorize 模型重构

**Files:**
- Modify: `opencontext/models/context.py:140-217`

**Step 1: 重写 Vectorize 类**

将 `text`/`images`/`videos`/`image_path` 字段替换为 `input: List[Dict[str, Any]]`。保留 `vector` 和 `content_format`。

删除 `VideoInput` 类（lines 140-144）—— 不再被 Vectorize 使用。但检查 `opencontext/server/routes/search.py` 和 retrieval tools 是否还直接使用 VideoInput —— 如果有，保留类定义但从 Vectorize 解耦。

替换 lines 147-217 的 Vectorize 类为：

```python
class Vectorize(BaseModel):
    """
    Vectorization configuration — supports text, image, video, and multimodal content.
    Uses Ark API content parts format as unified internal representation.
    """

    input: List[Dict[str, Any]] = Field(default_factory=list)
    vector: Optional[List[float]] = None
    content_format: ContentFormat = ContentFormat.TEXT

    def get_modality_string(self) -> str:
        """Return a human-readable modality descriptor based on input content types.

        Examples: "text", "text and image", "text and image and video", "image", etc.
        Falls back to "text" when no content is present.
        """
        types = {item.get("type") for item in self.input}
        parts: List[str] = []
        if "text" in types:
            parts.append("text")
        if "image_url" in types:
            parts.append("image")
        if "video_url" in types:
            parts.append("video")
        return " and ".join(parts) if parts else "text"

    def build_ark_input(self) -> List[Dict[str, Any]]:
        """Build the input list for the Ark multimodal embedding API.

        Local file paths in image_url/video_url items are converted to
        base64 data URIs since remote APIs cannot access local files.
        """
        result: List[Dict[str, Any]] = []
        for item in self.input:
            item_type = item.get("type")
            if item_type == "image_url":
                url = item["image_url"]["url"]
                if _is_local_path(url):
                    url = _file_to_data_uri(url)
                result.append({"type": "image_url", "image_url": {"url": url}})
            elif item_type == "video_url":
                vid_info = item["video_url"]
                url = vid_info["url"]
                if _is_local_path(url):
                    url = _file_to_data_uri(url)
                new_vid = {k: v for k, v in vid_info.items()}
                new_vid["url"] = url
                result.append({"type": "video_url", "video_url": new_vid})
            else:
                result.append(item)
        return result

    def get_text(self) -> Optional[str]:
        """Extract text content from input items. Returns concatenated text or None."""
        texts = [item["text"] for item in self.input if item.get("type") == "text"]
        return "\n".join(texts) if texts else None
```

注意：
- `get_vectorize_content()` 改名为 `get_text()` 并简化
- 删除 `_compat_image_path` validator（lines 159-164）
- `_is_local_path` 和 `_file_to_data_uri` 保持不动（lines 477-512）

**Step 2: 编译检查**

```bash
python -m py_compile opencontext/models/context.py
```

**Step 3: Commit**

```bash
git add opencontext/models/context.py
git commit -m "refactor(vectorize): replace text/images/videos with unified input field"
```

---

### Task 2: Embedding Client 适配

**Files:**
- Modify: `opencontext/llm/global_embedding_client.py:167-259`
- Modify: `opencontext/llm/llm_client.py:220-226`

**Step 1: 修改 global_embedding_client.py**

`_build_instruction()` (lines 167-186): 无需改动 —— 已通过 `vectorize.get_modality_string()` 间接访问。

`do_vectorize()` (lines 192-216): 无需改动 —— 已通过 `vectorize.build_ark_input()` 间接访问。

`do_vectorize_batch()` (line 257): 修改日志行，将 `v.text[:50]` 替换为从 input 提取：

```python
# Line 257 原代码:
f"(text={v.text[:50] if v.text else 'N/A'}...)"

# 改为:
f"(text={v.get_text()[:50] if v.get_text() else 'N/A'}...)"
```

**Step 2: 修改 llm_client.py**

`vectorize()` 方法 (lines 220-226): 将 `get_vectorize_content()` 替换为 `get_text()`：

```python
# 原代码 (lines 220-226):
async def vectorize(self, vectorize: Vectorize, **kwargs):
    if vectorize.vector:
        return
    content = vectorize.get_vectorize_content()
    if not content:
        return
    vectorize.vector = await self.generate_embedding(content, **kwargs)

# 改为:
async def vectorize(self, vectorize: Vectorize, **kwargs):
    if vectorize.vector:
        return
    content = vectorize.get_text()
    if not content:
        return
    vectorize.vector = await self.generate_embedding(content, **kwargs)
```

**Step 3: 编译检查**

```bash
python -m py_compile opencontext/llm/global_embedding_client.py
python -m py_compile opencontext/llm/llm_client.py
```

**Step 4: Commit**

```bash
git add opencontext/llm/global_embedding_client.py opencontext/llm/llm_client.py
git commit -m "refactor(embedding): adapt to new Vectorize input field"
```

---

### Task 3: TextChatProcessor 适配

**Files:**
- Modify: `opencontext/context_processing/processor/text_chat_processor.py:296-365`

**Step 1: 修改 _build_processed_context()**

将 lines 296-365 中构建 `vectorize_images`/`vectorize_videos` 和 Vectorize 创建的逻辑改为直接构建 content parts list。

原有的 media 解析逻辑（从 `related_media` 解析 image/video indices 到 URLs）保留不变。只修改最终组装部分。

找到 Vectorize 创建处（lines 360-365），替换为：

```python
# 构建 content parts list
ark_input = [
    {
        "type": "text",
        "text": f"{extracted_data.title}\n{extracted_data.summary}\n{' '.join(extracted_data.keywords)}",
    }
]
for img_url in vectorize_images or []:
    ark_input.append({"type": "image_url", "image_url": {"url": img_url}})
for vid_url in vectorize_videos or []:
    ark_input.append({"type": "video_url", "video_url": {"url": vid_url, "fps": 1.0}})

vectorize_format = (
    ContentFormat.MULTIMODAL if (vectorize_images or vectorize_videos) else ContentFormat.TEXT
)
```

然后将 ProcessedContext 中的 vectorize 参数改为：

```python
vectorize=Vectorize(
    input=ark_input,
    content_format=vectorize_format,
),
```

注意：`vectorize_images` 和 `vectorize_videos` 变量仍用于 `media_refs` 的构建（metadata），不要删除。只改 Vectorize 构建方式。

**Step 2: 清理 VideoInput import**

如果此文件 import 了 `VideoInput`，移除该 import（因为不再需要构建 VideoInput 对象）。

**Step 3: 编译检查**

```bash
python -m py_compile opencontext/context_processing/processor/text_chat_processor.py
```

**Step 4: Commit**

```bash
git add opencontext/context_processing/processor/text_chat_processor.py
git commit -m "refactor(processor): TextChatProcessor uses content parts list for Vectorize"
```

---

### Task 4: DocumentProcessor + Merger + Hierarchy Summary + Context Operations

**Files:**
- Modify: `opencontext/context_processing/processor/document_processor.py:255-258`
- Modify: `opencontext/context_processing/merger/merge_strategies.py:250`
- Modify: `opencontext/periodic_task/hierarchy_summary.py:1359-1362`
- Modify: `opencontext/server/context_operations.py:142`

**Step 1: 修改 document_processor.py**

Line 255-258，将 Vectorize 创建改为：

```python
# 原代码:
vectorize=Vectorize(
    content_format=ContentFormat.TEXT,
    text=chunk.text,
),

# 改为:
vectorize=Vectorize(
    input=[{"type": "text", "text": chunk.text}],
    content_format=ContentFormat.TEXT,
),
```

**Step 2: 修改 merge_strategies.py**

Line 250，将：

```python
# 原代码:
vectorize=Vectorize(text=extracted_data.title + " " + extracted_data.summary),

# 改为:
vectorize=Vectorize(
    input=[{"type": "text", "text": extracted_data.title + " " + extracted_data.summary}],
),
```

**Step 3: 修改 hierarchy_summary.py**

Lines 1359-1362，将：

```python
# 原代码:
vectorize = Vectorize(
    content_format=ContentFormat.TEXT,
    text=summary_body,
)

# 改为:
vectorize = Vectorize(
    input=[{"type": "text", "text": summary_body}],
    content_format=ContentFormat.TEXT,
)
```

**Step 4: 修改 context_operations.py**

Line 142，将：

```python
# 原代码:
query_vectorize = Vectorize(text=query)

# 改为:
query_vectorize = Vectorize(input=[{"type": "text", "text": query}])
```

**Step 5: 编译检查**

```bash
python -m py_compile opencontext/context_processing/processor/document_processor.py
python -m py_compile opencontext/context_processing/merger/merge_strategies.py
python -m py_compile opencontext/periodic_task/hierarchy_summary.py
python -m py_compile opencontext/server/context_operations.py
```

**Step 6: Commit**

```bash
git add opencontext/context_processing/processor/document_processor.py \
        opencontext/context_processing/merger/merge_strategies.py \
        opencontext/periodic_task/hierarchy_summary.py \
        opencontext/server/context_operations.py
git commit -m "refactor: adapt all Vectorize creation sites to content parts format"
```

---

### Task 5: Retrieval Tools 适配

**Files:**
- Modify: `opencontext/tools/retrieval_tools/base_context_retrieval_tool.py:114-121`
- Modify: `opencontext/tools/retrieval_tools/knowledge_retrieval_tool.py:127-134`
- Modify: `opencontext/tools/retrieval_tools/hierarchical_event_tool.py:317-324`
- Modify: `opencontext/tools/retrieval_tools/document_management_tool.py:153`

所有 retrieval tools 中构建 Vectorize 的模式相同。将分离字段改为 content parts list。

**Step 1: 修改 base_context_retrieval_tool.py**

Lines 114-121，将 Vectorize 创建改为：

```python
# 原代码:
vectorize = Vectorize(
    text=query,
    images=[image_url] if image_url else None,
    videos=[VideoInput(url=video_url)] if video_url else None,
    content_format=...,
)

# 改为:
ark_input = [{"type": "text", "text": query}]
if image_url:
    ark_input.append({"type": "image_url", "image_url": {"url": image_url}})
if video_url:
    ark_input.append({"type": "video_url", "video_url": {"url": video_url}})
has_multimodal = bool(image_url or video_url)
vectorize = Vectorize(
    input=ark_input,
    content_format=ContentFormat.MULTIMODAL if has_multimodal else ContentFormat.TEXT,
)
```

**Step 2: 对 knowledge_retrieval_tool.py 做相同改动** (lines 127-134)

**Step 3: 对 hierarchical_event_tool.py 做相同改动** (lines 317-324)

**Step 4: 修改 document_management_tool.py**

Line 153，简单的文本查询：

```python
# 原代码:
vectorize = Vectorize(text=query)

# 改为:
vectorize = Vectorize(input=[{"type": "text", "text": query}])
```

**Step 5: 清理所有文件的 VideoInput import**（如果不再需要）

**Step 6: 编译检查**

```bash
python -m py_compile opencontext/tools/retrieval_tools/base_context_retrieval_tool.py
python -m py_compile opencontext/tools/retrieval_tools/knowledge_retrieval_tool.py
python -m py_compile opencontext/tools/retrieval_tools/hierarchical_event_tool.py
python -m py_compile opencontext/tools/retrieval_tools/document_management_tool.py
```

**Step 7: Commit**

```bash
git add opencontext/tools/retrieval_tools/
git commit -m "refactor(tools): adapt retrieval tools to new Vectorize input format"
```

---

### Task 6: 存储层清理

**Files:**
- Modify: `opencontext/storage/backends/vikingdb_backend.py:660-665,797-808,1188-1207`
- Modify: `opencontext/storage/backends/qdrant_backend.py:148-149,516-522`

**Step 1: VikingDB — 清理 _context_to_doc_format()**

Lines 798-808，将 `vectorize.text`、`vectorize.images`、`vectorize.videos` 的存储改为从 input 提取 text：

```python
# 原代码 (lines 798-808):
if context.vectorize.text:
    doc[FIELD_DOCUMENT] = context.vectorize.text
doc[FIELD_CONTENT_MODALITIES] = context.vectorize.get_modality_string()
if context.vectorize.images:
    doc["images"] = json.dumps(context.vectorize.images, ensure_ascii=False)
if context.vectorize.videos:
    doc["videos"] = json.dumps(
        [v.model_dump() for v in context.vectorize.videos], ensure_ascii=False
    )

# 改为:
text = context.vectorize.get_text()
if text:
    doc[FIELD_DOCUMENT] = text
doc[FIELD_CONTENT_MODALITIES] = context.vectorize.get_modality_string()
# images/videos 不再单独存储 — media_refs in metadata 已覆盖
```

**Step 2: VikingDB — 清理 _doc_to_processed_context()**

Lines 1188-1207，删除 images/videos 重建逻辑。改为只重建 text 到 input 格式：

```python
# 原代码 (lines 1188-1207): 完整的 images/videos JSON 解析重建
# 删除全部 images_raw 和 videos_raw 相关代码

# 改为（在现有的 vectorize_dict 构建处）:
# vectorize_dict 已有 "vector" 字段
doc_text = fields.pop(FIELD_DOCUMENT, None)
if doc_text:
    vectorize_dict["input"] = [{"type": "text", "text": doc_text}]

# 删除 images_raw 和 videos_raw 的解析（lines 1189-1207）
```

注意：由 `fields.pop("images", None)` 和 `fields.pop("videos", None)` 仍然需要 pop 掉这些字段（避免进入 metadata），但不需要重建到 vectorize_dict。简单 pop 并丢弃即可：

```python
fields.pop("images", None)
fields.pop("videos", None)
```

**Step 3: VikingDB — 清理 collection schema**

Lines 664-665，删除 images/videos 字段定义：

```python
# 删除:
{"FieldName": "images", "FieldType": "string"},
{"FieldName": "videos", "FieldType": "string"},
```

注意：已存在的 collection 不受影响（VikingDB 忽略未声明的已有字段）。

**Step 4: Qdrant — 适配 text 存储**

Lines 148-149，将 `vectorize.text` 改为 `vectorize.get_text()`：

```python
# 原代码:
if context.vectorize.content_format == ContentFormat.TEXT:
    payload[FIELD_DOCUMENT] = context.vectorize.text

# 改为:
text = context.vectorize.get_text()
if text:
    payload[FIELD_DOCUMENT] = text
```

Lines 516-522，重建时用 input 格式：

```python
# 原代码:
if document:
    vectorize_dict["text"] = document

# 改为:
if document:
    vectorize_dict["input"] = [{"type": "text", "text": document}]
```

**Step 5: 编译检查**

```bash
python -m py_compile opencontext/storage/backends/vikingdb_backend.py
python -m py_compile opencontext/storage/backends/qdrant_backend.py
```

**Step 6: Commit**

```bash
git add opencontext/storage/backends/vikingdb_backend.py \
        opencontext/storage/backends/qdrant_backend.py
git commit -m "refactor(storage): stop storing images/videos separately, use Vectorize.input"
```

---

### Task 7: Search 响应模型 + 请求模型

**Files:**
- Modify: `opencontext/server/search/models.py:25-92,98-123`

**Step 1: 修改 EventSearchRequest**

将 `query` 类型改为 content parts list，删除 `image_url`/`video_url` 顶层字段：

```python
# 原代码 (lines 27-40):
query: Optional[str] = Field(default=None, ...)
image_url: Optional[str] = Field(default=None, ...)
video_url: Optional[str] = Field(default=None, ...)

# 改为:
query: Optional[List[Dict[str, Any]]] = Field(
    default=None,
    description=(
        "Multimodal search query in OpenAI content parts format. "
        "Example: [{\"type\": \"text\", \"text\": \"...\"}, "
        "{\"type\": \"image_url\", \"image_url\": {\"url\": \"...\"}}]"
    ),
)
```

需要在文件顶部添加 import：`from typing import Any, Dict, List, Optional`（检查是否已有）。

**Step 2: 修改 validator**

Lines 79-92 的 `validate_search_criteria()`：

```python
# 原代码检查:
has_query = bool(self.query)
has_image = bool(self.image_url)
has_video = bool(self.video_url)
...
if not any([has_query, has_image, has_video, has_ids, has_time, has_levels]):

# 改为:
has_query = bool(self.query)
has_ids = bool(self.event_ids)
has_time = self.time_range is not None
has_levels = bool(self.hierarchy_levels)
if not any([has_query, has_ids, has_time, has_levels]):
    raise ValueError(...)
```

**Step 3: 修改 EventNode**

Lines 98-122，添加 `media_refs` 字段：

```python
class EventNode(BaseModel):
    # ... 现有字段 (lines 108-116) ...

    # Multimodal media references (L0 events only, summaries have empty list)
    media_refs: List[Dict[str, str]] = Field(default_factory=list)

    # Search-hit fields ...
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    score: Optional[float] = None
```

**Step 4: 编译检查**

```bash
python -m py_compile opencontext/server/search/models.py
```

**Step 5: Commit**

```bash
git add opencontext/server/search/models.py
git commit -m "feat(search): multimodal query format + media_refs in EventNode"
```

---

### Task 8: Search 路由 + 缓存模型

**Files:**
- Modify: `opencontext/server/routes/search.py:141-164,404-440,443-472`
- Modify: `opencontext/server/cache/models.py:20-31`

**Step 1: 修改搜索查询构建**

Lines 141-164，替换 Vectorize 构建逻辑：

```python
# 原代码 (lines 141-151):
elif request.query or request.image_url or request.video_url:
    has_multimodal = bool(request.image_url or request.video_url)
    vectorize = Vectorize(
        text=request.query,
        images=[request.image_url] if request.image_url else None,
        videos=[VideoInput(url=request.video_url)] if request.video_url else None,
        content_format=(
            ContentFormat.MULTIMODAL if has_multimodal else ContentFormat.TEXT
        ),
    )
    await do_vectorize(vectorize, role="query")

# 改为:
elif request.query:
    # Detect if multimodal
    query_types = {item.get("type") for item in request.query}
    has_multimodal = bool(query_types & {"image_url", "video_url"})
    vectorize = Vectorize(
        input=request.query,
        content_format=(
            ContentFormat.MULTIMODAL if has_multimodal else ContentFormat.TEXT
        ),
    )
    await do_vectorize(vectorize, role="query")
```

清理 VideoInput import（如果此文件有）。

**Step 2: 修改 _to_search_hit_node()**

Lines 422-440，添加 media_refs 提取：

```python
def _to_search_hit_node(ctx: ProcessedContext, score: float) -> EventNode:
    props = ctx.properties
    extracted = ctx.extracted_data

    # Extract media_refs from metadata
    media_refs = []
    if ctx.metadata and ctx.metadata.get("media_refs"):
        media_refs = ctx.metadata["media_refs"]

    return EventNode(
        id=ctx.id,
        title=extracted.title if extracted else None,
        summary=extracted.summary if extracted else None,
        keywords=extracted.keywords if extracted and extracted.keywords else [],
        entities=extracted.entities if extracted and extracted.entities else [],
        score=score,
        hierarchy_level=props.hierarchy_level if props else 0,
        time_bucket=props.time_bucket if props else None,
        parent_id=_normalize_parent_id(props),
        event_time=_format_timestamp(props.event_time if props else None),
        create_time=_format_timestamp(props.create_time if props else None),
        is_search_hit=True,
        media_refs=media_refs,
    )
```

**Step 3: 修改 _to_context_node()**

Lines 404-419，同样添加 media_refs：

```python
def _to_context_node(ctx: ProcessedContext) -> EventNode:
    props = ctx.properties
    extracted = ctx.extracted_data

    media_refs = []
    if ctx.metadata and ctx.metadata.get("media_refs"):
        media_refs = ctx.metadata["media_refs"]

    return EventNode(
        id=ctx.id,
        hierarchy_level=props.hierarchy_level if props else 0,
        time_bucket=props.time_bucket if props else None,
        parent_id=_normalize_parent_id(props),
        title=extracted.title if extracted else None,
        summary=extracted.summary if extracted else None,
        event_time=_format_timestamp(props.event_time if props else None),
        create_time=_format_timestamp(props.create_time if props else None),
        is_search_hit=False,
        media_refs=media_refs,
    )
```

**Step 4: 修改 _track_accessed_safe()**

Lines 450-464，在 items dict 中添加 media_refs：

```python
items.append({
    "id": er.id,
    "context_type": EVENT_TYPE,
    "title": er.title,
    "summary": er.summary,
    "keywords": er.keywords,
    "score": er.score,
    "event_time": er.event_time,
    "create_time": er.create_time,
    "media_refs": er.media_refs,  # 新增
})
```

**Step 5: 修改 RecentlyAccessedItem**

`opencontext/server/cache/models.py` lines 20-31，添加字段：

```python
class RecentlyAccessedItem(BaseModel):
    id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    context_type: str
    keywords: List[str] = Field(default_factory=list)
    accessed_ts: float
    score: Optional[float] = None
    event_time: Optional[str] = None
    create_time: Optional[str] = None
    media_refs: List[Dict[str, str]] = Field(default_factory=list)  # 新增
```

需要在文件顶部添加 `Dict` import（检查是否已有）。

**Step 6: 编译检查**

```bash
python -m py_compile opencontext/server/routes/search.py
python -m py_compile opencontext/server/cache/models.py
```

**Step 7: Commit**

```bash
git add opencontext/server/routes/search.py opencontext/server/cache/models.py
git commit -m "feat(search): media_refs in response + content parts query format"
```

---

### Task 9: API 文档更新

**Files:**
- Modify: `docs/curls.sh` (search 相关的 curl 示例)

**Step 1: 更新搜索 curl 示例**

找到搜索相关的 curl 命令（约 lines 260-302），将 `query`/`image_url`/`video_url` 顶层字段改为 content parts 格式：

```bash
# 原格式:
# "query": "team outing photos",
# "image_url": "https://example.com/reference-image.jpg",

# 改为:
# "query": [
#   {"type": "text", "text": "team outing photos"},
#   {"type": "image_url", "image_url": {"url": "https://example.com/reference-image.jpg"}}
# ],
```

添加纯文本搜索示例：

```bash
# "query": [{"type": "text", "text": "昨天的会议"}],
```

添加以图搜事件示例：

```bash
# "query": [{"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}],
```

**Step 2: Commit**

```bash
git add docs/curls.sh
git commit -m "docs: update search API examples to content parts format"
```

---

### Task 10: 最终验证 + MODULE.md 更新

**Files:**
- Modify: `opencontext/server/MODULE.md` (如果搜索 API 文档在此)
- Modify: `opencontext/models/MODULE.md` (Vectorize 文档)
- Modify: `opencontext/storage/object_storage/MODULE.md` (如有相关)

**Step 1: 全量编译检查**

```bash
python -m py_compile opencontext/models/context.py
python -m py_compile opencontext/llm/global_embedding_client.py
python -m py_compile opencontext/llm/llm_client.py
python -m py_compile opencontext/context_processing/processor/text_chat_processor.py
python -m py_compile opencontext/context_processing/processor/document_processor.py
python -m py_compile opencontext/context_processing/merger/merge_strategies.py
python -m py_compile opencontext/periodic_task/hierarchy_summary.py
python -m py_compile opencontext/server/context_operations.py
python -m py_compile opencontext/tools/retrieval_tools/base_context_retrieval_tool.py
python -m py_compile opencontext/tools/retrieval_tools/knowledge_retrieval_tool.py
python -m py_compile opencontext/tools/retrieval_tools/hierarchical_event_tool.py
python -m py_compile opencontext/tools/retrieval_tools/document_management_tool.py
python -m py_compile opencontext/storage/backends/vikingdb_backend.py
python -m py_compile opencontext/storage/backends/qdrant_backend.py
python -m py_compile opencontext/server/search/models.py
python -m py_compile opencontext/server/routes/search.py
python -m py_compile opencontext/server/cache/models.py
```

**Step 2: 更新 MODULE.md 文件**

按 CLAUDE.md 的维护规则，更新受影响模块的 MODULE.md：
- `opencontext/models/MODULE.md` — 更新 Vectorize 的字段说明
- `opencontext/server/MODULE.md` — 更新搜索 API 请求/响应格式

**Step 3: Commit**

```bash
git add -A
git commit -m "docs: update MODULE.md for Vectorize refactor and search multimodal"
```

---

## 任务依赖关系

```
Task 1 (Vectorize 模型) ──┬── Task 2 (Embedding Client)
                          ├── Task 3 (TextChatProcessor)
                          ├── Task 4 (Doc/Merger/Hierarchy/Ops)
                          ├── Task 5 (Retrieval Tools)
                          └── Task 6 (Storage Layer)

Task 7 (Search Models) ──── Task 8 (Search Routes + Cache)

Task 9 (API Docs) ← depends on Task 7, 8

Task 10 (Final Verify) ← depends on all above
```

Tasks 2-6 可以在 Task 1 完成后并行执行。
Tasks 7-8 独立于 Tasks 2-6，可以并行。
