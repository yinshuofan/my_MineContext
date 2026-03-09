# Search API 多模态适配设计

## 背景

Push API 已支持多模态（图片/视频），数据正确存储在 VikingDB 中（media_refs + 多模态融合向量）。但 Search API 存在两个问题：

1. **响应缺失**：`EventNode` 没有 `media_refs` 字段，`_to_search_hit_node()` 转换时丢弃了 metadata 中的媒体引用
2. **查询格式不一致**：搜索用独立的 `image_url`/`video_url` 顶层字段，与 Push API 的 OpenAI content parts 格式不统一
3. **Vectorize 模型冗余**：`text`/`images`/`videos` 分离字段在 `build_ark_input()` 时重新拼装成 content parts list，存储后重建的这些字段无任何下游消费者

## 方案

方案 B：响应修复 + 查询链路审计与补全 + Vectorize 重构。

## 设计

### 1. 响应模型变更

`EventNode` 新增 `media_refs` 字段：

```python
# opencontext/server/search/models.py
class EventNode(BaseModel):
    # ... 现有字段 ...
    media_refs: List[Dict[str, str]] = Field(default_factory=list)
    # 示例: [{"type": "image", "url": "https://..."}, {"type": "video", "url": "https://..."}]
```

- L0 事件节点携带 media_refs，L1/L2/L3 摘要节点为空列表（摘要本身没有媒体）
- 不在摘要节点聚合子节点的 media_refs，前端通过 children 树 drill-down 查看

### 2. 转换层修复

四个修复点，堵住数据丢失断点：

**2.1 `_to_search_hit_node()`**：从 `ProcessedContext.metadata` 提取 media_refs

**2.2 `_to_context_node()`**：祖先节点同样提取（保持一致性，摘要通常为空）

**2.3 `_track_accessed_safe()`**：写入 Redis 的最近访问记录携带 media_refs

**2.4 `RecentlyAccessedItem`**：缓存模型新增 `media_refs` 字段

### 3. 查询格式统一

#### 3.1 请求格式改为 OpenAI content parts

```json
{
  "query": [
    {"type": "text", "text": "找会议截图"},
    {"type": "image_url", "image_url": {"url": "https://..."}},
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}
  ]
}
```

- `EventSearchRequest.query` 类型改为 `Optional[List[Dict[str, Any]]]`
- 删除 `image_url`/`video_url` 顶层字段
- 不保留纯字符串向后兼容
- Base64 data URI 直接透传给 Ark API，搜索查询是临时的不需要上传对象存储

#### 3.2 验证逻辑调整

至少需要一个搜索条件：query / event_ids / time_range / hierarchy_levels。

### 4. Vectorize 模型重构

核心改动：用 `input: List[Dict]`（Ark API content parts 格式）替代分离的 `text`/`images`/`videos` 字段。

#### 审计结论

- `vectorize.text`/`images`/`videos` 在检索后无任何下游消费者，仅 `vectorize.vector` 被使用
- `metadata.media_refs` 已包含相同的媒体 URL，可完全替代
- VikingDB 无需 schema 迁移，缺失字段自动处理

#### 新 Vectorize 结构

```python
class Vectorize(BaseModel):
    input: List[Dict[str, Any]] = Field(default_factory=list)  # Ark API content parts
    vector: Optional[List[float]] = None
    content_format: ContentFormat = ContentFormat.TEXT

    def build_ark_input(self) -> List[Dict[str, Any]]:
        """返回 input，对 local path 做 data URI 转换"""
        # 扫描 input list，将 local path 转为 data URI，其余原样返回

    def get_modality_string(self) -> str:
        """从 input list 推导模态字符串"""
        # 扫描 type 字段，返回 "text and image and video" 格式
```

#### Push 侧统一

TextChatProcessor 直接构建 content parts list：

```python
ark_input = [{"type": "text", "text": f"{title}\n{summary}\n{keywords}"}]
for url in resolved_images:
    ark_input.append({"type": "image_url", "image_url": {"url": url}})
for url in resolved_videos:
    ark_input.append({"type": "video_url", "video_url": {"url": url, "fps": 1.0}})
vectorize = Vectorize(input=ark_input, content_format=ContentFormat.MULTIMODAL)
```

#### Search 侧统一

```python
vectorize = Vectorize(input=request.query, content_format=ContentFormat.MULTIMODAL)
await do_vectorize(vectorize, role="query")
raw_results = await storage.search(query=vectorize, ...)
```

#### 存储层清理

- VikingDB：停止写入/重建 `images`/`videos` 字段
- 保留 `content_modalities`（已索引，未来可用于模态过滤）
- 清除 collection schema 中的 images/videos 字段定义

### 5. 多模态查询链路确认

审计确认查询链路完整无缺口：

- `image_url`/`video_url` content parts → `Vectorize.input` → `build_ark_input()` → Ark 多模态 embedding API → 融合向量 → VikingDB 检索
- 三种查询模式已覆盖：纯文字、以图搜事件、混合查询
- Base64 data URI 由 Ark API 原生支持，零额外处理

## 影响范围

| 文件 | 改动类型 |
|------|---------|
| `opencontext/models/context.py` | Vectorize 重构（input 替代 text/images/videos） |
| `opencontext/server/search/models.py` | EventNode 加 media_refs；EventSearchRequest query 改类型 |
| `opencontext/server/routes/search.py` | 转换函数修复；查询构建改用 content parts |
| `opencontext/server/cache/models.py` | RecentlyAccessedItem 加 media_refs |
| `opencontext/context_processing/processor/text_chat_processor.py` | 构建 content parts list |
| `opencontext/context_processing/processor/document_processor.py` | 构建 content parts list |
| `opencontext/context_processing/merger/merge_strategies.py` | Vectorize 创建适配 |
| `opencontext/periodic_task/hierarchy_summary.py` | Vectorize 创建适配 |
| `opencontext/llm/global_embedding_client.py` | 适配新 Vectorize |
| `opencontext/tools/retrieval_tools/*.py` | Vectorize 创建适配 |
| `opencontext/server/context_operations.py` | Vectorize 创建适配 |
| `opencontext/storage/backends/vikingdb_backend.py` | 停止写入/重建 images/videos |
| `opencontext/storage/backends/qdrant_backend.py` | 适配新 Vectorize |
| `docs/curls.sh` | 更新 search API curl 示例 |

## 不做的事情

- 不加模态过滤（YAGNI）
- 不在 L1/L2/L3 摘要节点聚合子节点 media_refs
- 搜索 base64 不上传对象存储
- 不加文件大小校验（Ark API 有自身限制）
- 不保留 query 的纯字符串向后兼容
