# Search Web 页面多模态适配 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 更新搜索页面和记忆缓存页面，适配 Search API 多模态改动（query content parts 格式 + media_refs 展示）。

**Architecture:** 纯前端改动（HTML/JS/CSS），涉及两个 Jinja2 模板文件和一处后端 metadata 修复。搜索表单新增图片 URL 输入，查询转为 content parts 格式发送；搜索结果和记忆缓存渲染媒体引用缩略图/链接。

**Tech Stack:** Jinja2 模板、Bootstrap 5、vanilla JS、feather icons

---

### Task 1: 修复 SearchMetadata.query 序列化

**背景：** `SearchMetadata.query` 是 `Optional[str]`，但 `search.py` 现在传入 `request.query`（`List[Dict]`）。Pydantic v2 会把 list 强制转为字符串表示 `"[{'type': 'text', ...}]"`，前端 metadata 显示会乱码。

**Files:**
- Modify: `opencontext/server/routes/search.py`

**Step 1: 修改 metadata.query 赋值**

在 `search_events()` 函数中，将 `metadata=SearchMetadata(query=request.query, ...)` 改为提取文本预览：

```python
# 提取 query 文本预览用于 metadata 展示
query_text_for_meta = None
if request.query:
    text_parts = [item["text"] for item in request.query if item.get("type") == "text"]
    non_text = [item.get("type") for item in request.query if item.get("type") != "text"]
    preview = " ".join(text_parts) if text_parts else ""
    if non_text:
        modalities = []
        if "image_url" in non_text:
            modalities.append("[图片]")
        if "video_url" in non_text:
            modalities.append("[视频]")
        preview = (preview + " " + " ".join(modalities)).strip()
    query_text_for_meta = preview or None
```

然后所有三处 `SearchMetadata(query=request.query, ...)` 替换为 `SearchMetadata(query=query_text_for_meta, ...)`。共 3 处：成功响应（~line 94）、超时响应（~line 107）、异常响应（~line 120）。

**Step 2: 验证**

```bash
python -m py_compile opencontext/server/routes/search.py
```

**Step 3: Commit**

```bash
git add opencontext/server/routes/search.py
git commit -m "fix(search): extract text preview for SearchMetadata.query"
```

---

### Task 2: 搜索页面 query 格式适配

**背景：** `vector_search.html` 的 `handleSearch()` 在 line 180 发送 `requestBody.query = query`（纯字符串），但后端现在期望 `List[Dict]` content parts 格式。需要将文本包装为 `[{"type": "text", "text": "..."}]`。

**Files:**
- Modify: `opencontext/web/templates/vector_search.html`

**Step 1: 修改 handleSearch() 中的 query 构建**

将 line 180:
```javascript
if (hasQuery) requestBody.query = query;
```

改为：
```javascript
if (hasQuery) requestBody.query = [{"type": "text", "text": query}];
```

**Step 2: 修改 displayMetadata() 中的 query 显示**

`metadata.query` 现在是文本预览字符串（Task 1 已修复后端），无需额外改动。确认 line 245 `escapeHtml(metadata.query)` 仍正确即可。

**Step 3: 验证**

浏览器打开 `/vector_search`，输入文字查询，检查 Network 面板中请求 body 是否为 content parts 格式。

**Step 4: Commit**

```bash
git add opencontext/web/templates/vector_search.html
git commit -m "fix(web): wrap search query in content parts format"
```

---

### Task 3: 搜索页面新增图片 URL 输入

**背景：** 后端已支持多模态搜索查询（文字+图片+视频），前端需要提供图片输入方式。采用 URL 输入框，支持 HTTPS URL 和 base64 data URI。不做文件上传（YAGNI，搜索场景以 URL 为主）。

**Files:**
- Modify: `opencontext/web/templates/vector_search.html`

**Step 1: 在表单中 query textarea 下方添加图片 URL 输入**

在 `<div class="form-text">支持自然语言查询</div>` 的 `</div>`（line 19 的父 div 闭合处）后面，添加：

```html
<div class="mb-3">
    <label for="imageUrl" class="form-label">图片 URL（可选）</label>
    <input type="text" class="form-control" id="imageUrl" placeholder="https://... 或 data:image/...;base64,...">
    <div class="form-text">支持 HTTPS 图片链接或 base64 data URI，与文字组合进行多模态搜索</div>
</div>
```

**Step 2: 修改 handleSearch() 构建 content parts**

替换现有的 query 构建逻辑。读取 imageUrl，将文字和图片组合成 content parts list：

```javascript
const imageUrl = document.getElementById('imageUrl').value.trim();
const hasQuery = query.length > 0;
const hasImage = imageUrl.length > 0;
const hasLevels = selectedLevels.length > 0;
const hasTimeRange = startTime || endTime;

if (!hasQuery && !hasImage && !hasLevels && !hasTimeRange) {
    displayError('请至少输入查询内容、选择层级过滤或设置时间范围');
    return;
}
```

构建 query content parts：
```javascript
if (hasQuery || hasImage) {
    const queryParts = [];
    if (hasQuery) queryParts.push({"type": "text", "text": query});
    if (hasImage) queryParts.push({"type": "image_url", "image_url": {"url": imageUrl}});
    requestBody.query = queryParts;
}
```

**Step 3: 更新提示文案**

将 textarea 的 `<div class="form-text">支持自然语言查询</div>` 改为 `<div class="form-text">支持自然语言查询，可与图片组合进行多模态搜索</div>`。

将搜索结果初始提示从 `输入查询内容或选择层级过滤` 改为 `输入查询内容、图片 URL 或选择过滤条件，然后点击"开始搜索"`。

**Step 4: Commit**

```bash
git add opencontext/web/templates/vector_search.html
git commit -m "feat(web): add image URL input for multimodal search"
```

---

### Task 4: 搜索结果渲染 media_refs

**背景：** `EventNode` 现在携带 `media_refs`（如 `[{"type": "image", "url": "..."}, {"type": "video", "url": "..."}]`）。需要在搜索命中卡片中展示媒体缩略图和链接。

**Files:**
- Modify: `opencontext/web/templates/vector_search.html`

**Step 1: 添加 CSS 样式**

在 `<style>` 块中添加：

```css
.media-refs { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.media-ref-thumb { width: 80px; height: 80px; object-fit: cover; border-radius: 4px; border: 1px solid #dee2e6; cursor: pointer; }
.media-ref-thumb:hover { opacity: 0.8; box-shadow: 0 2px 6px rgba(0,0,0,0.15); }
.media-ref-video { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 4px; background: #f0f0f0; font-size: 0.8rem; color: #495057; text-decoration: none; }
.media-ref-video:hover { background: #e2e6ea; }
```

**Step 2: 添加 renderMediaRefs() 函数**

在 `escapeHtml()` 函数之后添加：

```javascript
function renderMediaRefs(mediaRefs) {
    if (!mediaRefs || mediaRefs.length === 0) return '';
    const items = mediaRefs.map(ref => {
        if (ref.type === 'image') {
            return `<a href="${escapeHtml(ref.url)}" target="_blank" rel="noopener"><img src="${escapeHtml(ref.url)}" class="media-ref-thumb" alt="image" loading="lazy" onerror="this.style.display='none'"></a>`;
        } else if (ref.type === 'video') {
            return `<a href="${escapeHtml(ref.url)}" target="_blank" rel="noopener" class="media-ref-video"><i data-feather="play-circle" style="width:14px;height:14px;"></i> 视频</a>`;
        }
        return '';
    }).filter(Boolean).join('');
    return items ? `<div class="media-refs">${items}</div>` : '';
}
```

**Step 3: 在 renderSearchHitCard() 中调用**

在 `${keywords ? ...}` 行之后、`<div class="d-flex justify-content-between ...">` 行之前，插入：

```javascript
${renderMediaRefs(event.media_refs)}
```

即在关键词和底部操作栏之间展示媒体缩略图。

**Step 4: Commit**

```bash
git add opencontext/web/templates/vector_search.html
git commit -m "feat(web): render media_refs thumbnails in search results"
```

---

### Task 5: 记忆缓存页面渲染 media_refs

**背景：** `RecentlyAccessedItem` 现在携带 `media_refs`。在"最近访问"列表中，如果事件有媒体引用，显示小图标提示。

**Files:**
- Modify: `opencontext/web/templates/memory_cache.html`

**Step 1: 在 renderRecentlyAccessed() 中添加 media 展示**

在 `renderRecentlyAccessed()` 函数中，`item.summary` 渲染行之后，添加 media_refs 展示：

```javascript
const mediaHtml = (item.media_refs && item.media_refs.length > 0)
    ? `<div class="mt-1">${item.media_refs.map(ref => {
        if (ref.type === 'image') return `<a href="${ref.url}" target="_blank" rel="noopener"><img src="${ref.url}" style="width:40px;height:40px;object-fit:cover;border-radius:3px;border:1px solid #dee2e6;margin-right:4px;" loading="lazy" onerror="this.style.display='none'"></a>`;
        if (ref.type === 'video') return `<a href="${ref.url}" target="_blank" rel="noopener" class="badge bg-dark me-1"><i data-feather="play-circle" style="width:10px;height:10px;"></i> 视频</a>`;
        return '';
      }).join('')}</div>`
    : '';
```

将 `mediaHtml` 插入到 summary 行之后：

```javascript
${item.summary ? `<small class="text-muted">...</small>` : ''}
${mediaHtml}
```

**Step 2: Commit**

```bash
git add opencontext/web/templates/memory_cache.html
git commit -m "feat(web): render media_refs in memory cache recently accessed"
```

---

### Task 6: 最终验证

**Step 1: 编译检查**

```bash
python -m py_compile opencontext/server/routes/search.py
python -m py_compile opencontext/server/search/models.py
```

**Step 2: 页面功能清单**

- [ ] `/vector_search` 页面纯文字搜索正常（query 以 content parts 格式发送）
- [ ] `/vector_search` 填写图片 URL 后可组合搜索
- [ ] 搜索结果中 L0 事件的 media_refs 图片显示为缩略图
- [ ] 搜索结果中 media_refs 视频显示为链接按钮
- [ ] metadata 区域 query 显示为可读文本（非 JSON 数组）
- [ ] `/memory_cache` 最近访问中有媒体的条目显示缩略图
