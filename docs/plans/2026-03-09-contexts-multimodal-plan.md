# Contexts 页面多模态展示 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 更新 contexts 列表页和详情页，展示多模态内容（media_refs 图片/视频）。

**Architecture:** 纯前端改动（2 个 Jinja2 模板）。列表页在卡片 header 添加媒体图标；详情页 metadata 区域智能渲染 media_refs 为实际图片/视频，原始数据区兼容 TOS URL。

**Tech Stack:** Jinja2 模板、Bootstrap 5、feather icons

---

### Task 1: 列表页卡片添加媒体图标

**背景：** `contexts.html` 的卡片 header 只显示类型 badge 和层级 badge。当 `context.metadata` 包含 `media_refs` 时，用户无法从列表区分哪些 context 有图片/视频。需要在类型 badge 旁添加 feather icon 指示。

**数据结构：** `context.metadata` 是 `Dict[str, Any]`，其中 `media_refs` 是 `List[Dict]`，每项有 `type` 字段（`"image"` 或 `"video"`）。

**Files:**
- Modify: `opencontext/web/templates/contexts.html:90-97`

**Step 1: 在类型 badge 后添加媒体图标**

在 `contexts.html` 的 card-header `<div>` 中（line 91-96），在 hierarchy level badge 的 `{% endif %}` 之后、`</div>` 之前，添加 media_refs 图标检测：

```html
                    {% if context.metadata and context.metadata.get('media_refs') %}
                    {% set media_types = context.metadata['media_refs'] | map(attribute='type') | list %}
                    {% if 'image' in media_types %}
                    <span data-feather="image" style="width:14px;height:14px;color:#6c757d;vertical-align:-2px;" class="ms-1" title="含图片"></span>
                    {% endif %}
                    {% if 'video' in media_types %}
                    <span data-feather="video" style="width:14px;height:14px;color:#6c757d;vertical-align:-2px;" class="ms-1" title="含视频"></span>
                    {% endif %}
                    {% endif %}
```

即在 line 96 的 `{% endif %}` 之后、line 97 的 `</div>` 之前插入。

**Step 2: 验证**

浏览器打开 `/contexts`，找到有 media_refs 的 context 卡片，确认图标显示正确。

**Step 3: Commit**

```bash
git add opencontext/web/templates/contexts.html
git commit -m "feat(web): add media type icons to context list cards"
```

---

### Task 2: 详情页 metadata 区域智能渲染 media_refs

**背景：** `context_detail.html` 的 metadata 区域（line 68-88）遍历 `context.metadata.items()`，用通用逻辑渲染值。`media_refs` 是 `List[Dict]`，当前被渲染为一堆 `<span class="badge">` 的 dict 对象，不可读。需要在循环中特判 `key == "media_refs"`，渲染为实际图片/视频。

**数据结构：** `media_refs` = `[{"type": "image", "url": "https://..."}, {"type": "video", "url": "https://..."}]`

**Files:**
- Modify: `opencontext/web/templates/context_detail.html:72-86`

**Step 1: 在 metadata 循环中添加 media_refs 特判**

将 line 72-86 的 metadata 循环替换为：

```html
                    {% for key, value in context.metadata.items() %}
                        <li class="list-group-item">
                            <strong>{{ key }}:</strong>
                            {% if key == 'media_refs' and value is iterable and value is not string %}
                                <div class="mt-2 d-flex flex-wrap gap-2">
                                    {% for ref in value %}
                                        {% if ref.type == 'image' and ref.url %}
                                            <a href="{{ ref.url }}" target="_blank" rel="noopener">
                                                <img src="{{ ref.url }}" style="max-width:200px;max-height:200px;object-fit:cover;border-radius:6px;border:1px solid #dee2e6;" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='inline'">
                                                <span style="display:none" class="text-muted small">图片加载失败</span>
                                            </a>
                                        {% elif ref.type == 'video' and ref.url %}
                                            <div style="max-width:320px;">
                                                <video controls preload="metadata" style="width:100%;border-radius:6px;border:1px solid #dee2e6;">
                                                    <source src="{{ ref.url }}">
                                                </video>
                                            </div>
                                        {% else %}
                                            <pre class="mb-0 bg-light p-2 rounded">{{ ref | tojson(indent=2) }}</pre>
                                        {% endif %}
                                    {% endfor %}
                                </div>
                            {% elif value is mapping %}
                                <pre class="mb-0 bg-light p-2 rounded">{{ value | tojson(indent=2) }}</pre>
                            {% elif value is iterable and value is not string %}
                                <div>
                                    {% for item in value %}
                                        <span class="badge bg-secondary">{{ item }}</span>
                                    {% endfor %}
                                </div>
                            {% else %}
                                <small class="text-muted">{{ value }}</small>
                            {% endif %}
                        </li>
                    {% endfor %}
```

**Step 2: 验证**

浏览器打开一个含 media_refs 的 context 详情页，确认图片内联显示、视频有播放器、点击图片新窗口打开。

**Step 3: Commit**

```bash
git add opencontext/web/templates/context_detail.html
git commit -m "feat(web): smart render media_refs in context detail metadata"
```

---

### Task 3: 详情页原始数据区兼容 TOS URL

**背景：** `context_detail.html` 原始数据区（line 177-190）用 `/files/{{ raw_context.content_path }}` 渲染图片/视频。TOS 上传的媒体 `content_path` 是完整 HTTPS URL（如 `https://bucket.tos.volces.com/...`），拼接 `/files/` 前缀会导致 404。需要判断 `content_path` 是否以 `http` 开头，是则直接用，否则拼 `/files/`。

**Files:**
- Modify: `opencontext/web/templates/context_detail.html:177-183`

**Step 1: 修改图片和视频的 src 路径逻辑**

将 line 177-183 替换为：

```html
                                {% if raw_context.content_format == 'image' and raw_context.content_path %}
                                    {% if raw_context.content_path.startswith('http') %}
                                    <img src="{{ raw_context.content_path }}" class="img-fluid rounded" alt="Image" loading="lazy" style="max-width:400px;">
                                    {% else %}
                                    <img src="/files/{{ raw_context.content_path }}" class="img-fluid rounded" alt="Image" loading="lazy" style="max-width:400px;">
                                    {% endif %}
                                {% elif raw_context.content_format == 'video' and raw_context.content_path %}
                                    {% if raw_context.content_path.startswith('http') %}
                                    <video controls class="img-fluid rounded" style="max-width:480px;">
                                        <source src="{{ raw_context.content_path }}">
                                    </video>
                                    {% else %}
                                    <video controls class="img-fluid rounded" style="max-width:480px;">
                                        <source src="/files/{{ raw_context.content_path }}" type="video/mp4">
                                    </video>
                                    {% endif %}
```

**Step 2: 验证**

打开一个 TOS 上传媒体的 context 详情，展开"原始数据"折叠区，确认图片/视频正常加载。

**Step 3: Commit**

```bash
git add opencontext/web/templates/context_detail.html
git commit -m "fix(web): support TOS URLs in context detail raw data section"
```

---

### Task 4: 最终验证

**Step 1: 编译检查（无 Python 改动，可跳过）**

本次全部是模板改动，无 Python 文件修改。

**Step 2: 页面功能清单**

- [ ] `/contexts` 列表页：含图片的 context 卡片显示 image 图标
- [ ] `/contexts` 列表页：含视频的 context 卡片显示 video 图标
- [ ] `/contexts` 列表页：无媒体的 context 卡片无多余图标
- [ ] 详情页 metadata 区域：media_refs 中图片渲染为 `<img>` 内联
- [ ] 详情页 metadata 区域：media_refs 中视频渲染为 `<video>` 播放器
- [ ] 详情页 metadata 区域：其他 metadata 键值对渲染不变
- [ ] 详情页原始数据：TOS URL 的 content_path 正常加载图片/视频
- [ ] 详情页原始数据：本地 content_path 仍通过 `/files/` 前缀正常加载
