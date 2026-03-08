# 搜索页媒体上传功能 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 搜索页支持上传图片/视频文件到 TOS，用返回的 HTTPS URL 进行多模态搜索。

**Architecture:** 新增轻量媒体上传端点 `POST /api/media/upload`，接收 multipart 文件，通过现有 `get_object_storage()` 上传至 TOS（或 local fallback），返回 URL。前端统一媒体区域支持粘贴图片、拖放文件、文件选择、视频 URL 输入，上传后用 TOS URL 构建搜索 query。

**Tech Stack:** FastAPI UploadFile, 现有 object_storage 模块 (S3CompatibleBackend / LocalBackend), vanilla JS

---

## 现有基础设施

- **对象存储**: `get_object_storage()` → `IObjectStorage.upload(data: bytes, key: str, content_type: str) -> str`
- **push.py 参考**: `_upload_or_save_media()` 已有格式校验、大小限制、key 生成、本地回退逻辑
- **本地文件服务**: `/files/uploads/...` 路由已配置，LocalBackend 存储的文件可直接访问
- **当前前端**: `vector_search.html` 已有粘贴图片功能（存 base64 data URI），需改为上传到后端

---

### Task 1: 新增媒体上传端点

**Files:**
- Create: `opencontext/server/routes/media.py`
- Modify: `opencontext/server/api.py`

**Step 1: 创建 media.py 路由文件**

```python
# opencontext/server/routes/media.py
"""
Media Upload API — upload images/videos to object storage for search queries.
"""

import uuid

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import JSONResponse

from opencontext.server.middleware.auth import auth_dependency
from opencontext.storage.object_storage import get_object_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/media", tags=["media"])

_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "bmp"}
_VIDEO_EXTS = {"mp4", "avi", "mov", "webm"}
_ALL_EXTS = _IMAGE_EXTS | _VIDEO_EXTS
_MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB
_MAX_VIDEO_BYTES = 50 * 1024 * 1024   # 50 MB

_MIME_MAP = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
    "mp4": "video/mp4", "avi": "video/x-msvideo", "mov": "video/quicktime",
    "webm": "video/webm",
}

_MEDIA_UPLOAD_DIR = "./uploads/media"


@router.post("/upload")
async def upload_media(
    file: UploadFile = File(...),
    user_id: str = Query(default="default"),
    _auth: str = auth_dependency,
):
    """
    Upload an image or video file for use in search queries.
    Returns the accessible URL of the uploaded file.
    """
    # Extract and validate extension
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALL_EXTS:
        return JSONResponse(
            status_code=400,
            content={"error": f"不支持的文件格式: .{ext}，支持: {', '.join(sorted(_ALL_EXTS))}"},
        )

    # Read file data
    data = await file.read()

    # Validate size
    media_type = "image" if ext in _IMAGE_EXTS else "video"
    max_size = _MAX_IMAGE_BYTES if media_type == "image" else _MAX_VIDEO_BYTES
    if len(data) > max_size:
        max_mb = max_size // (1024 * 1024)
        return JSONResponse(
            status_code=400,
            content={"error": f"文件过大（{len(data) / 1024 / 1024:.1f} MB），{media_type} 最大 {max_mb} MB"},
        )

    # Generate storage key
    key = f"search-media/{user_id}/{uuid.uuid4().hex}.{ext}"
    content_type = _MIME_MAP.get(ext, file.content_type or "application/octet-stream")

    # Upload to object storage or fallback to local
    obj_storage = get_object_storage()
    if obj_storage:
        try:
            url = await obj_storage.upload(data, key, content_type)
            logger.info(f"Media uploaded to object storage: {key} ({len(data)} bytes)")
        except Exception as e:
            logger.error(f"Object storage upload failed: {e}, falling back to local")
            url = await _save_local(data, key)
    else:
        url = await _save_local(data, key)

    return {"url": url, "type": media_type, "filename": filename}


async def _save_local(data: bytes, key: str) -> str:
    """Fallback: save file locally and return the path."""
    import os

    # key is "search-media/{user_id}/{uuid}.{ext}", flatten to just filename
    filename = key.rsplit("/", 1)[-1]
    local_dir = _MEDIA_UPLOAD_DIR
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    with open(local_path, "wb") as f:
        f.write(data)

    # Return path accessible via /files/ route
    return os.path.abspath(local_path)
```

**Step 2: 在 api.py 注册路由**

在 `opencontext/server/api.py` 中，找到 router include 块，添加:

```python
from .routes import media
router.include_router(media.router)
```

放在 `router.include_router(search.router)` 附近即可。

**Step 3: 验证**

```bash
python -m py_compile opencontext/server/routes/media.py
python -m py_compile opencontext/server/api.py
```

**Step 4: Commit**

```bash
git add opencontext/server/routes/media.py opencontext/server/api.py
git commit -m "feat(api): add POST /api/media/upload for search media files"
```

---

### Task 2: 重构搜索页媒体区域

**Files:**
- Modify: `opencontext/web/templates/vector_search.html`

**背景:** 当前页面只支持粘贴图片（存 base64 data URI 直传搜索）。需要改为：
1. 粘贴图片 → 上传到 `/api/media/upload` → 拿 URL
2. 拖放图片/视频 → 上传 → 拿 URL
3. 点击选择文件 → 上传 → 拿 URL
4. 视频 URL 手动输入（已有 HTTPS 链接的场景）
5. 预览：图片显示缩略图，视频显示文件名+图标
6. 搜索时用 URL（而非 base64）构建 content part

**Step 1: 替换 HTML 媒体区域**

替换当前 `imagePasteZone` 区域（line 21-35）为:

```html
<div class="mb-3">
    <label class="form-label">搜索媒体（可选）</label>
    <div id="mediaPasteZone" class="paste-zone" tabindex="0">
        <div id="mediaPlaceholder" class="paste-placeholder">
            <i data-feather="upload" style="width: 24px; height: 24px;" class="mb-1"></i>
            <div>粘贴图片 / 拖放文件 / <a href="#" id="mediaFileLink" class="text-decoration-none">选择文件</a></div>
            <small class="text-muted">支持图片和视频（jpg, png, gif, mp4, mov...）</small>
        </div>
        <div id="mediaPreview" class="paste-preview d-none">
            <div id="mediaPreviewContent"></div>
            <button type="button" class="btn-close paste-remove" id="mediaRemoveBtn" aria-label="移除"></button>
        </div>
        <input type="file" id="mediaFileInput" class="d-none" accept="image/*,video/*">
    </div>
    <div id="mediaUploadProgress" class="progress mt-1 d-none" style="height: 4px;">
        <div class="progress-bar" role="progressbar" style="width: 0%"></div>
    </div>
    <div class="mt-2">
        <div class="input-group input-group-sm">
            <span class="input-group-text" style="font-size: 0.8rem;">视频 URL</span>
            <input type="text" class="form-control" id="videoUrlInput" placeholder="https://... (已有视频链接可直接粘贴)">
            <button class="btn btn-outline-secondary" type="button" id="videoUrlConfirmBtn" style="font-size: 0.8rem;">确认</button>
        </div>
    </div>
</div>
```

**Step 2: 更新 CSS**

替换当前 `.paste-zone` 相关样式，更新/新增:

```css
.paste-zone { border: 2px dashed #dee2e6; border-radius: 6px; padding: 12px; text-align: center; cursor: pointer; transition: border-color 0.2s, background-color 0.2s; outline: none; position: relative; min-height: 80px; display: flex; align-items: center; justify-content: center; }
.paste-zone:hover, .paste-zone:focus { border-color: #0d6efd; background-color: #f8f9ff; }
.paste-zone.drag-over { border-color: #198754; background-color: #f0fff4; }
.paste-zone.has-media { border-style: solid; border-color: #198754; padding: 8px; }
.paste-zone.uploading { border-color: #ffc107; background-color: #fffdf0; pointer-events: none; opacity: 0.8; }
.paste-placeholder { color: #6c757d; font-size: 0.85rem; }
.paste-preview { position: relative; display: inline-block; }
.paste-preview-img { max-width: 100%; max-height: 120px; border-radius: 4px; }
.paste-remove { position: absolute; top: -6px; right: -6px; background: #fff; border-radius: 50%; box-shadow: 0 1px 4px rgba(0,0,0,0.2); }
.paste-video-info { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: #f8f9fa; border-radius: 4px; font-size: 0.85rem; }
```

**Step 3: 替换 JS 媒体处理逻辑**

替换 `let pastedImageDataUri = null;` 和整个 `initImagePasteZone()` 函数为:

```javascript
let uploadedMedia = null;  // { url: string, type: 'image'|'video', filename: string }

function initMediaZone() {
    const zone = document.getElementById('mediaPasteZone');
    const placeholder = document.getElementById('mediaPlaceholder');
    const preview = document.getElementById('mediaPreview');
    const previewContent = document.getElementById('mediaPreviewContent');
    const removeBtn = document.getElementById('mediaRemoveBtn');
    const fileInput = document.getElementById('mediaFileInput');
    const fileLink = document.getElementById('mediaFileLink');
    const progressBar = document.getElementById('mediaUploadProgress');
    const videoUrlInput = document.getElementById('videoUrlInput');
    const videoUrlConfirmBtn = document.getElementById('videoUrlConfirmBtn');

    // Click "选择文件" link → open file dialog
    fileLink.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });

    // Click zone (but not on link/button) → open file dialog
    zone.addEventListener('click', function(e) {
        if (e.target === zone || e.target.closest('.paste-placeholder')) {
            fileInput.click();
        }
    });

    // File input change
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
            fileInput.value = '';  // Reset so same file can be re-selected
        }
    });

    // Paste — on form level to catch paste anywhere
    document.getElementById('eventSearchForm').addEventListener('paste', function(e) {
        // Don't capture paste if user is typing in video URL input
        if (document.activeElement === videoUrlInput) return;

        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;

        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                uploadFile(item.getAsFile());
                return;
            }
        }
    });

    // Drag and drop
    zone.addEventListener('dragover', function(e) { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', function() { zone.classList.remove('drag-over'); });
    zone.addEventListener('drop', function(e) {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0 && (files[0].type.startsWith('image/') || files[0].type.startsWith('video/'))) {
            uploadFile(files[0]);
        }
    });

    // Video URL confirm
    videoUrlConfirmBtn.addEventListener('click', function() {
        const url = videoUrlInput.value.trim();
        if (!url) return;
        setMedia({ url: url, type: 'video', filename: url.split('/').pop() || 'video' });
        videoUrlInput.value = '';
    });
    videoUrlInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') { e.preventDefault(); videoUrlConfirmBtn.click(); }
    });

    // Remove
    removeBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        clearMedia();
    });

    async function uploadFile(file) {
        zone.classList.add('uploading');
        progressBar.classList.remove('d-none');
        progressBar.querySelector('.progress-bar').style.width = '30%';

        try {
            const formData = new FormData();
            formData.append('file', file);

            progressBar.querySelector('.progress-bar').style.width = '60%';

            const resp = await fetch('/api/media/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await resp.json();

            progressBar.querySelector('.progress-bar').style.width = '100%';

            if (!resp.ok) {
                throw new Error(data.error || data.detail || '上传失败');
            }

            setMedia(data);  // { url, type, filename }
        } catch (err) {
            console.error('Upload failed:', err);
            alert('媒体上传失败: ' + err.message);
        } finally {
            zone.classList.remove('uploading');
            setTimeout(() => {
                progressBar.classList.add('d-none');
                progressBar.querySelector('.progress-bar').style.width = '0%';
            }, 500);
        }
    }

    function setMedia(media) {
        uploadedMedia = media;
        if (media.type === 'image') {
            previewContent.innerHTML = `<img src="${escapeHtml(media.url)}" class="paste-preview-img" alt="preview">`;
        } else {
            previewContent.innerHTML = `
                <div class="paste-video-info">
                    <i data-feather="film" style="width:20px;height:20px;"></i>
                    <span>${escapeHtml(media.filename)}</span>
                </div>`;
            feather.replace();
        }
        preview.classList.remove('d-none');
        placeholder.classList.add('d-none');
        zone.classList.add('has-media');
    }

    function clearMedia() {
        uploadedMedia = null;
        previewContent.innerHTML = '';
        preview.classList.add('d-none');
        placeholder.classList.remove('d-none');
        zone.classList.remove('has-media');
    }
}
```

**Step 4: 更新 DOMContentLoaded 和 handleSearch**

在 `DOMContentLoaded` 中替换 `initImagePasteZone()` 为 `initMediaZone()`。

在 `handleSearch()` 中，替换图片 URL 读取和 query 构建逻辑:

```javascript
// 旧代码:
// const imageUrl = pastedImageDataUri || '';
// const hasImage = imageUrl.length > 0;
// ...
// if (hasImage) queryParts.push({"type": "image_url", "image_url": {"url": imageUrl}});

// 新代码:
const hasMedia = !!uploadedMedia;
// ...validation 中 hasImage 替换为 hasMedia...
if (!hasQuery && !hasMedia && !hasLevels && !hasTimeRange) { ... }
// ...query 构建:
if (hasQuery || hasMedia) {
    const queryParts = [];
    if (hasQuery) queryParts.push({"type": "text", "text": query});
    if (hasMedia) {
        if (uploadedMedia.type === 'image') {
            queryParts.push({"type": "image_url", "image_url": {"url": uploadedMedia.url}});
        } else {
            queryParts.push({"type": "video_url", "video_url": {"url": uploadedMedia.url, "fps": 1.0}});
        }
    }
    requestBody.query = queryParts;
}
```

**Step 5: Commit**

```bash
git add opencontext/web/templates/vector_search.html
git commit -m "feat(web): unified media upload zone with TOS upload, video support"
```

---

### Task 3: 验证

**Step 1: 编译检查**

```bash
python -m py_compile opencontext/server/routes/media.py
python -m py_compile opencontext/server/api.py
```

**Step 2: 功能清单**

- [ ] `POST /api/media/upload` 接受图片文件，返回 URL
- [ ] `POST /api/media/upload` 接受视频文件，返回 URL
- [ ] 格式校验：不支持的格式返回 400
- [ ] 大小校验：超限返回 400
- [ ] TOS 不可用时回退到本地存储
- [ ] 搜索页：粘贴图片 → 自动上传 → 显示缩略图预览
- [ ] 搜索页：拖放图片/视频 → 上传 → 预览
- [ ] 搜索页：点击选择文件 → 上传 → 预览
- [ ] 搜索页：输入视频 URL → 确认 → 显示视频信息
- [ ] 搜索页：点击 X 移除媒体
- [ ] 搜索时图片 URL 作为 `image_url` content part 发送
- [ ] 搜索时视频 URL 作为 `video_url` content part（带 fps: 1.0）发送
- [ ] 纯文字搜索仍正常
