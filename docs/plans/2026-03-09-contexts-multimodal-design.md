# Contexts 页面多模态展示 Design

## 问题

`/contexts` 页面无法展示多模态内容：
1. 列表页卡片无任何媒体标识，无法区分纯文本和含图片/视频的 context
2. 详情页 `metadata.media_refs`（TOS URL）被当作普通 JSON 显示，未渲染为实际媒体
3. 详情页原始数据区 `/files/{{ content_path }}` 只适用本地文件，TOS 上传的媒体无法显示

## 设计

### 列表页 (`contexts.html`)

在卡片标题旁添加媒体图标指示。当 `context.metadata` 中有 `media_refs` 时，根据类型显示 feather icon（`image` / `video`）。不显示实际缩略图，保持卡片轻量。

### 详情页 (`context_detail.html`)

1. **metadata 区域智能渲染**：在 metadata 渲染循环中，当 `key == "media_refs"` 时，不渲染 JSON，而是：
   - 图片 → `<img>` 内联显示，限制最大宽度，点击新窗口打开
   - 视频 → `<video>` 播放器
   - 其他 metadata 键值对保持原样

2. **原始数据区兼容 TOS URL**：当 `content_path` 以 `http://` 或 `https://` 开头时，直接用完整 URL 渲染，不拼 `/files/` 前缀。

## 改动范围

- `opencontext/web/templates/contexts.html` — 列表卡片添加媒体图标
- `opencontext/web/templates/context_detail.html` — metadata 智能渲染 + raw data URL 兼容

纯前端改动，无后端改动。
