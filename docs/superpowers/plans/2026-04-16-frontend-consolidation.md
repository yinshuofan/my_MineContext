# Frontend Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate 5 debug/inspection pages into 2 focused console pages under `scripts/devtools/`, plus refactor the `contexts.html` page from server-side to client-side rendering.

**Architecture:** Two new pages (Agent Console + Memory Explorer) with symmetric master-detail + tab layout, served from `scripts/devtools/`. Two new backend APIs (`GET /api/contexts`, `GET /api/users`) power the client-side rendering. Existing pages remain untouched except `contexts.html`.

**Tech Stack:** Python/FastAPI (backend APIs), Jinja2 + Bootstrap 5 + vanilla JS (frontend), pytest (backend tests)

**Spec:** `docs/superpowers/specs/2026-04-16-frontend-consolidation-design.md`

---

## Task 1: `GET /api/contexts` API Endpoint

Extract the query/filter/pagination logic from `web.py:read_contexts()` (lines 37-188) into a JSON API endpoint on `context.py`.

**Files:**
- Modify: `opencontext/server/routes/context.py:107-110` (insert new endpoint before `GET /api/contexts/{context_id}`)
- Test: `tests/server/routes/test_context_api.py`

- [ ] **Step 1: Write test for GET /api/contexts**

Create `tests/server/routes/test_context_api.py`:

```python
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.get_available_context_types = AsyncMock(
        return_value=["event", "knowledge", "document", "profile", "agent_profile"]
    )
    storage.get_filtered_context_count = AsyncMock(return_value=2)

    # Build minimal ProcessedContext-like objects
    def make_ctx(id, ctx_type, title, user_id="user1"):
        from opencontext.models.context import (
            ProcessedContext,
            ContextProperties,
            ExtractedData,
            VectorizeData,
        )
        from opencontext.models.enums import ContextType
        from opencontext.utils.time_utils import now as tz_now

        ts = tz_now()
        ctx = ProcessedContext(
            properties=ContextProperties(
                id=id,
                create_time=ts,
                update_time=ts,
                user_id=user_id,
                device_id="default",
                agent_id="default",
            ),
            extracted_data=ExtractedData(
                title=title,
                summary=f"Summary for {title}",
                context_type=ContextType(ctx_type),
            ),
            vectorize=VectorizeData(),
        )
        return ctx

    ctx1 = make_ctx("id1", "event", "Event One")
    ctx2 = make_ctx("id2", "knowledge", "Knowledge One")
    storage.get_all_processed_contexts = AsyncMock(
        return_value={"event": [ctx1], "knowledge": [ctx2]}
    )
    return storage


@pytest.mark.unit
class TestGetContextsAPI:

    @pytest.mark.asyncio
    async def test_returns_paginated_json(self, mock_storage):
        with (
            patch("opencontext.server.routes.context.get_storage", return_value=mock_storage),
            patch(
                "opencontext.server.routes.context.get_context_lab",
                return_value=MagicMock(),
            ),
            patch("opencontext.server.routes.context.auth_dependency", ""),
        ):
            from opencontext.cli import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/contexts", params={"page": 1, "limit": 10})

            assert resp.status_code == 200
            data = resp.json()["data"]
            assert "contexts" in data
            assert "total" in data
            assert "page" in data
            assert data["page"] == 1
            assert isinstance(data["contexts"], list)

    @pytest.mark.asyncio
    async def test_filters_out_profile_types(self, mock_storage):
        with (
            patch("opencontext.server.routes.context.get_storage", return_value=mock_storage),
            patch(
                "opencontext.server.routes.context.get_context_lab",
                return_value=MagicMock(),
            ),
            patch("opencontext.server.routes.context.auth_dependency", ""),
        ):
            from opencontext.cli import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/contexts")

            # Should have called get_filtered_context_count with types excluding profile/agent_profile
            call_args = mock_storage.get_filtered_context_count.call_args
            types_used = call_args.kwargs.get("context_types") or call_args.args[0]
            assert "profile" not in types_used
            assert "agent_profile" not in types_used
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/server/routes/test_context_api.py -v
```

Expected: FAIL — no `GET /api/contexts` endpoint exists yet.

- [ ] **Step 3: Implement GET /api/contexts endpoint**

In `opencontext/server/routes/context.py`, add the following imports (if not already present) near the top:

```python
import math
from opencontext.storage.global_storage import get_storage
```

Then insert **before** the `GET /api/contexts/{context_id}` handler (before line 110):

```python
@router.get("/api/contexts")
async def list_contexts(
    page: int = Query(1, ge=1),
    limit: int = Query(15, ge=1, le=100),
    type: str | None = Query(None),
    user_id: str | None = Query(None),
    device_id: str | None = Query(None),
    agent_id: str | None = Query(None),
    hierarchy_level: int | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    _auth: str = auth_dependency,
):
    """Paginated context list as JSON. Replaces the server-rendered /contexts logic."""
    storage = get_storage()
    if not storage:
        return convert_resp(data=None, code=503, status="error", message="Storage unavailable")

    # Build filter dict
    storage_filter = {}
    if hierarchy_level is not None:
        storage_filter["hierarchy_level"] = hierarchy_level
    if start_date:
        ts = _parse_date(start_date)
        if ts is not None:
            storage_filter.setdefault("create_time_ts", {})["$gte"] = ts
    if end_date:
        ts = _parse_date(end_date, is_end=True)
        if ts is not None:
            storage_filter.setdefault("create_time_ts", {})["$lt"] = ts

    # Filter out profile types
    all_types = await storage.get_available_context_types()
    excluded = {"profile", "agent_profile", "agent_base_profile"}
    available_types = [t for t in all_types if t not in excluded]
    context_types = [type] if type and type in available_types else available_types

    # Count + pagination
    total_count = await storage.get_filtered_context_count(
        context_types=context_types,
        filter=storage_filter,
        user_id=user_id,
        device_id=device_id,
        agent_id=agent_id,
    )
    total_pages = max(1, math.ceil(total_count / limit))
    page = min(page, total_pages)
    offset = (page - 1) * limit

    # Fetch
    results = await storage.get_all_processed_contexts(
        context_types=context_types,
        limit=limit + offset,
        offset=0,
        need_vector=False,
        filter=storage_filter,
        user_id=user_id,
        device_id=device_id,
        agent_id=agent_id,
        skip_slice=True,
    )
    all_contexts = []
    for ctx_list in results.values():
        all_contexts.extend(ctx_list)

    # Sort by create_time desc, then slice
    all_contexts.sort(key=lambda c: c.properties.create_time, reverse=True)
    page_contexts = all_contexts[offset : offset + limit]

    # Serialize (lightweight: exclude embedding and raw_contexts)
    items = []
    for ctx in page_contexts:
        try:
            model = ProcessedContextModel.from_processed_context(ctx, project_root)
            d = model.model_dump(exclude={"embedding", "raw_contexts"})
            items.append(d)
        except Exception as e:
            logger.warning(f"Failed to serialize context: {e}")

    return convert_resp(
        data={
            "contexts": items,
            "page": page,
            "limit": limit,
            "total": total_count,
            "total_pages": total_pages,
            "context_types": available_types,
        }
    )


def _parse_date(date_str: str, is_end: bool = False) -> float | None:
    """Parse date string to timestamp. For end dates, adds 1 day (date-only) or 1 minute (datetime)."""
    from datetime import timedelta
    from opencontext.utils.time_utils import now as tz_now

    tz = tz_now().tzinfo
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            from datetime import datetime

            dt = datetime.strptime(date_str, fmt)
            dt = dt.replace(tzinfo=tz)
            if is_end:
                dt += timedelta(days=1) if fmt == "%Y-%m-%d" else timedelta(minutes=1)
            return dt.timestamp()
        except ValueError:
            continue
    return None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/server/routes/test_context_api.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opencontext/server/routes/context.py tests/server/routes/test_context_api.py
git commit -m "feat(api): add GET /api/contexts paginated list endpoint"
```

---

## Task 2: `GET /api/users` API Endpoint

Add a storage method to query distinct `(user_id, device_id, agent_id)` tuples from the profiles table, then expose as an API endpoint.

**Files:**
- Modify: `opencontext/storage/base_storage.py` (add interface method to `IDocumentStorageBackend`)
- Modify: `opencontext/storage/backends/sqlite_backend.py` (implement)
- Modify: `opencontext/storage/backends/mysql_backend.py` (implement)
- Modify: `opencontext/storage/unified_storage.py` (add routing method)
- Create: `opencontext/server/routes/users_api.py`
- Modify: `opencontext/server/api.py:14-32` (add import) and `api.py:42-57` (include router)
- Test: `tests/server/routes/test_users_api.py`

- [ ] **Step 1: Add `list_distinct_users()` to IDocumentStorageBackend**

Read `opencontext/storage/base_storage.py` and find the `IDocumentStorageBackend` class. Add this abstract method after the existing methods:

```python
async def list_distinct_users(self) -> list[dict[str, str]]:
    """Return unique (user_id, device_id, agent_id) tuples from the profiles table."""
    raise NotImplementedError
```

- [ ] **Step 2: Implement in SQLite backend**

Read `opencontext/storage/backends/sqlite_backend.py`, find the profiles-related methods. Add:

```python
async def list_distinct_users(self) -> list[dict[str, str]]:
    conn = self._get_connection()
    cursor = conn.execute(
        "SELECT DISTINCT user_id, device_id, agent_id FROM profiles ORDER BY user_id"
    )
    return [
        {"user_id": row[0], "device_id": row[1], "agent_id": row[2]}
        for row in cursor.fetchall()
    ]
```

- [ ] **Step 3: Implement in MySQL backend**

Read `opencontext/storage/backends/mysql_backend.py`, find the profiles-related methods. Add:

```python
async def list_distinct_users(self) -> list[dict[str, str]]:
    conn = self._get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT DISTINCT user_id, device_id, agent_id FROM profiles ORDER BY user_id"
            )
            return [
                {"user_id": row["user_id"], "device_id": row["device_id"], "agent_id": row["agent_id"]}
                for row in cursor.fetchall()
            ]
    finally:
        conn.commit()
```

Note: MySQL backend uses `DictCursor` — verify by reading the `_get_connection()` method. The `finally: conn.commit()` follows the MVCC snapshot pattern documented in CLAUDE.md.

- [ ] **Step 4: Add routing method to UnifiedStorage**

Read `opencontext/storage/unified_storage.py`. Add near the other profile-related methods:

```python
async def list_distinct_users(self) -> list[dict[str, str]]:
    """Return unique (user_id, device_id, agent_id) tuples from the document backend."""
    return await asyncio.to_thread(self._document_backend.list_distinct_users)
```

Note: `list_distinct_users` on the backend is a sync method (uses thread-local connections). Wrap with `asyncio.to_thread` following the same pattern as other `UnifiedStorage` methods that delegate to the document backend. Check how existing methods like `upsert_profile` handle this.

- [ ] **Step 5: Create route file and write test**

Create `opencontext/server/routes/users_api.py`:

```python
from fastapi import APIRouter
from opencontext.server.routes.common import auth_dependency, convert_resp
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["users"])


@router.get("/api/users")
async def list_users(_auth: str = auth_dependency):
    """Return unique (user_id, device_id, agent_id) tuples."""
    storage = get_storage()
    if not storage:
        return convert_resp(data=None, code=503, status="error", message="Storage unavailable")
    try:
        users = await storage.list_distinct_users()
        return convert_resp(data={"users": users, "total": len(users)})
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        return convert_resp(data=None, code=500, status="error", message=str(e))
```

Create `tests/server/routes/test_users_api.py`:

```python
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.mark.unit
class TestListUsersAPI:

    @pytest.mark.asyncio
    async def test_returns_user_list(self):
        mock_storage = MagicMock()
        mock_storage.list_distinct_users = AsyncMock(
            return_value=[
                {"user_id": "u1", "device_id": "default", "agent_id": "default"},
                {"user_id": "u2", "device_id": "mobile", "agent_id": "agent1"},
            ]
        )
        with (
            patch("opencontext.server.routes.users_api.get_storage", return_value=mock_storage),
            patch("opencontext.server.routes.users_api.auth_dependency", ""),
        ):
            from opencontext.server.routes.users_api import list_users

            result = await list_users(_auth="")
            data = result["data"]
            assert data["total"] == 2
            assert data["users"][0]["user_id"] == "u1"
```

- [ ] **Step 6: Register router in api.py**

Read `opencontext/server/api.py`. Add import alongside existing imports (around line 14-32):

```python
from opencontext.server.routes import users_api
```

Add router include alongside existing includes (around line 42-57):

```python
router.include_router(users_api.router)
```

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/server/routes/test_users_api.py -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add opencontext/storage/base_storage.py opencontext/storage/backends/sqlite_backend.py opencontext/storage/backends/mysql_backend.py opencontext/storage/unified_storage.py opencontext/server/routes/users_api.py opencontext/server/api.py tests/server/routes/test_users_api.py
git commit -m "feat(api): add GET /api/users endpoint for unique user triples"
```

---

## Task 3: Refactor `contexts.html` to Client-Side Rendering

Convert `contexts.html` from Jinja2 server-rendered to client-side JS fetch, and simplify `web.py:read_contexts()` to a static template render.

**Files:**
- Modify: `opencontext/server/routes/web.py:37-188` (simplify `read_contexts`)
- Modify: `opencontext/web/templates/contexts.html` (rewrite to client-side rendering)

- [ ] **Step 1: Simplify `read_contexts()` in web.py**

Read `opencontext/server/routes/web.py`. Replace the `read_contexts` function (lines 37-188) with:

```python
@router.get("/contexts", response_class=HTMLResponse)
async def read_contexts(request: Request):
    """Render contexts page — data loaded client-side via GET /api/contexts."""
    return templates.TemplateResponse("contexts.html", {"request": request, "title": "上下文列表"})
```

This removes all the server-side query/filter/pagination logic (now handled by `GET /api/contexts` from Task 1). Remove any imports that were only used by the old `read_contexts` function (e.g., `math`, `ProcessedContextModel`, `get_storage`) — but verify they aren't used elsewhere in the file first.

- [ ] **Step 2: Rewrite `contexts.html` for client-side rendering**

Read the current `opencontext/web/templates/contexts.html` to understand the full HTML structure. Then rewrite it to:

1. Keep `{% extends "base.html" %}` and `{% block content %}`
2. Keep the filter bar HTML structure, but replace Jinja2 template variables with static defaults (empty inputs, all-types dropdown populated by JS)
3. Replace the Jinja2-rendered card grid with an empty container `<div id="contextGrid" class="row">` 
4. Replace the Jinja2-rendered pagination with an empty `<div id="paginationArea">`
5. Add JavaScript in `{% block scripts %}` that:

```javascript
// State
let currentPage = 1;
let currentLimit = 15;

document.addEventListener('DOMContentLoaded', () => {
    loadContextTypes();
    loadContexts(1);
});

async function loadContextTypes() {
    // Fetch available types for the filter dropdown
    const resp = await fetch('/api/context_types');
    const data = await resp.json();
    const select = document.getElementById('typeFilter');
    select.innerHTML = '<option value="">全部</option>';
    (data.data || []).forEach(t => {
        select.innerHTML += `<option value="${esc(t)}">${esc(t)}</option>`;
    });
}

function getFilterParams() {
    const params = new URLSearchParams();
    const fields = {
        type: 'typeFilter', user_id: 'userIdFilter',
        device_id: 'deviceIdFilter', agent_id: 'agentIdFilter',
        hierarchy_level: 'levelFilter',
    };
    for (const [key, id] of Object.entries(fields)) {
        const val = document.getElementById(id)?.value?.trim();
        if (val) params.set(key, val);
    }
    // Date range
    const sd = document.getElementById('startDate')?.value;
    const ed = document.getElementById('endDate')?.value;
    if (sd) params.set('start_date', sd);
    if (ed) params.set('end_date', ed);
    return params;
}

async function loadContexts(page) {
    currentPage = page;
    const params = getFilterParams();
    params.set('page', page);
    params.set('limit', currentLimit);

    const resp = await fetch(`/api/contexts?${params}`);
    const json = await resp.json();
    const data = json.data;

    renderContextCards(data.contexts || []);
    renderPagination(data.page, data.total_pages, data.total);
}

function renderContextCards(contexts) {
    const grid = document.getElementById('contextGrid');
    if (!contexts.length) {
        grid.innerHTML = '<div class="col-12 text-center text-muted py-5">暂无数据</div>';
        return;
    }
    grid.innerHTML = contexts.map(ctx => renderContextCard(ctx)).join('');
    feather.replace();
}
```

The `renderContextCard(ctx)` function should produce the same HTML card structure as the existing Jinja2 template (type badge, level badge, media icons, truncated ID, title, summary, user/device/agent IDs, create time, view/delete buttons). Port the card HTML structure from the current `contexts.html` lines 86-134, converting Jinja2 expressions to JS template literals.

The `renderPagination(page, totalPages, total)` function should produce the same pagination bar: per-page selector, page navigation with prev/next + numbered buttons, jump-to-page input. Port the HTML structure from `contexts.html` lines 155-210.

Also port these utility functions from the current `contexts.html` JS:
- `viewContext(contextId, contextType)` (lines 560-579)
- `copyId(el, fullId)` (lines 532-558) with `fallbackCopyText` (lines 488-513) and `showCopyFeedback` (lines 515-529)
- `esc()` — use the same `document.createElement('div').textContent` pattern from agents.html

Add the delete handler as a function `deleteContext(id, type, el)` that calls `POST /contexts/delete` and removes the card from DOM on success.

- [ ] **Step 3: Verify in browser**

```bash
uv run opencontext start
```

Open `http://localhost:1733/contexts`. Verify:
- Filter bar works (type dropdown, user_id, date range)
- Cards render with correct type badges and content
- Pagination works (page navigation, per-page selector, jump-to-page)
- View detail button opens context detail
- Delete button removes card
- Click-to-copy IDs work

- [ ] **Step 4: Commit**

```bash
git add opencontext/server/routes/web.py opencontext/web/templates/contexts.html
git commit -m "refactor(contexts): convert from server-side to client-side rendering"
```

---

## Task 4: Devtools Infrastructure

Create the `scripts/devtools/` directory structure, base template, shared JS utilities, routes, and mount them in the app.

**Files:**
- Create: `scripts/devtools/routes.py`
- Create: `scripts/devtools/templates/base.html`
- Create: `scripts/devtools/static/css/console.css`
- Create: `scripts/devtools/static/js/shared.js`
- Modify: `opencontext/server/api.py` (import and include devtools router)
- Modify: `opencontext/cli.py:312-333` (add static mount + template directory)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p scripts/devtools/templates scripts/devtools/static/js scripts/devtools/static/css
```

- [ ] **Step 2: Create `scripts/devtools/routes.py`**

```python
"""Devtools console page routes."""
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/console", tags=["devtools-console"])

templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/agents", response_class=HTMLResponse)
async def agent_console(request: Request):
    return templates.TemplateResponse("agent_console.html", {"request": request})


@router.get("/memory", response_class=HTMLResponse)
async def memory_explorer(request: Request):
    return templates.TemplateResponse("memory_explorer.html", {"request": request})
```

- [ ] **Step 3: Create `scripts/devtools/templates/base.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Console{% endblock %} - MineContext</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <link href="/console/static/css/console.css" rel="stylesheet">
    {% block head %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-dark bg-dark fixed-top">
        <div class="container-fluid">
            <a class="navbar-brand" href="/console/agents">MineContext</a>
            <div class="d-flex align-items-center">
                <ul class="nav nav-pills me-3">
                    <li class="nav-item">
                        <a class="nav-link console-nav-link" href="/console/agents">Agent 调试台</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link console-nav-link" href="/console/memory">记忆浏览器</a>
                    </li>
                </ul>
                <button class="btn btn-outline-secondary btn-sm" onclick="window.apiAuth && window.apiAuth.showAPIKeyManager()">
                    <i data-feather="key" style="width:14px;height:14px"></i> API Key
                </button>
            </div>
        </div>
    </nav>

    <main class="console-main">
        {% block content %}{% endblock %}
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/feather-icons@4.28.0/dist/feather.min.js"></script>
    <script src="/static/js/api_auth.js"></script>
    <script src="/console/static/js/shared.js"></script>
    <script>
        // Highlight active nav link
        document.querySelectorAll('.console-nav-link').forEach(link => {
            if (window.location.pathname.startsWith(link.getAttribute('href'))) {
                link.classList.add('active');
            }
        });
        feather.replace();
    </script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

- [ ] **Step 4: Create `scripts/devtools/static/css/console.css`**

```css
body { background: #f8f9fa; }

.console-main {
    padding-top: 60px;
    height: 100vh;
    overflow: hidden;
}

/* Master-detail layout */
.console-container {
    display: flex;
    height: calc(100vh - 56px);
}

.console-sidebar {
    width: 190px;
    min-width: 190px;
    border-right: 1px solid #dee2e6;
    background: #fff;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.console-sidebar .sidebar-header {
    padding: 8px 10px;
    border-bottom: 1px solid #dee2e6;
}

.console-sidebar .sidebar-list {
    flex: 1;
    overflow-y: auto;
}

.console-sidebar .sidebar-footer {
    padding: 6px 10px;
    border-top: 1px solid #dee2e6;
    font-size: 12px;
    color: #6c757d;
}

.sidebar-item {
    padding: 8px 10px;
    cursor: pointer;
    border-left: 3px solid transparent;
    text-decoration: none;
    color: inherit;
    display: block;
}

.sidebar-item:hover { background: #f8f9fa; }

.sidebar-item.active {
    background: #e8f0fe;
    border-left-color: #0d6efd;
}

.sidebar-item .item-primary { font-size: 13px; }
.sidebar-item .item-secondary { font-size: 11px; color: #6c757d; }

/* Tab content area */
.console-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.console-content .tab-content {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
}

/* Context cards */
.ctx-card {
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 8px;
    background: #fff;
    cursor: pointer;
    transition: box-shadow 0.15s;
}
.ctx-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }

/* Level badges */
.badge-l0 { background: #198754; }
.badge-l1 { background: #0d6efd; }
.badge-l2 { background: #6f42c1; }
.badge-l3 { background: #dc3545; }

/* Type badges */
.badge-event { background: #ffc107; color: #000; }
.badge-knowledge { background: #6f42c1; }
.badge-document { background: #198754; }
.badge-profile { background: #0d6efd; }

/* Responsive */
@media (max-width: 768px) {
    .console-sidebar { display: none; }
    .console-content { width: 100%; }
}
```

- [ ] **Step 5: Create `scripts/devtools/static/js/shared.js`**

```javascript
/**
 * Shared utilities for devtools console pages.
 */

// --- HTML escaping ---
function esc(str) {
    if (str == null) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function escAttr(str) {
    return esc(str).replace(/'/g, '&#39;');
}

// --- Clipboard ---
function fallbackCopyText(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
}

function copyToClipboard(text, feedbackEl) {
    const doCopy = navigator.clipboard
        ? navigator.clipboard.writeText(text)
        : Promise.resolve(fallbackCopyText(text));
    doCopy.then(() => {
        if (feedbackEl) {
            const orig = feedbackEl.textContent;
            feedbackEl.textContent = '已复制';
            setTimeout(() => { feedbackEl.textContent = orig; }, 1000);
        }
    });
}

// --- Date formatting ---
function formatDatetime(dtStr) {
    if (!dtStr) return '';
    return dtStr.substring(0, 19).replace('T', ' ');
}

function formatDateShort(dtStr) {
    if (!dtStr) return '';
    return dtStr.substring(0, 16).replace('T', ' ');
}

// --- Context type colors ---
const TYPE_COLORS = {
    event: 'warning',
    knowledge: 'purple',
    document: 'success',
    profile: 'primary',
    agent_profile: 'primary',
    daily_summary: 'info',
    weekly_summary: 'info',
    monthly_summary: 'info',
};

function typeBadgeClass(type) {
    return TYPE_COLORS[type] || 'secondary';
}

// --- Hierarchy level ---
const LEVEL_LABELS = { 0: 'L0 原始', 1: 'L1 日摘要', 2: 'L2 周摘要', 3: 'L3 月摘要' };
const LEVEL_SHORT = { 0: 'L0', 1: 'L1', 2: 'L2', 3: 'L3' };
const LEVEL_BADGE_CLASS = { 0: 'badge-l0', 1: 'badge-l1', 2: 'badge-l2', 3: 'badge-l3' };

function levelBadge(level) {
    const label = LEVEL_SHORT[level] || `L${level}`;
    const cls = LEVEL_BADGE_CLASS[level] || 'bg-secondary';
    return `<span class="badge ${cls}">${label}</span>`;
}

// --- View context detail ---
function viewContext(contextId, contextType) {
    fetch('/contexts/detail', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: contextId, context_type: contextType }),
    })
    .then(resp => resp.text())
    .then(html => {
        document.open();
        document.write(html);
        document.close();
    })
    .catch(err => alert('Failed to load detail: ' + err.message));
}

// --- Pagination renderer ---
function renderPaginationHTML(page, totalPages, total, limit, onPageClick) {
    if (totalPages <= 1) return `<span class="text-muted small">共 ${total} 条</span>`;

    let html = `<div class="d-flex align-items-center justify-content-between flex-wrap gap-2">`;
    html += `<span class="text-muted small">共 ${total} 条，第 ${page}/${totalPages} 页</span>`;
    html += `<nav><ul class="pagination pagination-sm mb-0">`;

    // Prev
    html += `<li class="page-item ${page <= 1 ? 'disabled' : ''}">`;
    html += `<a class="page-link" href="#" onclick="${onPageClick}(${page - 1});return false;">‹</a></li>`;

    // Page numbers with window
    const start = Math.max(1, page - 2);
    const end = Math.min(totalPages, page + 2);
    if (start > 1) {
        html += `<li class="page-item"><a class="page-link" href="#" onclick="${onPageClick}(1);return false;">1</a></li>`;
        if (start > 2) html += `<li class="page-item disabled"><span class="page-link">…</span></li>`;
    }
    for (let i = start; i <= end; i++) {
        html += `<li class="page-item ${i === page ? 'active' : ''}">`;
        html += `<a class="page-link" href="#" onclick="${onPageClick}(${i});return false;">${i}</a></li>`;
    }
    if (end < totalPages) {
        if (end < totalPages - 1) html += `<li class="page-item disabled"><span class="page-link">…</span></li>`;
        html += `<li class="page-item"><a class="page-link" href="#" onclick="${onPageClick}(${totalPages});return false;">${totalPages}</a></li>`;
    }

    // Next
    html += `<li class="page-item ${page >= totalPages ? 'disabled' : ''}">`;
    html += `<a class="page-link" href="#" onclick="${onPageClick}(${page + 1});return false;">›</a></li>`;

    html += `</ul></nav></div>`;
    return html;
}

// --- Collapsible card wrapper (from memory_cache.html pattern) ---
function wrapCard(title, icon, content, sectionId) {
    return `
    <div class="card mb-3">
        <div class="card-header d-flex justify-content-between align-items-center"
             style="cursor:pointer" data-bs-toggle="collapse" data-bs-target="#${sectionId}-body">
            <span><i data-feather="${icon}" style="width:16px;height:16px"></i> ${esc(title)}</span>
            <i data-feather="chevron-down" style="width:14px;height:14px"></i>
        </div>
        <div id="${sectionId}-body" class="collapse show">
            <div class="card-body">${content}</div>
        </div>
    </div>`;
}

// --- Context card renderer (for Contexts tab, shared by both pages) ---
function renderContextCardHTML(ctx) {
    const typeClass = typeBadgeClass(ctx.context_type);
    const mediaIcons = (ctx.metadata?.media_refs || []).length > 0
        ? ' <i data-feather="image" style="width:12px;height:12px"></i>' : '';
    const truncId = (ctx.id || '').substring(0, 8);
    const summary = (ctx.summary || '').substring(0, 120);

    return `
    <div class="col-md-6 mb-2">
        <div class="ctx-card" onclick="viewContext('${escAttr(ctx.id)}','${escAttr(ctx.context_type)}')">
            <div class="d-flex justify-content-between align-items-start mb-1">
                <div>
                    <span class="badge bg-${typeClass}">${esc(ctx.context_type)}</span>
                    ${ctx.hierarchy_level > 0 ? levelBadge(ctx.hierarchy_level) : ''}
                    ${mediaIcons}
                </div>
                <small class="text-muted" style="cursor:pointer"
                    onclick="event.stopPropagation();copyToClipboard('${escAttr(ctx.id)}',this)"
                    title="${esc(ctx.id)}">${esc(truncId)}…</small>
            </div>
            <div class="fw-bold small">${esc(ctx.title || '(无标题)')}</div>
            <div class="text-muted small text-truncate">${esc(summary)}</div>
            <div class="d-flex justify-content-between align-items-center mt-1">
                <small class="text-muted">${formatDateShort(ctx.create_time)}</small>
                <button class="btn btn-outline-danger btn-sm py-0 px-1" style="font-size:11px"
                    onclick="event.stopPropagation();deleteContextItem('${escAttr(ctx.id)}','${escAttr(ctx.context_type)}',this)">
                    删除</button>
            </div>
        </div>
    </div>`;
}

// --- Delete context ---
async function deleteContextItem(id, type, el) {
    if (!confirm('确定删除此 context?')) return;
    const resp = await fetch('/contexts/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, context_type: type }),
    });
    if (resp.ok) {
        const card = el.closest('.col-md-6');
        if (card) card.remove();
    } else {
        alert('删除失败');
    }
}

// --- Feather icons refresh ---
function refreshIcons() {
    if (typeof feather !== 'undefined') feather.replace();
}
```

- [ ] **Step 6: Mount devtools static files and register router**

Read `opencontext/cli.py` (specifically the `_setup_static_files` function around lines 312-331 and `app.include_router` at line 333). Add the devtools static mount and router registration.

In `_setup_static_files()`, add after the existing static mount:

```python
# Devtools console static files
devtools_static = Path(__file__).parents[1] / "scripts" / "devtools" / "static"
if devtools_static.exists():
    app.mount("/console/static", StaticFiles(directory=str(devtools_static)), name="console_static")
```

In `opencontext/server/api.py`, add the import and include:

```python
# At the top, with other imports — use a relative path to locate scripts/devtools
import sys
from pathlib import Path
_devtools_path = Path(__file__).parents[2] / "scripts" / "devtools"
if _devtools_path.exists():
    sys.path.insert(0, str(_devtools_path.parent))
    from scripts.devtools.routes import router as devtools_router
```

Wait — this import path is fragile. A more robust approach: add the devtools router registration in `cli.py` alongside the static mount, since `cli.py` already handles app setup:

In `cli.py`, after `app.include_router(api_router)` at line 333, add:

```python
# Devtools console routes
devtools_routes = Path(__file__).parents[1] / "scripts" / "devtools" / "routes.py"
if devtools_routes.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("devtools_routes", devtools_routes)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    app.include_router(mod.router)
```

This dynamically imports the devtools router only if it exists, keeping it fully optional.

- [ ] **Step 7: Verify devtools infrastructure works**

```bash
uv run opencontext start
```

Open `http://localhost:1733/console/agents` — should see the base template with top nav and empty content area. Open `http://localhost:1733/console/memory` — same. Verify:
- Top nav shows both links, active link is highlighted
- API Key button is present
- No JS errors in browser console

- [ ] **Step 8: Commit**

```bash
git add scripts/devtools/ opencontext/cli.py
git commit -m "feat(devtools): add console infrastructure (routes, base template, shared.js)"
```

---

## Task 5: Agent Console — Template + Tab 1 (Basic Info)

Create the Agent Console page with left sidebar agent list and Tab 1 (basic info, profile, base events).

**Files:**
- Create: `scripts/devtools/templates/agent_console.html`
- Create: `scripts/devtools/static/js/agent_console.js`

- [ ] **Step 1: Create `agent_console.html` template**

```html
{% extends "base.html" %}
{% block title %}Agent 调试台{% endblock %}

{% block content %}
<div class="console-container">
    <!-- Left sidebar: Agent list -->
    <div class="console-sidebar">
        <div class="sidebar-header">
            <input type="text" class="form-control form-control-sm" id="agentSearch"
                placeholder="搜索 Agent..." oninput="filterAgentList(this.value)">
        </div>
        <div class="sidebar-header p-1">
            <button class="btn btn-primary btn-sm w-100" onclick="showCreateAgentModal()">
                <i data-feather="plus" style="width:14px;height:14px"></i> 新建
            </button>
        </div>
        <div class="sidebar-list" id="agentList"></div>
        <div class="sidebar-footer" id="agentCount"></div>
    </div>

    <!-- Right: Tab content -->
    <div class="console-content">
        <ul class="nav nav-tabs px-2 pt-1" id="agentTabs">
            <li class="nav-item">
                <a class="nav-link active" data-bs-toggle="tab" href="#tab-info">基本信息</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#tab-batches">消息追踪</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#tab-contexts">关联 Contexts</a>
            </li>
        </ul>
        <div class="tab-content" id="agentTabContent">
            <!-- Tab 1: Basic Info -->
            <div class="tab-pane fade show active" id="tab-info">
                <div id="infoArea" class="text-center text-muted py-5">请从左侧选择一个 Agent</div>
            </div>
            <!-- Tab 2: Message Tracing -->
            <div class="tab-pane fade" id="tab-batches">
                <div id="batchesArea"></div>
            </div>
            <!-- Tab 3: Related Contexts -->
            <div class="tab-pane fade" id="tab-contexts">
                <div id="agentContextsArea"></div>
            </div>
        </div>
    </div>
</div>

<!-- Create Agent Modal -->
<div class="modal fade" id="createAgentModal" tabindex="-1">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header"><h5>新建 Agent</h5><button type="button" class="btn-close" data-bs-dismiss="modal"></button></div>
            <div class="modal-body">
                <div class="mb-3"><label class="form-label">Agent ID *</label><input type="text" class="form-control" id="newAgentId"></div>
                <div class="mb-3"><label class="form-label">名称 *</label><input type="text" class="form-control" id="newAgentName"></div>
                <div class="mb-3"><label class="form-label">描述</label><input type="text" class="form-control" id="newAgentDesc"></div>
            </div>
            <div class="modal-footer"><button class="btn btn-primary" onclick="createAgent()">创建</button></div>
        </div>
    </div>
</div>

<!-- Add Event Modal -->
<div class="modal fade" id="addEventModal" tabindex="-1">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header"><h5>添加 Base Event</h5><button type="button" class="btn-close" data-bs-dismiss="modal"></button></div>
            <div class="modal-body">
                <div class="mb-3"><label class="form-label">标题 *</label><input type="text" class="form-control" id="newEventTitle"></div>
                <div class="mb-3"><label class="form-label">摘要 *</label><textarea class="form-control" id="newEventSummary" rows="3"></textarea></div>
                <div class="mb-3"><label class="form-label">时间</label><input type="datetime-local" class="form-control" id="newEventTime"></div>
                <div class="mb-3"><label class="form-label">关键词</label><input type="text" class="form-control" id="newEventKeywords" placeholder="逗号分隔"></div>
                <div class="mb-3"><label class="form-label">重要性 <span id="importanceVal">5</span></label><input type="range" class="form-range" id="newEventImportance" min="0" max="10" value="5" oninput="document.getElementById('importanceVal').textContent=this.value"></div>
            </div>
            <div class="modal-footer"><button class="btn btn-primary" onclick="addBaseEvent()">添加</button></div>
        </div>
    </div>
</div>

<!-- Batch Import Modal -->
<div class="modal fade" id="batchImportModal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header"><h5>批量导入 Base Events</h5><button type="button" class="btn-close" data-bs-dismiss="modal"></button></div>
            <div class="modal-body">
                <div class="mb-2"><a href="#" onclick="fillBatchExample();return false;">填充示例</a></div>
                <textarea class="form-control" id="batchJsonInput" rows="12" placeholder='{"events": [...]}'></textarea>
                <div id="batchError" class="text-danger mt-2 d-none"></div>
            </div>
            <div class="modal-footer"><button class="btn btn-primary" id="batchSubmitBtn" onclick="submitBatchImport()">导入</button></div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script src="/console/static/js/agent_console.js"></script>
{% endblock %}
```

- [ ] **Step 2: Create `agent_console.js` — agent list + Tab 1 logic**

Create `scripts/devtools/static/js/agent_console.js`. This file ports the agent list, detail, profile, and base events logic from `opencontext/web/templates/agents.html` (lines 150-704).

Structure the file as:

```javascript
/**
 * Agent Console — all tab logic.
 * Dependencies: shared.js (esc, escAttr, refreshIcons, etc.)
 */

// ===== State =====
let selectedAgentId = null;
let _allAgents = [];       // full list for client-side filtering
let _eventTreeMap = {};    // id → tree node, for cascade delete
let _currentEvents = [];   // flat event list from API

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    loadAgentList();
    // Support deep-linking: /console/agents?agent_id=xxx
    const params = new URLSearchParams(window.location.search);
    const deepLink = params.get('agent_id');
    if (deepLink) {
        // Will be selected after list loads — see loadAgentList
        selectedAgentId = deepLink;
    }
});
```

Then implement these functions, porting the logic from `agents.html`:

**Agent list functions** (port from agents.html lines 161-198):
- `loadAgentList()` — `GET /api/agents`, stores in `_allAgents`, calls `renderAgentList`, auto-selects `selectedAgentId` if set from deep link
- `renderAgentList(agents)` — renders `sidebar-item` divs into `#agentList`, updates `#agentCount`
- `filterAgentList(query)` — filters `_allAgents` by agent_id/name, re-renders
- `selectAgent(agentId)` — sets `selectedAgentId`, highlights item, calls `loadAgentDetail`

**Tab 1: Basic Info** (port from agents.html lines 202-373):
- `loadAgentDetail(agentId)` — parallel fetch agent, profile, events; calls `renderInfoTab`
- `renderInfoTab(agent, profile, events)` — renders into `#infoArea` with info card, profile card, events card
- `renderInfoCard(agent)` — Agent ID, Name (editable), Description (editable)
- `startEditInfo()` / `saveInfo()` — inline edit for name/description, `PUT /api/agents/{id}`
- `deleteAgent()` — `DELETE /api/agents/{id}`, resets UI, reloads list
- `renderProfileCard(profile)` — factual_profile textarea
- `startEditProfile()` / `saveProfile()` — edit/save factual_profile, `POST /api/agents/{id}/base/profile`
- `renderEventsCard(events)` — calls `buildEventTree`, renders tree nodes
- `buildEventTree(events)` — port from agents.html lines 492-527, builds parent-child tree from refs + hierarchy_level, populates `_eventTreeMap`
- `renderEventNode(node)` — port from agents.html lines 529-569, recursive tree node with level badges, toggle, delete
- `toggleEvtChildren(id, el)` — toggle children visibility

**Agent CRUD modals** (port from agents.html lines 401-703):
- `showCreateAgentModal()` / `createAgent()` — creates agent via `POST /api/agents`
- `showAddEventModal()` / `addBaseEvent()` — adds event via `POST /api/agents/{id}/base/events`
- `showBatchImportModal()` / `fillBatchExample()` / `submitBatchImport()` — batch import JSON
- `_collectDescendantIds(node)` / `deleteEvent(eventId)` — cascade delete with optimistic UI

**Key modifications from agents.html:**
- All functions use DOM IDs from the new template (same IDs, no conflicts since this is a separate page)
- `feather.replace()` calls replaced with `refreshIcons()` from shared.js
- `esc()` and `escAttr()` from shared.js (not redefined)

- [ ] **Step 3: Verify Tab 1 in browser**

```bash
uv run opencontext start
```

Open `http://localhost:1733/console/agents`. Verify:
- Agent list loads and shows in left sidebar
- Search filter works
- Clicking an agent shows basic info in Tab 1
- Inline edit for name/description works
- Base profile edit works
- Base events tree renders with correct hierarchy
- Create Agent modal works
- Add Event / Batch Import modals work
- Delete agent/event works

- [ ] **Step 4: Commit**

```bash
git add scripts/devtools/templates/agent_console.html scripts/devtools/static/js/agent_console.js
git commit -m "feat(devtools): add Agent Console page with Tab 1 (Basic Info)"
```

---

## Task 6: Agent Console — Tab 2 (Message Tracing) + Tab 3 (Contexts)

Add message tracing and contexts functionality to the Agent Console page.

**Files:**
- Modify: `scripts/devtools/static/js/agent_console.js` (add Tab 2 + Tab 3 functions)

- [ ] **Step 1: Add Tab 2 (Message Tracing) logic to `agent_console.js`**

Port the chat batches logic from `opencontext/web/templates/chat_batches.html` (lines 92-325). Add to the bottom of `agent_console.js`:

```javascript
// ===== Tab 2: Message Tracing =====
let batchPage = 1;
const batchLimit = 20;
let expandedBatchId = null;
```

Implement these functions:

- `initBatchesTab()` — called when Tab 2 becomes active or agent changes. Renders the filter bar HTML into `#batchesArea` (User ID, Device ID, Start Time, End Time inputs — Agent ID shown as disabled with selected agent). Calls `loadBatches(1)`.
- `getBatchFilterParams()` — reads filter inputs, returns URLSearchParams with `agent_id` locked to `selectedAgentId`
- `loadBatches(page)` — `GET /api/chat-batches?{params}`, renders batch table + pagination into `#batchesArea`
- `renderBatchRow(batch)` — port from chat_batches.html lines 160-175. Returns `<tr>` with expand toggle, batch_id (truncated), user_id, message count badge, created time.
- `toggleBatchExpand(batchId)` — port from chat_batches.html lines 177-239. Accordion behavior (only one expanded at a time). Fetches batch detail + contexts in parallel.
- `renderBatchMessages(messages)` — port from chat_batches.html lines 242-266. Role-colored bubbles (user=blue, assistant=grey, system=yellow), multimodal content badges.
- `renderBatchContextCard(ctx)` — port from chat_batches.html lines 268-290. Type-colored context card with keywords and importance.

**Key modifications from chat_batches.html:**
- Agent ID filter is always locked to `selectedAgentId`
- DOM IDs prefixed where needed to avoid conflicts (e.g., `batchFilterUserId` instead of `filterUserId`)
- `esc()`, `escAttr()` from shared.js

Also add a tab-switch listener so Tab 2 loads data when activated:

```javascript
// In the DOMContentLoaded handler:
document.getElementById('agentTabs').addEventListener('shown.bs.tab', (e) => {
    if (!selectedAgentId) return;
    if (e.target.getAttribute('href') === '#tab-batches') initBatchesTab();
    if (e.target.getAttribute('href') === '#tab-contexts') initContextsTab();
});
```

- [ ] **Step 2: Add Tab 3 (Related Contexts) logic to `agent_console.js`**

```javascript
// ===== Tab 3: Related Contexts =====
let ctxPage = 1;
const ctxLimit = 15;
```

Implement:

- `initContextsTab()` — renders filter bar into `#agentContextsArea` (Type dropdown, User ID, Device ID, Hierarchy Level, Start/End Time — Agent ID locked). Calls `loadAgentContexts(1)`.
- `getCtxFilterParams()` — reads filters, returns URLSearchParams with `agent_id` locked
- `loadAgentContexts(page)` — `GET /api/contexts?{params}`, renders card grid + pagination
- Uses `renderContextCardHTML(ctx)` from shared.js for each card
- Uses `renderPaginationHTML(...)` from shared.js for pagination

Also populate the type dropdown by fetching `GET /api/context_types` on init.

**Cross-page link:** When rendering user_id in batch rows or context cards, make it a clickable link:

```javascript
function userLink(userId, deviceId, agentId) {
    if (!userId || userId === 'default') return esc(userId || '');
    const href = `/console/memory?user_id=${encodeURIComponent(userId)}&device_id=${encodeURIComponent(deviceId || 'default')}&agent_id=${encodeURIComponent(agentId || 'default')}`;
    return `<a href="${href}" title="在记忆浏览器中查看">${esc(userId)}</a>`;
}
```

- [ ] **Step 3: Verify Tabs 2 + 3 in browser**

Open `http://localhost:1733/console/agents`. Select an agent, then:
- Switch to "消息追踪" tab — verify filter bar shows, batches load, expand works, messages render
- Switch to "关联 Contexts" tab — verify filter bar shows, context cards load, pagination works
- Click a user_id link — verify it navigates to `/console/memory?user_id=...`

- [ ] **Step 4: Commit**

```bash
git add scripts/devtools/static/js/agent_console.js
git commit -m "feat(devtools): add Agent Console Tab 2 (message tracing) + Tab 3 (contexts)"
```

---

## Task 7: Memory Explorer — Template + Tab 1 (Memory Snapshot)

Create the Memory Explorer page with left sidebar user list and Tab 1 (memory snapshot).

**Files:**
- Create: `scripts/devtools/templates/memory_explorer.html`
- Create: `scripts/devtools/static/js/memory_explorer.js`

- [ ] **Step 1: Create `memory_explorer.html` template**

Same master-detail layout as `agent_console.html`, but with user list in left sidebar and memory-specific tabs:

```html
{% extends "base.html" %}
{% block title %}记忆浏览器{% endblock %}

{% block content %}
<div class="console-container">
    <!-- Left sidebar: User list -->
    <div class="console-sidebar">
        <div class="sidebar-header">
            <input type="text" class="form-control form-control-sm" id="userSearch"
                placeholder="搜索用户..." oninput="filterUserList(this.value)">
        </div>
        <div class="sidebar-list" id="userList"></div>
        <div class="sidebar-footer" id="userCount"></div>
    </div>

    <!-- Right: Tab content -->
    <div class="console-content">
        <ul class="nav nav-tabs px-2 pt-1" id="memoryTabs">
            <li class="nav-item">
                <a class="nav-link active" data-bs-toggle="tab" href="#tab-snapshot">记忆快照</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#tab-search">语义搜索</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#tab-mcontexts">Contexts 列表</a>
            </li>
        </ul>
        <div class="tab-content" id="memoryTabContent">
            <div class="tab-pane fade show active" id="tab-snapshot">
                <div id="snapshotArea" class="text-center text-muted py-5">请从左侧选择一个用户</div>
            </div>
            <div class="tab-pane fade" id="tab-search">
                <div id="searchArea"></div>
            </div>
            <div class="tab-pane fade" id="tab-mcontexts">
                <div id="memoryContextsArea"></div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script src="/console/static/js/memory_explorer.js"></script>
{% endblock %}
```

- [ ] **Step 2: Create `memory_explorer.js` — user list + Tab 1 logic**

Create `scripts/devtools/static/js/memory_explorer.js`:

```javascript
/**
 * Memory Explorer — all tab logic.
 * Dependencies: shared.js
 */

// ===== State =====
let selectedUser = null;  // { user_id, device_id, agent_id }
let _allUsers = [];

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    loadUserList();
    // Deep-link support
    const params = new URLSearchParams(window.location.search);
    const uid = params.get('user_id');
    if (uid) {
        selectedUser = {
            user_id: uid,
            device_id: params.get('device_id') || 'default',
            agent_id: params.get('agent_id') || 'default',
        };
    }
});
```

**User list functions:**
- `loadUserList()` — `GET /api/users`, stores in `_allUsers`, calls `renderUserList`, auto-selects `selectedUser` if set from deep link
- `renderUserList(users)` — renders sidebar items: `user_id` as primary text, `device_id / agent_id` as secondary. Updates `#userCount`.
- `filterUserList(query)` — filters `_allUsers` by user_id substring, re-renders
- `selectUser(user)` — sets `selectedUser`, highlights item, calls `loadSnapshot`

**Tab 1: Memory Snapshot** — port from `opencontext/web/templates/memory_cache.html` (lines 87-304):

- `loadSnapshot()` — renders config bar + result area into `#snapshotArea`. Config bar has: include checkboxes (Agent Prompt, Profile, Events, Accessed), Recent Days, Max Today Events, Max Accessed, Force Refresh. Calls `querySnapshot()` on first load.
- `querySnapshot()` — reads config form, builds URLSearchParams with `user_id/device_id/agent_id` from `selectedUser`. `GET /api/memory-cache?{params}`. Calls `renderSnapshotResults`.
- `renderSnapshotResults(data)` — dispatches to section renderers based on non-null data fields

Port these renderers from memory_cache.html:
- `renderProfile(profile)` — uses `wrapCard()` from shared.js
- `renderAgentPrompt(agentPrompt)` — factual_profile + behavioral_profile in `wrapCard()`
- `renderRecentlyAccessed(items)` — type-badged items with title, summary, score, access time, media thumbnails
- `renderTodayEvents(events)` — time-prefixed event list
- `renderDailySummaries(summaries)` — date-badged summary cards

**Key modifications from memory_cache.html:**
- No user_id/device_id/agent_id inputs (come from left panel selection)
- Config params remain as a compact form row within the tab
- Uses `wrapCard()`, `esc()`, `refreshIcons()` from shared.js

Tab switch listener:

```javascript
document.getElementById('memoryTabs').addEventListener('shown.bs.tab', (e) => {
    if (!selectedUser) return;
    if (e.target.getAttribute('href') === '#tab-search') initSearchTab();
    if (e.target.getAttribute('href') === '#tab-mcontexts') initMemoryContextsTab();
});
```

- [ ] **Step 3: Verify in browser**

Open `http://localhost:1733/console/memory`. Verify:
- User list loads from API
- Search filter works
- Clicking a user loads memory snapshot in Tab 1
- All sections render (profile, agent prompt, events, summaries, accessed)
- Config options work (recent days, max counts, force refresh)
- Collapsible cards expand/collapse

- [ ] **Step 4: Commit**

```bash
git add scripts/devtools/templates/memory_explorer.html scripts/devtools/static/js/memory_explorer.js
git commit -m "feat(devtools): add Memory Explorer page with Tab 1 (memory snapshot)"
```

---

## Task 8: Memory Explorer — Tab 2 (Semantic Search) + Tab 3 (Contexts)

Add full semantic search and contexts list to the Memory Explorer page.

**Files:**
- Modify: `scripts/devtools/static/js/memory_explorer.js` (add Tab 2 + Tab 3 functions)

- [ ] **Step 1: Add Tab 2 (Semantic Search) logic**

Port the full vector search functionality from `opencontext/web/templates/vector_search.html` (lines 191-742). Add to `memory_explorer.js`:

```javascript
// ===== Tab 2: Semantic Search =====
let uploadedMediaList = [];
```

Implement:

- `initSearchTab()` — renders the full search form into `#searchArea`: query textarea, media upload zone, event ID textarea, top-K input, hierarchy level checkboxes, drill direction select, start/end time, user/device/agent ID (disabled, from selection). Calls `initMediaZone()`.
- `initMediaZone()` — port from vector_search.html lines 202-386. Full media upload functionality: paste handler, drag-drop, file picker, video URL input, upload to `/api/media/upload`, preview with remove buttons.
  - Nested functions: `uploadFile(file)`, `addMedia(media)`, `removeMedia(index)`, `clearMedia()`, `renderMediaPreview()`
- `handleSearch(event)` — port from vector_search.html lines 388-477. Builds search request body with multipart query array. `POST /api/search` with `user_id/device_id/agent_id` injected from `selectedUser`. Calls `displaySearchResults`.
- `displaySearchMetadata(metadata)` — shows query text, hit count, search time
- `displaySearchResults(events)` — renders result tree using `renderSearchTreeNode`
- `renderSearchTreeNode(node)` — dispatches to `renderSearchHitCard` or `renderSearchContextCard` plus `renderSearchChildren`
- `renderSearchHitCard(event)` — port from vector_search.html lines 579-636. Full hit card with level badge, score, keywords, media refs, expandable summary, view detail button.
- `renderSearchContextCard(node)` — port from vector_search.html lines 638-666. Lighter context card for ancestors/descendants.
- `renderSearchChildren(node)` — recursive children rendering
- `toggleSearchChildren(id, el)` / `toggleSearchSummary(id, link)` — toggle helpers
- `renderMediaRefs(mediaRefs)` — port from vector_search.html lines 694-705

**Key modifications from vector_search.html:**
- `user_id`, `device_id`, `agent_id` are locked to `selectedUser` (disabled inputs)
- DOM IDs prefixed with `search-` where needed to avoid conflicts
- Uses `esc()`, `escAttr()`, `levelBadge()`, `refreshIcons()` from shared.js
- `escapeHtml` references changed to `esc`
- `viewContext` from shared.js

- [ ] **Step 2: Add Tab 3 (Contexts List) logic**

Same pattern as Agent Console Tab 3 but filtered by user triple:

```javascript
// ===== Tab 3: Contexts List =====
let mCtxPage = 1;
const mCtxLimit = 15;
```

Implement:

- `initMemoryContextsTab()` — renders filter bar into `#memoryContextsArea` (Type dropdown, Hierarchy Level, Start/End Time — user triple locked). Calls `loadMemoryContexts(1)`.
- `getMemoryCtxFilterParams()` — reads filters, returns URLSearchParams with user triple from `selectedUser`
- `loadMemoryContexts(page)` — `GET /api/contexts?{params}`, renders card grid + pagination using `renderContextCardHTML` and `renderPaginationHTML` from shared.js

**Cross-page link:** When `agent_id` is not "default" in context cards, show a link icon:

```javascript
function agentLink(agentId) {
    if (!agentId || agentId === 'default') return '';
    const href = `/console/agents?agent_id=${encodeURIComponent(agentId)}`;
    return ` <a href="${href}" title="在 Agent 调试台中查看" class="text-decoration-none"><i data-feather="external-link" style="width:12px;height:12px"></i></a>`;
}
```

- [ ] **Step 3: Verify Tabs 2 + 3 in browser**

Open `http://localhost:1733/console/memory`. Select a user, then:
- Switch to "语义搜索" tab — verify:
  - Query textarea works
  - Media upload (paste, drag-drop, file select) works
  - Video URL input works
  - Search executes and result tree renders with hit/context cards
  - Hierarchy level checkboxes and drill direction work
  - View detail button works
- Switch to "Contexts 列表" tab — verify:
  - Cards load with filters
  - Pagination works
  - Cross-page agent link works

- [ ] **Step 4: Commit**

```bash
git add scripts/devtools/static/js/memory_explorer.js
git commit -m "feat(devtools): add Memory Explorer Tab 2 (semantic search) + Tab 3 (contexts)"
```

---

## Task 9: Cross-Page Linking + Final Polish

Verify cross-page deep linking works end-to-end and fix any issues.

**Files:**
- Modify: `scripts/devtools/static/js/agent_console.js` (verify deep link handling)
- Modify: `scripts/devtools/static/js/memory_explorer.js` (verify deep link handling)

- [ ] **Step 1: Verify deep link flows**

Test these flows in the browser:

1. **Agent → Memory**: On Agent Console, select an agent. Go to Tab 2 (Message Tracing) or Tab 3 (Contexts). Click a user_id link. Verify it navigates to `/console/memory?user_id=...&device_id=...&agent_id=...` and auto-selects the user in the left panel, loading their memory snapshot.

2. **Memory → Agent**: On Memory Explorer, select a user. Go to Tab 3 (Contexts). Find a context card with a non-default agent_id. Click the agent link icon. Verify it navigates to `/console/agents?agent_id=...` and auto-selects the agent.

3. **Direct URL access**: Open `/console/agents?agent_id=someAgent` directly. Verify the agent is auto-selected after the list loads. Open `/console/memory?user_id=someUser&device_id=default&agent_id=default` directly. Verify the user is auto-selected.

- [ ] **Step 2: Fix any deep-link timing issues**

The most common issue: the deep-link URL param is read before the list finishes loading. Both pages should handle this by storing the deep-link target and checking it after the list renders. Verify this is implemented in `loadAgentList()` and `loadUserList()`:

```javascript
// In loadAgentList(), after rendering:
if (selectedAgentId) {
    selectAgent(selectedAgentId);
}

// In loadUserList(), after rendering:
if (selectedUser) {
    const match = _allUsers.find(u =>
        u.user_id === selectedUser.user_id &&
        u.device_id === selectedUser.device_id &&
        u.agent_id === selectedUser.agent_id
    );
    if (match) selectUser(match);
}
```

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest -m unit -v
```

Ensure all existing tests still pass and the new tests from Tasks 1-2 pass.

- [ ] **Step 4: Final browser verification**

Open both console pages and verify the complete user journey:
- `/console/agents` — agent list, all 3 tabs, modals, cross-page links
- `/console/memory` — user list, all 3 tabs, media upload, cross-page links
- Old pages still work at their original URLs (`/agents`, `/contexts`, `/vector_search`, etc.)

- [ ] **Step 5: Commit**

```bash
git add scripts/devtools/
git commit -m "feat(devtools): add cross-page linking and finalize console pages"
```
