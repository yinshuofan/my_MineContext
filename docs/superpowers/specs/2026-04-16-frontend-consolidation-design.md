# Frontend Consolidation Design: 5 Pages → 2 Pages

## Context

MineContext needs to provide reference frontend pages for integration into an agent creation platform. The current 10-entry sidebar with separate pages for each function is too fragmented for platform admins. This spec consolidates 5 debug/inspection pages into 2 focused pages.

**Target users:** Platform admins/ops — debugging agent memory and inspecting user data. NOT end-users. System settings and monitoring are excluded.

**Delivery model:** Reference implementation only (embedding method C). The platform builds its own frontend using MineContext APIs. These pages demonstrate the complete data flow and interaction patterns.

## Pages In Scope

**Consolidated into 2 new pages:**
- `agents.html` → Agent Debug Console
- `chat_batches.html` → Agent Debug Console
- `contexts.html` → Both pages (filtered by agent or user)
- `memory_cache.html` → Memory Explorer
- `vector_search.html` → Memory Explorer

**Existing pages:** All original templates and routes are kept untouched. The new pages are created in a separate location and coexist with the old pages.

## Overall Architecture

### Navigation

Replace the current 10-item sidebar with a top navigation bar containing 2 entries:

| Entry | Route | Description |
|-------|-------|-------------|
| Agent 调试台 | `/console/agents` | Agent inspection + message tracing + related contexts |
| 记忆浏览器 | `/console/memory` | User memory snapshot + semantic search + contexts list |

The console pages use a new `console/base.html` template with a top navbar containing two tab-style links + API Key button. No sidebar.

### Unified Layout Pattern

Both pages use an identical master-detail + tab layout:

```
┌─────────────────────────────────────────────────┐
│  MineContext   [Agent 调试台]  [记忆浏览器]   🔑 │  ← top nav
├──────────┬──────────────────────────────────────┤
│          │  [Tab A]  [Tab B]  [Tab C]           │  ← tabs
│  Master  ├──────────────────────────────────────┤
│  List    │                                      │
│  (190px) │  Tab Content Area                    │
│          │                                      │
│          │                                      │
└──────────┴──────────────────────────────────────┘
```

- Left panel: 190px fixed-width list with search input
- Right panel: Bootstrap tab navigation + tab content area
- Responsive: left panel collapses on small screens

## Page 1: Agent Debug Console (`/console/agents`)

### Left Panel — Agent List

Reuses the current `agents.html` agent list logic:
- `GET /api/agents` to load list
- Search input filters client-side by agent_id and name
- Selected agent highlighted with left blue border
- Shows agent count at bottom
- "Create Agent" button at top (opens modal)

### Tab 1: Basic Info (基本信息)

Combines the current agent detail + base profile + base events sections from `agents.html`.

**Content:**
- Info cards row: Agent ID, Name, Description (inline-editable)
- Base Profile card: `factual_profile` textarea (editable), fetched via `GET /api/agents/{id}/base/profile`, saved via `POST /api/agents/{id}/base/profile`
- Base Events card: hierarchical event tree (same tree rendering as current `agents.html`), with level badges (L0-L3), collapsible children, add/batch-import/cascade-delete

**Write operations preserved:**
- Agent CRUD: `POST/PUT/DELETE /api/agents/{id}`
- Base profile: `GET/POST /api/agents/{id}/base/profile`
- Base events: `GET/POST /api/agents/{id}/base/events`, `DELETE /api/agents/{id}/base/events/{eventId}`

**Modals preserved:**
- Create Agent modal (`agent_id`, `name`, `description`)
- Add Event modal (`title`, `summary`, `event_time`, `keywords`, `importance`)
- Batch Import modal (JSON textarea with example fill)

### Tab 2: Message Tracing (消息追踪)

Integrates `chat_batches.html` functionality, pre-filtered by the selected agent.

**Content:**
- Filter bar: User ID, Device ID, Start Time, End Time (Agent ID is auto-set from selected agent, shown as disabled input)
- Batch table: Batch ID (truncated), User ID, Message Count badge, Created Time
- Expandable row detail (same as current `chat_batches.html`):
  - Left column: chat message bubbles (role-colored: user=blue, assistant=grey, system=yellow), multimodal content badges
  - Right column: associated context cards (type-colored borders)
- Pagination (page/limit from API response)

**API endpoints:**
- `GET /api/chat-batches?agent_id={selected}&{filters}` — list with agent pre-filter
- `GET /api/chat-batches/{id}` — batch detail
- `GET /api/chat-batches/{id}/contexts` — associated contexts

**Key difference from standalone `chat_batches.html`:** Agent ID is always locked to the selected agent from the left panel. User can still filter by user_id, device_id, and time range within that agent scope.

### Tab 3: Related Contexts (关联 Contexts)

A client-side rendered version of the contexts list, filtered by the selected agent's `agent_id`.

**Content:**
- Filter bar: Type dropdown, User ID, Device ID, Hierarchy Level dropdown, Start Time, End Time (Agent ID auto-set)
- Context card grid (responsive: 1/2 columns based on viewport)
- Each card: type badge, level badge, media icons, truncated ID (click-to-copy), title, summary (120 chars), user/device IDs, create time
- Card click → view detail (navigates to `context_detail.html`)
- Delete button per card
- Pagination: page selector, page numbers, jump-to-page

**Key change from current `contexts.html`:** This tab is **client-side rendered**, not server-side. It calls the same storage APIs but via JavaScript fetch. This aligns with the other 4 client-side pages and avoids the server-rendered outlier.

**API endpoints:**
- New: `GET /api/contexts?agent_id={selected}&type=&user_id=&hierarchy_level=&start_date=&end_date=&page=1&limit=15` — paginated context list (REST API version of what the server-side route currently does)
- `POST /contexts/detail` — view detail (existing)
- `POST /contexts/delete` — delete context (existing)

## Page 2: Memory Explorer (`/console/memory`)

### Left Panel — User List

A new component that displays unique `(user_id, device_id, agent_id)` combinations from existing context data.

**Content:**
- Search input filters client-side by user_id
- Each list item shows: `user_id` (primary), `device_id / agent_id` (secondary line, smaller text)
- Selected item highlighted with left blue border
- Shows user combination count at bottom

**API endpoint (new):**
- `GET /api/users` — returns list of unique `(user_id, device_id, agent_id)` tuples from stored contexts

### Tab 1: Memory Snapshot (记忆快照)

Integrates `memory_cache.html` functionality. User triple is auto-set from left panel selection.

**Content:**
- Config bar: Include section checkboxes (Agent Prompt, Profile, Events, Accessed), Recent Days input, Max Today Events, Max Accessed, Force Refresh checkbox
- Result sections (collapsible cards, same rendering as current `memory_cache.html`):
  - Agent Prompt: `factual_profile` + `behavioral_profile` preformatted blocks
  - User Profile: `factual_profile` preformatted block
  - Recently Accessed: type-badged items with title, summary, score, access time, media thumbnails
  - Today Events: time-prefixed event list with green border
  - Daily Summaries: date-badged summary cards

**API endpoint:**
- `GET /api/memory-cache?user_id={}&device_id={}&agent_id={}&{config_params}` (existing)

**Key difference from standalone `memory_cache.html`:** The user_id/device_id/agent_id inputs are removed; these come from the left panel selection. The config parameters (include sections, recent_days, max counts, force_refresh) remain as a compact config bar within the tab.

### Tab 2: Semantic Search (语义搜索)

Full `vector_search.html` functionality, pre-filtered by the selected user triple.

**Content (all features preserved):**
- Query textarea (text, max 2000 chars)
- Multimodal media input: paste/drag-drop/file-select for images + videos, video URL input, up to 10 files, upload progress bar
- Event ID exact lookup textarea
- Top-K slider (1-100, default 10)
- Hierarchy Level checkboxes (L0-L3)
- Drill direction select (up/down/both/none)
- Start Time, End Time
- User/Device/Agent ID auto-set from left panel (shown as disabled inputs)
- Result tree: hit cards (green border, score, keywords, media, expand/collapse summary) + context cards (dashed border, grey), recursive tree rendering with level-colored borders

**API endpoints:**
- `POST /api/search` (existing) — user_id/device_id/agent_id injected from selection
- `POST /api/media/upload` (existing)
- `POST /contexts/detail` (existing) — for "view detail" button

### Tab 3: Contexts List (Contexts 列表)

Same as Agent Debug Console's Tab 3, but filtered by user triple instead of agent_id.

**Content:** Identical card grid, filters (Type, Hierarchy Level, Time range), pagination, click-to-detail, delete.

**API endpoint:**
- `GET /api/contexts?user_id={}&device_id={}&agent_id={}&{filters}` (same new API as Page 1 Tab 3)

## New API Requirements

### 1. `GET /api/users` — User Triple List

Returns unique `(user_id, device_id, agent_id)` combinations from stored context data.

**Response:**
```json
{
  "users": [
    {"user_id": "user_001", "device_id": "default", "agent_id": "default"},
    {"user_id": "user_002", "device_id": "mobile", "agent_id": "customer_service_01"}
  ],
  "total": 2
}
```

**Implementation:** Aggregate from vector DB context metadata. Deduplicate on the three-field tuple.

### 2. `GET /api/contexts` — Paginated Context List API

REST API version of the server-side `/contexts` route logic. Enables client-side rendering in both pages' "Contexts" tabs.

**Query parameters:** `type`, `user_id`, `device_id`, `agent_id`, `hierarchy_level`, `start_date`, `end_date`, `page` (default 1), `limit` (default 15)

**Response:**
```json
{
  "contexts": [
    {
      "id": "...",
      "context_type": "event",
      "title": "...",
      "summary": "...",
      "user_id": "...",
      "device_id": "...",
      "agent_id": "...",
      "hierarchy_level": 0,
      "create_time": "2026-04-16T14:30:00",
      "metadata": {"media_refs": [...]},
      "keywords": [...]
    }
  ],
  "page": 1,
  "limit": 15,
  "total": 128,
  "total_pages": 9
}
```

**Implementation:** Extract the query/filter/pagination logic from `web.py:read_contexts()` into a new API route handler. The server-rendered `/contexts` route can be removed or kept as a redirect.

## Cross-Page Linking

- **Agent Console → Memory Explorer:** In Tab 3 (Contexts) or Tab 2 (Chat Batches), clicking a `user_id` navigates to `/console/memory?user_id={}&device_id={}&agent_id={}`, which auto-selects the user in Memory Explorer's left panel.
- **Memory Explorer → Agent Console:** In Tab 3 (Contexts) or Tab 1 (Memory Snapshot), if `agent_id` is not "default", show a link icon next to agent_id that navigates to `/console/agents?agent_id={}`, which auto-selects the agent.

Both pages read URL query parameters on load to support deep-linking and cross-page navigation.

## Technical Approach

### What to reuse

The new pages reuse all existing JavaScript logic from the current pages. The consolidation is structural (combining templates) not functional (rewriting logic).

- `agents.html`: Agent list, detail rendering, base profile edit, base events tree — all JS logic moves to Agent Console Tab 1
- `chat_batches.html`: Batch table, expandable detail, message rendering — moves to Agent Console Tab 2
- `memory_cache.html`: Config form, section rendering — moves to Memory Explorer Tab 1
- `vector_search.html`: Search form, media upload, result tree — moves to Memory Explorer Tab 2
- `contexts.html`: Card rendering, pagination, filters — converted from server-side to client-side, shared by both pages' Tab 3

### Shared utilities to deduplicate

Current pages have duplicated utility functions. Extract once into a shared JS file:

- `escapeHtml()` / `esc()` — duplicated in agents, chat_batches, vector_search
- `feather.replace()` calls after DOM mutations — all pages
- Date/time formatting helpers
- Click-to-copy ID function
- Context card rendering (used in contexts, chat_batches, vector_search)

### File structure

All new files live under `scripts/devtools/`, fully separate from the main application code.

```
scripts/devtools/
  templates/
    base.html              ← new: standalone layout (top nav, no sidebar)
    agent_console.html     ← new: Agent Debug Console
    memory_explorer.html   ← new: Memory Explorer
  static/
    js/shared.js           ← new: deduplicated utilities
    js/agent_console.js    ← new: Agent Console logic
    js/memory_explorer.js  ← new: Memory Explorer logic
    css/console.css        ← new: console-specific styles
  routes.py                ← new: page routes (/console/agents, /console/memory)

opencontext/server/routes/
  contexts_api.py          ← new: GET /api/contexts endpoint
  users_api.py             ← new: GET /api/users endpoint
  api.py                   ← add new API route includes (existing file, minimal change)
```

### Coexistence with existing pages

- All original templates, routes, and static files are **untouched**
- Old pages remain accessible at their original URLs (`/agents`, `/contexts`, `/vector_search`, etc.)
- New console pages are served at `/console/agents` and `/console/memory`
- `scripts/devtools/routes.py` exports a FastAPI `APIRouter` that is mounted by `api.py`
- `scripts/devtools/templates/` and `scripts/devtools/static/` are registered as separate Jinja2 template directory and static mount at `/console/static`
- The 2 new APIs (`GET /api/contexts`, `GET /api/users`) are additive — no existing API is changed

### devtools base.html (new layout)

A new base template for the console pages, independent of the existing `base.html`:

- Top navbar with "MineContext" brand + two tab-style page links + API Key button
- No sidebar
- Main content area uses full width with `padding-top` for navbar clearance
- Loads Bootstrap 5, Feather Icons, and `/console/static/js/shared.js`
- Reuses the existing `api_auth.js` from `/static/js/api_auth.js` for API key injection
