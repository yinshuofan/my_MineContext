# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MineContext is a **memory backend service** that captures, processes, stores, and retrieves context/memory from multiple sources (chat logs, documents, web links). It uses vector embeddings and LLM-powered analysis to organize information into 5 typed memory contexts with type-specific update strategies and storage routing.

## Common Commands

```bash
# Install dependencies (uv recommended)
uv sync

# Start the server (default port 1733)
uv run opencontext start
uv run opencontext start --config config/config.yaml
uv run opencontext start --port 1733 --host 0.0.0.0
uv run opencontext start --workers 4   # multi-process mode

# Code formatting
black opencontext --line-length 100
isort opencontext
pre-commit run --all-files

# Docker
docker-compose up
```

There is no test suite in this project currently. To verify changes compile-check with:
```bash
python -m py_compile opencontext/path/to/file.py
```

## Architecture

### 5-Type Context System

All data flows through a unified `ProcessedContext` model, but each type has its own update strategy and storage destination. This is the central design decision — understand it before modifying any storage, processing, or retrieval code.

| Type | Update Strategy | Storage | Description |
|------|----------------|---------|-------------|
| `profile` | OVERWRITE | Relational DB | User's own preferences, habits, communication style. Key: `(user_id, agent_id)` |
| `entity` | OVERWRITE | Relational DB | Other people, projects, teams, organizations. Key: `(user_id, entity_name)` |
| `document` | OVERWRITE | Vector DB | Uploaded files and web links. Overwrite = delete old chunks + insert new. Tracked by `source_file_key` |
| `event` | APPEND | Vector DB | Immutable activity records, meetings, status changes. Never modified after creation |
| `knowledge` | APPEND_MERGE | Vector DB | Reusable concepts, procedures, patterns. Similar entries merged to avoid duplication |

Defined in `opencontext/models/enums.py`:
- `ContextType` enum — the 5 types
- `UpdateStrategy` enum — OVERWRITE, APPEND, APPEND_MERGE
- `CONTEXT_UPDATE_STRATEGIES` dict — type → strategy mapping
- `CONTEXT_STORAGE_BACKENDS` dict — type → `"document_db"` or `"vector_db"`
- `ContextDescriptions` dict — rich descriptions with `key_indicators`, `examples`, `classification_priority` (used by LLM prompts for classification)

### Data Pipeline

```
Input → Processor → ProcessedContext → context_operations (routes by type) → Storage
```

1. **Process** (`opencontext/context_processing/`): `ProcessorFactory` routes by `ContextSource`:
   - `LOCAL_FILE`, `VAULT`, `WEB_LINK` → `DocumentProcessor`
   - `CHAT_LOG`, `INPUT` → `TextChatProcessor`
   - Entity extraction → `EntityProcessor`

2. **Route** (`opencontext/server/context_operations.py`): The central routing point. Reads `CONTEXT_UPDATE_STRATEGIES` and `CONTEXT_STORAGE_BACKENDS` to decide:
   - profile/entity → convert to `ProfileData`/`EntityData` → relational DB upsert
   - document → delete old chunks by `source_file_key` → insert new chunks to vector DB
   - event → append to vector DB (immutable)
   - knowledge → run through `ContextMerger` (similarity-based deduplication)

3. **Store** (`opencontext/storage/`): `UnifiedStorage` wraps dual backends:
   - **Document DB** (MySQL or SQLite) — profiles table, entities table, plus context metadata
   - **Vector DB** (VikingDB, ChromaDB, or Qdrant) — document/event/knowledge embeddings

### Hierarchical Event Indexing

Events support a 4-level time-based hierarchy for efficient historical retrieval:
- **L0** — raw individual events
- **L1** — daily summaries (`time_bucket: "2026-02-21"`)
- **L2** — weekly summaries (`time_bucket: "2026-W08"`)
- **L3** — monthly summaries (`time_bucket: "2026-02"`)

Key fields on `ContextProperties`: `hierarchy_level`, `parent_id`, `children_ids`, `time_bucket`.

`HierarchicalEventTool` retrieval algorithm: search L1-L3 summaries top-down → drill through `children_ids` to L0 → merge with direct L0 semantic search as fallback → deduplicate by ID keeping higher score.

Periodic task `hierarchy_summary.py` generates summaries on schedule (daily/weekly/monthly).

### Retrieval Tools

4 type-aligned tools registered in `opencontext/tools/tool_definitions.py`:

| Tool | Types Searched | Backend | Strategy |
|------|---------------|---------|----------|
| `ProfileEntityTool` | profile, entity | Relational DB | Exact name lookup + text search |
| `DocumentRetrievalTool` | document | Vector DB | Semantic similarity search |
| `KnowledgeRetrievalTool` | knowledge, event (L0) | Vector DB | Semantic similarity search |
| `HierarchicalEventTool` | event (all levels) | Vector DB | Summary drill-down + L0 fallback |

### Storage Access

**Always use `get_storage()` from `opencontext/storage/global_storage.py`** to get the `UnifiedStorage` instance. This returns the object with all profile/entity/hierarchy methods.

Do NOT use `GlobalStorage.get_instance()` or `get_global_storage()` — these return the raw `GlobalStorage` wrapper which lacks `upsert_profile()`, `get_entity()`, `search_hierarchy()`, etc.

### Server Layer

- **Entry point**: `opencontext/cli.py` → `opencontext start`
- **Framework**: FastAPI on Uvicorn, port 1733
- **Orchestrator**: `opencontext/server/opencontext.py` → `OpenContext`
- **Auth**: API key via `X-API-Key` header, disabled by default

### Key Data Models (`opencontext/models/context.py`)

- `ProcessedContext` — the universal intermediate format all processors produce
- `ContextProperties` — includes `hierarchy_level`, `parent_id`, `children_ids`, `time_bucket`, `source_file_key`, `enable_merge`
- `ProfileData` — relational DB model for user profiles (PK: `user_id + agent_id`)
- `EntityData` — relational DB model for entities (PK: `user_id + entity_name`)
- `ProcessedContextModel` — API response model, must mirror all fields from `ContextProperties` that should be exposed

### LLM Integration (`opencontext/llm/`)

Three global singletons accessed via `get_instance()`:
- `GlobalEmbeddingClient` — text embeddings (Volcengine Doubao default)
- `GlobalVLMClient` — vision/image understanding
- `LLMClient` — text generation

All use OpenAI-compatible API format.

### Prompts (`config/prompts_en.yaml`, `config/prompts_zh.yaml`)

LLM prompts for extraction, classification, merging, and hierarchy summarization. Loaded by `PromptManager`. When modifying the type system, all prompts referencing context types must be updated in both language files.

### Scheduling (`opencontext/periodic_task/`)

APScheduler with Redis backend. Registered tasks:
- `MemoryCompressionTask` — deduplicates similar contexts (knowledge type only)
- `DataCleanupTask` — retention-based cleanup
- `HierarchySummaryTask` — generates L1/L2/L3 event summaries

## Configuration

- Main config: `config/config.yaml` — YAML with `${ENV_VAR:default}` syntax
- Environment variables: copy `.env.example` to `.env`
- Key env vars: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `MYSQL_HOST`, `MYSQL_PASSWORD`, `REDIS_HOST`
- Service mode (`service_mode.enabled: true`): stateless deployment requiring external Redis + MySQL/vector DB

## Code Style

- Python 3.10+, Black (line length 100), isort (profile "black")
- Package name in imports: `opencontext`
- Logging: `loguru` via `from opencontext.utils.logging_utils import get_logger`
- Pre-commit hooks auto-format on commit

## Pitfalls and Lessons Learned

These are real bugs encountered during development. Check for them when modifying related code.

### Storage singleton confusion
`get_storage()` returns `UnifiedStorage` (has profile/entity/hierarchy methods). `GlobalStorage.get_instance()` and `get_global_storage()` return `GlobalStorage` (lacks those methods). Always use `get_storage()` from `opencontext.storage.global_storage`.

### Qdrant Range filter only supports numeric/datetime
Qdrant's `models.Range(gte=..., lte=...)` does NOT work on string fields like `time_bucket`. Use in-code string comparison filtering instead (over-fetch with `top_k * 3`, then filter). See `qdrant_backend.py` `search_by_hierarchy()` for the pattern.

### ProcessedContextModel must declare all exposed fields
If you add a field to `ContextProperties` and want it in API responses, you must also declare it on `ProcessedContextModel` and update `from_processed_context()`. Missing declarations cause silent Pydantic field drops.

### MySQL lastrowid is 0 for VARCHAR primary keys
`cursor.lastrowid` only returns meaningful values for auto-increment integer PKs. For UUID/VARCHAR PKs (like the entities table), always SELECT back the persisted ID instead of relying on lastrowid.

### Use timezone-aware datetime everywhere
`datetime.utcfromtimestamp()` is deprecated in Python 3.12+. Use `datetime.fromtimestamp(ts, tz=datetime.timezone.utc)` instead. The same applies to `datetime.utcnow()` — use `datetime.now(tz=datetime.timezone.utc)`.

### Context merger only handles knowledge type
The merger (`context_merger.py`) now only processes `knowledge` contexts. Profile/entity use relational DB overwrite, document uses delete+insert, event is immutable append. Do not route non-knowledge types through the merger.

### Prompt files must stay in sync
`config/prompts_en.yaml` and `config/prompts_zh.yaml` must have identical keys. When adding/removing context types or changing classification logic, update both files and all `ContextDescriptions` in `enums.py`.

### EntityData.relationships stored in metadata JSON
The `relationships` field on `EntityData` is not a separate DB column — it's serialized inside the `metadata` JSON column. Keep this in mind when querying relationships.

## Extending the System

- **New context type**: Add to `ContextType` enum, `UpdateStrategy`, both mapping dicts, `ContextDescriptions`, `ContextSimpleDescriptions`, and update all prompt files
- **New processor**: Extend `BaseContextProcessor`, implement `can_process()` and `process()`, register in `ProcessorFactory`
- **New storage backend**: Implement `IVectorStorageBackend` or `IDocumentStorageBackend` from `base_storage.py`, register in factory
- **New retrieval tool**: Extend `BaseTool`, register in `tool_definitions.py` and `tools_executor.py`
