# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MineContext is a **memory backend** system that captures, processes, stores, and retrieves context/memory from multiple sources (chat logs, documents, screenshots, web links). It uses vector embeddings and LLM-powered analysis to organize information into typed memory contexts. Originally derived from a desktop office assistant (OpenContext), it now serves as a standalone memory processing service.

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

There is no test suite in this project currently.

## Architecture

### Data Pipeline: Capture → Process → Store → Consume

The system follows a 4-stage pipeline:

1. **Capture** (`opencontext/context_capture/`): Collects raw data from sources (screenshots, chat logs, documents, web links, folder monitoring). Each capture component implements `ICaptureComponent` and produces `RawContextProperties`.

2. **Process** (`opencontext/context_processing/`): Transforms raw data into structured `ProcessedContext`. The `ProcessorFactory` routes by `ContextSource`:
   - `SCREENSHOT` → `ScreenshotProcessor`
   - `LOCAL_FILE`, `VAULT`, `WEB_LINK` → `DocumentProcessor`
   - `CHAT_LOG`, `INPUT` → `TextChatProcessor`

   Processing includes chunking, entity/keyword extraction via LLM, and embedding vectorization. The context merger handles deduplication and similarity-based merging.

3. **Store** (`opencontext/storage/`): `UnifiedStorage` abstracts over dual backends:
   - **Document DB** (MySQL or SQLite) for structured context data
   - **Vector DB** (VikingDB, DashVector, Qdrant, or ChromaDB) for semantic search

   Access via `GlobalStorage.get_instance()`.

4. **Consume** (`opencontext/context_consumption/`): Agent-based chat with workflow stages (Intent → Context Retrieval → Execution → Reflection), smart completions, and content generation (tips, todos, reports).

### Server Layer

- **Entry point**: `opencontext/cli.py` → `opencontext start` command
- **Framework**: FastAPI on Uvicorn, port 1733
- **Main class**: `opencontext/server/opencontext.py` → `OpenContext` orchestrates all components
- **Routes**: `opencontext/server/routes/` — context CRUD, agent chat, completions, documents, conversations, messages, push API, etc.
- **Auth**: API key via `X-API-Key` header (`opencontext/server/middleware/auth.py`), disabled by default

### Key Data Models (`opencontext/models/`)

- **`RawContextProperties`**: Raw input with `source`, `content_format`, `content_text/content_path`, multi-user fields (`user_id`, `device_id`, `agent_id`)
- **`ProcessedContext`**: Stored context with `ContextProperties`, `ExtractedData` (title, summary, keywords, entities, importance, confidence), `Vectorize` (embedding text + vector)
- **`ContextType` enum**: `entity_context`, `activity_context`, `intent_context`, `semantic_context`, `procedural_context`, `state_context`, `knowledge_context`
- **`ContextSource` enum**: `screenshot`, `vault`, `local_file`, `web_link`, `input`, `chat_log`

### LLM Integration (`opencontext/llm/`)

Three global singletons accessed via `get_instance()`:
- `GlobalEmbeddingClient` — text embeddings (Volcengine Doubao default)
- `GlobalVLMClient` — vision/image understanding
- `LLMClient` — text generation

All use OpenAI-compatible API format.

### Singletons Pattern

Core components use singleton pattern with `get_instance()`:
- `GlobalConfig` — configuration from YAML + env vars
- `GlobalStorage` — unified storage access
- `GlobalEmbeddingClient`, `GlobalVLMClient` — LLM clients

### Scheduling (`opencontext/scheduler/`, `opencontext/periodic_task/`)

APScheduler with Redis backend for distributed task scheduling. Tasks include `MemoryCompression` (deduplication) and `DataCleanup`. Multi-user support via `UserKeyBuilder` composite keys.

## Configuration

- Main config: `config/config.yaml` — YAML with `${ENV_VAR:default}` syntax
- Environment variables: copy `.env.example` to `.env`
- Key env vars: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `MYSQL_HOST`, `MYSQL_PASSWORD`, `REDIS_HOST`, `VIKINGDB_ACCESS_KEY_ID`
- Service mode (`service_mode.enabled: true`): stateless deployment requiring external Redis + MySQL/vector DB

## Code Style

- Python 3.10+, Black (line length 100), isort (profile "black")
- Package name in imports: `opencontext`
- Logging: `loguru` via `from opencontext.utils.logging_utils import get_logger`
- Pre-commit hooks auto-format on commit

## Extending the System

- **New capture source**: Implement `ICaptureComponent` from `opencontext/interfaces/capture_interface.py`, return `List[RawContextProperties]`
- **New processor**: Extend `BaseContextProcessor` from `opencontext/context_processing/processor/base_processor.py`, implement `can_process()` and `process()`
- **New storage backend**: Add to `opencontext/storage/backends/`, register in factory
