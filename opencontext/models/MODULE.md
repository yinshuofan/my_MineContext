# opencontext/models/ -- Core data models and enums for the 4-type context system.

## File Overview

| File | Responsibility |
|------|---------------|
| `enums.py` | All enums (`ContextType`, `ContextSource`, `UpdateStrategy`, etc.), type-to-strategy/storage mappings, context description dicts, and helper functions for prompt formatting |
| `context.py` | Pydantic models: pipeline intermediates (`ProcessedContext`, `ContextProperties`, `ExtractedData`), API response models (`ProcessedContextModel`, `RawContextModel`), relational DB models (`ProfileData`), and utilities (`Chunk`, `Vectorize`, `KnowledgeContextMetadata`) |
| `__init__.py` | Re-exports: `ContextProperties`, `ExtractedData`, `ProcessedContext`, `ProfileData`, `RawContextProperties`, `ContextSource`, `ContentFormat` |

## Key Enums (enums.py)

### ContextSource(str, Enum)
```
VAULT = "vault"  |  LOCAL_FILE = "local_file"  |  WEB_LINK = "web_link"
INPUT = "input"  |  CHAT_LOG = "chat_log"
```

### FileType(str, Enum)
Documents: `PDF`, `DOCX`, `DOC`, `PPTX`, `PPT`
Spreadsheets: `FAQ_XLSX`, `XLSX`, `XLS`, `CSV`, `JSONL`, `PARQUET`
Images: `PNG`, `JPG`, `JPEG`, `GIF`, `BMP`, `WEBP`
Text: `MD`, `TXT`

### ContentFormat(str, Enum)
```
TEXT = "text"  |  IMAGE = "image"  |  VIDEO = "video"  |  MULTIMODAL = "multimodal"  |  FILE = "file"
```

- `VIDEO`: Single video content (used when only video is present)
- `MULTIMODAL`: Mixed-modality content (text + image and/or video combined)

### ContextType(str, Enum)
```
PROFILE = "profile"              |  DOCUMENT = "document"
EVENT = "event"                  |  KNOWLEDGE = "knowledge"
DAILY_SUMMARY = "daily_summary"  |  WEEKLY_SUMMARY = "weekly_summary"  |  MONTHLY_SUMMARY = "monthly_summary"
AGENT_EVENT = "agent_event"
AGENT_DAILY_SUMMARY = "agent_daily_summary"  |  AGENT_WEEKLY_SUMMARY = "agent_weekly_summary"  |  AGENT_MONTHLY_SUMMARY = "agent_monthly_summary"
AGENT_PROFILE = "agent_profile"
AGENT_BASE_PROFILE = "agent_base_profile"
```

- `DAILY_SUMMARY`, `WEEKLY_SUMMARY`, `MONTHLY_SUMMARY`: User-side hierarchy summaries (L1/L2/L3). Formerly stored as `EVENT` with `hierarchy_level > 0`; now have their own types.
- `AGENT_EVENT`: Agent-observed events (same role as `EVENT` but for agent memory owner).
- `AGENT_DAILY_SUMMARY`, `AGENT_WEEKLY_SUMMARY`, `AGENT_MONTHLY_SUMMARY`: Agent-side hierarchy summaries.
- `AGENT_PROFILE`: Agent's perception and knowledge about a specific user. Stored in relational DB (same as `PROFILE`) with `context_type="agent_profile"`. Extracted by `AgentMemoryProcessor` and routed to `_store_profile()` alongside `PROFILE`.
- `AGENT_BASE_PROFILE`: Agent's pre-configured base profile, set via the `/api/agents/{agent_id}/base/profile` endpoint. Stored in relational DB with `user_id="__base__"` and `context_type="agent_base_profile"`. Distinguished from per-user `AGENT_PROFILE` by both the sentinel `user_id` and the dedicated context type.

### UpdateStrategy(str, Enum)
```
OVERWRITE = "overwrite"  |  APPEND = "append"  |  APPEND_MERGE = "append_merge"
```

### VaultType(str, Enum)
```
DAILY_REPORT = "DailyReport"  |  WEEKLY_REPORT = "WeeklyReport"  |  NOTE = "Note"
```

### CompletionType(Enum) -- note: `Enum`, not `str, Enum`
```
SEMANTIC_CONTINUATION = "semantic_continuation"  |  TEMPLATE_COMPLETION = "template_completion"
REFERENCE_SUGGESTION = "reference_suggestion"    |  CONTEXT_AWARE = "context_aware"
```

## Mapping Dicts (enums.py)

```python
CONTEXT_UPDATE_STRATEGIES = {
    PROFILE: OVERWRITE, DOCUMENT: OVERWRITE,
    EVENT: APPEND, KNOWLEDGE: APPEND_MERGE,
    DAILY_SUMMARY: APPEND, WEEKLY_SUMMARY: APPEND, MONTHLY_SUMMARY: APPEND,
    AGENT_EVENT: APPEND,
    AGENT_DAILY_SUMMARY: APPEND, AGENT_WEEKLY_SUMMARY: APPEND, AGENT_MONTHLY_SUMMARY: APPEND,
    AGENT_PROFILE: OVERWRITE,
    AGENT_BASE_PROFILE: OVERWRITE,
}

CONTEXT_STORAGE_BACKENDS = {
    PROFILE: "document_db",
    DOCUMENT: "vector_db", EVENT: "vector_db", KNOWLEDGE: "vector_db",
    DAILY_SUMMARY: "vector_db", WEEKLY_SUMMARY: "vector_db", MONTHLY_SUMMARY: "vector_db",
    AGENT_EVENT: "vector_db",
    AGENT_DAILY_SUMMARY: "vector_db", AGENT_WEEKLY_SUMMARY: "vector_db", AGENT_MONTHLY_SUMMARY: "vector_db",
    AGENT_PROFILE: "document_db",
    AGENT_BASE_PROFILE: "document_db",
}

MEMORY_OWNER_TYPES = {
    "user":  [EVENT, DAILY_SUMMARY, WEEKLY_SUMMARY, MONTHLY_SUMMARY],        # index 0=L0, 1=L1, 2=L2, 3=L3
    "agent": [AGENT_EVENT, AGENT_DAILY_SUMMARY, AGENT_WEEKLY_SUMMARY, AGENT_MONTHLY_SUMMARY],
}

SYSTEM_GENERATED_TYPES = {
    DAILY_SUMMARY, WEEKLY_SUMMARY, MONTHLY_SUMMARY,
    AGENT_DAILY_SUMMARY, AGENT_WEEKLY_SUMMARY, AGENT_MONTHLY_SUMMARY,
}

STRUCTURED_FILE_TYPES = {XLSX, XLS, CSV, JSONL, PARQUET, FAQ_XLSX}
```

`MEMORY_OWNER_TYPES` maps a `memory_owner` string (`"user"` or `"agent"`) to an ordered list of 4 ContextTypes (L0-L3). Used by search, cache, and hierarchy tools to resolve types dynamically instead of hardcoding `EVENT`.

`SYSTEM_GENERATED_TYPES` is a guard set. `get_context_type_for_analysis()` falls back to `KNOWLEDGE` if the LLM output matches a system-generated type, preventing LLM from classifying user input as a summary type.

### ContextDescriptions / ContextSimpleDescriptions
Both are `Dict[ContextType, dict]`. `ContextDescriptions` includes `key_indicators`, `examples`, `classification_priority` (used in LLM prompts). `ContextSimpleDescriptions` has `name`, `description`, `purpose`. Both now include entries for all 12 context types (including `AGENT_PROFILE`).

## Helper Functions (enums.py)

| Function | Returns | Used By |
|----------|---------|---------|
| `get_context_type_options() -> list[str]` | All `ContextType` values | Validation |
| `get_context_descriptions() -> str` | Formatted `"- type: description"` lines | General |
| `validate_context_type(context_type: str) -> bool` | Whether string is valid type | Validation |
| `get_context_type_for_analysis(context_type_str: str) -> ContextType` | Parsed type, falls back to `KNOWLEDGE` | LLM output parsing |
| `get_context_type_choices_for_tools() -> list[str]` | Same as `get_context_type_options()` | Tool parameter enums |
| `get_context_type_descriptions_for_prompts() -> str` | Backtick-formatted type descriptions | Prompt templates |
| `get_context_type_descriptions_for_extraction() -> str` | Descriptions with indicators + examples | LLM classification prompts |
| `get_context_type_descriptions_for_retrieval() -> str` | Compact type descriptions | Query processing |

## Key Pydantic Models (context.py)

### RawContextProperties
Input metadata before processing. Fields:
| Field | Type | Default |
|-------|------|---------|
| `content_format` | `ContentFormat` | required |
| `source` | `ContextSource` | required |
| `create_time` | `datetime` | required |
| `object_id` | `str` | `uuid4()` |
| `content_path` | `Optional[str]` | `None` |
| `content_type` | `Optional[str]` | `None` |
| `content_text` | `Optional[str]` | `None` |
| `filter_path` | `Optional[str]` | `None` |
| `additional_info` | `Optional[Dict[str, Any]]` | `None` |
| `enable_merge` | `bool` | `True` |
| `user_id`, `device_id`, `agent_id` | `Optional[str]` | `None` |

Methods: `to_dict() -> Dict`, `from_dict(cls, data) -> RawContextProperties`

### ExtractedData
LLM extraction results. Fields:
| Field | Type | Default |
|-------|------|---------|
| `title` | `Optional[str]` | `None` |
| `summary` | `Optional[str]` | `None` |
| `keywords` | `List[str]` | `[]` |
| `entities` | `List[str]` | `[]` |
| `context_type` | `ContextType` | required |
| `confidence` | `int` | `0` |
| `importance` | `int` | `0` |

Methods: `to_dict() -> Dict`, `from_dict(cls, data) -> ExtractedData`

### ContextProperties
Tracking and hierarchy metadata. Fields:
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `raw_properties` | `list[RawContextProperties]` | `[]` | |
| `create_time` | `datetime` | required | |
| `event_time_start` | `datetime` | required | Start of event time range. Can be future |
| `event_time_end` | `datetime` | required | End of event time range. Equals `event_time_start` for point-in-time events |
| `update_time` | `datetime` | required | |
| `is_processed` | `bool` | `False` | |
| `has_compression` | `bool` | `False` | |
| `call_count` | `int` | `0` | Updated on retrieval |
| `merge_count` | `int` | `0` | |
| `duration_count` | `int` | `1` | |
| `enable_merge` | `bool` | `False` | |
| `last_call_time` | `Optional[datetime]` | `None` | |
| `file_path` | `Optional[str]` | `None` | Document tracking |
| `raw_type` | `Optional[str]` | `None` | e.g. `"vaults"` |
| `raw_id` | `Optional[str]` | `None` | |
| `user_id`, `device_id`, `agent_id` | `Optional[str]` | `None` | 3-key identifier |
| `hierarchy_level` | `int` | `0` | 0=raw, 1=daily, 2=weekly, 3=monthly |
| `refs` | `Dict[str, List[str]]` | `{}` | Flexible bidirectional reference map. Keys are ContextType values (e.g. `"daily_summary"`, `"event"`), values are lists of context IDs. See below. |

**`refs` field**: A single bidirectional reference map linking contexts across hierarchy levels. Key = the ContextType of the referenced contexts, value = list of context IDs.
- **Downward** (summary → children): e.g. `refs: {"event": ["id1", "id2"]}` on a DAILY_SUMMARY means it was generated from those L0 events.
- **Upward** (child → parent summary): e.g. `refs: {"daily_summary": ["sum-id"]}` on an EVENT means it was summarized by that daily summary. Backfilled by `batch_update_refs()` after summary storage.

### VideoInput
Video input model for multimodal embedding. Fields:
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `url` | `str` | required | HTTP URL, TOS path, or `data:video/...;base64,...` |
| `fps` | `float` | `1.0` | Frame extraction rate (0.2-5.0). Lower = fewer tokens, higher = more detail |

Note: `VideoInput` is kept for reference but is no longer used by `Vectorize` directly. Video inputs are now represented as content parts dicts: `{"type": "video_url", "video_url": {"url": "...", "fps": 1.0}}`.

### Vectorize
Unified embedding configuration using Ark API content parts format. Fields:
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `input` | `List[Dict[str, Any]]` | `[]` | Ark API content parts list (see format below) |
| `vector` | `Optional[List[float]]` | `None` | Pre-computed embedding vector |
| `content_format` | `ContentFormat` | `ContentFormat.TEXT` | |

**`input` format** (OpenAI content parts, same format used by Ark multimodal embedding API):
```python
[
    {"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {"url": "https://... or data:image/...;base64,..."}},
    {"type": "video_url", "video_url": {"url": "https://...", "fps": 1.0}},
]
```

Methods:
- `get_text() -> Optional[str]` -- extracts and joins all text parts from `input`; returns `None` if no text parts. Used for display/logging and storage text fields
- `get_modality_string() -> str` -- scans `input` list `type` fields, returns e.g. `"text"`, `"text and image"`, `"text and image and video"`. Used to generate the `{modality}` placeholder in embedding instructions
- `build_ark_input() -> List[Dict]` -- returns `input` with local file paths converted to base64 data URIs (images and videos). HTTPS URLs and data URIs pass through unchanged

### ProcessedContext
The universal intermediate format all processors produce. Fields:
| Field | Type | Default |
|-------|------|---------|
| `id` | `str` | `uuid4()` |
| `properties` | `ContextProperties` | required |
| `extracted_data` | `ExtractedData` | required |
| `vectorize` | `Vectorize` | required |
| `metadata` | `Optional[Dict[str, Any]]` | `{}` |

Key methods:
- `get_vectorize_content() -> str` -- delegates to `vectorize`
- `get_llm_context_string() -> str` -- formatted string for LLM input (id, title, summary, keywords, entities, type, metadata, times, hierarchy info)
- `to_dict() -> Dict`, `dump_json() -> str`, `from_dict(cls, data)`, `from_json(cls, json_str)`

### ProcessedContextModel
API response model. Mirrors `ProcessedContext` fields as serialized types (datetimes as `str`, enums as `str`, embedding as `Optional[List[float]]`). Includes hierarchy fields.
Key methods: `from_processed_context(cls, pc: ProcessedContext, project_root: Path) -> ProcessedContextModel`, `from_dict(cls, data: Dict[str, Any]) -> ProcessedContextModel`

### RawContextModel
API response model for raw context. Fields: `object_id`, `content_format`, `source`, `create_time` (all `str`), plus optional `content_path`, `content_text`, `additional_info`.
Key method: `from_raw_context_properties(cls, rcp: RawContextProperties, project_root: Path) -> RawContextModel`

### ProfileData
Relational DB model. Composite PK: `(user_id, device_id, agent_id)`. Note: The DB `profiles` table uses a 4-column PK `(user_id, device_id, agent_id, context_type)`, but `ProfileData` does not include `context_type` or `refs` -- those are handled at the storage layer.
| Field | Type | Default |
|-------|------|---------|
| `user_id` | `str` | required |
| `device_id` | `str` | `"default"` |
| `agent_id` | `str` | `"default"` |
| `factual_profile` | `str` | required |
| `behavioral_profile` | `Optional[str]` | `None` | Behavioral profile text (Phase 2) |
| `keywords` | `List[str]` | `[]` |
| `entities` | `List[str]` | `[]` |
| `importance` | `int` | `0` |
| `metadata` | `Optional[Dict[str, Any]]` | `{}` |
| `created_at` | `datetime` | `now(utc)` |
| `updated_at` | `datetime` | `now(utc)` |

Methods: `to_dict() -> Dict`, `from_dict(cls, data) -> ProfileData`

### Chunk
Document chunk. Fields: `text` (Optional[str]), `image` (Optional[bytes]), `chunk_index` (int), `keywords` (List[str]), `entities` (List[str]).

### KnowledgeContextMetadata
Fields: `knowledge_source`, `knowledge_file_path`, `knowledge_title`, `knowledge_raw_id` (all `str`, default `""`).

## Cross-Module Dependencies

**Imports from:**
- `pydantic` (BaseModel, Field)
- `opencontext.utils.logging_utils` (get_logger)

**Depended on by (nearly everything):**
- `opencontext/context_processing/` -- processors produce `ProcessedContext`
- `opencontext/storage/` -- stores/retrieves `ProcessedContext`, `ProfileData`
- `opencontext/server/` -- uses API response models, routes by `ContextType`/`CONTEXT_STORAGE_BACKENDS`
- `opencontext/tools/` -- uses `ContextType`, description helpers
- `opencontext/config/prompt_manager.py` -- calls enum description helpers
- `opencontext/llm/` -- uses `ExtractedData`, `ContextType`

## Conventions and Constraints

1. **ProcessedContextModel must mirror ContextProperties**: If you add a field to `ContextProperties`, also add it to `ProcessedContextModel` and update `from_processed_context()`, or it will be silently dropped from API responses.
2. **All types must stay in sync**: Adding/removing a `ContextType` value requires updating `CONTEXT_UPDATE_STRATEGIES`, `CONTEXT_STORAGE_BACKENDS`, `MEMORY_OWNER_TYPES` (if event-family), `SYSTEM_GENERATED_TYPES` (if system-generated), `ContextDescriptions`, `ContextSimpleDescriptions`, and both prompt YAML files. Profile-family types (`PROFILE`, `AGENT_PROFILE`, `AGENT_BASE_PROFILE`) route to `document_db` and use `_store_profile()` in `opencontext.py`.
3. **3-key identifier required**: `ProfileData` always requires `(user_id, device_id, agent_id)`. Defaults are `"default"` for `device_id` and `agent_id`.
4. **Timezone-aware datetimes**: Use `datetime.now(tz=datetime.timezone.utc)`, never `datetime.utcnow()`.
5. **`get_context_type_for_analysis()` falls back to KNOWLEDGE**: Unrecognized type strings from LLM output default to `KNOWLEDGE`, not `EVENT`.
6. **Enum values are lowercase strings**: All `str, Enum` classes use lowercase values matching their `.value` attribute.
