# opencontext/models/ -- Core data models and enums for the 5-type context system.

## File Overview

| File | Responsibility |
|------|---------------|
| `enums.py` | All enums (`ContextType`, `ContextSource`, `UpdateStrategy`, etc.), type-to-strategy/storage mappings, context description dicts, and helper functions for prompt formatting |
| `context.py` | Pydantic models: pipeline intermediates (`ProcessedContext`, `ContextProperties`, `ExtractedData`), API response models (`ProcessedContextModel`, `RawContextModel`), relational DB models (`ProfileData`, `EntityData`), and utilities (`Chunk`, `Vectorize`, `KnowledgeContextMetadata`) |
| `__init__.py` | Re-exports: `ContextProperties`, `EntityData`, `ExtractedData`, `ProcessedContext`, `ProfileData`, `RawContextProperties`, `ContextSource`, `ContentFormat` |

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
TEXT = "text"  |  IMAGE = "image"  |  FILE = "file"
```

### ContextType(str, Enum)
```
PROFILE = "profile"  |  ENTITY = "entity"  |  DOCUMENT = "document"
EVENT = "event"      |  KNOWLEDGE = "knowledge"
```

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
    PROFILE: OVERWRITE, ENTITY: OVERWRITE, DOCUMENT: OVERWRITE,
    EVENT: APPEND, KNOWLEDGE: APPEND_MERGE,
}

CONTEXT_STORAGE_BACKENDS = {
    PROFILE: "document_db", ENTITY: "document_db",
    DOCUMENT: "vector_db", EVENT: "vector_db", KNOWLEDGE: "vector_db",
}

STRUCTURED_FILE_TYPES = {XLSX, XLS, CSV, JSONL, PARQUET, FAQ_XLSX}
```

### ContextDescriptions / ContextSimpleDescriptions
Both are `Dict[ContextType, dict]`. `ContextDescriptions` includes `key_indicators`, `examples`, `classification_priority` (used in LLM prompts). `ContextSimpleDescriptions` has `name`, `description`, `purpose`.

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
| `event_time` | `datetime` | required | Can be future |
| `update_time` | `datetime` | required | |
| `is_processed` | `bool` | `False` | |
| `has_compression` | `bool` | `False` | |
| `call_count` | `int` | `0` | Updated on retrieval |
| `merge_count` | `int` | `0` | |
| `duration_count` | `int` | `1` | |
| `enable_merge` | `bool` | `False` | |
| `is_happened` | `bool` | `False` | |
| `last_call_time` | `Optional[datetime]` | `None` | |
| `file_path` | `Optional[str]` | `None` | Document tracking |
| `raw_type` | `Optional[str]` | `None` | e.g. `"vaults"` |
| `raw_id` | `Optional[str]` | `None` | |
| `user_id`, `device_id`, `agent_id` | `Optional[str]` | `None` | 3-key identifier |
| `hierarchy_level` | `int` | `0` | 0=raw, 1=daily, 2=weekly, 3=monthly |
| `parent_id` | `Optional[str]` | `None` | Parent summary context ID; backfilled by `batch_set_parent_id()` after hierarchy summary generation, enables upward traversal (L0 → L1/L2/L3) |
| `children_ids` | `List[str]` | `[]` | Child context IDs; set during hierarchy summary generation, enables downward traversal and drill-down |
| `time_bucket` | `Optional[str]` | `None` | e.g. `"2026-02-21"`, `"2026-W08"` |
| `source_file_key` | `Optional[str]` | `None` | `"user_id:file_path"` format |

### Vectorize
Embedding configuration. Fields: `content_format` (ContentFormat, default `ContentFormat.TEXT`), `image_path` (Optional[str]), `text` (Optional[str]), `vector` (Optional[List[float]]).
Method: `get_vectorize_content() -> str`

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
API response model. Mirrors `ProcessedContext` fields as serialized types (datetimes as `str`, enums as `str`, embedding as `Optional[List[float]]`). Includes hierarchy fields and `source_file_key`.
Key methods: `from_processed_context(cls, pc: ProcessedContext, project_root: Path) -> ProcessedContextModel`, `from_dict(cls, data: Dict[str, Any]) -> ProcessedContextModel`

### RawContextModel
API response model for raw context. Fields: `object_id`, `content_format`, `source`, `create_time` (all `str`), plus optional `content_path`, `content_text`, `additional_info`.
Key method: `from_raw_context_properties(cls, rcp: RawContextProperties, project_root: Path) -> RawContextModel`

### ProfileData
Relational DB model. Composite PK: `(user_id, device_id, agent_id)`.
| Field | Type | Default |
|-------|------|---------|
| `user_id` | `str` | required |
| `device_id` | `str` | `"default"` |
| `agent_id` | `str` | `"default"` |
| `content` | `str` | required |
| `summary` | `Optional[str]` | `None` | **Deprecated** — DB column exists for backward compatibility but profile processor always stores `None`. Not exposed in API responses. |
| `keywords` | `List[str]` | `[]` |
| `entities` | `List[str]` | `[]` |
| `importance` | `int` | `0` |
| `metadata` | `Optional[Dict[str, Any]]` | `{}` |
| `created_at` | `datetime` | `now(utc)` |
| `updated_at` | `datetime` | `now(utc)` |

Methods: `to_dict() -> Dict`, `from_dict(cls, data) -> ProfileData`

### EntityData
Relational DB model. Unique key: `(user_id, device_id, agent_id, entity_name)`.
| Field | Type | Default |
|-------|------|---------|
| `id` | `str` | `uuid4()` |
| `user_id` | `str` | required |
| `device_id` | `str` | `"default"` |
| `agent_id` | `str` | `"default"` |
| `entity_name` | `str` | required |
| `entity_type` | `Optional[str]` | `None` |
| `content` | `str` | required |
| `summary` | `Optional[str]` | `None` |
| `keywords` | `List[str]` | `[]` |
| `aliases` | `List[str]` | `[]` |
| `relationships` | `Dict[str, List[str]]` | `{}` |
| `metadata` | `Optional[Dict[str, Any]]` | `{}` |
| `created_at`, `updated_at` | `datetime` | `now(utc)` |

Methods: `to_dict() -> Dict`, `from_dict(cls, data) -> EntityData`

Note: `relationships` is stored inside the `metadata` JSON column in the DB, not as a separate column.

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
- `opencontext/storage/` -- stores/retrieves `ProcessedContext`, `ProfileData`, `EntityData`
- `opencontext/server/` -- uses API response models, routes by `ContextType`/`CONTEXT_STORAGE_BACKENDS`
- `opencontext/tools/` -- uses `ContextType`, description helpers
- `opencontext/config/prompt_manager.py` -- calls enum description helpers
- `opencontext/llm/` -- uses `ExtractedData`, `ContextType`

## Conventions and Constraints

1. **ProcessedContextModel must mirror ContextProperties**: If you add a field to `ContextProperties`, also add it to `ProcessedContextModel` and update `from_processed_context()`, or it will be silently dropped from API responses.
2. **All 5 types must stay in sync**: Adding/removing a `ContextType` value requires updating `CONTEXT_UPDATE_STRATEGIES`, `CONTEXT_STORAGE_BACKENDS`, `ContextDescriptions`, `ContextSimpleDescriptions`, and both prompt YAML files.
3. **3-key identifier required**: `ProfileData` and `EntityData` always require `(user_id, device_id, agent_id)`. Defaults are `"default"` for `device_id` and `agent_id`.
4. **Timezone-aware datetimes**: Use `datetime.now(tz=datetime.timezone.utc)`, never `datetime.utcnow()`.
5. **`get_context_type_for_analysis()` falls back to KNOWLEDGE**: Unrecognized type strings from LLM output default to `KNOWLEDGE`, not `EVENT`.
6. **Enum values are lowercase strings**: All `str, Enum` classes use lowercase values matching their `.value` attribute.
