# context_processing/ -- Processes raw input into typed ProcessedContext objects and merges similar knowledge entries

## File Overview

| File | Responsibility |
|------|----------------|
| `__init__.py` | Re-exports `BaseContextProcessor`, `DocumentProcessor`, `processor_factory`, `ContextMerger` |
| **processor/** | |
| `processor/base_processor.py` | Abstract base class for all processors; implements `IContextProcessor` interface |
| `processor/processor_factory.py` | Factory for creating processor instances; global singleton `processor_factory` |
| `processor/text_chat_processor.py` | Extracts structured memories from chat logs via LLM analysis |
| `processor/document_processor.py` | Processes documents (PDF, DOCX, images, CSV, XLSX, JSONL, MD, TXT) |
| `processor/entity_processor.py` | Standalone functions for validating and persisting entities to relational DB |
| `processor/profile_processor.py` | Standalone functions for LLM-driven profile merge-before-write to relational DB |
| `processor/document_converter.py` | Converts documents to images and analyzes page structure (PDF, DOCX, PPTX, MD) |
| **chunker/** | |
| `chunker/__init__.py` | Re-exports `BaseChunker`, `ChunkingConfig`, `StructuredFileChunker`, `FAQChunker`, `DocumentTextChunker` |
| `chunker/chunkers.py` | Base chunker ABC and structured file chunkers (CSV, XLSX, JSONL, FAQ) |
| `chunker/document_text_chunker.py` | Semantic text chunking with LLM-assisted splitting |
| **merger/** | |
| `merger/__init__.py` | Re-exports `ContextMerger`, `ContextTypeAwareStrategy`, `KnowledgeMergeStrategy`, `StrategyFactory` |
| `merger/context_merger.py` | Knowledge-only similarity merge processor with periodic compression |
| `merger/merge_strategies.py` | Merge strategy ABC and `KnowledgeMergeStrategy` implementation |

## Class Hierarchy

```
IContextProcessor (interface: opencontext/interfaces/processor_interface.py)
  +-- BaseContextProcessor (ABC)
        +-- TextChatProcessor        (processor/text_chat_processor.py)
        +-- DocumentProcessor        (processor/document_processor.py)
        +-- ContextMerger            (merger/context_merger.py)

BaseChunker (ABC)                    (chunker/chunkers.py)
  +-- StructuredFileChunker          (chunker/chunkers.py)
  +-- FAQChunker                     (chunker/chunkers.py)
  +-- DocumentTextChunker            (chunker/document_text_chunker.py)

ContextTypeAwareStrategy (ABC)       (merger/merge_strategies.py)
  +-- KnowledgeMergeStrategy         (merger/merge_strategies.py)

DocumentConverter                    (processor/document_converter.py)
PageInfo                             (processor/document_converter.py)

ChunkingConfig                       (chunker/chunkers.py)
StrategyFactory                      (merger/merge_strategies.py)
ProcessorFactory                     (processor/processor_factory.py)
```

## Key Classes and Functions

### ProcessorFactory (`processor/processor_factory.py`)

Global singleton: `processor_factory = ProcessorFactory()`

Registers built-in processors on init:
- `"document_processor"` -> `DocumentProcessor`
- `"text_chat_processor"` -> `TextChatProcessor`

```python
class ProcessorFactory:
    def register_processor_type(self, type_name: str, processor_class: Type[IContextProcessor]) -> bool
    def create_processor(self, type_name: str) -> Optional[IContextProcessor]
    def create_processor_with_validation(self, type_name: str) -> Optional[IContextProcessor]
    def get_registered_types(self) -> List[str]
    def is_type_registered(self, type_name: str) -> bool
```

### BaseContextProcessor (`processor/base_processor.py`)

Abstract base implementing `IContextProcessor`. All processors inherit from this.

Key abstract methods subclasses must implement:
- `get_description(self) -> str`
- `can_process(self, context: Any) -> bool`
- `process(self, context: Any) -> List[ProcessedContext]`

All concrete processors (`TextChatProcessor`, `DocumentProcessor`, `ContextMerger`) return `List[ProcessedContext]` from `process()`, consistent with the interface contract.

Provided methods:
```python
def get_name(self) -> str                    # returns class name by default; subclasses override
def get_version(self) -> str                 # returns "1.0.0"
@property
def is_initialized(self) -> bool             # returns _is_initialized
def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool
def validate_config(self, config: Dict[str, Any]) -> bool   # default returns True
def batch_process(self, contexts: List[Any]) -> Dict[str, List[ProcessedContext]]
def set_callback(self, callback: Optional[Callable[[List[ProcessedContext]], None]]) -> bool
def get_statistics(self) -> Dict[str, Any]
def reset_statistics(self) -> bool           # resets stats to zero
def shutdown(self) -> bool
def _extract_object_id(self, context: Any, processed_contexts: List[ProcessedContext]) -> str  # used by batch_process
def _invoke_callback(self, processed_contexts: List[ProcessedContext]) -> None  # invokes callback if set
```

State fields: `config: dict`, `_is_initialized: bool`, `_callback: Optional[Callable]`, `_processing_stats: dict`

### TextChatProcessor (`processor/text_chat_processor.py`)

Processes `ContextSource.CHAT_LOG` + `ContentFormat.TEXT` inputs.

```python
def can_process(self, context: RawContextProperties) -> bool
async def process(self, context: RawContextProperties) -> List[ProcessedContext]
async def _process_async(self, raw_context: RawContextProperties) -> List[ProcessedContext]
```

Pipeline:
1. Calls LLM with prompt `processing.extraction.chat_analyze`
2. Parses JSON response -> `memories` array
3. Builds `ProcessedContext` per memory via `_build_processed_context()`
4. Persists extracted entities across all memories via `_persist_entities()` (calls `entity_processor.refresh_entities`)
5. Returns `List[ProcessedContext]` — callback invocation is handled by `ContextProcessorManager`

The `_build_processed_context` method validates and sanitizes all LLM output fields (context_type, title, summary, keywords, importance 0-10, confidence 0-100, event_time). Sets `enable_merge = True` only for `ContextType.KNOWLEDGE`.

### DocumentProcessor (`processor/document_processor.py`)

Processes `ContextSource.LOCAL_FILE`, `ContextSource.WEB_LINK`, and `ContextSource.INPUT`.

Supported formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.csv`, `.jsonl`, `.md`, `.txt`

```python
def can_process(self, context: RawContextProperties) -> bool
async def process(self, context: RawContextProperties) -> List[ProcessedContext]
def real_process(self, raw_context: RawContextProperties) -> List[ProcessedContext]
```

`process()` delegates to `real_process()` via `asyncio.to_thread()` and returns the result directly — no internal storage calls. Storage persistence is handled by the caller (typically `ContextProcessorManager` via its callback).

Four processing paths inside `real_process()`:

| Condition | Method | Strategy |
|-----------|--------|----------|
| Structured (CSV/XLSX/JSONL) | `_process_structured_document()` | Chunker-based (StructuredFileChunker or FAQChunker) |
| Text content (INPUT source) | `_process_text_content()` | DocumentTextChunker |
| Plain text (.txt) | `_process_txt_file()` | Read UTF-8 content -> DocumentTextChunker (no VLM) |
| Visual (PDF/DOCX/images/PPT/MD) | `_process_visual_document()` | Page-by-page: text extraction + VLM for visual pages |

Note: `.txt` files are routed from within `_process_document_page_by_page()` before page analysis begins.

All paths produce `List[ProcessedContext]` via `_create_contexts_from_chunks()`.

Internal components:
- `_document_converter: DocumentConverter` -- converts documents to images, analyzes pages
- `_structured_chunker: StructuredFileChunker` -- CSV/XLSX/JSONL chunking
- `_faq_chunker: FAQChunker` -- FAQ Excel chunking
- `_document_chunker: DocumentTextChunker` -- semantic text chunking

Config keys (from `processing.document_processor`): `batch_size`, `batch_timeout`
Config keys (from `document_processing`): `enabled`, `dpi`, `vlm_batch_size`, `text_threshold_per_page`

### entity_processor (`processor/entity_processor.py`)

Standalone module (no class). Two public functions:

```python
def validate_and_clean_entities(raw_entities: list) -> Dict[str, EntityData]
async def refresh_entities(
    entities_info: Dict[str, EntityData],
    context_text: str,
    user_id: str = "default",
    device_id: str = "default",
    agent_id: str = "default",
) -> List[str]
```

`validate_and_clean_entities`: Converts list of dicts (with `name`, `type`, `description`, `aliases`, `metadata`) into `Dict[str, EntityData]`.

`refresh_entities`: For each entity, checks if it exists in relational DB via `storage.get_entity()`. If exists, merges content/aliases/keywords. Upserts via `storage.upsert_entity()`. Always requires all three identifiers (`user_id`, `device_id`, `agent_id`).

### profile_processor (`processor/profile_processor.py`)

Standalone module (no class). Two public functions:

```python
def refresh_profile(
    new_content: str,
    new_summary: Optional[str],
    new_keywords: Optional[List[str]],
    new_entities: Optional[List[str]],
    new_importance: int,
    new_metadata: Optional[Dict[str, Any]],
    user_id: str = "default",
    device_id: str = "default",
    agent_id: str = "default",
) -> bool
```

`refresh_profile`: Checks if a profile exists via `storage.get_profile()`. If exists, calls `_merge_profile_with_llm()` to intelligently merge old and new data using the `merging.overwrite_merge` prompt, then upserts the merged result. If no existing profile, writes directly. Falls back to direct overwrite if LLM merge fails. **Note**: `summary` is always stored as `None` — profile no longer uses the summary field.

`_merge_profile_with_llm`: Internal function. Loads `merging.overwrite_merge` prompt group, serializes old/new profile data as JSON (excluding `summary`), calls `generate_with_messages()` (sync), parses response JSON expecting `content`, `keywords`, `entities`, `importance` fields.

Called from: `OpenContext._store_profile()` in `opencontext/server/opencontext.py`.

### DocumentConverter (`processor/document_converter.py`)

Converts documents to PIL images and analyzes page structure.

```python
class DocumentConverter:
    def __init__(self, dpi: int = 200)
    def convert_to_images(self, file_path: str) -> List[Image.Image]      # PDF, images, PPTX
    def analyze_pdf_pages(self, file_path: str, text_threshold: int = 50) -> List[PageInfo]
    def analyze_docx_pages(self, file_path: str) -> List[PageInfo]
    def analyze_markdown_pages(self, file_path: str, chars_per_group: int = 2000) -> List[PageInfo]

class PageInfo:
    page_number: int
    text: str
    has_visual_elements: bool     # True if page contains images/charts -> needs VLM
    doc_images: List[Image.Image] # Embedded images (DOCX/MD only)
```

PDF analysis: uses `pypdf` to extract text and detect images per page. Pages with images or less text than threshold are flagged `has_visual_elements=True`.

DOCX analysis: splits by page breaks or character count (2000 chars), extracts embedded images from paragraph XML.

Markdown analysis: groups by `#`/`##` headings + character count, extracts local/remote `![](path)` images.

PPTX: converts to PDF via LibreOffice, then processes as PDF.

### BaseChunker and Subclasses (`chunker/chunkers.py`)

```python
class ChunkingConfig:
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    batch_size: int = 100
    enable_caching: bool = True

class BaseChunker(ABC):
    def chunk(self, context: RawContextProperties) -> Iterator[Chunk]   # abstract
    def chunk_to_list(self, context: RawContextProperties) -> List[Chunk]

class StructuredFileChunker(BaseChunker):
    # Handles CSV (streaming via pd.read_csv chunksize), XLSX (sheet-by-sheet), JSONL (line batches)
    def chunk(self, context: RawContextProperties) -> Iterator[Chunk]

class FAQChunker(BaseChunker):
    # Treats each Q&A row in Excel as a separate Chunk
    def chunk(self, context: RawContextProperties) -> Iterator[Chunk]
```

### DocumentTextChunker (`chunker/document_text_chunker.py`)

Primary entry point is `chunk_text()`, NOT the inherited `chunk()` method (which raises `NotImplementedError`).

```python
class DocumentTextChunker(BaseChunker):
    def chunk_text(self, texts: List[str], document_title: str = None) -> List[Chunk]
    def chunk(self, _context) -> Iterator[Chunk]  # raises NotImplementedError
```

Chunking strategy:
- Short documents (< 10,000 chars): `_global_semantic_chunking()` -- single LLM call with prompt `document_processing.global_semantic_chunking`
- Long documents (>= 10,000 chars): `_fallback_chunking()` -- accumulates text buffers, splits each via LLM with prompt `document_processing.text_chunking`, falls back to mechanical sentence splitting on failure

### ContextMerger (`merger/context_merger.py`)

Extends `BaseContextProcessor`. Only processes `ContextType.KNOWLEDGE`.

```python
class ContextMerger(BaseContextProcessor):
    def can_process(self, context: ProcessedContext) -> bool             # True only for KNOWLEDGE
    def find_merge_target(self, context: ProcessedContext) -> Optional[ProcessedContext]
    def merge_multiple(self, target: ProcessedContext, sources: List[ProcessedContext]) -> Optional[ProcessedContext]
    def periodic_memory_compression(self, interval_seconds: int)
    def periodic_memory_compression_for_user(self, user_id, device_id=None, agent_id=None, interval_seconds=1800)
    def intelligent_memory_cleanup(self)
    def memory_reinforcement(self, context_ids: List[str])
    def get_memory_statistics(self) -> Dict[str, Any]
```

Merge target finding:
1. Vectorizes the context via `do_vectorize()`
2. If intelligent merging enabled and strategy exists: queries vector DB for candidates, scores each via `KnowledgeMergeStrategy.can_merge()`, picks highest score
3. Fallback: queries for top-2 similar, checks similarity > threshold (default 0.85)

Periodic compression (`periodic_memory_compression_for_user`):
1. Fetches recent KNOWLEDGE contexts with `enable_merge=True` and `has_compression=False`
2. Groups by cosine similarity using `_group_contexts_by_similarity()`
3. Merges groups with `merge_multiple()`, upserts merged, deletes originals

Config keys (from `processing.context_merger`): `similarity_threshold` (default 0.85), `use_intelligent_merging` (default True), `enable_memory_management` (default False), `knowledge_retention_days` (default 30), `knowledge_similarity_threshold` (default 0.8), `knowledge_max_merge_count` (default 3)

### Merge Strategies (`merger/merge_strategies.py`)

```python
class ContextTypeAwareStrategy(ABC):
    def __init__(self, config: dict)  # reads {type}_similarity_threshold, {type}_retention_days, {type}_max_merge_count
    def can_merge(self, target: ProcessedContext, source: ProcessedContext) -> Tuple[bool, float]
    def merge_contexts(self, target: ProcessedContext, sources: List[ProcessedContext]) -> Optional[ProcessedContext]
    def calculate_forgetting_probability(self, context: ProcessedContext) -> float
    def should_cleanup(self, context: ProcessedContext) -> bool

class KnowledgeMergeStrategy(ContextTypeAwareStrategy):
    # can_merge: requires keyword_overlap >= 0.3 OR entity_overlap >= 0.3, AND vector_sim > threshold
    # merge_contexts: combines entities/keywords by frequency (top 10), uses highest-importance title/summary
    # Final score = entity_overlap*0.3 + keyword_overlap*0.3 + vector_sim*0.4

class StrategyFactory:
    def __init__(self, config: dict)  # creates KnowledgeMergeStrategy only
    def get_supported_types(self) -> List[ContextType]
    def get_strategy(self, context_type: ContextType) -> Optional[ContextTypeAwareStrategy]
```

## Internal Data Flow

### Chat Processing Flow
```
RawContextProperties (CHAT_LOG + TEXT)
  -> ContextProcessorManager.process()
    -> TextChatProcessor.process()
      -> _process_async()
        -> LLM call (chat_analyze prompt) -> JSON with memories[]
        -> _build_processed_context() per memory -> List[ProcessedContext]
        -> _persist_entities() -> entity_processor.refresh_entities() -> relational DB
      -> return List[ProcessedContext]
    -> Manager invokes callback -> OpenContext._handle_processed_context()
```

### Document Processing Flow
```
RawContextProperties (LOCAL_FILE / WEB_LINK / INPUT)
  -> ContextProcessorManager.process()
    -> DocumentProcessor.process()
      -> real_process() via asyncio.to_thread()
        -> route by document type:
           |-- Structured (CSV/XLSX/JSONL) -> StructuredFileChunker/FAQChunker.chunk() -> List[Chunk]
           |-- Text (INPUT source) -> DocumentTextChunker.chunk_text() -> List[Chunk]
           |-- Visual (PDF/DOCX/images/PPT/MD):
                 -> DocumentConverter.analyze_*_pages() -> List[PageInfo]
                 -> text pages: use extracted text directly
                 -> visual pages: VLM analysis -> extracted text
                 -> merge all pages -> DocumentTextChunker.chunk_text() -> List[Chunk]
        -> _create_contexts_from_chunks(chunks) -> List[ProcessedContext]
      -> return List[ProcessedContext]
    -> Manager invokes callback -> OpenContext._handle_processed_context()
```

**Note**: Processors are pure data transformers — they return `List[ProcessedContext]` without performing storage writes or callback invocations. The `ContextProcessorManager` centrally handles callback dispatch after collecting results.

### Knowledge Merge Flow
```
ProcessedContext (KNOWLEDGE, enable_merge=True)
  -> ContextMerger.find_merge_target()
    -> vectorize context
    -> query vector DB for candidates
    -> KnowledgeMergeStrategy.can_merge() per candidate -> (bool, score)
    -> return best target
  -> ContextMerger.merge_multiple(target, sources)
    -> KnowledgeMergeStrategy.merge_contexts() or _merge_with_llm()
    -> merged ProcessedContext
```

## Cross-Module Dependencies

### Imports FROM other modules
| Dependency | Used by | Purpose |
|------------|---------|---------|
| `opencontext.models.context` | All files | `ProcessedContext`, `RawContextProperties`, `ContextProperties`, `ExtractedData`, `Vectorize`, `Chunk`, `EntityData` |
| `opencontext.models.enums` | All files | `ContextType`, `ContextSource`, `ContentFormat`, `FileType`, `STRUCTURED_FILE_TYPES` |
| `opencontext.config.global_config` | Processors, chunker | `get_config()`, `get_prompt_group()`, `get_prompt_manager()` |
| `opencontext.llm.global_vlm_client` | TextChat, Document, Merger, Profile | `generate_with_messages_async()`, `generate_with_messages()` |
| `opencontext.llm.global_embedding_client` | Merger | `do_vectorize()`, `do_vectorize_async()` |
| `opencontext.storage.global_storage` | Entity, Profile, Merger | `get_storage()` -> `UnifiedStorage` |
| `opencontext.interfaces.processor_interface` | BaseProcessor | `IContextProcessor` interface |
| `opencontext.utils.json_parser` | TextChat, Document, Profile, Merger | `parse_json_from_response()` |
| `opencontext.monitoring.monitor` | Document | `record_processing_error()`, `record_processing_metrics()` |

### Depended on BY other modules
| Consumer | What it uses |
|----------|-------------|
| `opencontext.server.opencontext` (OpenContext) | `processor_factory.create_processor()`, `ContextMerger` |
| `opencontext.server.routes/push.py` | `DocumentProcessor.process_async()`, `TextChatProcessor` via factory |
| `opencontext.periodic_task.memory_compression` | `ContextMerger.periodic_memory_compression_for_user()` |
| `opencontext.server.routes` | Entity functions called indirectly via TextChatProcessor |
| `opencontext.context_capture.folder_monitor` | `DocumentProcessor.get_supported_formats()` for file type filtering |

## Extension Points

### Adding a new processor
1. Create a class extending `BaseContextProcessor` in `processor/`
2. Implement `get_description()`, `can_process(context)`, and `process(context)`
3. Register in `ProcessorFactory._register_built_in_processors()` with a string key
4. If it needs async processing, add `process_async()` following DocumentProcessor's pattern

### Adding a new chunker
1. Create a class extending `BaseChunker` in `chunker/`
2. Implement `chunk(self, context: RawContextProperties) -> Iterator[Chunk]`
3. Export from `chunker/__init__.py`
4. Instantiate in the processor that uses it (e.g., `DocumentProcessor.__init__`)

### Adding a new merge strategy
The merger currently only supports KNOWLEDGE type. To add a new type:
1. Create a class extending `ContextTypeAwareStrategy` in `merger/merge_strategies.py`
2. Implement `get_context_type()`, `can_merge()`, `merge_contexts()`
3. Register in `StrategyFactory.__init__` with the corresponding `ContextType`
4. Ensure the context type's `UpdateStrategy` is `APPEND_MERGE` in `enums.py`

## Conventions and Constraints

- **Merger only handles KNOWLEDGE**: `ContextMerger.can_process()` returns `False` for all non-KNOWLEDGE types. Document uses delete+insert, event is immutable append. Do not route other types through the merger. Profile and entity have their own dedicated merge logic in `profile_processor.py` and `entity_processor.py` respectively.

- **Profile persistence uses LLM merge**: `profile_processor.refresh_profile()` calls LLM with the `merging.overwrite_merge` prompt to intelligently merge new profile data with existing records before writing. This requires all 3 identifiers (`user_id`, `device_id`, `agent_id`). If LLM fails, falls back to direct overwrite.

- **DocumentTextChunker uses `chunk_text()`, not `chunk()`**: Unlike other chunkers, `DocumentTextChunker.chunk()` raises `NotImplementedError`. Always call `chunk_text(texts: List[str])` instead.

- **Entity persistence requires all 3 identifiers**: `refresh_entities()` and all `storage.upsert_entity()` / `storage.get_entity()` calls require `user_id`, `device_id`, `agent_id`. Omitting any causes argument mismatches.

- **`shutdown()` signature varies across subclasses**: `BaseContextProcessor.shutdown(self) -> bool` takes no args and returns bool. `DocumentProcessor.shutdown(self, _graceful: bool = False)` adds an extra parameter and returns None implicitly. This deviates from the base class contract.

- **Sync/async bridging in processors**: `TextChatProcessor.process()` detects whether an event loop is running and either creates a task or calls `asyncio.run()`. `DocumentProcessor._run_async_tasks()` uses `asyncio.run(asyncio.gather(...))` from the sync `real_process()` method (which runs inside `asyncio.to_thread`). Be careful when modifying this logic.

- **VLM batch size is configurable**: `DocumentProcessor._vlm_batch_size` controls how many pages are sent to VLM in parallel. Set via `document_processing.vlm_batch_size` config.

- **`_create_contexts_from_chunks` always produces `ContextType.DOCUMENT`**: All chunks from `DocumentProcessor` are typed as DOCUMENT regardless of file content. Classification into other types happens only in `TextChatProcessor` via LLM.

- **ProcessorFactory creates instances with parameterless constructors**: All processors read their own config from `GlobalConfig` inside `__init__()`. The factory's `config` and `**dependencies` parameters are deprecated and ignored.

- **ContextMerger._merge_with_llm has Chinese string dependency**: Checks `if "无需合并" in response:` ("no merge needed") to detect when LLM declines a merge. This works with Chinese prompts but will not match if using English prompts.
