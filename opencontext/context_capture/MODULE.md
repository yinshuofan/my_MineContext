# context_capture/ -- Input Capture Components for Various Data Sources

Provides a base class and concrete implementations for capturing raw context data from screenshots, web links, local folders, chat messages, and vault documents. Each component produces `RawContextProperties` objects that flow into the processing pipeline.

## File Overview

| File | Responsibility |
|------|---------------|
| `base.py` | `BaseCaptureComponent` -- abstract base implementing `ICaptureComponent` with lifecycle, threading, stats, and callback plumbing |
| `screenshot.py` | `ScreenshotCapture` -- periodic multi-monitor screen capture using `mss` |
| `web_link_capture.py` | `WebLinkCapture` -- converts URLs to Markdown (crawl4ai) or PDF (Playwright) |
| `folder_monitor.py` | `FolderMonitorCapture` -- watches local folders for file create/update/delete events |
| `text_chat.py` | `TextChatCapture` -- buffers chat messages in Redis, flushes when threshold reached |
| `vault_document_monitor.py` | `VaultDocumentMonitor` -- polls the vaults DB table for new/updated documents |
| `__init__.py` | Re-exports `BaseCaptureComponent`, `VaultDocumentMonitor`, `FolderMonitorCapture`, `TextChatCapture` (note: `ScreenshotCapture` and `WebLinkCapture` are NOT exported) |

## Class Hierarchy

```
ICaptureComponent (interface, opencontext/interfaces/capture_interface.py)
  |
  +-- BaseCaptureComponent (base.py) -- abstract, implements lifecycle + stats
        |
        +-- ScreenshotCapture (screenshot.py)
        +-- WebLinkCapture (web_link_capture.py)
        +-- FolderMonitorCapture (folder_monitor.py)
        +-- TextChatCapture (text_chat.py)
        +-- VaultDocumentMonitor (vault_document_monitor.py)
```

## Key Classes and Functions

### `BaseCaptureComponent` (`base.py`)

Abstract base class implementing `ICaptureComponent`. Manages lifecycle (init/start/stop), a periodic capture thread, callback dispatch, and statistics tracking. All public methods are thread-safe via `self._lock` (`threading.RLock`).

**Constructor:**
```python
def __init__(self, name: str, description: str, source_type: ContextSource)
```
Key fields: `_config: Dict`, `_running: bool`, `_capture_thread: Thread`, `_stop_event: threading.Event`, `_callback: Callable`, `_capture_interval: float` (default 1.0), `_capture_count: int`, `_error_count: int`, `_last_capture_time: Optional[datetime]`, `_last_error: Optional[str]`.

**Public API (all implemented in base):**

| Method | Signature | Description |
|--------|-----------|-------------|
| `initialize` | `(config: Dict[str, Any]) -> bool` | Validates config, calls `_initialize_impl` |
| `start` | `() -> bool` | Calls `_start_impl`, optionally starts `_capture_loop` thread |
| `stop` | `(graceful: bool = True) -> bool` | Stops thread (5s join timeout), calls `_stop_impl` |
| `capture` | `() -> List[RawContextProperties]` | Calls `_capture_impl`, invokes callback if results exist (supports both sync and async callbacks) |
| `set_callback` | `(callback: Callable[[List[RawContextProperties]], None])` | Sets data callback (may be sync or async) |
| `get_status` | `() -> Dict[str, Any]` | Base status merged with `_get_status_impl()` |
| `get_statistics` | `() -> Dict[str, Any]` | Base stats merged with `_get_statistics_impl()` |
| `validate_config` | `(config: Dict[str, Any]) -> bool` | Base checks + `_validate_config_impl()` |
| `get_config_schema` | `() -> Dict[str, Any]` | Base schema merged with `_get_config_schema_impl()` |
| `is_running` | `() -> bool` | Whether the component is currently running |
| `get_name` | `() -> str` | Returns `self._name` |
| `get_description` | `() -> str` | Returns `self._description` |
| `reset_statistics` | `() -> bool` | Resets capture/error counts, calls `_reset_statistics_impl()` |

**Abstract methods subclasses must implement:**

| Method | Signature |
|--------|-----------|
| `_initialize_impl` | `(config: Dict[str, Any]) -> bool` |
| `_start_impl` | `() -> bool` |
| `_stop_impl` | `(graceful: bool = True) -> bool` |
| `_capture_impl` | `() -> List[RawContextProperties]` |

**Optional overrides (default no-op):** `_get_config_schema_impl`, `_validate_config_impl`, `_get_status_impl`, `_get_statistics_impl`, `_reset_statistics_impl`.

### `ScreenshotCapture` (`screenshot.py`)

Captures screenshots from all monitors using the `mss` library. Supports configurable format (png/jpg), quality, region, deduplication, and max image size.

- `source_type`: `ContextSource.SCREENSHOT` (**Note**: `SCREENSHOT` does not exist in the `ContextSource` enum in `enums.py` -- this is a latent bug in the code)
- `_take_screenshot() -> list` -- returns list of `(bytes, format_str, details_dict)` per monitor
- `_create_new_context(screenshot_bytes, screenshot_format, timestamp, details) -> RawContextProperties`
- On graceful stop, flushes pending stable screenshots via callback

Config keys: `screenshot_format`, `screenshot_quality`, `screenshot_region`, `storage_path`, `dedup_enabled`, `similarity_threshold`, `max_image_size`.

### `WebLinkCapture` (`web_link_capture.py`)

Converts URLs to Markdown or PDF files, then wraps file paths as `RawContextProperties`.

- `source_type`: `ContextSource.WEB_LINK`
- Overrides `capture(urls: Optional[List[str]] = None) -> List[RawContextProperties]` to accept URL list
- Uses `ThreadPoolExecutor(max_workers)` for parallel conversion

| Method | Signature | Description |
|--------|-----------|-------------|
| `submit_url` | `(url: str) -> List[RawContextProperties]` | Convenience for single URL |
| `convert_url_to_markdown` | `(url: str, filename_hint: Optional[str]) -> Optional[Dict[str, str]]` | Uses `crawl4ai.AsyncWebCrawler` |
| `convert_url_to_pdf` | `(url: str, filename_hint: Optional[str]) -> Optional[Dict[str, str]]` | Uses `playwright` |

Config keys: `output_dir`, `mode` ("pdf" or "markdown"), `timeout`, `wait_until`, `max_workers`, `pdf_format`, `print_background`, `landscape`.

### `FolderMonitorCapture` (`folder_monitor.py`)

Monitors local folders for file changes via periodic polling with SHA-256 hash comparison.

- `source_type`: `ContextSource.LOCAL_FILE`
- Runs its own `_monitor_loop` thread (separate from base capture thread)
- Detects: `file_created`, `file_updated`, `file_deleted`
- On delete, calls `async _cleanup_file_context()` (bridged from sync thread via `run_coroutine_threadsafe`) to remove associated vector DB entries via `UnifiedStorage`
- File type support derived from `DocumentProcessor.get_supported_formats()`

| Method | Signature | Description |
|--------|-----------|-------------|
| `_scan_folder_files` | `(folder_path: str, recursive: bool) -> List[str]` | Lists supported files under size limit |
| `_detect_new_and_updated_files` | `(current_files: set) -> Tuple[List[str], List[str]]` | Compares against `_file_info_cache` |
| `_create_context_from_event` | `(event: Dict) -> Optional[RawContextProperties]` | Maps file ext to `ContentFormat` |
| `_get_content_format` | `(file_ext: str) -> Optional[ContentFormat]` | IMAGE / FILE / TEXT mapping |

Config keys: `watch_folder_paths`, `recursive`, `max_file_size`, `monitor_interval`, `initial_scan`, `capture_interval`.

### `TextChatCapture` (`text_chat.py`)

Stateless chat message buffer backed by Redis. Buffers messages per `(user_id, device_id, agent_id)` key, flushes to the processing pipeline when buffer reaches threshold.

- `source_type`: `ContextSource.CHAT_LOG`
- `_capture_impl()` returns `[]` (passive component; data flows through `push_message`)
- Redis key pattern: `chat:buffer:{user_id}:{device_id}:{agent_id}` (None mapped to `"_"`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `push_message` (async) | `(role, content, user_id, device_id, agent_id)` | Appends to Redis list; flushes if `>= buffer_size` |
| `process_messages_directly` (async) | `(messages, user_id, device_id, agent_id)` | Bypasses buffer, sends immediately via async callback |
| `flush_user_buffer` (async) | `(user_id, device_id, agent_id)` | Manual flush for a specific user |
| `get_user_buffer_length` (async) | `(user_id, device_id, agent_id) -> int` | Current buffer count |
| `get_buffer_stats` (async) | `() -> Dict[str, Any]` | Redis connectivity and buffer stats |

Config keys: `buffer_size` (default 4), `buffer_ttl` (default 86400), `redis` (host, port, password, db, key_prefix).

### `VaultDocumentMonitor` (`vault_document_monitor.py`)

Polls the vaults table in the relational DB for new/updated documents, generates `RawContextProperties` for the pipeline.

- `source_type`: `ContextSource.INPUT` (constructor), but `_create_context_from_event` produces `RawContextProperties` with `source=ContextSource.VAULT`
- Runs its own `_monitor_loop` thread
- Tracks `_processed_vault_ids: Set[int]` to avoid reprocessing

| Method | Signature | Description |
|--------|-----------|-------------|
| `_scan_existing_documents` | `() -> None` | Initial scan via `storage.get_vaults()` |
| `_scan_vault_changes` | `() -> None` | Periodic poll; compares `created_at`/`updated_at` against `_last_scan_time` |
| `_create_context_from_event` | `(event: Dict) -> Optional[RawContextProperties]` | Builds context from vault doc data |

Config keys: `monitor_interval` (default 5), `initial_scan` (default True).

## Internal Data Flow

```
External trigger (timer / API call / push_message)
       |
       v
BaseCaptureComponent.capture()
       |
       v
Subclass._capture_impl() --> List[RawContextProperties]
       |
       v
BaseCaptureComponent invokes self._callback(results)
       |
       v
Processing pipeline (registered via set_callback)
```

For **TextChatCapture**, the flow is different (all async):
```
push_message() --> Redis buffer --> threshold reached --> _flush_buffer()
       --> await _create_and_send_context() --> await self._callback([raw_context])
```

For **FolderMonitorCapture** and **VaultDocumentMonitor**, a separate monitor thread detects changes and queues events in `_document_events`. The base capture loop (or manual `capture()` call) dequeues and processes them.

## Extension Points

To add a new capture component:

1. Create a new file in `opencontext/context_capture/`
2. Subclass `BaseCaptureComponent`:
   ```python
   class MyCapture(BaseCaptureComponent):
       def __init__(self):
           super().__init__(
               name="MyCapture",
               description="...",
               source_type=ContextSource.XXX,  # pick from enums.py
           )

       def _initialize_impl(self, config: Dict[str, Any]) -> bool: ...
       def _start_impl(self) -> bool: ...
       def _stop_impl(self, graceful: bool = True) -> bool: ...
       def _capture_impl(self) -> List[RawContextProperties]: ...
   ```
3. Return `RawContextProperties` from `_capture_impl()` with appropriate `source`, `content_format`, and `content_text`/`content_path`
4. Register in `__init__.py` if it should be publicly importable
5. The component is activated by calling `initialize(config)` then `start()`, and connecting it to the pipeline via `set_callback()`

## Cross-Module Dependencies

**Imports from:**
- `opencontext.interfaces.capture_interface` -- `ICaptureComponent` (base interface)
- `opencontext.models.context` -- `RawContextProperties`
- `opencontext.models.enums` -- `ContextSource`, `ContentFormat`, `ContextType`
- `opencontext.storage.global_storage` -- `get_storage()` (used by `FolderMonitorCapture`, `VaultDocumentMonitor`)
- `opencontext.context_processing.processor.document_processor` -- `DocumentProcessor.get_supported_formats()` (used by `FolderMonitorCapture`)
- `opencontext.storage.redis_cache` -- `RedisCacheConfig`, `get_redis_cache` (used by `TextChatCapture`)
- `opencontext.utils.logger` -- `LogManager.get_logger()` (used by `screenshot.py` instead of the standard `get_logger` from `logging_utils`)
- External: `mss` (screenshot), `crawl4ai` (web markdown), `playwright` (web PDF), `PIL` (image processing)

**Depended on by:**
- `opencontext/server/routes/push.py` -- creates and uses `TextChatCapture` for chat message buffering
- `opencontext/server/opencontext.py` -- may instantiate capture components and wire callbacks

## Conventions and Constraints

- All subclasses must implement the four `_*_impl` abstract methods. Optional `_get_*_impl` methods default to no-op/empty dict.
- `BaseCaptureComponent` manages the capture thread internally. Subclasses that need their own background thread (like `FolderMonitorCapture`, `VaultDocumentMonitor`) should create it in `_start_impl` and stop it in `_stop_impl`, using a separate `threading.Event`.
- The `_callback` is the only bridge to the processing pipeline. Always check `if self._callback` before calling it.
- `TextChatCapture` has async methods (`push_message`, `flush_user_buffer`, `process_messages_directly`, `_create_and_send_context`, etc.).
- `WebLinkCapture` overrides the base `capture()` method to accept a `urls` parameter. This is the only subclass that changes the base method's signature.
- `FolderMonitorCapture._cleanup_file_context()` is async and uses `await` on `UnifiedStorage` calls to delete vector entries for deleted files. It is bridged from sync thread context via `_cleanup_file_context_sync()` using `asyncio.run_coroutine_threadsafe`. This is the only capture component that writes to storage.
- `BaseCaptureComponent.capture()` detects async callbacks via `inspect.isawaitable()` and schedules them on the running event loop when invoked from sync context (e.g., capture loop threads).
