# llm/ -- LLM Client Wrappers and Global Singletons

Provides OpenAI-compatible client wrappers for chat completion, embedding, and vision/tool-calling, exposed as thread-safe global singletons.

## File Overview

| File | Responsibility |
|------|---------------|
| `llm_client.py` | Core `LLMClient` class wrapping OpenAI sync/async APIs for chat and embedding |
| `global_embedding_client.py` | Thread-safe singleton for embedding operations |
| `global_vlm_client.py` | Thread-safe singleton for vision/chat LLM with automatic tool-call execution loop |
| `__init__.py` | Module docstring only (no public re-exports) |

## Key Classes and Functions

### `LLMClient` (`llm_client.py`)

Low-level client wrapping `openai.OpenAI` and `openai.AsyncOpenAI`. Instantiated with an `LLMType` (CHAT or EMBEDDING) that gates which methods are callable.

**Constructor:**
```python
def __init__(self, llm_type: LLMType, config: Dict[str, Any])
```
Fields: `self.model`, `self.api_key`, `self.base_url`, `self.timeout` (default 300), `self.provider` (default `"openai"`), `self.client` (sync), `self.async_client` (async).

**Chat methods** (require `LLMType.CHAT`):

| Method | Signature | Returns |
|--------|-----------|---------|
| `generate` | `(prompt: str, **kwargs)` | Convenience; wraps prompt in messages list, returns ChatCompletion response (same as `generate_with_messages`) |
| `generate_with_messages` | `(messages: List[Dict[str, Any]], **kwargs)` | OpenAI `ChatCompletion` response object |
| `generate_with_messages_async` | `(messages: List[Dict[str, Any]], **kwargs)` | Same, awaitable |
| `generate_with_messages_stream` | `(messages: List[Dict[str, Any]], **kwargs)` | Sync stream iterator |
| `generate_with_messages_stream_async` | `(messages: List[Dict[str, Any]], **kwargs)` | Async generator yielding chunks |

kwargs forwarded: `tools` (adds `tool_choice: "auto"`), `thinking` (Doubao: `reasoning_effort=minimal`; Dashscope: `extra_body={"thinking": {"type": ...}}`).

**Embedding methods** (require `LLMType.EMBEDDING`):

| Method | Signature | Returns |
|--------|-----------|---------|
| `generate_embedding` | `(text: str, **kwargs) -> List[float]` | Embedding vector |
| `generate_embedding_async` | `(text: str, **kwargs) -> List[float]` | Same, awaitable |
| `vectorize` | `(vectorize: Vectorize, **kwargs) -> None` | Sets `vectorize.vector` in-place |
| `vectorize_async` | `(vectorize: Vectorize, **kwargs) -> None` | Same, awaitable |

Both embedding methods apply optional `output_dim` truncation with L2 re-normalization.

**Validation:**
```python
def validate(self) -> tuple[bool, str]
```
Makes a test API call; returns `(success, message)`. Contains `_extract_error_summary()` helper that maps Volcengine/OpenAI error codes to concise messages.

### `LLMProvider` / `LLMType` (enums in `llm_client.py`)

- `LLMProvider`: `OPENAI`, `DOUBAO`
- `LLMType`: `CHAT`, `EMBEDDING`

### `GlobalEmbeddingClient` (`global_embedding_client.py`)

Thread-safe singleton (double-checked locking). Wraps a single `LLMClient(LLMType.EMBEDDING)`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_instance` | `() -> GlobalEmbeddingClient` | Class method; auto-initializes from `get_config("embedding_model")` on first call |
| `is_initialized` | `() -> bool` | Whether inner client is set |
| `reinitialize` | `(new_config: Optional[Dict] = None) -> bool` | Thread-safe hot-reload from config |
| `do_embedding` | `(text: str, **kwargs) -> List[float]` | Delegates to inner client |
| `do_vectorize` | `(vectorize: Vectorize, **kwargs) -> None` | Sync vectorize |
| `do_vectorize_async` | `(vectorize: Vectorize, **kwargs) -> None` | Async vectorize |

**Module-level convenience functions:** `is_initialized()`, `do_embedding()`, `do_vectorize()`, `do_vectorize_async()` -- all delegate to `GlobalEmbeddingClient.get_instance()`.

### `GlobalVLMClient` (`global_vlm_client.py`)

Thread-safe singleton. Wraps a `LLMClient(LLMType.CHAT)` configured from `get_config("vlm_model")`. Also holds a `ToolsExecutor` instance for automatic tool-call resolution.

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_instance` | `() -> GlobalVLMClient` | Auto-initializes on first call |
| `reset` | `() -> None` | Class method; resets singleton (for testing) |
| `reinitialize` | `() -> bool` | Thread-safe hot-reload |
| `generate_with_messages` | `(messages, enable_executor=True, max_calls=5, **kwargs) -> str` | Sync chat with auto tool-call loop |
| `generate_with_messages_async` | `(messages, enable_executor=True, max_calls=5, **kwargs) -> str` | Async variant; parallel tool execution via `asyncio.gather` |
| `generate_for_agent_async` | `(messages, tools=None, **kwargs)` | Returns raw response without auto-executing tool calls |
| `generate_stream_for_agent` | `(messages, tools=None, **kwargs)` | Async generator; yields stream chunks |
| `execute_tool_async` | `(tool_call) -> Any` | Executes a single OpenAI-format tool call |

**Tool-call loop logic:** After each LLM response, if `message.tool_calls` exists and `enable_executor=True`, executes all tool calls (sync: `ThreadPoolExecutor`; async: `asyncio.gather`), appends results as `role: "tool"` messages, and re-calls the LLM. Stops after `max_calls` iterations or when no tool calls are returned.

**Module-level convenience functions:** `is_initialized()`, `generate_with_messages()`, `generate_with_messages_async()`, `generate_for_agent_async()`, `generate_stream_for_agent()`.

**Note:** The module-level `is_initialized()` checks `_auto_initialized` (True after auto-init attempt, even on failure), NOT `_vlm_client is not None`. The instance method `is_initialized()` checks `_vlm_client is not None`. These have different semantics.

**Implementation details:**
- `generate_stream_for_agent` calls the private method `_vlm_client._openai_chat_completion_stream_async()` directly, bypassing the type check in `generate_with_messages_stream_async`.
- `_openai_chat_completion_stream_async` in `LLMClient` creates a local `AsyncOpenAI` client instead of using `self.async_client`.

## Cross-Module Dependencies

**Imports from:**
- `opencontext.config.global_config` -- `get_config()` for auto-initialization
- `opencontext.models.context` -- `Vectorize` model
- `opencontext.monitoring` -- `record_processing_stage`, `record_token_usage`
- `opencontext.tools.tools_executor` -- `ToolsExecutor` (lazy import in `GlobalVLMClient`)
- `opencontext.storage.unified_storage` -- `UnifiedStorage` (imported in `global_vlm_client.py`)
- `opencontext.utils.json_parser` -- `parse_json_from_response`
- `openai` -- `OpenAI`, `AsyncOpenAI`, `APIError`

**Depended on by:**
- `opencontext/context_processing/` -- uses `GlobalEmbeddingClient` for vectorization, `GlobalVLMClient` for text/screenshot analysis
- `opencontext/storage/backends/` -- vector backends call `do_vectorize()` for auto-vectorization when context lacks a vector
- `opencontext/server/` -- health checks call `validate()`; search strategies use embedding client; settings routes reinitialize clients
- `opencontext/periodic_task/hierarchy_summary.py` -- uses LLM for summary generation

## Conventions and Constraints

- Both singletons (`GlobalEmbeddingClient`, `GlobalVLMClient`) use double-checked locking with `threading.Lock`. `LLMClient` is a plain class, not a singleton. Do not bypass `get_instance()` for the singletons.
- `LLMType` gates method access: calling `generate_embedding` on a `CHAT`-type client raises `ValueError`. Do not mix types.
- Token usage is recorded via `record_token_usage` inside API call methods. The import is guarded by `try/except ImportError` for graceful degradation.
- The `provider` field affects `thinking` parameter handling: Doubao uses `reasoning_effort=minimal`, Dashscope uses `extra_body={"thinking": {"type": ...}}`. All other API calls use standard OpenAI format.
- `GlobalVLMClient._auto_initialize()` imports `ToolsExecutor` at call time to avoid circular imports. Do not move this to module level.
