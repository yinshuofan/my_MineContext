# opencontext/tools/ -- Retrieval and Operation Tools

Tool definitions, executor, and implementations for all retrieval and operation tools used by the intelligent search strategy and LLM-driven agentic workflows.

## File Overview

| File | Responsibility |
|------|---------------|
| `base.py` | `BaseTool` ABC -- defines the tool interface (`get_name`, `get_description`, `get_parameters`, `execute`, `get_definition`) |
| `tool_definitions.py` | Tool registry -- assembles tool definition lists (`CONTEXT_RETRIEVAL_TOOLS`, `ALL_PROFILE_TOOL_DEFINITIONS`, `WEB_SEARCH_TOOL_DEFINITION`, `ALL_TOOL_DEFINITIONS`) |
| `tools_executor.py` | `ToolsExecutor` -- name-based tool dispatch, sync/async execution, batch parallel execution |
| `profile_tools/__init__.py` | Exports `ProfileEntityTool` |
| `profile_tools/profile_entity_tool.py` | `ProfileEntityTool` -- entity CRUD and relationship queries against relational DB |
| `retrieval_tools/__init__.py` | Exports `BaseContextRetrievalTool` and the 4 concrete retrieval tool classes (`DocumentRetrievalTool`, `KnowledgeRetrievalTool`, `HierarchicalEventTool`, `ProfileRetrievalTool`) |
| `retrieval_tools/base_context_retrieval_tool.py` | `BaseContextRetrievalTool` -- primary base class for vector DB context retrieval tools; provides `_execute_search`, `_build_filters`, `_format_results` |
| `retrieval_tools/document_retrieval_tool.py` | `DocumentRetrievalTool` -- retrieves `document` contexts from vector DB |
| `retrieval_tools/knowledge_retrieval_tool.py` | `KnowledgeRetrievalTool` -- retrieves `knowledge` + L0 `event` contexts from vector DB, merges and deduplicates |
| `retrieval_tools/hierarchical_event_tool.py` | `HierarchicalEventTool` -- retrieves `event` contexts using L0-L3 hierarchy drill-down + direct L0 fallback |
| `retrieval_tools/profile_retrieval_tool.py` | `ProfileRetrievalTool` -- retrieves user profiles and entities from relational DB |
| `retrieval_tools/document_management_tool.py` | `DocumentManagementTool` -- admin operations (get/delete document by raw_type+raw_id); not registered as an LLM tool |
| `operation_tools/__init__.py` | Exports `WebSearchTool` |
| `operation_tools/web_search_tool.py` | `WebSearchTool` -- DuckDuckGo-based internet search |

## Class Hierarchy

```
BaseTool (ABC)                                      # base.py
├── ProfileEntityTool                               # profile_tools/
├── ProfileRetrievalTool                            # retrieval_tools/
├── HierarchicalEventTool                           # retrieval_tools/
├── WebSearchTool                                   # operation_tools/
└── BaseContextRetrievalTool                        # retrieval_tools/base_context_retrieval_tool.py
    ├── DocumentRetrievalTool                       # retrieval_tools/document_retrieval_tool.py
    └── KnowledgeRetrievalTool                      # retrieval_tools/knowledge_retrieval_tool.py

DocumentManagementTool (standalone, not BaseTool)   # retrieval_tools/document_management_tool.py
```

## Key Classes and Functions

### BaseTool (ABC) -- `base.py`

The interface all tools implement. `get_definition()` is concrete and builds the LLM function-calling schema from the three classmethods. Only `execute()` is `@abstractmethod`; the other three (`get_name`, `get_description`, `get_parameters`) are regular `@classmethod`s that subclasses override but are not enforced by ABC.

```python
class BaseTool(ABC):
    @classmethod
    def get_name(cls) -> str             # Tool name string (used as registry key) -- NOT abstract
    @classmethod
    def get_description(cls) -> str      # Natural language description for LLM -- NOT abstract
    @classmethod
    def get_parameters(cls) -> Dict[str, Any]  # JSON Schema for parameters -- NOT abstract
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]  # Run the tool -- ONLY abstract method
    @classmethod
    def get_definition(cls) -> Dict[str, Any]      # Returns {"name", "description", "parameters"}
```

### ToolsExecutor -- `tools_executor.py`

Central dispatcher. Holds a `_tools_map: Dict[str, BaseTool]` mapping tool names to instances. All 6 registered tools are instantiated in `__init__`.

```python
class ToolsExecutor:
    def __init__(self)                    # Builds _tools_map with all 6 tool instances
    async def run_async(self, tool_name: str, tool_input: Dict) -> Any  # Async: await tool.execute(**kwargs)
    async def batch_run_tools_async(self, tool_calls: List[Dict[str, Any]]) -> Any  # Parallel via asyncio.gather; expects OpenAI tool_call objects (with .function.name, .function.arguments attrs)
    def _validate_input(self, tool_input) -> Tuple[Dict|None, Dict|None]  # Normalize/validate
    def _handle_unknown_tool(self, tool_name: str) -> Dict[str, Any]      # Fuzzy suggestions via difflib
```

**Registered tools in `_tools_map`:**

| Name Key | Class |
|----------|-------|
| `"retrieve_document_context"` | `DocumentRetrievalTool` |
| `"retrieve_knowledge_context"` | `KnowledgeRetrievalTool` |
| `"retrieve_event_context"` | `HierarchicalEventTool` |
| `"retrieve_profile_context"` | `ProfileRetrievalTool` |
| `"profile_entity_tool"` | `ProfileEntityTool` |
| `"web_search"` | `WebSearchTool` |

### BaseContextRetrievalTool -- `retrieval_tools/base_context_retrieval_tool.py`

Primary base class for vector DB retrieval tools. Subclasses only need to set `CONTEXT_TYPE` and override `get_name()`/`get_description()`. The base `execute()` handles the full query-filter-search-format pipeline.

```python
class BaseContextRetrievalTool(BaseTool):
    CONTEXT_TYPE: ContextType = None                  # Subclass must set

    def __init__(self)                                # Creates ProfileEntityTool for entity normalization
    def _build_filters(self, filters: ContextRetrievalFilter) -> Dict[str, Any]
    def _execute_search(self, query: Optional[str], filters: ContextRetrievalFilter, top_k: int = 20) -> List[Tuple[ProcessedContext, float]]
    def _format_context_result(self, context: ProcessedContext, score: float, additional_fields: Dict = None) -> Dict[str, Any]
    def _format_results(self, search_results: List[Tuple[ProcessedContext, float]]) -> List[Dict[str, Any]]
    async def execute(self, **kwargs) -> List[Dict[str, Any]]  # Full pipeline: parse kwargs -> build filters -> search -> format
```

**Dataclasses used by BaseContextRetrievalTool:**

```python
@dataclass
class TimeRangeFilter:
    start: Optional[int] = None
    end: Optional[int] = None
    timezone: Optional[str] = None
    time_type: Optional[str] = "event_time_ts"

@dataclass
class ContextRetrievalFilter:
    time_range: Optional[TimeRangeFilter] = None
    entities: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    agent_id: Optional[str] = None
```

### DocumentRetrievalTool -- `retrieval_tools/document_retrieval_tool.py`

Minimal subclass. Sets `CONTEXT_TYPE = ContextType.DOCUMENT`, overrides name/description/parameters. All logic inherited from `BaseContextRetrievalTool`.

- Tool name: `"retrieve_document_context"`
- Searches: `document` contexts in vector DB

### KnowledgeRetrievalTool -- `retrieval_tools/knowledge_retrieval_tool.py`

Extends `BaseContextRetrievalTool` with a custom `execute()` that searches **both** KNOWLEDGE and L0 EVENT contexts, merges, deduplicates by ID (keeping higher score), and sorts.

- Tool name: `"retrieve_knowledge_context"`
- Searches: `knowledge` + `event` (L0 only) contexts in vector DB

```python
class KnowledgeRetrievalTool(BaseContextRetrievalTool):
    CONTEXT_TYPE = ContextType.KNOWLEDGE

    def _search_l0_events(self, query: str, filters: ContextRetrievalFilter, top_k: int) -> List[Tuple[ProcessedContext, float]]
    async def execute(self, **kwargs) -> List[Dict[str, Any]]  # Custom: knowledge search + L0 event search -> merge -> dedup -> sort
```

### HierarchicalEventTool -- `retrieval_tools/hierarchical_event_tool.py`

Standalone `BaseTool` subclass (does NOT extend `BaseContextRetrievalTool`). Implements the hierarchical drill-down retrieval algorithm for events.

- Tool name: `"retrieve_event_context"`
- Searches: `event` contexts at all hierarchy levels (L0-L3) in vector DB

```python
class HierarchicalEventTool(BaseTool):
    CONTEXT_TYPE = ContextType.EVENT

    # Helpers
    @staticmethod
    def _ts_to_day_bucket(ts: int) -> str
    @staticmethod
    def _ts_to_week_bucket(ts: int) -> str
    @staticmethod
    def _ts_to_month_bucket(ts: int) -> str
    def _search_summaries(self, level: int, time_bucket_start: Optional[str], time_bucket_end: Optional[str], user_id: Optional[str], top_k: int, device_id: Optional[str] = None, agent_id: Optional[str] = None) -> List[Tuple[ProcessedContext, float]]
    def _drill_down_children(self, parent_contexts: List[Tuple[ProcessedContext, float]], user_id: Optional[str] = None) -> List[Tuple[ProcessedContext, float, int]]
    def _direct_l0_search(self, query: str, user_id: Optional[str], device_id: Optional[str], agent_id: Optional[str], filters: Dict, top_k: int) -> List[Tuple[ProcessedContext, float, int]]
    @staticmethod
    def _format_result(context: ProcessedContext, score: float, hierarchy_level: int) -> Dict[str, Any]
    async def execute(self, **kwargs) -> List[Dict[str, Any]]
```

**Retrieval algorithm (execute):**
1. Convert `time_range` timestamps to day/week/month bucket strings
2. Search L1, L2, L3 summaries via `storage.search_hierarchy()`
3. BFS drill-down from matched summaries through `children_ids` to reach L0 events; blended score = `0.5 * child_score(0.3) + 0.5 * parent_score`
4. Direct L0 semantic search as fallback (catches un-aggregated events)
5. Merge both paths, deduplicate by context ID (keep higher score)
6. Sort by score descending, truncate to `top_k`

### ProfileRetrievalTool -- `retrieval_tools/profile_retrieval_tool.py`

Standalone `BaseTool` subclass for relational DB queries. Operation-based dispatch (similar to ProfileEntityTool).

- Tool name: `"retrieve_profile_context"`
- Searches: `profile` and `entity` data in relational DB

```python
class ProfileRetrievalTool(BaseTool):
    async def execute(self, **kwargs) -> Dict[str, Any]                     # Dispatch by "operation" param
    def _get_storage(self)                                                 # Get storage instance; raises RuntimeError if unavailable
    def _handle_get_profile(self, params: Dict[str, Any]) -> Dict[str, Any]     # storage.get_profile()
    def _handle_find_entity(self, params: Dict[str, Any]) -> Dict[str, Any]     # storage.get_entity()
    def _handle_search_entities(self, params: Dict[str, Any]) -> Dict[str, Any] # storage.search_entities()
    def _handle_list_entities(self, params: Dict[str, Any]) -> Dict[str, Any]   # storage.list_entities()
```

**Required params:** `operation` (enum: `get_profile`, `find_entity`, `search_entities`, `list_entities`), `user_id`

### ProfileEntityTool -- `profile_tools/profile_entity_tool.py`

Entity management tool with 4 operations + a `match_entity()` method used by other tools for entity normalization.

- Tool name: `"profile_entity_tool"`
- Searches: `entity` data in relational DB

```python
class ProfileEntityTool(BaseTool):
    def __init__(self, user_id="default", device_id="default", agent_id="default")
    async def execute(self, **kwargs) -> Dict[str, Any]              # Dispatch by "operation" param
    def _handle_find_exact(self, params, user_id, device_id, agent_id) -> Dict
    def _handle_find_similar(self, params, user_id, device_id, agent_id) -> Dict
    def _handle_list_entities(self, params, user_id, device_id, agent_id) -> Dict
    def _handle_check_relationships(self, params, user_id, device_id, agent_id) -> Dict
    def match_entity(self, entity_name: str, entity_type: str = None, top_k: int = 3, user_id=None, device_id=None, agent_id=None) -> Tuple[Optional[str], Optional[Dict]]
```

**Operations:** `find_exact_entity`, `find_similar_entity`, `list_entities`, `check_entity_relationships`

`match_entity()` is called by `BaseContextRetrievalTool._build_filters()` for entity normalization -- exact match first, then fuzzy search fallback.

### WebSearchTool -- `operation_tools/web_search_tool.py`

Internet search tool using DuckDuckGo (`ddgs` library). Not a retrieval tool -- it queries the web, not internal storage.

- Tool name: `"web_search"`

```python
class WebSearchTool(BaseTool):
    def __init__(self)                                          # Reads config from tools.operation_tools.web_search_tool
    async def execute(self, query: str, max_results: int = None, lang: str = "zh-cn", **kwargs) -> Dict[str, Any]
    def _search_duckduckgo(self, query: str, max_results: int, lang: str) -> List[Dict[str, Any]]
    def _get_region(self, lang: str) -> str
```

### DocumentManagementTool -- `retrieval_tools/document_management_tool.py`

Admin-only tool (NOT registered in `ToolsExecutor` or tool definitions). Used internally for document lifecycle management.

```python
class DocumentManagementTool:   # Does not extend BaseTool
    def get_document_by_id(self, raw_type: str, raw_id: str, return_chunks: bool = True) -> Dict[str, Any]
    def delete_document_chunks(self, raw_type: str, raw_id: str) -> Dict[str, Any]  # NOTE: actual deletion is TODO
    def _execute_document_search(self, query: str, context_types: List[str], filters: Dict, top_k: int = 10) -> List[Tuple[ProcessedContext, float]]
    def _aggregate_document_info(self, results: List[Tuple[ProcessedContext, float]]) -> Dict[str, Any]
    def _format_context_result(self, context: ProcessedContext, score: float, additional_fields: Dict[str, Any] = None) -> Dict[str, Any]
```

## Internal Data Flow

### LLM-driven tool execution (intelligent search strategy)

```
LLMContextStrategy (server/search/)
  │
  │  provides ALL_TOOL_DEFINITIONS to LLM as function schemas
  │
  ▼
LLM selects tool(s) + generates arguments
  │
  ▼
ToolsExecutor.batch_run_tools_async(tool_calls)
  │
  ├── Parses function name + JSON arguments from each tool_call
  ├── Calls run_async() for each in parallel (asyncio.gather)
  │     │
  │     ├── Looks up tool name in _tools_map
  │     ├── _validate_input() normalizes input
  │     └── await tool.execute(**kwargs)
  │
  ▼
Results: List[(tool_call_id, function_name, result_dict)]
```

### Vector DB retrieval tool execution (e.g. DocumentRetrievalTool)

```
execute(**kwargs)
  │
  ├── Parse query, entities, time_range, top_k, user_id/device_id/agent_id
  ├── Build ContextRetrievalFilter
  │
  ▼
_execute_search(query, filters, top_k)
  │
  ├── _build_filters(filters)
  │     ├── Time range → {"event_time_ts": {"$gte": ..., "$lte": ...}}
  │     └── Entities → ProfileEntityTool.match_entity() for normalization → {"entities": [...]}
  │
  ├── If query: Vectorize(text=query) → storage.search(query, context_types, filters, top_k, user_id, ...)
  └── If no query: storage.get_all_processed_contexts(context_types, limit, filter, user_id, ...)
  │
  ▼
_format_results(search_results)
  │
  └── For each (ProcessedContext, score):
        └── {"similarity_score", "context" (from get_llm_context_string()), "context_type", "context_description"}
```

### Hierarchical event retrieval

```
execute(**kwargs)
  │
  ├── Convert time_range to bucket strings (day/week/month)
  │
  ├── Path 1: Top-down hierarchy search
  │     ├── _search_summaries(level=1, day_buckets, user_id, top_k, device_id, agent_id)  → L1 hits
  │     ├── _search_summaries(level=2, week_buckets, user_id, top_k, device_id, agent_id) → L2 hits
  │     ├── _search_summaries(level=3, month_buckets, user_id, top_k, device_id, agent_id) → L3 hits
  │     └── _drill_down_children(all_hits)
  │           ├── BFS through children_ids via storage.get_contexts_by_ids()
  │           └── Blended score: 0.5 * child(0.3) + 0.5 * parent_score
  │
  ├── Path 2: Direct L0 fallback
  │     └── _direct_l0_search(query, filters={hierarchy_level: 0}, top_k)
  │           └── Vectorize → storage.search()
  │
  ├── Merge: Dict[context_id → (ctx, score, level)], keep higher score
  ├── Sort by score descending
  └── Truncate to top_k
```

## Tool Definition Registry -- `tool_definitions.py`

The registry groups tools into categories, consumed by `LLMContextStrategy` for function-calling:

```python
# Each entry is {"type": "function", "function": <ToolClass>.get_definition()}
CONTEXT_RETRIEVAL_TOOLS = [
    {"type": "function", "function": DocumentRetrievalTool.get_definition()},
    {"type": "function", "function": KnowledgeRetrievalTool.get_definition()},
    {"type": "function", "function": HierarchicalEventTool.get_definition()},
]
ALL_PROFILE_TOOL_DEFINITIONS = [
    {"type": "function", "function": ProfileRetrievalTool.get_definition()},
    {"type": "function", "function": ProfileEntityTool.get_definition()},
]
WEB_SEARCH_TOOL_DEFINITION = [
    {"type": "function", "function": WebSearchTool.get_definition()},
]
ALL_RETRIEVAL_TOOL_DEFINITIONS = CONTEXT_RETRIEVAL_TOOLS   # Alias
ALL_TOOL_DEFINITIONS = ALL_RETRIEVAL_TOOL_DEFINITIONS + ALL_PROFILE_TOOL_DEFINITIONS + WEB_SEARCH_TOOL_DEFINITION
```

## Cross-Module Dependencies

**Imports from other modules:**
- `opencontext.models.context` -- `ProcessedContext`, `Vectorize`
- `opencontext.models.enums` -- `ContextType`, `ContextSimpleDescriptions`
- `opencontext.storage.global_storage` -- `get_storage()` (all tools use this for storage access)
- `opencontext.config.global_config` -- `get_config()` (WebSearchTool only)
- `opencontext.utils.logging_utils` -- `get_logger()`
- `ddgs` -- DuckDuckGo search library (runtime/lazy import in `WebSearchTool._search_duckduckgo`)

**Depended on by:**
- `opencontext/server/search/intelligent_strategy.py` -- uses `ALL_TOOL_DEFINITIONS` and `ToolsExecutor` for LLM-driven agentic search
- `opencontext/server/search/fast_strategy.py` -- does NOT use tools; queries storage directly
- `opencontext/server/opencontext.py` -- may reference tools for certain operations

## Extension Points

### Adding a new retrieval tool

1. **Create the tool class** in `retrieval_tools/` or a new subdirectory:
   - For vector DB search: extend `BaseContextRetrievalTool`, set `CONTEXT_TYPE`, override `get_name()`, `get_description()`, optionally `get_parameters()` and `execute()`
   - For relational DB search: extend `BaseTool` directly, implement all abstract methods
   - For custom logic (like hierarchical drill-down): extend `BaseTool` directly

2. **Export it** from the relevant `__init__.py` (e.g., `retrieval_tools/__init__.py`)

3. **Register in `tool_definitions.py`**: add to the appropriate list (`CONTEXT_RETRIEVAL_TOOLS`, `ALL_PROFILE_TOOL_DEFINITIONS`, or a new list) and ensure it's included in `ALL_TOOL_DEFINITIONS`

4. **Register in `tools_executor.py`**: add an entry to `_tools_map` in `ToolsExecutor.__init__()`:
   ```python
   NewTool.get_name(): NewTool(),
   ```

5. **Storage access**: use the `storage` property pattern (lazy `get_storage()`) for thread-safety

### Adding a new operation tool

Same as above, but place in `operation_tools/` and add to `WEB_SEARCH_TOOL_DEFINITION` (or create a new category list).

## Conventions and Constraints

- **Storage access**: All tools use `get_storage()` from `opencontext.storage.global_storage`, never `GlobalStorage.get_instance()`. Access is via a `@property` for lazy initialization.
- **3-key identifier**: All profile/entity operations require `(user_id, device_id, agent_id)`. Defaults are `"default"`. Always pass all three to avoid positional argument mismatches.
- **Tool names are stable identifiers**: The string returned by `get_name()` is the key in `ToolsExecutor._tools_map` and appears in LLM function-call responses. Changing a tool name is a breaking change for the intelligent search strategy.
- **Return format**: Retrieval tools return `List[Dict[str, Any]]`. Operation/profile tools return `Dict[str, Any]` with a `success` boolean. Errors are returned inline (not raised), except for fatal errors.
- **Entity normalization**: `BaseContextRetrievalTool._build_filters()` calls `ProfileEntityTool.match_entity()` to normalize entity names before querying. This means entity filters go through exact-match-then-fuzzy-search before reaching the storage layer.
- **Thread safety**: `execute()` is `async` and called via `await` in `run_async()`. Tools must not share mutable state across calls. Storage connections are per-thread (see `_get_connection()` pattern in storage module).
- **DocumentManagementTool is not an LLM tool**: It is not registered in `tool_definitions.py` or `ToolsExecutor`. It is used internally for document admin operations.
- **Entity filtering is not yet effective at the storage layer**: `_build_filters()` normalizes entity names via `ProfileEntityTool.match_entity()`, but both VikingDB and Qdrant backends skip the `entities` filter key because entities are stored as JSON-serialized strings. To enable entity filtering, the storage format must be changed to native lists.
