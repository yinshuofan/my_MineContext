# opencontext/config/ -- Configuration loading, environment variable substitution, and prompt management.

## File Overview

| File | Responsibility |
|------|---------------|
| `config_manager.py` | Loads `config.yaml`, substitutes `${ENV_VAR:default}` patterns, merges user settings |
| `prompt_manager.py` | Loads prompt YAML files, supports dot-path lookup, user prompt overrides, import/export |
| `global_config.py` | Thread-safe singleton (`GlobalConfig`) that wraps `ConfigManager` + `PromptManager`, plus convenience functions |
| `__init__.py` | Re-exports: `ConfigManager`, `PromptManager`, `GlobalConfig`, `get_global_config`, `get_config`, `get_prompt` |

## Key Classes

### ConfigManager (config_manager.py)

Loads YAML config with environment variable substitution.

**State:**
- `_config: Optional[Dict[str, Any]]` -- parsed config dict
- `_config_path: Optional[str]` -- path to loaded config file
- `_env_vars: Dict[str, str]` -- captured environment variables

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `load_config` | `(config_path: Optional[str] = None) -> bool` | Load YAML, substitute env vars, merge user settings. Default path: `config/config.yaml`. Raises `FileNotFoundError` if path missing. |
| `get_config` | `() -> Optional[Dict[str, Any]]` | Return full config dict |
| `get_config_path` | `() -> Optional[str]` | Return config file path |
| `deep_merge` | `(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]` | Recursive dict merge (override wins) |
| `load_user_settings` | `() -> bool` | Load `user_setting_path` from config, merge into `_config` |
| `save_user_settings` | `(settings: Dict[str, Any]) -> bool` | Save specific keys (vlm_model, embedding_model, capture, processing, logging, prompts, content_generation) to user settings file and merge |
| `reset_user_settings` | `() -> bool` | Delete user settings file, reload config |

**Environment variable substitution** (`_replace_env_vars`):
- Pattern: `${VAR_NAME}` or `${VAR_NAME:default_value}`
- Regex: `\$\{([^}:]+)(?::([^}]*))?\}`
- Auto-converts `"true"`/`"false"` strings to Python booleans after substitution

### PromptManager (prompt_manager.py)

Loads and manages LLM prompt templates from YAML.

**State:**
- `prompts: dict` -- parsed prompt dict (base merged with user overrides)
- `prompt_config_path: str` -- path to base prompts file

**Constructor:** `__init__(self, prompt_config_path: str = None)` -- loads YAML, raises `FileNotFoundError` if file missing.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_prompt` | `(name: str, default: str = None) -> str` | Dot-path lookup (e.g. `"extraction.system_prompt"`). Returns `default` if not found or not a string. |
| `get_prompt_group` | `(name: str) -> Dict[str, str]` | Dot-path lookup returning a dict subtree. Returns `{}` if not found. |
| `get_context_type_descriptions` | `() -> str` | Delegates to `enums.get_context_type_descriptions_for_prompts()` |
| `get_context_type_descriptions_for_retrieval` | `() -> str` | Delegates to `enums.get_context_type_descriptions_for_retrieval()` |
| `get_user_prompts_path` | `() -> str \| None` | Derives `user_prompts_{lang}.yaml` path from base prompt filename |
| `load_user_prompts` | `() -> bool` | Deep-merge user prompt overrides into `self.prompts` |
| `save_prompts` | `(prompts_data: dict) -> bool` | Save to user prompts file with literal YAML style for multi-line strings |
| `export_prompts` | `() -> str` | Export current prompts as YAML string |
| `import_prompts` | `(yaml_content: str) -> bool` | Parse YAML string, save via `save_prompts` |
| `reset_user_prompts` | `() -> bool` | Delete user prompts file, reload base prompts |

### GlobalConfig (global_config.py)

Thread-safe singleton wrapping both managers. Uses double-checked locking (`threading.Lock`).

**State:**
- `_config_manager: Optional[ConfigManager]`
- `_prompt_manager: Optional[PromptManager]`
- `_config_path: Optional[str]`
- `_prompt_path: Optional[str]`
- `_auto_initialized: bool`

Note: `_language` is NOT set in `__init__`. It is set later in `_init_prompt_manager()` and `set_language()`. `get_language()` falls back to `"zh"` via `hasattr` check if the attribute doesn't exist yet.

**Key methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_instance` | `(cls) -> GlobalConfig` | Get singleton; auto-initializes on first access |
| `reset` | `(cls) -> None` | Reset singleton (for testing) |
| `initialize` | `(config_path: Optional[str] = None) -> bool` | Init both managers; called once at startup |
| `get_config` | `(path: Optional[str] = None) -> Optional[Dict[str, Any]]` | Dot-path config lookup (e.g. `"storage.vector_db"`) |
| `get_prompt` | `(name: str, default: Optional[str] = None) -> Optional[str]` | Delegates to `PromptManager.get_prompt()` |
| `get_prompt_group` | `(name: str) -> Dict[str, str]` | Delegates to `PromptManager.get_prompt_group()` |
| `get_language` | `() -> str` | Current language (`"zh"` or `"en"`) |
| `set_language` | `(language: str) -> bool` | Change language, save to user settings, reload prompts |
| `is_enabled` | `(module: str) -> bool` | Check `config[module]["enabled"]` |
| `is_initialized` | `() -> bool` | Whether config has been initialized |
| `set_config_manager` | `(config_manager: ConfigManager) -> None` | Manual setter (backward compat) |
| `set_prompt_manager` | `(prompt_manager: PromptManager) -> None` | Manual setter (backward compat) |
| `get_config_manager` | `() -> Optional[ConfigManager]` | Get inner ConfigManager |
| `get_prompt_manager` | `() -> Optional[PromptManager]` | Get inner PromptManager |

## Convenience Functions (global_config.py)

These are module-level shortcuts that delegate to `GlobalConfig.get_instance()`:

```python
get_global_config() -> GlobalConfig
get_config(path: Optional[str] = None) -> Optional[Dict[str, Any]]
get_language() -> str
get_prompt(name: str, default: Optional[str] = None) -> Optional[str]
get_prompt_group(name: str) -> Dict[str, str]
get_prompt_manager() -> PromptManager
is_initialized() -> bool
```

## Cross-Module Dependencies

**Imports from:**
- `pyyaml` (yaml.safe_load, yaml.dump)
- `python-dotenv` (load_dotenv) -- in ConfigManager
- `opencontext.utils.logging_utils` (get_logger) -- used by `config_manager.py` and `global_config.py`
- `loguru` (logger) -- used directly by `prompt_manager.py` (does not use `get_logger`)
- `opencontext.models.enums` -- PromptManager calls `get_context_type_descriptions_for_prompts()` and `get_context_type_descriptions_for_retrieval()`

**Depended on by (most modules):**
- `opencontext/server/` -- `cli.py` calls `GlobalConfig.initialize()` at startup
- `opencontext/llm/` -- reads model config via `get_config()`
- `opencontext/storage/` -- reads storage backend config
- `opencontext/context_processing/` -- reads processing config, gets prompts
- `opencontext/tools/` -- reads tool-related config
- `opencontext/scheduler/`, `opencontext/periodic_task/` -- reads task config

## Conventions and Constraints

1. **Always use `get_config()` or `get_global_config()`** to access config from application code. Do not instantiate `ConfigManager` directly outside of `GlobalConfig`.
2. **Dot-path syntax** for nested config: `get_config("storage.vector_db.type")` traverses the dict hierarchy. Returns `None` if any key is missing.
3. **Prompt dot-path syntax** works the same: `get_prompt("extraction.system_prompt")` navigates the prompt YAML tree.
4. **Env var substitution happens at load time**: `${VAR:default}` is resolved once during `load_config()`. Changing env vars after load has no effect.
5. **Boolean coercion**: After env var substitution, strings `"true"` and `"false"` (case-insensitive) are converted to Python `bool`. This means `${SOME_FLAG:false}` yields `False`, not the string `"false"`.
6. **User settings are overlay files**: `user_setting.yaml` (config) and `user_prompts_{lang}.yaml` (prompts) are deep-merged on top of base files. Only specific keys are saved by `save_user_settings`.
7. **Language must be `"zh"` or `"en"`**: `set_language()` rejects other values. Changing language reloads the prompt manager with the corresponding `prompts_{lang}.yaml` file.
8. **Prompt files must have matching keys**: `prompts_en.yaml` and `prompts_zh.yaml` must define the same prompt keys. Missing keys in one file cause `get_prompt()` to return `default`.
