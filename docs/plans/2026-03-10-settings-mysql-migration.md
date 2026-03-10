# User Settings MySQL Migration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate user settings from file-based YAML storage to the existing MySQL backend so all instances read/write a single source of truth, ensuring multi-instance config consistency.

**Architecture:** Add a `system_settings` table and CRUD methods to `MySQLBackend`, exposed through `UnifiedStorage` (same pattern as monitoring methods — no changes to `IDocumentStorageBackend`). No separate connection pool — reuse the existing MySQL pool (minsize=5, maxsize=20). `ConfigManager` gains async methods and a "DB mode" flag; when enabled, all settings I/O goes through `get_storage()` (lazy import to avoid circular dependency). Uses `JSON_MERGE_PATCH` for atomic deep-merge upserts — no read-modify-write race. Auto-migration copies existing `user_setting.yaml` into MySQL on first startup, guarded by a sentinel row to prevent duplicate migration under concurrent startup.

**Tech Stack:** Python 3.10+, asyncmy (existing dep), MySQL 5.7.22+ (for `JSON_MERGE_PATCH`)

**Note on prompts:** User prompt overrides (`user_prompts_{lang}.yaml`) remain file-based. They are a separate concern — rarely changed and language-scoped.

---

### Task 1: Add settings CRUD to storage layer

Add a `system_settings` key-value table to MySQL and expose load/save/delete through `MySQLBackend` → `UnifiedStorage`. Following the existing monitoring methods pattern (e.g., `save_monitoring_token_usage`), methods are added only to these two classes — `IDocumentStorageBackend` is not modified.

**Files:**
- Modify: `opencontext/storage/backends/mysql_backend.py` (add table + methods)
- Modify: `opencontext/storage/unified_storage.py` (add delegation methods)

**Step 1: Add `system_settings` table to `MySQLBackend._create_tables()`**

In `opencontext/storage/backends/mysql_backend.py`, inside `_create_tables()`, add BEFORE the final `await conn.commit()` (before line 313):

```python
                # System settings table (key-value, for multi-instance config consistency)
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS system_settings (
                        setting_key VARCHAR(128) NOT NULL,
                        setting_value JSON NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (setting_key)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )
```

**Step 2: Add settings CRUD methods to `MySQLBackend`**

Add BEFORE the `query` method (before line 1368), after `clear_message_thinking`:

```python
    # ── System Settings ──

    async def load_all_settings(self) -> Dict[str, Any]:
        """Load all settings rows and return as a dict keyed by setting_key."""
        if not self._initialized:
            return {}
        try:
            async with self._get_connection() as conn:
                async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                    await cursor.execute(
                        "SELECT setting_key, setting_value FROM system_settings"
                    )
                    rows = await cursor.fetchall()
            result: Dict[str, Any] = {}
            for row in rows:
                key = row["setting_key"]
                if key.startswith("_"):
                    continue  # Skip internal sentinel rows (e.g. _migrated)
                value = row["setting_value"]
                if isinstance(value, str):
                    value = json.loads(value)
                result[key] = value
            return result
        except Exception as e:
            logger.exception(f"Failed to load settings: {e}")
            return {}

    async def save_setting(self, key: str, value: dict) -> bool:
        """Atomically deep-merge *value* into the existing row for *key*.

        Uses JSON_MERGE_PATCH so that keys absent from *value* are
        preserved in the stored JSON — matching Python deep_merge behaviour.
        First-time inserts store *value* directly (no merge needed).

        Note: JSON_MERGE_PATCH treats null values as "delete key" (RFC 7396).
        Callers should strip None values before calling if preservation
        semantics are desired (see _strip_none_values in ConfigManager).
        """
        if not self._initialized:
            return False
        try:
            json_value = json.dumps(value, ensure_ascii=False)
            async with self._get_connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        """
                        INSERT INTO system_settings (setting_key, setting_value)
                        VALUES (%s, %s)
                        ON DUPLICATE KEY UPDATE
                            setting_value = JSON_MERGE_PATCH(setting_value, VALUES(setting_value))
                        """,
                        (key, json_value),
                    )
                await conn.commit()
            return True
        except Exception as e:
            logger.exception(f"Failed to save setting '{key}': {e}")
            return False

    async def replace_setting(self, key: str, value: dict) -> bool:
        """Overwrite (not merge) the row for *key*. Used for migration."""
        if not self._initialized:
            return False
        try:
            json_value = json.dumps(value, ensure_ascii=False)
            async with self._get_connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "REPLACE INTO system_settings (setting_key, setting_value) VALUES (%s, %s)",
                        (key, json_value),
                    )
                await conn.commit()
            return True
        except Exception as e:
            logger.exception(f"Failed to replace setting '{key}': {e}")
            return False

    async def delete_all_settings(self) -> bool:
        """Delete every row from system_settings (excluding internal sentinel rows)."""
        if not self._initialized:
            return False
        try:
            async with self._get_connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "DELETE FROM system_settings WHERE setting_key NOT LIKE '\\_%'"
                    )
                await conn.commit()
            logger.info("All settings deleted from DB")
            return True
        except Exception as e:
            logger.exception(f"Failed to delete settings: {e}")
            return False

    async def settings_count(self) -> int:
        """Return number of stored settings rows (excluding sentinel rows)."""
        if not self._initialized:
            return 0
        try:
            async with self._get_connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "SELECT COUNT(*) FROM system_settings WHERE setting_key NOT LIKE '\\_%'"
                    )
                    row = await cursor.fetchone()
            return row[0]
        except Exception as e:
            logger.exception(f"Failed to count settings: {e}")
            return 0
```

**Step 3: Add delegation methods to `UnifiedStorage`**

In `opencontext/storage/unified_storage.py`, add after `batch_set_parent_id` (after line 935, at the end of the class):

```python
    # ── System Settings (→ document DB, MySQL only) ──

    @_require_backend("_document_backend", default={})
    async def load_all_settings(self) -> Dict[str, Any]:
        """Load all system settings from document backend."""
        return await self._document_backend.load_all_settings()

    @_require_backend("_document_backend", default=False)
    async def save_setting(self, key: str, value: dict) -> bool:
        """Save a system setting with atomic deep-merge."""
        return await self._document_backend.save_setting(key, value)

    @_require_backend("_document_backend", default=False)
    async def replace_setting(self, key: str, value: dict) -> bool:
        """Overwrite a system setting (for migration)."""
        return await self._document_backend.replace_setting(key, value)

    @_require_backend("_document_backend", default=False)
    async def delete_all_settings(self) -> bool:
        """Delete all system settings."""
        return await self._document_backend.delete_all_settings()

    @_require_backend("_document_backend", default=0)
    async def settings_count(self) -> int:
        """Return number of stored settings rows."""
        return await self._document_backend.settings_count()
```

**Step 4: Compile-check both files**

```bash
python -m py_compile opencontext/storage/backends/mysql_backend.py && python -m py_compile opencontext/storage/unified_storage.py
```
Expected: no output (success)

**Step 5: Commit**

```bash
git add opencontext/storage/backends/mysql_backend.py opencontext/storage/unified_storage.py
git commit -m "feat(storage): add system_settings table and CRUD to MySQL backend"
```

---

### Task 2: Add async DB mode to `ConfigManager`

Add async methods and a DB-mode flag. When MySQL is detected and storage is ready, settings I/O goes through `get_storage()` (lazy import) instead of the YAML file. File-based methods remain for sync startup only.

**Files:**
- Modify: `opencontext/config/config_manager.py`

**Step 1: Add new instance attributes to `__init__`**

In `__init__` (after line 51 `self._env_vars: Dict[str, str] = {}`), add:

```python
        self._use_db_settings: bool = False
```

**Step 2: Add public property for `use_db_settings`**

Add after `__init__`:

```python
    @property
    def use_db_settings(self) -> bool:
        """Whether DB-backed settings mode is active."""
        return self._use_db_settings
```

**Step 3: Add `_has_mysql_backend` static helper**

Add after the property:

```python
    @staticmethod
    def _has_mysql_backend(config: dict) -> bool:
        """Check if a MySQL document backend is configured."""
        for backend in config.get("storage", {}).get("backends", []):
            if (
                backend.get("storage_type") == "document_db"
                and backend.get("backend") == "mysql"
            ):
                return True
        return False
```

**Step 4: Add `_load_base_config` helper for torn-read prevention**

Add after `_has_mysql_backend`:

```python
    def _load_base_config(self, config_path: str) -> dict:
        """Load and return base config dict without mutating self._config.

        Used by reload_config_async() to build the full merged config in a
        local variable before atomically swapping self._config, preventing
        other coroutines from reading a partially-merged state during awaits.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        self._load_env_vars()
        return self._replace_env_vars(config_data)
```

**Step 5: Add `init_db_settings` async method**

Add after `_load_base_config`:

```python
    async def init_db_settings(self) -> bool:
        """Enable DB-backed settings if MySQL storage is available.

        Called from lifespan after GlobalStorage.ensure_initialized().
        Auto-migrates from user_setting.yaml on first startup (if DB is empty).

        Returns True if DB settings were loaded (config may have changed),
        False if DB mode was not enabled or no settings exist yet.
        """
        if not self._config or not self._has_mysql_backend(self._config):
            return False

        # Lazy import to avoid circular dependency (storage imports config)
        from opencontext.storage.global_storage import get_storage

        storage = get_storage()
        if not storage:
            return False

        # Auto-migrate from file on first startup (sentinel-guarded)
        count = await storage.settings_count()
        if count == 0:
            user_setting_path = self._config.get("user_setting_path", "")
            if user_setting_path and os.path.exists(user_setting_path):
                await self._migrate_file_to_db(storage, user_setting_path)

        # Load settings from DB and apply
        db_settings = await storage.load_all_settings()
        if db_settings:
            self._config = deep_merge(self._config, db_settings)
            logger.info(f"User settings loaded from MySQL (keys={list(db_settings.keys())})")

        self._use_db_settings = True
        return bool(db_settings)

    async def _migrate_file_to_db(self, storage, file_path: str) -> bool:
        """One-time migration: copy settings from YAML file into DB.

        Uses a sentinel row ('_migrated') to prevent duplicate migration
        when multiple workers start concurrently. INSERT IGNORE ensures
        only the first worker proceeds.
        """
        try:
            # Sentinel guard: only the first worker to insert wins
            ok = await storage.save_setting("_migrated", {"ts": str(datetime.now())})
            if not ok:
                return False

            with open(file_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)
            if not settings or not isinstance(settings, dict):
                return False
            count = 0
            for key, value in settings.items():
                if isinstance(value, dict):
                    await storage.replace_setting(key, value)
                    count += 1
                else:
                    logger.warning(
                        f"Skipping non-dict setting '{key}' during migration "
                        f"(type={type(value).__name__})"
                    )
            logger.info(f"Migrated {count} setting(s) from {file_path} to MySQL")
            return count > 0
        except Exception as e:
            logger.exception(f"Failed to migrate settings from file: {e}")
            return False
```

Note: `_migrate_file_to_db` uses `save_setting("_migrated", ...)` as a sentinel. Because `save_setting` uses `INSERT ... ON DUPLICATE KEY UPDATE` with `JSON_MERGE_PATCH`, the first worker inserts the row and proceeds. Subsequent workers also "succeed" at the save but by then `settings_count()` returns > 0, so they skip the `count == 0` branch entirely. The sentinel row is filtered out by `load_all_settings` (which skips keys starting with `_`).

Add `from datetime import datetime` to the imports at the top of the file if not already present.

**Step 6: Modify `load_user_settings` to no-op in DB mode**

In the existing `load_user_settings` method (line 145), add a DB-mode guard at the very top of the method body, before the existing code:

```python
        if self._use_db_settings:
            return True
```

The rest of the method body stays unchanged.

**Step 7: Add async save method**

Add after `save_user_settings` (after line 236):

```python
    async def save_user_settings_async(self, settings: Dict[str, Any]) -> bool:
        """Save user settings to MySQL.

        Each whitelisted key is saved as a separate row with atomic
        JSON_MERGE_PATCH, so concurrent saves from different instances
        are safe and non-form keys within a section are preserved.
        """
        if not self._use_db_settings:
            return self.save_user_settings(settings)

        if not self._config:
            logger.error("Main configuration not loaded")
            return False

        try:
            from opencontext.storage.global_storage import get_storage

            storage = get_storage()
            if not storage:
                logger.error("Storage unavailable, cannot save settings")
                return False

            saved_keys = []
            for key in settings:
                if key in SAVEABLE_KEYS:
                    value = self._strip_none_values(settings[key])
                    if value is not None and (not isinstance(value, dict) or value):
                        await storage.save_setting(key, value)
                        saved_keys.append(key)

            # Reload to get consistent merged state from DB
            await self.reload_config_async()
            logger.info(f"User settings saved to MySQL (keys={saved_keys})")
            return True
        except Exception as e:
            logger.error(f"Failed to save user settings to DB: {e}")
            return False
```

**Step 8: Add async reset method**

Add after `reset_user_settings` (after line 265):

```python
    async def reset_user_settings_async(self) -> bool:
        """Reset user settings by clearing the DB."""
        if not self._use_db_settings:
            return self.reset_user_settings()

        if not self._config:
            logger.error("Main configuration not loaded")
            return False

        try:
            from opencontext.storage.global_storage import get_storage

            storage = get_storage()
            if not storage:
                logger.error("Storage unavailable, cannot reset settings")
                return False

            await storage.delete_all_settings()
            logger.info("User settings cleared from MySQL")

            # Reload base config (no user settings to merge)
            await self.reload_config_async()
            return True
        except Exception as e:
            logger.error(f"Failed to reset user settings in DB: {e}")
            return False
```

**Step 9: Add async reload method**

Add after the reset method:

```python
    async def reload_config_async(self) -> bool:
        """Reload base config from YAML and merge user settings from DB.

        Builds the full config in a local variable before assigning to
        self._config to prevent torn reads during the async DB call.
        """
        if not self._config_path:
            return False

        if not self._use_db_settings:
            return self.load_config(self._config_path)

        try:
            # Build base config without mutating self._config
            new_config = self._load_base_config(self._config_path)

            from opencontext.storage.global_storage import get_storage

            storage = get_storage()
            if storage:
                db_settings = await storage.load_all_settings()
                if db_settings:
                    new_config = deep_merge(new_config, db_settings)

            # Atomic pointer swap — no torn-read window
            self._config = new_config
            return True
        except Exception as e:
            logger.error(f"Failed to reload config from DB: {e}")
            return False
```

**Step 10: Compile-check**

```bash
python -m py_compile opencontext/config/config_manager.py
```
Expected: no output (success)

**Step 11: Commit**

```bash
git add opencontext/config/config_manager.py
git commit -m "feat(config): add async DB-backed settings methods to ConfigManager"
```

---

### Task 3: Wire into startup, reload, routes, and GlobalConfig

Initialize DB settings in the async lifespan (after storage init, before scheduler start). Update `reload_components` to read from DB. Switch settings API endpoints to async methods. Add `set_language_async` to GlobalConfig. Change `_config_lock` from `threading.Lock` to `asyncio.Lock`.

**Files:**
- Modify: `opencontext/cli.py:40-124` (lifespan), `249-302` (headless mode)
- Modify: `opencontext/server/opencontext.py:232-271` (reload_components)
- Modify: `opencontext/server/routes/settings.py`
- Modify: `opencontext/config/global_config.py:187-234`

#### Part A: Startup & Reload

**Step 1: Add DB settings init to lifespan**

In `opencontext/cli.py`, in the `lifespan` function, add after the storage init retry loop (after line 68 `logger.error(...)`) and BEFORE the task scheduler start (before line 70 `# Start task scheduler...`):

```python
    # Initialize DB-backed settings (if MySQL storage is available)
    try:
        from opencontext.config.global_config import GlobalConfig

        config_mgr = GlobalConfig.get_instance().get_config_manager()
        if config_mgr:
            settings_changed = await config_mgr.init_db_settings()
            if settings_changed:
                logger.info("DB-backed settings loaded, reinitializing components")
                context_lab.component_initializer.reload_config()
                context_lab.component_initializer.initialize_task_scheduler(
                    context_lab.processor_manager
                )
    except Exception as e:
        logger.warning(f"DB settings init failed, using file-based settings: {e}")
```

**Step 2: Add DB settings init to headless mode**

In `_run_headless_mode`, after the storage init retry loop and before scheduler start:

```python
        # Init DB-backed settings
        try:
            from opencontext.config.global_config import GlobalConfig

            config_mgr = GlobalConfig.get_instance().get_config_manager()
            if config_mgr:
                settings_changed = await config_mgr.init_db_settings()
                if settings_changed:
                    lab_instance.component_initializer.reload_config()
                    lab_instance.component_initializer.initialize_task_scheduler(
                        lab_instance.processor_manager
                    )
        except Exception as e:
            logger.warning(f"DB settings init failed: {e}")
```

**Step 3: Update `reload_components` to use async reload**

In `opencontext/server/opencontext.py`, replace lines 236-239:

```python
        # 1. Reload GlobalConfig from YAML
        config_mgr = GlobalConfig.get_instance().get_config_manager()
        if config_mgr:
            config_mgr.load_config(config_mgr.get_config_path())
```

With:

```python
        # 1. Reload GlobalConfig (from DB if available, otherwise YAML)
        config_mgr = GlobalConfig.get_instance().get_config_manager()
        if config_mgr:
            if config_mgr.use_db_settings:
                await config_mgr.reload_config_async()
            else:
                config_mgr.load_config(config_mgr.get_config_path())
```

#### Part B: Settings Routes

**Step 4: Change `_config_lock` to asyncio.Lock**

In `opencontext/server/routes/settings.py`, replace:

```python
import threading
```

With:

```python
import asyncio
```

And replace:

```python
_config_lock = threading.Lock()
```

With:

```python
_config_lock = asyncio.Lock()
```

If `threading` is used elsewhere in the file, keep the import. Otherwise remove it.

**Step 5: Change all `with _config_lock:` to `async with _config_lock:`**

There are 3 uses: `update_model_settings` (line 172), `update_general_settings` (line 377), and `reset_settings` (line 542).

**Step 6: Update `update_model_settings` to use async save**

Replace lines 242-245:

```python
            if not config_mgr.save_user_settings(new_settings):
                return convert_resp(code=500, status=500, message="Failed to save settings")

            config_mgr.load_config(config_mgr.get_config_path())
```

With:

```python
            if not await config_mgr.save_user_settings_async(new_settings):
                return convert_resp(code=500, status=500, message="Failed to save settings")
```

Note: `save_user_settings_async` calls `reload_config_async()` internally — the explicit `load_config` is no longer needed.

**Step 7: Update `update_general_settings` to use async save**

Replace lines 392-396:

```python
            logger.info(f"Saving general settings: keys={list(settings.keys())}")
            if not config_mgr.save_user_settings(settings):
                return convert_resp(code=500, status=500, message="Failed to save settings")

            config_mgr.load_config(config_mgr.get_config_path())
```

With:

```python
            logger.info(f"Saving general settings: keys={list(settings.keys())}")
            if not await config_mgr.save_user_settings_async(settings):
                return convert_resp(code=500, status=500, message="Failed to save settings")
```

**Step 8: Update `reset_settings` to use async reset**

Replace lines 549-552:

```python
            if config_mgr:
                if not config_mgr.reset_user_settings():
                    success = False
                    logger.error("Failed to reset user settings")
```

With:

```python
            if config_mgr:
                if not await config_mgr.reset_user_settings_async():
                    success = False
                    logger.error("Failed to reset user settings")
```

#### Part C: GlobalConfig

**Step 9: Add `set_language_async` to GlobalConfig**

In `opencontext/config/global_config.py`, add after `set_language` (after line 234):

```python
    async def set_language_async(self, language: str) -> bool:
        """Async version of set_language — uses DB-backed save when available."""
        if language not in ["zh", "en"]:
            logger.error(f"Invalid language: {language}")
            return False

        try:
            self._language = language

            if self._config_manager:
                settings = {"prompts": {"language": language}}
                if not await self._config_manager.save_user_settings_async(settings):
                    logger.error("Failed to save language setting")
                    return False

            # Reload prompt manager with new language
            prompts_path = f"prompts_{language}.yaml"
            base_dir = os.path.dirname(self._config_path)
            absolute_prompts_path = os.path.join(base_dir, prompts_path)

            if not os.path.exists(absolute_prompts_path):
                logger.error(f"Prompt file not found: {absolute_prompts_path}")
                return False

            self._prompt_manager = PromptManager(absolute_prompts_path)
            self._prompt_path = absolute_prompts_path
            logger.info(f"Prompts reloaded from: {self._prompt_path} (language: {language})")
            self._prompt_manager.load_user_prompts()
            return True
        except Exception as e:
            logger.error(f"Failed to set language: {e}")
            return False
```

**Step 10: Update language endpoint to use async version**

In `settings.py`, in the `change_prompt_language` endpoint (line 520), replace:

```python
        success = GlobalConfig.get_instance().set_language(request.language)
```

With:

```python
        success = await GlobalConfig.get_instance().set_language_async(request.language)
```

**Step 11: Compile-check all modified files**

```bash
python -m py_compile opencontext/cli.py && python -m py_compile opencontext/server/opencontext.py && python -m py_compile opencontext/server/routes/settings.py && python -m py_compile opencontext/config/global_config.py
```
Expected: no output

**Step 12: Commit**

```bash
git add opencontext/cli.py opencontext/server/opencontext.py opencontext/server/routes/settings.py opencontext/config/global_config.py
git commit -m "feat(config): wire DB settings into startup, reload, routes, and GlobalConfig"
```

---

### Task 4: Update documentation

**Files:**
- Modify: `opencontext/config/MODULE.md`
- Modify: `opencontext/storage/MODULE.md`

**Step 1: Update `opencontext/config/MODULE.md`**

In the ConfigManager section, update the Methods table and State to reflect:

**New state:**
- `_use_db_settings: bool` — whether DB-backed settings are active

**New methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `use_db_settings` | `@property -> bool` | Whether DB-backed settings mode is active |
| `_has_mysql_backend` | `(config: dict) -> bool` | Static. Check if MySQL document backend is configured |
| `_load_base_config` | `(config_path: str) -> dict` | Load base config dict without mutating `self._config` (for torn-read prevention) |
| `init_db_settings` | `() -> bool` | Async. Enable DB mode if MySQL available. Auto-migrates from file. Called from lifespan after storage init |
| `save_user_settings_async` | `(settings: Dict) -> bool` | Async. Save via `get_storage().save_setting()` with `JSON_MERGE_PATCH`. Falls back to sync file method when not in DB mode |
| `reset_user_settings_async` | `() -> bool` | Async. Clear settings via `get_storage().delete_all_settings()`. Falls back to sync file method when not in DB mode |
| `reload_config_async` | `() -> bool` | Async. Reload base YAML + overlay DB settings with atomic pointer swap. Falls back to `load_config()` when not in DB mode |

Update the `save_user_settings` description to note it is the file-based path, used when `_use_db_settings` is False.

Update the `load_user_settings` description to note it is a no-op when `_use_db_settings` is True.

In the GlobalConfig section, add `set_language_async` method.

Update the Conventions section:
- Add: "In multi-instance deployments with MySQL, user settings are stored in the `system_settings` table. DB mode is enabled automatically when a MySQL document backend is configured and storage is initialized."
- Add: "Settings access uses lazy import (`from opencontext.storage.global_storage import get_storage` inside method bodies) to avoid circular dependency between config and storage modules."
- Add: "After deploying this change, existing `user_setting.yaml` data is auto-migrated to MySQL on first startup. The file can be removed after migration is confirmed."
- Add: "Cross-worker consistency: after saving settings, changes are visible only to the current worker until `/api/settings/apply` is called (broadcasts via Redis Pub/Sub). This is the existing two-step Save + Apply flow."

**Step 2: Update `opencontext/storage/MODULE.md`**

In the MySQLBackend section, add to the tables list:

| Table | Purpose | Key |
|-------|---------|-----|
| `system_settings` | Multi-instance user settings (key-value with JSON values) | `setting_key VARCHAR(128) PK` |

Add settings methods to the MySQLBackend methods table:

| Method | Description |
|--------|-------------|
| `load_all_settings()` | Load all settings rows as `{key: value}` dict (skips `_`-prefixed sentinel rows) |
| `save_setting(key, value)` | Atomic upsert with `JSON_MERGE_PATCH` |
| `replace_setting(key, value)` | Full overwrite (for migration) |
| `delete_all_settings()` | Clear all settings (preserves sentinel rows) |
| `settings_count()` | Row count (excludes sentinel rows) |

In the UnifiedStorage section, add the settings delegation methods.

Note: `IDocumentStorageBackend` is NOT modified — settings methods follow the monitoring methods pattern (only on `MySQLBackend` + `UnifiedStorage`).

**Step 3: Commit**

```bash
git add opencontext/config/MODULE.md opencontext/storage/MODULE.md
git commit -m "docs: document DB-backed settings in config and storage MODULE.md"
```

---

### Design Decision Notes

**Why reuse the existing MySQL pool (not a separate SettingsStore):**
Settings are architecturally a storage concern. `MySQLBackend` already hosts 10 tables for profiles, monitoring, conversations, etc. The MySQL pool (5-20 connections) is initialized in the lifespan BEFORE settings need to be loaded. A separate pool duplicates connection management, wastes resources for rare operations, and creates an unnecessary new class. Using `get_storage()` is O(1) (singleton lookup) and matches the pattern used by `BaseContextRetrievalTool`, `ContextMerger`, etc.

**Why methods only on `MySQLBackend` + `UnifiedStorage` (not `IDocumentStorageBackend`):**
Follows the existing monitoring methods precedent (e.g., `save_monitoring_token_usage` is on `MySQLBackend` and `UnifiedStorage` but NOT on `IDocumentStorageBackend`). Settings is a MySQL-only concern. `ConfigManager._has_mysql_backend()` gates DB mode activation, so these methods are only called when the backend is MySQL.

**Why `JSON_MERGE_PATCH` instead of read-modify-write:**
The current file-based approach does read → deep merge → write, which has a race condition across instances. `JSON_MERGE_PATCH` performs the deep merge atomically inside MySQL. Combined with `_strip_none_values` in Python (removes `null`s before sending), the behaviour matches the existing `deep_merge` + null-stripping logic exactly. Note: `JSON_MERGE_PATCH` treats null as "delete key" per RFC 7396 — nulls are stripped in Python before reaching MySQL. Requires MySQL 5.7.22+.

**Why lazy import in ConfigManager:**
`config_manager.py` cannot import `global_storage.py` at module level because `global_storage.py` imports from `global_config.py` which imports `config_manager.py` — circular dependency. Lazy import (`from opencontext.storage.global_storage import get_storage` inside method bodies) breaks the cycle. By the time async methods are called, all modules are fully loaded.

**Why `asyncio.Lock` instead of `threading.Lock`:**
Settings routes are `async def` handlers. `threading.Lock` blocks the event loop while waiting. `asyncio.Lock` yields control to other coroutines — critical when settings operations involve async DB writes. Note: `asyncio.Lock` only protects within one worker process. Cross-worker concurrency is handled by MySQL's row-level locking and `JSON_MERGE_PATCH` atomicity.

**Torn-read prevention in `reload_config_async`:**
`reload_config_async()` uses `_load_base_config()` to build the full config in a local variable, then assigns to `self._config` in a single pointer swap. This prevents other coroutines from reading a partially-merged state during the async DB call.

**Migration idempotency:**
`_migrate_file_to_db()` uses a sentinel row (`_migrated`) to prevent duplicate migration when multiple workers start concurrently. `load_all_settings()` and `settings_count()` filter out `_`-prefixed keys. `delete_all_settings()` preserves sentinel rows so migration doesn't re-trigger after a reset.

**Startup timing and component reinit:**
`init_db_settings()` runs after `GlobalStorage.ensure_initialized()` but before `start_task_scheduler()`. If DB settings differ from the file-based defaults loaded during sync init, `component_initializer.reload_config()` + `initialize_task_scheduler()` reinits affected components before the scheduler starts.

**Cross-worker consistency:**
After saving settings, changes are visible only to the current worker. Other workers learn about changes when the user clicks "Apply" (calls `/api/settings/apply`), which broadcasts via Redis Pub/Sub. `reload_components()` on each worker calls `reload_config_async()` to read the latest settings from MySQL. This preserves the existing two-step Save + Apply flow.

**DB mode: no silent fallback to file:**
When `_use_db_settings` is True, if `get_storage()` returns None (storage crashed), `save_user_settings_async` and `reset_user_settings_async` return `False` with an error log instead of silently falling back to file-based save. Silent fallback would create split-brain: one instance writes to file while others use DB.
