# Remove user_setting.yaml File Layer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the file-based `user_setting.yaml` layer. Settings are read from `config.yaml` (defaults) + DB `system_settings` table (user overrides). Support both MySQL and SQLite as document backends for settings storage.

**Architecture:** Two-layer config model — `config.yaml` provides defaults, the document DB backend (`system_settings` table) provides user overrides via `init_db_settings()` at startup. The file-based middle layer (`user_setting.yaml`) is removed entirely. Both MySQL and SQLite backends implement the settings CRUD methods. Migration code (`_migrate_file_to_db`) is kept temporarily for transition.

**Tech Stack:** Python, YAML, MySQL (JSON_MERGE_PATCH), SQLite (Python-side deep_merge), aiosqlite

---

### Task 1: Add settings support to SQLite backend

**Files:**
- Modify: `opencontext/storage/backends/sqlite_backend.py`

SQLite backend currently has no `system_settings` table and no settings CRUD methods. Add them following the MySQL backend pattern, with one key difference: SQLite uses Python-side `deep_merge` for atomic merging (no `JSON_MERGE_PATCH` in SQLite < 3.45), which is safe because SQLite deployments are single-instance and WAL mode serializes writes.

**Step 1: Add `system_settings` table to `_create_tables()`**

In `_create_tables()`, add the table creation **before** `await conn.commit()` (line 323). Insert right before the commit:

```python
        # System settings table (key-value, for user config overrides)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_settings (
                setting_key TEXT NOT NULL PRIMARY KEY,
                setting_value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
```

**Step 2: Add `deep_merge` import**

Add to the imports at the top of the file (after the existing imports, around line 25):

```python
from opencontext.utils.dict_utils import deep_merge
```

**Step 3: Add 5 settings CRUD methods**

Add these methods before the `query()` method (before line 1939). Place them after the `clear_message_thinking()` method (after line 1937), with a section comment:

```python
    # ── System Settings ──

    async def load_all_settings(self) -> Dict[str, Any]:
        """Load all settings rows and return as a dict keyed by setting_key."""
        if not self._initialized:
            return {}
        conn = self._connection
        try:
            cursor = await conn.execute(
                "SELECT setting_key, setting_value FROM system_settings"
                " WHERE setting_key NOT LIKE '\\_%'"
            )
            rows = await cursor.fetchall()
            result: Dict[str, Any] = {}
            for row in rows:
                value = row["setting_value"]
                if isinstance(value, str):
                    value = json.loads(value)
                result[row["setting_key"]] = value
            return result
        except Exception as e:
            logger.exception(f"Failed to load settings: {e}")
            return {}

    async def save_setting(self, key: str, value: dict) -> bool:
        """Save a setting with deep-merge semantics.

        Fetches the existing value, merges in Python via deep_merge(),
        and writes back. This is safe for SQLite because WAL mode
        serializes writes and this is a single-instance deployment.
        """
        if not self._initialized:
            return False
        conn = self._connection
        try:
            # Fetch existing value for merge
            cursor = await conn.execute(
                "SELECT setting_value FROM system_settings WHERE setting_key = ?",
                (key,),
            )
            row = await cursor.fetchone()

            if row:
                existing = json.loads(row["setting_value"]) if isinstance(row["setting_value"], str) else row["setting_value"]
                merged = deep_merge(existing, value)
                json_value = json.dumps(merged, ensure_ascii=False)
                await conn.execute(
                    "UPDATE system_settings SET setting_value = ?, updated_at = CURRENT_TIMESTAMP WHERE setting_key = ?",
                    (json_value, key),
                )
            else:
                json_value = json.dumps(value, ensure_ascii=False)
                await conn.execute(
                    "INSERT INTO system_settings (setting_key, setting_value) VALUES (?, ?)",
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
        conn = self._connection
        try:
            json_value = json.dumps(value, ensure_ascii=False)
            await conn.execute(
                "INSERT OR REPLACE INTO system_settings (setting_key, setting_value) VALUES (?, ?)",
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
        conn = self._connection
        try:
            await conn.execute(
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
        conn = self._connection
        try:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM system_settings WHERE setting_key NOT LIKE '\\_%'"
            )
            row = await cursor.fetchone()
            return row[0]
        except Exception as e:
            logger.exception(f"Failed to count settings: {e}")
            return 0
```

**Step 4: Verify compile**

Run: `uv run python -m py_compile opencontext/storage/backends/sqlite_backend.py`
Expected: No output (success)

**Step 5: Commit**

```bash
git add opencontext/storage/backends/sqlite_backend.py
git commit -m "feat(storage): add settings CRUD methods to SQLite backend

Add system_settings table and 5 CRUD methods (load_all_settings,
save_setting, replace_setting, delete_all_settings, settings_count)
matching the MySQL backend interface. Uses Python-side deep_merge
for save_setting instead of JSON_MERGE_PATCH."
```

---

### Task 2: Generalize ConfigManager to support any document backend

**Files:**
- Modify: `opencontext/config/config_manager.py`

Change `_has_mysql_backend()` to `_has_document_backend()` so that `init_db_settings()` activates for both MySQL and SQLite. Then remove file-based sync methods and update async fallbacks.

**Step 1: Rename `_has_mysql_backend()` to `_has_document_backend()`**

```python
# BEFORE (lines 60-69):
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

# AFTER:
    @staticmethod
    def _has_document_backend(config: dict) -> bool:
        """Check if a document backend (MySQL or SQLite) is configured."""
        for backend in config.get("storage", {}).get("backends", []):
            if backend.get("storage_type") == "document_db":
                return True
        return False
```

**Step 2: Update `init_db_settings()` to call the renamed method**

```python
# BEFORE (line 92):
        if not self._config or not self._has_mysql_backend(self._config):

# AFTER:
        if not self._config or not self._has_document_backend(self._config):
```

**Step 3: Remove `load_user_settings()` call from `load_config()`**

Remove lines 175–179:

```python
# REMOVE:
        if not self.load_user_settings():
            logger.warning(
                f"User settings not loaded (path={self._config.get('user_setting_path', 'N/A')}), "
                "using main config defaults"
            )
```

After removal, `load_config()` ends with:

```python
        self._config = config_data
        self._config_path = found_config_path
        logger.info(f"Configuration loaded successfully: {self._config_path}")

        return True
```

**Step 4: Delete the `load_user_settings()` method**

Delete the entire method (lines 246–275).

**Step 5: Delete the `save_user_settings()` method**

Delete the entire sync method (lines 284–339).

**Step 6: Delete the `reset_user_settings()` method**

Delete the entire sync method (lines 379–406).

**Step 7: Update `save_user_settings_async()` fallback**

```python
# BEFORE:
        if not self._use_db_settings:
            return self.save_user_settings(settings)

# AFTER:
        if not self._use_db_settings:
            logger.error("Cannot save settings: DB-backed settings not enabled")
            return False
```

**Step 8: Update `reset_user_settings_async()` fallback**

```python
# BEFORE:
        if not self._use_db_settings:
            return self.reset_user_settings()

# AFTER:
        if not self._use_db_settings:
            logger.error("Cannot reset settings: DB-backed settings not enabled")
            return False
```

**Step 9: Update `reload_config_async()` fallback comment**

```python
# BEFORE:
        if not self._use_db_settings:
            return self.load_config(self._config_path)

# AFTER:
        if not self._use_db_settings:
            # Non-DB mode: reload base config only (no user overrides)
            return self.load_config(self._config_path)
```

**Step 10: Verify compile**

Run: `uv run python -m py_compile opencontext/config/config_manager.py`
Expected: No output (success)

**Step 11: Commit**

```bash
git add opencontext/config/config_manager.py
git commit -m "refactor(config): remove file-based settings, support any document backend

- Rename _has_mysql_backend() to _has_document_backend() so
  init_db_settings() activates for both MySQL and SQLite.
- Remove sync methods (load/save/reset_user_settings) and the
  load_user_settings() call from load_config().
- Async method fallbacks log error instead of delegating to
  deleted sync methods. Migration code kept for transition."
```

---

### Task 3: Remove sync `set_language()` from GlobalConfig

**Files:**
- Modify: `opencontext/config/global_config.py`

The sync `set_language()` method (lines 187–234) calls the now-deleted `save_user_settings()`. It has **zero callers** — the API route uses `set_language_async()` exclusively. Remove it.

**Step 1: Delete the sync `set_language()` method**

Delete lines 187–234 — the entire `set_language()` method. Keep `set_language_async()` (lines 236–267) which is the active code path.

**Step 2: Verify compile**

Run: `uv run python -m py_compile opencontext/config/global_config.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add opencontext/config/global_config.py
git commit -m "refactor(config): remove sync set_language() — only async path remains

The sync set_language() called the now-deleted save_user_settings()
and had zero callers (API route uses set_language_async())."
```

---

### Task 4: Update documentation

**Files:**
- Modify: `opencontext/config/MODULE.md`
- Modify: `opencontext/storage/MODULE.md`
- Modify: `CLAUDE.md`

**Step 1: Update config/MODULE.md — ConfigManager methods table**

- Remove rows: `load_user_settings`, `save_user_settings`, `reset_user_settings`
- Rename `_has_mysql_backend` → `_has_document_backend` with updated description: "Check if any document backend (MySQL or SQLite) is configured"
- Update `save_user_settings_async` description: "Returns False with error log when DB mode is not active" (remove file fallback mention)
- Update `reset_user_settings_async` description: same change
- Update `_use_db_settings` state description: "True after `init_db_settings()` enables DB-backed settings (MySQL or SQLite)"

**Step 2: Update config/MODULE.md — GlobalConfig methods table**

Remove `set_language` row. Keep only `set_language_async`.

**Step 3: Update config/MODULE.md — Conventions section**

Update convention #6:

```markdown
6. **User settings are DB-backed**: User overrides are stored in the document backend's `system_settings` table (MySQL or SQLite) and merged on top of `config.yaml` at startup via `init_db_settings()`. Only keys in `SAVEABLE_KEYS` are persisted. Prompt overrides (`user_prompts_{lang}.yaml`) remain file-based.
```

**Step 4: Update storage/MODULE.md**

Add SQLite settings methods to the SQLite backend section (matching the existing MySQL settings docs). Note the difference: SQLite uses Python-side `deep_merge` instead of `JSON_MERGE_PATCH`.

**Step 5: Update CLAUDE.md — Configuration section**

Add a bullet:

```markdown
- User settings: stored in `system_settings` table (MySQL or SQLite document backend), not in `user_setting.yaml`
```

**Step 6: Commit**

```bash
git add opencontext/config/MODULE.md opencontext/storage/MODULE.md CLAUDE.md
git commit -m "docs: update docs for DB-backed settings with SQLite support"
```

---

## Scope Boundaries

**Kept (not touched):**
- `_migrate_file_to_db()` and migration logic in `init_db_settings()` — transitional, will be removed once all deployments have migrated
- `user_setting_path` key in `config.yaml` / `config-docker.yaml` — still referenced by migration code
- `SAVEABLE_KEYS` — still used by `save_user_settings_async()`
- `_strip_none_values()` — still used by `save_user_settings_async()`
- `_load_base_config()` — still used by `reload_config_async()`
- Prompt system (`user_prompts_{lang}.yaml`) — independent, remains file-based
- `UnifiedStorage` delegation methods — already generic, no changes needed
- `base_storage.py` (`IDocumentStorageBackend`) — not modified, settings methods are duck-typed (matching monitoring methods pattern)

**Removed:**
- `load_user_settings()` — sync file reader
- `save_user_settings()` — sync file writer
- `reset_user_settings()` — sync file deleter
- `set_language()` — sync method that called `save_user_settings()`
- `load_user_settings()` call from `load_config()`
- Sync fallback paths in async methods (replaced with error log)
- `_has_mysql_backend()` — replaced by `_has_document_backend()`

## Design Decisions

**Why Python-side `deep_merge` for SQLite instead of `JSON_MERGE_PATCH`?**
- `JSON_MERGE_PATCH()` requires SQLite 3.45.0+ (2023-12), not universally available
- SQLite is single-instance; WAL mode serializes writes, so read-merge-write is safe
- Matches the existing SQLite backend pattern (uses `json.dumps/loads`, no native JSON functions)
- `deep_merge` utility already exists in the codebase (`opencontext/utils/dict_utils.py`)

**Why `_has_document_backend()` checks `storage_type` only (not backend name)?**
- Any document backend should support settings — the methods are implemented on both MySQL and SQLite
- Avoids hardcoding backend names, making it extensible if new document backends are added
