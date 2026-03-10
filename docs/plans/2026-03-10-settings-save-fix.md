# Settings Page Default Values Fix

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix scheduler settings page showing 0 for hierarchy summary values, and prevent saving from silently dropping config keys not present in the form.

**Architecture:** Three layered fixes: (1) add missing `hierarchy_summary` section to `config-docker.yaml` so the API always returns values for the form, (2) change `save_user_settings` from whole-key replacement to `deep_merge` so non-form keys like `user_key_config` and `backfill_days` survive saves, with null-stripping to safely handle empty form fields, (3) change JS `getInt`/`getFloat` to return `null` instead of `0` for empty inputs so unfilled fields are excluded rather than persisted as zero.

**Tech Stack:** Python 3.10+, JavaScript (vanilla), YAML

**Note:** After applying these fixes, the user must reset the corrupted `persist/user_setting.yaml` — either delete the file, remove the `scheduler` key from it, or use the "Reset" button on the settings page. The existing `hierarchy_summary: {interval: 0, timeout: 0, task_ttl: 0}` values will otherwise override the corrected defaults.

---

### Task 1: Add `hierarchy_summary` to `config-docker.yaml`

This is the root cause. The docker config has no `hierarchy_summary` section, so the API returns no values for it, leaving form fields empty.

**Files:**
- Modify: `config/config-docker.yaml:154-164`

**Step 1: Add hierarchy_summary section after data_cleanup**

In `config/config-docker.yaml`, add the `hierarchy_summary` block after the `data_cleanup` section (after line 164), matching the values from `config/config.yaml`:

```yaml
    hierarchy_summary:
      enabled: "${HIERARCHY_SUMMARY_ENABLED:true}"
      trigger_mode: "user_activity"
      interval: 86400
      timeout: 600
      task_ttl: 172800
      backfill_days: 7
```

**Step 2: Verify YAML syntax**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('config/config-docker.yaml'))" && echo "OK"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add config/config-docker.yaml
git commit -m "fix(config): add missing hierarchy_summary to docker config"
```

---

### Task 2: Change `save_user_settings` to deep merge + null stripping

Currently `save_user_settings` does whole-key replacement at the top level: `user_settings["scheduler"] = settings["scheduler"]`. This permanently loses any config keys not present in the form submission (e.g. `user_key_config`, `backfill_days`).

Fix: deep_merge new values into existing user_setting.yaml, and strip `None` values beforehand so that empty form fields (which produce `null` in JSON after Task 3) are excluded from the merge rather than overwriting existing values with `None`.

**Files:**
- Modify: `opencontext/config/config_manager.py:174-224`

**Step 1: Add `_strip_none_values` helper method**

In `ConfigManager` class, add before `save_user_settings` (before line 174):

```python
    @staticmethod
    def _strip_none_values(d):
        """Recursively remove None values from nested dicts."""
        if not isinstance(d, dict):
            return d
        return {k: ConfigManager._strip_none_values(v) for k, v in d.items() if v is not None}
```

**Step 2: Change save logic from replacement to deep merge**

In `save_user_settings`, replace lines 200-206:

```python
            # Update with new settings (only whitelisted keys).
            # Note: This does whole-key replacement at the user_setting.yaml level,
            # not deep_merge. Callers must send complete section objects.
            # The deep_merge only happens when applying user_settings to _config.
            for key in settings:
                if key in SAVEABLE_KEYS:
                    user_settings[key] = settings[key]
```

With:

```python
            # Update with new settings (only whitelisted keys).
            # Deep merge: preserves keys in user_setting.yaml that are not present
            # in the incoming settings (e.g. keys not exposed in the UI form).
            # None values are stripped first so empty form fields don't overwrite
            # existing values with null.
            for key in settings:
                if key in SAVEABLE_KEYS:
                    value = self._strip_none_values(settings[key])
                    if isinstance(user_settings.get(key), dict) and isinstance(value, dict):
                        user_settings[key] = deep_merge(user_settings[key], value)
                    else:
                        user_settings[key] = value
```

**Step 3: Compile-check**

Run:
```bash
python -m py_compile opencontext/config/config_manager.py
```
Expected: no output (success)

**Step 4: Commit**

```bash
git add opencontext/config/config_manager.py
git commit -m "fix(config): use deep_merge in save_user_settings to preserve non-form keys"
```

---

### Task 3: Fix `getInt` / `getFloat` to return `null` for empty fields

Currently `getInt` returns `0` for empty inputs (`parseInt('') || 0` → `NaN || 0` → `0`). This means saving a form with unfilled fields writes `0` to the config. Combined with Task 2's null stripping, returning `null` instead causes empty fields to be excluded from the save entirely.

**Files:**
- Modify: `opencontext/web/static/js/settings.js:34-40`

**Step 1: Fix `getInt`**

Replace lines 34-36:

```javascript
function getInt(id) {
    return parseInt(getVal(id)) || 0;
}
```

With:

```javascript
function getInt(id) {
    const v = parseInt(getVal(id));
    return isNaN(v) ? null : v;
}
```

**Step 2: Fix `getFloat`**

Replace lines 38-40:

```javascript
function getFloat(id) {
    return parseFloat(getVal(id)) || 0;
}
```

With:

```javascript
function getFloat(id) {
    const v = parseFloat(getVal(id));
    return isNaN(v) ? null : v;
}
```

**Step 3: Commit**

```bash
git add opencontext/web/static/js/settings.js
git commit -m "fix(settings): return null instead of 0 for empty numeric form fields"
```

---

### Post-fix: Reset corrupted user settings

The existing `persist/user_setting.yaml` contains `hierarchy_summary: {interval: 0, timeout: 0, task_ttl: 0}` from a previous save. These values will override the corrected defaults until cleared.

**Option A** — Delete the file and let the next save recreate it:
```bash
rm persist/user_setting.yaml
```

**Option B** — Use the settings page "Reset" button (calls `POST /api/settings/reset`)

**Option C** — Manually edit `persist/user_setting.yaml` to remove the `scheduler` key

After resetting, reload the settings page to verify hierarchy summary shows correct defaults: interval=86400, timeout=600, task_ttl=172800.
