#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""One-shot cleanup: strip stale `trigger_mode` fields from the `scheduler`
row in `system_settings`.

Background
----------
Before the scheduler trigger_mode refactor, users could edit each task's
`trigger_mode` from the settings UI. Those values were persisted to
`system_settings.setting_value` as JSON. After the refactor, `trigger_mode`
became a code-declared handler contract; YAML values are stripped with a
deprecation warning at startup, and `GET /api/settings/general` hides
residual fields from the UI.

But the stale fields remain in the DB because `save_user_settings_async`
uses deep-merge semantics — saves never remove a key, they only add/update.
So the record keeps its old `trigger_mode` strings forever, even though
nothing reads them.

This script loads the scheduler row, pops every
`tasks.<name>.trigger_mode` field, and writes the result back with raw
SQL (overwrite, not merge). Safe to run multiple times: if no stale fields
are found it exits without touching the DB.

Usage
-----

    uv run python scripts/cleanup_stale_trigger_mode.py             # dry run
    uv run python scripts/cleanup_stale_trigger_mode.py --apply     # actually write

Environment
-----------
Reads `config/config.yaml` + `.env` the same way the main app does (via
`GlobalConfig`). Detects MySQL vs SQLite from the `storage.stores` config
section and connects directly to the document DB using `pymysql` or
`sqlite3`. Does NOT go through `ConfigManager.save_user_settings_async`
because that would merge instead of overwrite.

Exit codes
----------
    0  success (or nothing to clean up)
    1  no scheduler row found (nothing to do)
    2  setup/connection error
    3  dry run would change rows (useful for CI gating)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make sure we can import from the project root when this script is run
# directly (e.g. `python scripts/cleanup_stale_trigger_mode.py`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from opencontext.config.global_config import GlobalConfig  # noqa: E402


def _find_document_store_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first enabled document_db entry from config['storage']['backends']."""
    storage = config.get("storage", {})
    backends = storage.get("backends", []) or []
    for backend in backends:
        if not isinstance(backend, dict):
            continue
        if backend.get("storage_type") != "document_db":
            continue
        if backend.get("enabled", True) is False:
            continue
        return backend
    return None


def _strip_trigger_mode(scheduler_value: dict[str, Any]) -> list[tuple[str, str]]:
    """Mutate *scheduler_value* in place: pop `trigger_mode` from every task.

    Returns a list of (task_name, removed_value) pairs describing what was
    stripped. Empty list means nothing to do.
    """
    changes: list[tuple[str, str]] = []
    tasks = scheduler_value.get("tasks")
    if not isinstance(tasks, dict):
        return changes
    for task_name, task_cfg in tasks.items():
        if not isinstance(task_cfg, dict):
            continue
        if "trigger_mode" in task_cfg:
            changes.append((task_name, str(task_cfg.pop("trigger_mode"))))
    return changes


def _cleanup_mysql(store_config: dict[str, Any], apply_changes: bool) -> int:
    """Connect to MySQL and clean up. Returns exit code."""
    import pymysql

    conn_params = {
        "host": store_config.get("host", "localhost"),
        "port": int(store_config.get("port", 3306)),
        "user": store_config.get("user", "root"),
        "password": store_config.get("password", ""),
        "database": store_config.get("database", "opencontext"),
        "charset": store_config.get("charset", "utf8mb4"),
    }
    print(
        f"[backend] mysql @ {conn_params['host']}:{conn_params['port']}/{conn_params['database']}"
    )

    try:
        conn = pymysql.connect(**conn_params)
    except Exception as e:
        print(f"[error] MySQL connection failed: {e}", file=sys.stderr)
        return 2

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT setting_value FROM system_settings WHERE setting_key = %s",
                ("scheduler",),
            )
            row = cursor.fetchone()
            if row is None:
                print("[info] no 'scheduler' row in system_settings — nothing to clean up")
                return 1

            raw = row[0]
            value = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(value, dict):
                print(
                    f"[error] scheduler setting_value is not a dict: {type(value).__name__}",
                    file=sys.stderr,
                )
                return 2

            changes = _strip_trigger_mode(value)
            if not changes:
                print("[info] no stale trigger_mode fields found — DB already clean")
                return 0

            print(f"[found] {len(changes)} stale trigger_mode fields:")
            for task_name, old_value in changes:
                print(f"  - tasks.{task_name}.trigger_mode = {old_value!r}")

            if not apply_changes:
                print("[dry-run] re-run with --apply to write changes")
                return 3

            new_json = json.dumps(value, ensure_ascii=False)
            cursor.execute(
                "UPDATE system_settings SET setting_value = %s WHERE setting_key = %s",
                (new_json, "scheduler"),
            )
            conn.commit()
            print(f"[ok] updated scheduler row; {len(changes)} fields removed")
            return 0
    finally:
        conn.close()


def _cleanup_sqlite(store_config: dict[str, Any], apply_changes: bool) -> int:
    """Connect to SQLite and clean up. Returns exit code."""
    import sqlite3

    raw_path = store_config.get("path") or store_config.get("database")
    if not raw_path:
        print("[error] SQLite store config has no 'path' or 'database' key", file=sys.stderr)
        return 2

    db_path = Path(raw_path).expanduser().resolve()
    if not db_path.exists():
        print(f"[error] SQLite file not found: {db_path}", file=sys.stderr)
        return 2

    print(f"[backend] sqlite @ {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT setting_value FROM system_settings WHERE setting_key = ?",
            ("scheduler",),
        )
        row = cursor.fetchone()
        if row is None:
            print("[info] no 'scheduler' row in system_settings — nothing to clean up")
            return 1

        raw = row["setting_value"]
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, dict):
            print(
                f"[error] scheduler setting_value is not a dict: {type(value).__name__}",
                file=sys.stderr,
            )
            return 2

        changes = _strip_trigger_mode(value)
        if not changes:
            print("[info] no stale trigger_mode fields found — DB already clean")
            return 0

        print(f"[found] {len(changes)} stale trigger_mode fields:")
        for task_name, old_value in changes:
            print(f"  - tasks.{task_name}.trigger_mode = {old_value!r}")

        if not apply_changes:
            print("[dry-run] re-run with --apply to write changes")
            return 3

        new_json = json.dumps(value, ensure_ascii=False)
        conn.execute(
            "UPDATE system_settings SET setting_value = ?,"
            " updated_at = CURRENT_TIMESTAMP"
            " WHERE setting_key = ?",
            (new_json, "scheduler"),
        )
        conn.commit()
        print(f"[ok] updated scheduler row; {len(changes)} fields removed")
        return 0
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strip stale scheduler.tasks.<name>.trigger_mode fields from system_settings.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this flag the script runs in dry-run mode.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()

    # Bootstrap GlobalConfig so env var substitution works the same way as
    # the running server. We only need config["storage"]["stores"] — this
    # does NOT touch any other subsystem (no scheduler, no Redis, no LLM).
    try:
        gc = GlobalConfig.get_instance()
        gc.initialize(args.config)
        config = gc.get_config()
        if not isinstance(config, dict):
            print("[error] GlobalConfig returned no config", file=sys.stderr)
            return 2
    except Exception as e:
        print(f"[error] failed to load config: {e}", file=sys.stderr)
        return 2

    store = _find_document_store_config(config)
    if store is None:
        print(
            "[error] no document_db backend found in config.storage.backends",
            file=sys.stderr,
        )
        return 2

    backend = store.get("backend", "").lower()
    backend_config = store.get("config") or {}
    if not isinstance(backend_config, dict):
        print(
            "[error] document_db backend 'config' section is not a dict",
            file=sys.stderr,
        )
        return 2

    if backend == "mysql":
        return _cleanup_mysql(backend_config, apply_changes=args.apply)
    if backend == "sqlite":
        return _cleanup_sqlite(backend_config, apply_changes=args.apply)

    print(f"[error] unsupported document_db backend: {backend!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
