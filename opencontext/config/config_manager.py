#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


"""
Configuration manager, responsible for loading and managing system configurations
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from opencontext.utils.dict_utils import deep_merge
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

SAVEABLE_KEYS = {
    "llm",
    "vlm_model",
    "embedding_model",
    "capture",
    "processing",
    "logging",
    "prompts",
    "document_processing",
    "scheduler",
    "memory_cache",
    "tools",
}


class ConfigManager:
    """
    Configuration Manager

    Responsible for loading and managing system configurations
    """

    def __init__(self):
        """Initialize the configuration manager"""
        self._config: Optional[Dict[str, Any]] = None
        self._config_path: Optional[str] = None
        self._env_vars: Dict[str, str] = {}
        self._use_db_settings: bool = False

    @property
    def use_db_settings(self) -> bool:
        """Whether DB-backed settings mode is active."""
        return self._use_db_settings

    @staticmethod
    def _has_document_backend(config: dict) -> bool:
        """Check if a document backend (MySQL or SQLite) is configured."""
        for backend in config.get("storage", {}).get("backends", []):
            if backend.get("storage_type") == "document_db":
                return True
        return False

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

    async def init_db_settings(self) -> bool:
        """Enable DB-backed settings if MySQL storage is available.

        Called from lifespan after GlobalStorage.ensure_initialized().
        Auto-migrates from user_setting.yaml on first startup (if DB is empty).

        Returns True if DB settings were loaded (config may have changed),
        False if DB mode was not enabled or no settings exist yet.
        """
        if not self._config or not self._has_document_backend(self._config):
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

        Migration is idempotent — if multiple workers race on startup,
        they all write the same data via replace_setting, and the end
        state is correct. A sentinel row ('_migrated') marks migration
        as complete so subsequent startups skip it.
        """
        try:
            # Sentinel: marks migration as started/done
            ok = await storage.save_setting(
                "_migrated", {"ts": str(datetime.now(tz=timezone.utc))}
            )
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

    def load_config(self, config_path: Optional[str] = None) -> bool:
        """
        Load configuration
        """
        found_config_path = None

        if not config_path:
            config_path = "config/config.yaml"
        if config_path and os.path.exists(config_path):
            found_config_path = config_path
        else:
            raise FileNotFoundError(f"Specified configuration file does not exist: {config_path}")

        with open(found_config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        self._load_env_vars()
        config_data = self._replace_env_vars(config_data)
        self._config = config_data
        self._config_path = found_config_path
        logger.info(f"Configuration loaded successfully: {self._config_path}")

        return True

    def _load_env_vars(self) -> None:
        """Load environment variables from system and .env file"""
        # Load .env file if it exists (does not override existing env vars)
        load_dotenv()

        for key, value in os.environ.items():
            self._env_vars[key] = value

    def _replace_env_vars(self, config_data: Any) -> Any:
        """
        Replace environment variable references in the configuration

        Supported formats:
        - ${VAR}: Simple variable substitution
        - ${VAR:default}: Use default value if the variable does not exist

        Args:
            config_data (Any): Configuration data

        Returns:
            Any: Configuration data after replacement
        """
        if isinstance(config_data, dict):
            return {k: self._replace_env_vars(v) for k, v in config_data.items()}
        elif isinstance(config_data, list):
            return [self._replace_env_vars(item) for item in config_data]
        elif isinstance(config_data, str):
            pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

            def replace_match(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ""
                env_value = self._env_vars.get(var_name)
                return env_value if env_value is not None else default_value

            replaced = re.sub(pattern, replace_match, config_data)
            low = replaced.strip().lower()
            if low == "true":
                return True
            if low == "false":
                return False
            return replaced
        else:
            return config_data

    def get_config(self) -> Optional[Dict[str, Any]]:
        """
        Get configuration

        Returns:
            Optional[Dict[str, Any]]: Configuration dictionary, or None if not loaded
        """
        return self._config

    def get_config_path(self) -> Optional[str]:
        """
        Get configuration file path

        Returns:
            Optional[str]: Configuration file path, or None if not loaded
        """
        return self._config_path

    @staticmethod
    def _strip_none_values(d):
        """Recursively remove None values from nested dicts."""
        if not isinstance(d, dict):
            return d
        return {k: ConfigManager._strip_none_values(v) for k, v in d.items() if v is not None}

    async def save_user_settings_async(self, settings: Dict[str, Any]) -> bool:
        """Save user settings to MySQL.

        Each whitelisted key is saved as a separate row with atomic
        JSON_MERGE_PATCH, so concurrent saves from different instances
        are safe and non-form keys within a section are preserved.
        """
        if not self._use_db_settings:
            logger.error("Cannot save settings: DB-backed settings not enabled")
            return False

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

    async def reset_user_settings_async(self) -> bool:
        """Reset user settings by clearing the DB."""
        if not self._use_db_settings:
            logger.error("Cannot reset settings: DB-backed settings not enabled")
            return False

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

    async def reload_config_async(self) -> bool:
        """Reload base config from YAML and merge user settings from DB.

        Builds the full config in a local variable before assigning to
        self._config to prevent torn reads during the async DB call.
        """
        if not self._config_path:
            return False

        if not self._use_db_settings:
            # Non-DB mode: reload base config only (no user overrides)
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
