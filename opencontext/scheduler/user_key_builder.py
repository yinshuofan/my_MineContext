#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
User Key Builder Implementation

Builds user identification keys based on configurable dimensions.
"""

from typing import Dict, List, Optional

from opencontext.scheduler.base import IUserKeyBuilder, UserKeyConfig


class UserKeyBuilder(IUserKeyBuilder):
    """
    User Key Builder

    Builds user identification keys from configurable dimensions:
    - 3 key mode: user_id:device_id:agent_id
    - 2 key mode: user_id:device_id (when use_agent_id=False)
    - 1 key mode: user_id (when use_device_id=False and use_agent_id=False)
    """

    KEY_SEPARATOR = ":"

    def __init__(self, config: Optional[UserKeyConfig] = None):
        """
        Initialize the UserKeyBuilder.

        Args:
            config: UserKeyConfig instance, or None for defaults
        """
        if config is None:
            config = UserKeyConfig()

        self._use_user_id = config.use_user_id  # Always True
        self._use_device_id = config.use_device_id
        self._use_agent_id = config.use_agent_id
        self._default_device_id = config.default_device_id
        self._default_agent_id = config.default_agent_id

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "UserKeyBuilder":
        """
        Create a UserKeyBuilder from a configuration dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            UserKeyBuilder instance
        """
        config = UserKeyConfig.from_dict(config_dict)
        return cls(config)

    def build_key(
        self, user_id: str, device_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> str:
        """
        Build a user key from the given dimensions.

        Args:
            user_id: User identifier (required)
            device_id: Device identifier (optional, uses default if not provided)
            agent_id: Agent identifier (optional, uses default if not provided)

        Returns:
            A string key combining the specified dimensions

        Examples:
            # 3 key mode (default)
            build_key("user1", "device1", "agent1") -> "user1:device1:agent1"

            # 2 key mode (use_agent_id=False)
            build_key("user1", "device1", "agent1") -> "user1:device1"

            # With defaults
            build_key("user1") -> "user1:default:default"
        """
        if not user_id:
            raise ValueError("user_id is required")

        parts = [user_id]

        if self._use_device_id:
            parts.append(device_id or self._default_device_id)

        if self._use_agent_id:
            parts.append(agent_id or self._default_agent_id)

        return self.KEY_SEPARATOR.join(parts)

    def parse_key(self, user_key: str) -> Dict[str, Optional[str]]:
        """
        Parse a user key back into its dimensions.

        Args:
            user_key: The combined user key string

        Returns:
            Dictionary with user_id, device_id, agent_id

        Examples:
            # 3 key mode
            parse_key("user1:device1:agent1") ->
                {"user_id": "user1", "device_id": "device1", "agent_id": "agent1"}

            # 2 key mode
            parse_key("user1:device1") ->
                {"user_id": "user1", "device_id": "device1", "agent_id": None}
        """
        if not user_key:
            return {"user_id": None, "device_id": None, "agent_id": None}

        parts = user_key.split(self.KEY_SEPARATOR)
        result: Dict[str, Optional[str]] = {
            "user_id": None,
            "device_id": None,
            "agent_id": None,
        }

        idx = 0

        # user_id is always first
        if idx < len(parts):
            result["user_id"] = parts[idx]
            idx += 1

        # device_id is second if enabled
        if self._use_device_id and idx < len(parts):
            result["device_id"] = parts[idx]
            idx += 1

        # agent_id is third if enabled
        if self._use_agent_id and idx < len(parts):
            result["agent_id"] = parts[idx]

        return result

    def get_key_dimensions(self) -> List[str]:
        """
        Get the list of dimensions currently in use.

        Returns:
            List of dimension names

        Examples:
            # 3 key mode
            get_key_dimensions() -> ["user_id", "device_id", "agent_id"]

            # 2 key mode
            get_key_dimensions() -> ["user_id", "device_id"]
        """
        dims = ["user_id"]

        if self._use_device_id:
            dims.append("device_id")

        if self._use_agent_id:
            dims.append("agent_id")

        return dims

    def get_key_count(self) -> int:
        """
        Get the number of dimensions in use.

        Returns:
            Number of dimensions (1, 2, or 3)
        """
        return len(self.get_key_dimensions())

    @property
    def use_device_id(self) -> bool:
        """Whether device_id dimension is enabled"""
        return self._use_device_id

    @property
    def use_agent_id(self) -> bool:
        """Whether agent_id dimension is enabled"""
        return self._use_agent_id

    @property
    def default_device_id(self) -> str:
        """Default value for device_id"""
        return self._default_device_id

    @property
    def default_agent_id(self) -> str:
        """Default value for agent_id"""
        return self._default_agent_id
