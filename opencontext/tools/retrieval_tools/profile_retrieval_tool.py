#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Profile retrieval tool - retrieves user profiles from the relational DB.

This tool operates on the relational database (not vector DB), providing access to
user profile data via UnifiedStorage's profile methods.
"""

from typing import Any

from opencontext.storage.global_storage import get_storage
from opencontext.tools.base import BaseTool
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProfileRetrievalTool(BaseTool):
    """Retrieves user profiles from the relational DB.

    Supports one operation:
    - get_profile: Fetch a user's profile by user_id and agent_id
    """

    @classmethod
    def get_name(cls) -> str:
        return "retrieve_profile_context"

    @classmethod
    def get_description(cls) -> str:
        return """Retrieve user profiles from the relational database.

**Supported operations:**
- **get_profile**: Fetch a user's profile by user_id (and optional \
agent_id). Returns the full profile dict including preferences, \
history, and metadata.

**When to use this tool:**
- When you need user profile information (preferences, settings, identity)

**Note:** This tool queries the relational DB directly and does \
NOT perform vector similarity search."""

    @classmethod
    def get_parameters(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["get_profile"],
                    "description": "The operation to perform",
                },
                "user_id": {
                    "type": "string",
                    "description": "User ID (required for all operations)",
                },
                "device_id": {
                    "type": "string",
                    "description": "Device ID (defaults to 'default')",
                    "default": "default",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Agent ID (defaults to 'default')",
                    "default": "default",
                },
            },
            "required": ["operation", "user_id"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute the requested profile retrieval operation."""
        operation = kwargs.get("operation")
        user_id = kwargs.get("user_id")

        if not operation:
            return {"success": False, "error": "operation is required", "operation": None}

        if not user_id:
            return {"success": False, "error": "user_id is required", "operation": operation}

        operation_handlers = {
            "get_profile": self._handle_get_profile,
        }

        handler = operation_handlers.get(operation)
        if not handler:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}",
                "operation": operation,
                "supported_operations": list(operation_handlers.keys()),
            }

        try:
            return await handler(kwargs)
        except Exception as e:
            logger.error(f"ProfileRetrievalTool failed - operation={operation}: {e}", exc_info=True)
            return {"success": False, "error": str(e), "operation": operation}

    def _get_storage(self):
        """Get storage instance, raising an error if unavailable."""
        storage = get_storage()
        if storage is None:
            raise RuntimeError("Storage not initialized")
        return storage

    async def _handle_get_profile(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch user profile from relational DB."""
        user_id = params["user_id"]
        device_id = params.get("device_id", "default")
        agent_id = params.get("agent_id", "default")

        storage = self._get_storage()
        profile = await storage.get_profile(user_id, device_id, agent_id)

        if not profile:
            return {
                "success": False,
                "data": None,
                "operation": "get_profile",
                "error": (
                    f"No profile found for user_id='{user_id}', "
                    f"device_id='{device_id}', agent_id='{agent_id}'"
                ),
            }

        return {"success": True, "data": profile, "operation": "get_profile"}
