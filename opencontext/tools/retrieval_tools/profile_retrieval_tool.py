#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Profile retrieval tool - retrieves user profiles and entity profiles from the relational DB.

This tool operates on the relational database (not vector DB), providing access to
user profile data and entity information via UnifiedStorage's profile/entity methods.
"""

from typing import Any, Dict

from opencontext.storage.global_storage import get_storage
from opencontext.tools.base import BaseTool
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProfileRetrievalTool(BaseTool):
    """Retrieves user profiles and entity profiles from the relational DB.

    Supports four operations:
    - get_profile: Fetch a user's profile by user_id and agent_id
    - find_entity: Look up a specific entity by exact name
    - search_entities: Search entities by text query (fuzzy matching)
    - list_entities: List entities with optional type filtering
    """

    @classmethod
    def get_name(cls) -> str:
        return "retrieve_profile_context"

    @classmethod
    def get_description(cls) -> str:
        return """Retrieve user profiles and entity profiles from the relational database.

**Supported operations:**
- **get_profile**: Fetch a user's profile by user_id (and optional agent_id). Returns the full profile dict including preferences, history, and metadata.
- **find_entity**: Look up a specific entity by exact name. Returns the entity's details including type, aliases, description, and metadata.
- **search_entities**: Search entities by a text query with fuzzy matching. Useful when you don't know the exact entity name.
- **list_entities**: List all entities for a user, optionally filtered by entity type (person, project, team, organization, other).

**When to use this tool:**
- When you need user profile information (preferences, settings, identity)
- When looking up known entities (people, projects, teams, organizations)
- When searching for entities related to a topic or keyword
- When exploring what entities exist for a user

**Note:** This tool queries the relational DB directly and does NOT perform vector similarity search."""

    @classmethod
    def get_parameters(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["get_profile", "find_entity", "search_entities", "list_entities"],
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
                "entity_name": {
                    "type": "string",
                    "description": "Exact entity name to look up (required for find_entity)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query text for fuzzy entity matching (required for search_entities)",
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["person", "project", "team", "organization", "other"],
                    "description": "Entity type filter for list_entities (optional)",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                },
            },
            "required": ["operation", "user_id"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the requested profile/entity retrieval operation."""
        operation = kwargs.get("operation")
        user_id = kwargs.get("user_id")

        if not operation:
            return {"success": False, "error": "operation is required", "operation": None}

        if not user_id:
            return {"success": False, "error": "user_id is required", "operation": operation}

        operation_handlers = {
            "get_profile": self._handle_get_profile,
            "find_entity": self._handle_find_entity,
            "search_entities": self._handle_search_entities,
            "list_entities": self._handle_list_entities,
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

    async def _handle_get_profile(self, params: Dict[str, Any]) -> Dict[str, Any]:
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
                "error": f"No profile found for user_id='{user_id}', device_id='{device_id}', agent_id='{agent_id}'",
            }

        return {"success": True, "data": profile, "operation": "get_profile"}

    async def _handle_find_entity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Look up a specific entity by exact name."""
        user_id = params["user_id"]
        device_id = params.get("device_id", "default")
        agent_id = params.get("agent_id", "default")
        entity_name = params.get("entity_name", "")

        if not entity_name:
            return {
                "success": False,
                "data": None,
                "operation": "find_entity",
                "error": "entity_name is required for find_entity operation",
            }

        storage = self._get_storage()
        entity = await storage.get_entity(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            entity_name=entity_name,
        )

        if not entity:
            return {
                "success": False,
                "data": None,
                "operation": "find_entity",
                "error": f"Entity '{entity_name}' not found for user_id='{user_id}'",
            }

        return {"success": True, "data": entity, "operation": "find_entity"}

    async def _handle_search_entities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search entities by text query with fuzzy matching."""
        user_id = params["user_id"]
        device_id = params.get("device_id", "default")
        agent_id = params.get("agent_id", "default")
        query = params.get("query", "")

        if not query:
            return {
                "success": False,
                "data": None,
                "operation": "search_entities",
                "error": "query is required for search_entities operation",
            }

        top_k = min(max(params.get("top_k", 5), 1), 100)

        storage = self._get_storage()
        results = await storage.search_entities(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            query_text=query,
            limit=top_k,
        )

        return {
            "success": True,
            "data": results,
            "operation": "search_entities",
            "count": len(results),
        }

    async def _handle_list_entities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List entities with optional type filtering."""
        user_id = params["user_id"]
        device_id = params.get("device_id", "default")
        agent_id = params.get("agent_id", "default")
        entity_type = params.get("entity_type")
        top_k = min(max(params.get("top_k", 5), 1), 100)

        storage = self._get_storage()
        results = await storage.list_entities(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            entity_type=entity_type,
            limit=top_k,
        )

        return {
            "success": True,
            "data": results,
            "operation": "list_entities",
            "count": len(results),
        }
