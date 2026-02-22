#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Profile Entity Tool — adapted for relational DB storage.

Entities are now stored in the relational DB (entities table) via UnifiedStorage,
not in the vector DB. This tool provides entity lookup, search, and relationship queries.
"""

from typing import Any, Dict, List, Optional, Tuple

from opencontext.storage.global_storage import get_storage
from opencontext.tools.base import BaseTool
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProfileEntityTool(BaseTool):
    """Unified entity management tool — relational DB backed"""

    def __init__(self, user_id: str = "default"):
        super().__init__()
        self._storage = None
        self.user_id = user_id

    @property
    def storage(self):
        """Lazy storage access — avoids init-order issues."""
        if self._storage is None:
            self._storage = get_storage()
            if self._storage is None:
                raise RuntimeError("Storage not initialized")
        return self._storage

    @classmethod
    def get_name(cls) -> str:
        return "profile_entity_tool"

    @classmethod
    def get_description(cls) -> str:
        return """Entity profile management tool for finding, matching, and exploring entities (people, projects, organizations, etc.) and their relationships.

**Supported operations:**
- **find_exact_entity**: Lookup entity by exact name or alias
- **find_similar_entity**: Search entities by text query (fuzzy matching)
- **list_entities**: List all entities for the user, optionally filtered by type
- **check_entity_relationships**: Check if two entities are related

**Entity types supported:**
- person, project, team, organization, other
"""

    @classmethod
    def get_parameters(cls) -> Dict[str, Any]:
        """Get tool parameter definitions"""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "find_exact_entity",
                        "find_similar_entity",
                        "list_entities",
                        "check_entity_relationships",
                    ],
                    "description": "Operation type",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Entity name to search",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Entity type filter (person, project, team, organization, other)",
                },
                "entity1": {
                    "type": "string",
                    "description": "First entity in relationship check",
                },
                "entity2": {
                    "type": "string",
                    "description": "Second entity in relationship check",
                },
                "query": {
                    "type": "string",
                    "description": "Search query text for fuzzy matching",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier for multi-user filtering",
                },
            },
            "required": ["operation"],
            "additionalProperties": False,
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute entity operation"""
        operation = kwargs.get("operation")
        user_id = kwargs.get("user_id", self.user_id)

        operation_handlers = {
            "find_exact_entity": self._handle_find_exact,
            "find_similar_entity": self._handle_find_similar,
            "list_entities": self._handle_list_entities,
            "check_entity_relationships": self._handle_check_relationships,
        }

        handler = operation_handlers.get(operation)
        if not handler:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}",
                "supported_operations": list(operation_handlers.keys()),
            }

        try:
            return handler(kwargs, user_id)
        except Exception as e:
            logger.error(f"Failed to execute entity operation - {operation}: {e}", exc_info=True)
            return {"success": False, "error": str(e), "operation": operation}

    def _handle_find_exact(self, params: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Handle exact search operation — queries relational DB"""
        entity_name = params.get("entity_name", "")
        if not entity_name:
            return {
                "success": False,
                "error": "entity_name is required for find_exact_entity operation",
            }

        result = self.storage.get_entity(user_id, entity_name)
        if not result:
            return {"success": False, "error": f"Entity '{entity_name}' not found"}
        return {"success": True, "entity_info": result, "entity_name": entity_name}

    def _handle_find_similar(self, params: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Handle similar/fuzzy search operation — uses relational DB text search"""
        query = params.get("query") or params.get("entity_name", "")
        if not query:
            return {
                "success": False,
                "error": "query or entity_name is required for find_similar_entity operation",
            }

        top_k = min(max(params.get("top_k", 10), 1), 100)
        results = self.storage.search_entities(user_id, query, limit=top_k)
        if not results:
            return {"success": False, "error": f"No entities found matching '{query}'"}
        return {"success": True, "entities": results}

    def _handle_list_entities(self, params: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Handle list entities operation"""
        entity_type = params.get("entity_type")
        top_k = min(max(params.get("top_k", 100), 1), 1000)
        results = self.storage.list_entities(user_id, entity_type=entity_type, limit=top_k)
        return {"success": True, "entities": results, "count": len(results)}

    def _handle_check_relationships(self, params: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Handle relationship checking between two entities"""
        entity1_name = params.get("entity1", "")
        entity2_name = params.get("entity2", "")

        if not entity1_name or not entity2_name:
            return {"success": False, "error": "Both entity1 and entity2 are required"}

        entity1 = self.storage.get_entity(user_id, entity1_name)
        entity2 = self.storage.get_entity(user_id, entity2_name)

        if not entity1 or not entity2:
            return {
                "success": False,
                "related": False,
                "error": "One or both entities not found",
            }

        # Check metadata for relationship info
        meta1 = entity1.get("metadata", {}) or {}
        meta2 = entity2.get("metadata", {}) or {}

        relationships1 = meta1.get("relationships", {})
        relationships2 = meta2.get("relationships", {})

        # Check if entity2 is in entity1's relationships
        for rel_type, rel_list in relationships1.items():
            if isinstance(rel_list, list) and entity2_name in rel_list:
                return {
                    "success": True,
                    "related": True,
                    "relationship_type": rel_type,
                    "direction": f"{entity1_name} -> {entity2_name}",
                }

        # Check reverse
        for rel_type, rel_list in relationships2.items():
            if isinstance(rel_list, list) and entity1_name in rel_list:
                return {
                    "success": True,
                    "related": True,
                    "relationship_type": rel_type,
                    "direction": f"{entity2_name} -> {entity1_name}",
                }

        return {"success": True, "related": False}

    def match_entity(
        self,
        entity_name: str,
        entity_type: str = None,
        top_k: int = 3,
        user_id: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Intelligent entity matching — exact first, then fuzzy search.

        Args:
            entity_name: Name to match
            entity_type: Optional type filter
            top_k: Max fuzzy results
            user_id: User identifier (falls back to self.user_id if None)

        Returns:
            Tuple[Optional[str], Optional[Dict]]: (matched entity name, entity dict)
        """
        uid = user_id if user_id is not None else self.user_id

        # 1. Try exact match
        result = self.storage.get_entity(uid, entity_name)
        if result:
            return result.get("entity_name", entity_name), result

        # 2. Search by text
        similar = self.storage.search_entities(uid, entity_name, limit=top_k)
        if similar:
            # Return the first match
            best = similar[0]
            return best.get("entity_name", entity_name), best

        return None, None
