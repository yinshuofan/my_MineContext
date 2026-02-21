#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Entity processing module — adapted for relational DB storage.

Entities are now stored in the relational DB (profiles/entities tables)
via UnifiedStorage.upsert_entity(), not in the vector DB.
"""
import asyncio
import datetime
import json
from typing import Dict, List, Optional, Tuple

from opencontext.models.context import EntityData
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.json_parser import parse_json_from_response

logger = get_logger(__name__)


def validate_and_clean_entities(raw_entities) -> Dict[str, EntityData]:
    """Validate and clean entity list, return dict of entity_name -> EntityData"""
    if not isinstance(raw_entities, list):
        logger.warning(f"Entity is not a list type: {type(raw_entities)}, using empty list")
        return {}
    entities_info = {}
    for entity in raw_entities:
        if isinstance(entity, dict) and "name" in entity:
            name = str(entity["name"]).strip()
            entity_type = entity.get("type", "other")
            if name:
                aliases = entity.get("aliases", [])
                if name not in aliases:
                    aliases.append(name)
                entity_info = EntityData(
                    user_id="",  # Will be set by caller
                    entity_name=name,
                    entity_type=entity_type,
                    content=entity.get("description", ""),
                    aliases=aliases,
                    metadata=entity.get("metadata", {}),
                )
                entities_info[name] = entity_info
            else:
                logger.warning(f"Skipping invalid entity type {type(entity)}: {entity}")
    return entities_info


async def refresh_entities(
    entities_info: Dict[str, EntityData],
    context_text: str,
    user_id: str = "default",
) -> List[str]:
    """
    Entity processing main workflow — upserts entities to relational DB.

    Args:
        entities_info: Dict of entity_name -> EntityData
        context_text: The source context text (for LLM merge if needed)
        user_id: User ID for storage

    Returns:
        List of processed entity names
    """
    from opencontext.storage.global_storage import get_global_storage

    storage = get_global_storage()
    processed_names = []

    for entity_name, entity_data in entities_info.items():
        try:
            entity_data.user_id = user_id

            # Check if entity already exists
            existing = storage.get_entity(user_id, entity_name)
            if existing:
                # Merge: combine old content with new description
                old_content = existing.get("content", "")
                new_content = entity_data.content
                if new_content and new_content not in old_content:
                    merged_content = f"{old_content}\n{new_content}".strip()
                else:
                    merged_content = old_content

                # Merge aliases
                old_aliases = existing.get("aliases", [])
                merged_aliases = list(set(old_aliases + entity_data.aliases))

                # Merge keywords
                old_keywords = existing.get("keywords", [])
                merged_keywords = list(set(old_keywords + entity_data.keywords))

                storage.upsert_entity(
                    user_id=user_id,
                    entity_name=entity_name,
                    content=merged_content,
                    entity_type=entity_data.entity_type or existing.get("entity_type"),
                    summary=entity_data.summary or existing.get("summary"),
                    keywords=merged_keywords,
                    aliases=merged_aliases,
                    metadata=entity_data.metadata,
                )
            else:
                # Create new entity
                storage.upsert_entity(
                    user_id=user_id,
                    entity_name=entity_name,
                    content=entity_data.content,
                    entity_type=entity_data.entity_type,
                    summary=entity_data.summary,
                    keywords=entity_data.keywords,
                    aliases=entity_data.aliases,
                    metadata=entity_data.metadata,
                )

            processed_names.append(entity_name)

        except Exception as e:
            logger.error(f"Entity processing failed for {entity_name}: {e}")

    return processed_names
