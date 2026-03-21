#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Profile processing module — LLM-driven intelligent merge before overwrite.

Profiles are stored in the relational DB (profiles table) via
UnifiedStorage.upsert_profile(). When an existing profile is found,
new information is intelligently merged with the old using an LLM call
before writing, rather than blindly overwriting.
"""

import json
from typing import Any, Dict, List, Optional

from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def refresh_profile(
    new_factual_profile: str,
    new_entities: Optional[List[str]],
    new_importance: int,
    new_metadata: Optional[Dict[str, Any]],
    user_id: str = "default",
    device_id: str = "default",
    agent_id: str = "default",
    context_type: str = "profile",
) -> bool:
    """
    Profile processing main workflow — merges with existing profile via LLM, then upserts.

    If no existing profile is found, writes directly.
    If LLM merge fails, falls back to direct overwrite.

    Args:
        new_factual_profile: New factual profile content text
        new_entities: New related entity names
        new_importance: New importance score (0-10)
        new_metadata: New metadata dict
        user_id: User ID for storage
        device_id: Device ID for storage
        agent_id: Agent ID for storage
        context_type: Profile context type ("profile" or "agent_profile")

    Returns:
        True if profile was stored successfully
    """
    from opencontext.storage.global_storage import get_storage

    storage = get_storage()

    try:
        existing = await storage.get_profile(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            context_type=context_type,
        )

        # Fallback: first interaction profile for an agent inherits from base profile
        if existing is None and context_type == "agent_profile" and user_id != "__base__":
            existing = await storage.get_profile(
                "__base__", device_id, agent_id, context_type="agent_base_profile"
            )

        if existing:
            merged = await _merge_profile_with_llm(existing, {
                "factual_profile": new_factual_profile,
                "entities": new_entities or [],
                "importance": new_importance,
            })

            if merged:
                return await storage.upsert_profile(
                    user_id=user_id,
                    device_id=device_id,
                    agent_id=agent_id,
                    factual_profile=merged.get("factual_profile", new_factual_profile),
                    entities=merged.get("entities", new_entities),
                    importance=merged.get("importance", new_importance),
                    metadata=new_metadata,
                    context_type=context_type,
                )
            else:
                logger.warning("LLM merge failed, falling back to direct overwrite")

        # No existing profile or LLM merge failed — direct write
        return await storage.upsert_profile(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            factual_profile=new_factual_profile,
            entities=new_entities,
            importance=new_importance,
            metadata=new_metadata,
            context_type=context_type,
        )

    except Exception as e:
        logger.error(f"Profile processing failed for user={user_id}: {e}")
        return False


async def _merge_profile_with_llm(
    existing: Dict[str, Any],
    new_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to intelligently merge new profile data into an existing profile.

    Args:
        existing: Existing profile dict from storage (has factual_profile, entities, etc.)
        new_data: New profile data dict

    Returns:
        Merged profile dict with factual_profile/entities/importance,
        or None if LLM call fails.
    """
    from opencontext.config.global_config import get_prompt_group
    from opencontext.llm.global_vlm_client import generate_with_messages
    from opencontext.utils.json_parser import parse_json_from_response

    try:
        prompt_group = get_prompt_group("merging.overwrite_merge")
        if not prompt_group or "user" not in prompt_group:
            logger.error("Prompt 'merging.overwrite_merge' not found")
            return None

        existing_content = json.dumps(
            {
                "factual_profile": existing.get("factual_profile", ""),
                "entities": existing.get("entities", []),
                "importance": existing.get("importance", 0),
            },
            ensure_ascii=False,
            indent=2,
        )
        new_content = json.dumps(new_data, ensure_ascii=False, indent=2)

        user_prompt = prompt_group["user"].format(
            existing_content=existing_content,
            new_content=new_content,
        )

        messages = []
        system_prompt = prompt_group.get("system")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = await generate_with_messages(messages)
        if not response:
            logger.warning("LLM returned empty response for profile merge")
            return None

        merged = parse_json_from_response(response)
        if not merged or not isinstance(merged, dict):
            logger.warning(f"Failed to parse LLM merge response as JSON")
            return None

        if "factual_profile" not in merged:
            logger.warning("LLM merge response missing 'factual_profile' field")
            return None

        return merged

    except Exception as e:
        logger.error(f"LLM profile merge failed: {e}")
        return None
