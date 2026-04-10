import json
from typing import Any

from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

_EMPTY_MEDIA_REFS_SENTINELS = {"", "default", "null", "none"}


def normalize_media_refs(value: Any) -> list[dict[str, Any]]:
    """Normalize media_refs into a list of dicts."""
    if value is None:
        return []

    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in _EMPTY_MEDIA_REFS_SENTINELS:
            return []
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Dropping invalid media_refs string: {stripped!r}")
            return []

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        logger.warning(f"Dropping invalid media_refs payload type: {type(parsed).__name__}")
        return []

    normalized: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            logger.warning(f"Dropping invalid media_refs item type: {type(item).__name__}")
    return normalized
