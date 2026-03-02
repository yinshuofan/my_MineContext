#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


"""
JSON parsing utility functions
"""

import json
import re
from typing import Any, Optional

import json_repair
from loguru import logger


def parse_json_from_response(response: str) -> Optional[Any]:
    """
    Parse JSON object from LLM text response.

    Handles code blocks and plain JSON strings, including fixing common format issues.

    Args:
        response (str): LLM text response or JSON string

    Returns:
        Optional[Any]: Parsed JSON object, None if parsing fails
    """
    if not isinstance(response, str):
        return None

    # Remove leading and trailing whitespace
    response = response.strip()

    # Strategy 1: Direct parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from code blocks
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Regex match JSON structure
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", response)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Use json_repair library
    try:
        return json_repair.loads(response)
    except (json.JSONDecodeError, ValueError):
        logger.error(f"Failed to parse JSON from response: {response}")

    return None
