#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Entity normalization tool base class
"""

from abc import ABC, abstractmethod
from typing import Any

from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseTool(ABC):
    """Base class for entity tools"""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Get tool name"""

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """Get tool description"""

    @classmethod
    @abstractmethod
    def get_parameters(cls) -> dict[str, Any]:
        """Get tool parameters definition"""

    @abstractmethod
    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute tool operation"""

    @classmethod
    def get_definition(cls) -> dict[str, Any]:
        """Get tool definition for LLM calls"""
        return {
            "name": cls.get_name(),
            "description": cls.get_description(),
            "parameters": cls.get_parameters(),
        }
