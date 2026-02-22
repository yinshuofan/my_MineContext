# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from difflib import get_close_matches
from typing import Any, Dict, List

from opencontext.tools.base import BaseTool
from opencontext.tools.operation_tools import *
from opencontext.tools.profile_tools import ProfileEntityTool
from opencontext.tools.retrieval_tools import *
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ToolsExecutor:
    def __init__(self):
        self._tools_map: Dict[str, BaseTool] = {
            # Context retrieval tools (vector DB)
            DocumentRetrievalTool.get_name(): DocumentRetrievalTool(),
            KnowledgeRetrievalTool.get_name(): KnowledgeRetrievalTool(),
            HierarchicalEventTool.get_name(): HierarchicalEventTool(),
            # Profile retrieval tools (relational DB)
            ProfileRetrievalTool.get_name(): ProfileRetrievalTool(),
            ProfileEntityTool.get_name(): ProfileEntityTool(),
            # Operation tools
            WebSearchTool.get_name(): WebSearchTool(),
        }

    def _validate_input(self, tool_input):
        """Normalize and validate tool input.

        Returns:
            (normalized_input, None) on success, or (None, error_dict) on failure.
        """
        if (
            isinstance(tool_input, list)
            and len(tool_input) == 1
            and isinstance(tool_input[0], dict)
        ):
            tool_input = tool_input[0]

        if not isinstance(tool_input, dict):
            return None, {
                "error": (
                    f"Tool parameter format error: expected dict, "
                    f"got {type(tool_input).__name__}"
                ),
                "message": "Tool parameters must be in dictionary format",
                "received_type": type(tool_input).__name__,
            }
        return tool_input, None

    def _handle_unknown_tool(self, tool_name: str) -> Dict[str, Any]:
        """Handle unknown tool call with suggestions."""
        available_tools = list(self._tools_map.keys())
        suggestions = get_close_matches(tool_name, available_tools, n=3, cutoff=0.6)
        suggestion_text = f"Suggested tools: {', '.join(suggestions)}" if suggestions else ""

        error_msg = f"Unknown tool: {tool_name}. {suggestion_text}"
        logger.warning(error_msg)

        available_tools_text = f"Available tools: {', '.join(available_tools[:10])}" + (
            "..." if len(available_tools) > 10 else ""
        )
        return {
            "error": error_msg,
            "message": "This tool does not exist, please use system-provided tools",
            "available_tools": available_tools_text,
            "suggestions": suggestions,
        }

    async def run_async(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        if tool_name not in self._tools_map:
            return self._handle_unknown_tool(tool_name)

        tool = self._tools_map[tool_name]
        tool_input, error = self._validate_input(tool_input)
        if error:
            return error
        return await asyncio.to_thread(tool.execute, **tool_input)

    def run(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        if tool_name not in self._tools_map:
            return self._handle_unknown_tool(tool_name)

        tool = self._tools_map[tool_name]
        tool_input, error = self._validate_input(tool_input)
        if error:
            return error
        return tool.execute(**tool_input)

    async def batch_run_tools_async(self, tool_calls: List[Dict[str, Any]]) -> Any:
        results = []
        tool_call_info = []
        tasks = []
        for tc in tool_calls:
            function_name = tc.function.name
            function_args = json.loads(tc.function.arguments)
            tasks.append(self.run_async(function_name, function_args))
            tool_call_info.append((tc.id, function_name))

        res = await asyncio.gather(*tasks)
        for i, r in enumerate(res):
            results.append((tool_call_info[i][0], tool_call_info[i][1], r))
        return results
