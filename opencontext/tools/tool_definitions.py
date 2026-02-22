"""
Tool: tool_definitions
"""

from opencontext.tools.operation_tools import *
from opencontext.tools.profile_tools import *
from opencontext.tools.retrieval_tools import *

# Context retrieval tools (vector DB-based)
CONTEXT_RETRIEVAL_TOOLS = [
    {"type": "function", "function": DocumentRetrievalTool.get_definition()},
    {"type": "function", "function": KnowledgeRetrievalTool.get_definition()},
    {"type": "function", "function": HierarchicalEventTool.get_definition()},
]

# Profile retrieval tools (relational DB-based)
ALL_PROFILE_TOOL_DEFINITIONS = [
    {"type": "function", "function": ProfileRetrievalTool.get_definition()},
    {"type": "function", "function": ProfileEntityTool.get_definition()},
]

WEB_SEARCH_TOOL_DEFINITION = [
    {"type": "function", "function": WebSearchTool.get_definition()},
]

ALL_RETRIEVAL_TOOL_DEFINITIONS = CONTEXT_RETRIEVAL_TOOLS

ALL_TOOL_DEFINITIONS = (
    ALL_RETRIEVAL_TOOL_DEFINITIONS + ALL_PROFILE_TOOL_DEFINITIONS + WEB_SEARCH_TOOL_DEFINITION
)
