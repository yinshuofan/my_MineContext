# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Document context retrieval tool
Retrieves document and file content chunks from the vector DB
"""

from opencontext.models.enums import ContextType
from opencontext.tools.retrieval_tools.base_context_retrieval_tool import BaseContextRetrievalTool


class DocumentRetrievalTool(BaseContextRetrievalTool):
    """
    Document context retrieval tool

    Retrieves document and file content chunks including:
    - Uploaded document content (PDF, Word, text files, etc.)
    - Local file content indexed from monitored folders
    - Web link content fetched and processed from URLs
    - Document chunks with semantic embeddings for similarity search

    Supports both semantic search (with query) and filter-based retrieval (without query)
    """

    CONTEXT_TYPE = ContextType.DOCUMENT

    @classmethod
    def get_name(cls) -> str:
        """Get tool name"""
        return "retrieve_document_context"

    @classmethod
    def get_description(cls) -> str:
        """Get tool description"""
        return """Retrieve document and file content chunks from the vector DB.

**What this tool retrieves:**
- Content chunks from uploaded documents (PDF, Word, Markdown, text files, etc.)
- Indexed content from local files and monitored folders
- Fetched and processed content from web links and URLs
- Semantically embedded document passages for similarity-based retrieval
- File metadata including source paths, titles, and document structure

**When to use this tool:**
- When you need to find specific information within uploaded documents or files
- When searching for content from previously ingested web pages or articles
- When looking for passages or sections relevant to a particular topic
- When you want to retrieve document chunks that semantically match a query
- When needing to locate references, definitions, or details stored in files

**Two modes of operation:**
1. **With query** (semantic search): Provide a natural language query to find semantically relevant document chunks
   - Example: "API authentication requirements", "project budget estimates", "deployment instructions"
2. **Without query** (filter-only): Retrieve document chunks based on time range and/or entities
   - Example: Get all documents added this week, or documents mentioning specific topics

**Filter options:**
- Time range filtering (by event_time, create_time, or update_time)
- Entity filtering (find documents mentioning specific people, projects, technologies)
- Configurable result count (top_k: 1-100, default 5)

**Best for:**
- Searching through uploaded documents and files for relevant content
- Finding information from web links and online resources
- Retrieving document passages that match a semantic query
- Locating specific details across a large collection of ingested documents"""

    @classmethod
    def get_parameters(cls):
        """Get tool parameters with document-specific descriptions"""
        base_params = super().get_parameters()

        # Customize query description for document context
        base_params["properties"]["query"]["description"] = (
            "Natural language query for semantic search of document content chunks. "
            "Examples: 'API rate limiting policy', 'quarterly revenue figures', "
            "'installation prerequisites for the backend service'. "
            "Leave empty to perform filter-only retrieval."
        )

        return base_params
