# -*- coding: utf-8 -*-

"""
Unified Search API - Request/Response Models
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class SearchStrategy(str, Enum):
    """Search strategy enumeration"""

    FAST = "fast"  # Direct parallel search, zero LLM reasoning calls
    INTELLIGENT = "intelligent"  # LLM-driven agentic search with tool selection and validation


class TimeRange(BaseModel):
    """Time range filter"""

    start: Optional[int] = Field(default=None, description="Start timestamp in Unix epoch seconds")
    end: Optional[int] = Field(default=None, description="End timestamp in Unix epoch seconds")

    @model_validator(mode="after")
    def validate_range(self) -> "TimeRange":
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")
        return self


class UnifiedSearchRequest(BaseModel):
    """Unified search API request"""

    query: str = Field(..., description="Natural language search query")
    strategy: SearchStrategy = Field(
        default=SearchStrategy.FAST,
        description="Search strategy: 'fast' for direct parallel search, 'intelligent' for LLM-driven agentic search",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum results per context type",
    )
    context_types: Optional[List[str]] = Field(
        default=None,
        description="Context types to search. None means all 5 types. "
        "Values: profile, entity, document, event, knowledge",
    )
    time_range: Optional[TimeRange] = Field(
        default=None,
        description="Optional time range filter (Unix epoch seconds)",
    )
    user_id: Optional[str] = Field(
        default=None, description="User identifier for multi-user filtering"
    )
    device_id: Optional[str] = Field(
        default=None, description="Device identifier for multi-user filtering"
    )
    agent_id: Optional[str] = Field(
        default=None, description="Agent identifier for multi-user filtering"
    )


# ── Response Models ──


class ProfileResult(BaseModel):
    """Profile search result from relational DB"""

    user_id: str
    agent_id: str
    content: str
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityResult(BaseModel):
    """Entity search result from relational DB"""

    id: str
    entity_name: str
    entity_type: Optional[str] = None
    content: str
    summary: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    score: float = 1.0


class VectorResult(BaseModel):
    """Vector search result (shared by document/event/knowledge types)"""

    id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None  # get_llm_context_string() output
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    context_type: str
    score: float
    create_time: Optional[str] = None
    event_time: Optional[str] = None

    # Hierarchy info (event type only)
    hierarchy_level: int = 0
    time_bucket: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    parent_summary: Optional[str] = None  # Parent summary attached by fast search

    # Document source tracking
    source_file_key: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


class TypedResults(BaseModel):
    """Search results grouped by context type"""

    profile: Optional[ProfileResult] = None
    entities: List[EntityResult] = Field(default_factory=list)
    documents: List[VectorResult] = Field(default_factory=list)
    events: List[VectorResult] = Field(default_factory=list)
    knowledge: List[VectorResult] = Field(default_factory=list)


class SearchMetadata(BaseModel):
    """Metadata about the search execution"""

    strategy: str
    query: str
    total_results: int
    search_time_ms: float
    types_searched: List[str]


class UnifiedSearchResponse(BaseModel):
    """Unified search API response"""

    success: bool
    results: TypedResults = Field(default_factory=TypedResults)
    metadata: SearchMetadata
