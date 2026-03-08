# -*- coding: utf-8 -*-

"""
Event Search API - Request/Response Models
"""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class TimeRange(BaseModel):
    """Time range filter"""

    start: Optional[int] = Field(default=None, description="Start timestamp in Unix epoch seconds")
    end: Optional[int] = Field(default=None, description="End timestamp in Unix epoch seconds")

    @model_validator(mode="after")
    def validate_range(self) -> "TimeRange":
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")
        return self


class EventSearchRequest(BaseModel):
    """Event search API request"""

    query: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Semantic search query text. At least one of query/event_ids/time_range/hierarchy_levels must be provided.",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Optional image URL for multimodal search. Can be an HTTP URL or data:image/...;base64,... string.",
    )
    video_url: Optional[str] = Field(
        default=None,
        description="Optional video URL for multimodal search. Can be an HTTP URL or data:video/...;base64,... string.",
    )
    event_ids: Optional[List[str]] = Field(
        default=None,
        description="Exact event IDs to retrieve",
    )
    time_range: Optional[TimeRange] = Field(
        default=None,
        description="Time range filter (Unix epoch seconds)",
    )
    hierarchy_levels: Optional[List[int]] = Field(
        default=None,
        description="Filter by hierarchy levels: 0=raw events, 1=daily, 2=weekly, 3=monthly",
    )
    drill_up: bool = Field(
        default=True,
        description="Whether to recursively fetch ancestor summaries for each result",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results",
    )
    score_threshold: Optional[float] = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1). Results below this score are filtered out.",
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

    @model_validator(mode="after")
    def validate_search_criteria(self) -> "EventSearchRequest":
        has_query = self.query is not None and len(self.query.strip()) > 0
        has_image = self.image_url is not None and len(self.image_url.strip()) > 0
        has_video = self.video_url is not None and len(self.video_url.strip()) > 0
        has_ids = self.event_ids is not None and len(self.event_ids) > 0
        has_time = self.time_range is not None
        has_levels = self.hierarchy_levels is not None and len(self.hierarchy_levels) > 0
        if not (has_query or has_image or has_video or has_ids or has_time or has_levels):
            raise ValueError(
                "At least one of query, image_url, video_url, event_ids, time_range, "
                "or hierarchy_levels must be provided"
            )
        return self


# ── Response Models ──


class EventNode(BaseModel):
    """
    A node in the event hierarchy tree.

    All fields live on this single model to avoid Pydantic V2's polymorphic
    serialization issue (subclass fields silently dropped when serialized
    through a List[BaseClass] annotation). Search-hit fields (score,
    keywords, entities) are only populated when is_search_hit=True.
    """

    id: str
    hierarchy_level: int = 0
    time_bucket: Optional[str] = None
    parent_id: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    event_time: Optional[str] = None
    create_time: Optional[str] = None
    is_search_hit: bool = False
    children: List["EventNode"] = Field(default_factory=list)

    # Search-hit fields (populated only when is_search_hit=True)
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    score: Optional[float] = None


class SearchMetadata(BaseModel):
    """Metadata about the search execution"""

    query: Optional[str] = None
    total_results: int
    search_time_ms: float


class EventSearchResponse(BaseModel):
    """Event search API response"""

    success: bool
    events: List[EventNode] = Field(default_factory=list)
    metadata: SearchMetadata
