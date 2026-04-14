"""
Event Search API - Request/Response Models
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TimeRange(BaseModel):
    """Time range filter"""

    start: int | None = Field(default=None, description="Start timestamp in Unix epoch seconds")
    end: int | None = Field(default=None, description="End timestamp in Unix epoch seconds")

    @model_validator(mode="after")
    def validate_range(self) -> "TimeRange":
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")
        return self


class EventSearchRequest(BaseModel):
    """Event search API request"""

    query: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Multimodal search query in OpenAI content parts format. "
            'Example: [{"type": "text", "text": "..."}, '
            '{"type": "image_url", "image_url": {"url": "..."}}]'
        ),
    )
    event_ids: list[str] | None = Field(
        default=None,
        description="Exact event IDs to retrieve",
    )
    time_range: TimeRange | None = Field(
        default=None,
        description="Time range filter (Unix epoch seconds)",
    )
    hierarchy_levels: list[int] | None = Field(
        default=None,
        description="Filter by hierarchy levels: 0=raw events, 1=daily, 2=weekly, 3=monthly",
    )
    drill: Literal["none", "up", "down", "both"] = Field(
        default="up",
        description=(
            "Drill direction for hierarchy traversal: "
            "'none' (no traversal), 'up' (ancestors only), "
            "'down' (descendants only), 'both' (ancestors + descendants)"
        ),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score (0-1). Results below this "
            "score are filtered out. Default: no threshold."
        ),
    )
    user_id: str | None = Field(
        default=None, description="User identifier for multi-user filtering"
    )
    device_id: str | None = Field(
        default=None, description="Device identifier for multi-user filtering"
    )
    agent_id: str | None = Field(
        default=None, description="Agent identifier for multi-user filtering"
    )

    @model_validator(mode="after")
    def validate_search_criteria(self) -> "EventSearchRequest":
        has_query = bool(self.query)
        has_ids = bool(self.event_ids)
        has_time = self.time_range is not None
        has_levels = bool(self.hierarchy_levels)
        if not any([has_query, has_ids, has_time, has_levels]):
            raise ValueError(
                "At least one search criterion required: "
                "query, event_ids, time_range, or hierarchy_levels"
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
    refs: dict[str, list[str]] = Field(default_factory=dict)
    title: str | None = None
    summary: str | None = None
    event_time_start: str | None = None
    event_time_end: str | None = None
    create_time: str | None = None
    is_search_hit: bool = False
    children: list["EventNode"] = Field(default_factory=list)

    # Multimodal media references (L0 events carry media, summaries have empty list)
    media_refs: list[dict[str, Any]] = Field(default_factory=list)

    # Agent commentary (populated when agent annotated this event)
    agent_commentary: str | None = None

    # Search-hit fields (populated only when is_search_hit=True)
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    score: float | None = None


class SearchMetadata(BaseModel):
    """Metadata about the search execution"""

    query: str | None = None
    total_results: int
    search_time_ms: float


class EventSearchResponse(BaseModel):
    """Event search API response"""

    success: bool
    events: list[EventNode] = Field(default_factory=list)
    metadata: SearchMetadata
