"""
User Memory Cache — Request/Response Models
"""

from typing import Any

from pydantic import BaseModel, Field


class SimpleProfile(BaseModel):
    """Simplified profile for cache response (no user_id/device_id/agent_id)."""

    factual_profile: str
    behavioral_profile: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecentlyAccessedItem(BaseModel):
    """A memory that was recently returned in search results."""

    id: str
    title: str | None = None
    summary: str | None = None
    context_type: str
    keywords: list[str] = Field(default_factory=list)
    accessed_ts: float  # Unix timestamp of last access
    score: float | None = None  # Last search relevance score
    event_time_start: str | None = None
    create_time: str | None = None
    media_refs: list[dict[str, Any]] = Field(default_factory=list)


class RecentMemoryItem(BaseModel):
    """A memory created within the recent time window (internal use for snapshot building)."""

    id: str
    title: str | None = None
    summary: str | None = None
    context_type: str
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    importance: int = 0
    create_time: str | None = None
    event_time_start: str | None = None


class DailySummaryItem(BaseModel):
    """An L1 daily summary for a specific day (internal use for snapshot building)."""

    id: str
    event_time_start: str  # e.g. "2026-02-21T00:00:00"
    summary: str | None = None
    children_count: int = 0


class SimpleDailySummary(BaseModel):
    """Simplified daily summary for cache response."""

    event_time_start: str
    title: str | None = None
    summary: str | None = None


class SimpleTodayEvent(BaseModel):
    """Simplified today event for cache response."""

    title: str | None = None
    summary: str | None = None
    event_time_start: str | None = None


class UserMemoryCacheResponse(BaseModel):
    """Response for GET /api/memory-cache"""

    success: bool
    user_id: str
    device_id: str = "default"
    agent_id: str
    profile: SimpleProfile | None = None
    agent_prompt: SimpleProfile | None = None
    recently_accessed: list[RecentlyAccessedItem] | None = None
    daily_summaries: list[SimpleDailySummary] | None = None
    today_events: list[SimpleTodayEvent] | None = None
