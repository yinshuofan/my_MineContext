# -*- coding: utf-8 -*-

"""
User Memory Cache — Request/Response Models
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SimpleProfile(BaseModel):
    """Simplified profile for cache response (no user_id/device_id/agent_id/summary)."""

    content: str
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecentlyAccessedItem(BaseModel):
    """A memory that was recently returned in search results."""

    id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    context_type: str
    keywords: List[str] = Field(default_factory=list)
    accessed_ts: float  # Unix timestamp of last access
    score: Optional[float] = None  # Last search relevance score
    event_time: Optional[str] = None
    create_time: Optional[str] = None


class RecentMemoryItem(BaseModel):
    """A memory created within the recent time window (internal use for snapshot building)."""

    id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    context_type: str
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    importance: int = 0
    create_time: Optional[str] = None
    event_time: Optional[str] = None


class DailySummaryItem(BaseModel):
    """An L1 daily summary for a specific day (internal use for snapshot building)."""

    id: str
    time_bucket: str  # e.g. "2026-02-21"
    summary: Optional[str] = None
    children_count: int = 0


class SimpleDailySummary(BaseModel):
    """Simplified daily summary for cache response."""

    time_bucket: str
    summary: Optional[str] = None


class SimpleTodayEvent(BaseModel):
    """Simplified today event for cache response."""

    title: Optional[str] = None
    summary: Optional[str] = None
    event_time: Optional[str] = None


class UserMemoryCacheResponse(BaseModel):
    """Response for GET /api/memory-cache"""

    success: bool
    user_id: str
    device_id: str = "default"
    agent_id: str
    profile: Optional[SimpleProfile] = None
    recently_accessed: List[RecentlyAccessedItem] = Field(default_factory=list)
    daily_summaries: List[SimpleDailySummary] = Field(default_factory=list)
    today_events: List[SimpleTodayEvent] = Field(default_factory=list)
