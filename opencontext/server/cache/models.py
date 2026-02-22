# -*- coding: utf-8 -*-

"""
User Memory Cache — Request/Response Models
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from opencontext.server.search.models import EntityResult, ProfileResult


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
    """A memory created within the recent time window."""

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
    """An L1 daily summary for a specific day."""

    id: str
    time_bucket: str  # e.g. "2026-02-21"
    summary: Optional[str] = None
    children_count: int = 0


class RecentMemories(BaseModel):
    """Hierarchical recent memories to prevent token explosion."""

    today_events: List[RecentMemoryItem] = Field(default_factory=list)
    daily_summaries: List[DailySummaryItem] = Field(default_factory=list)
    recent_documents: List[RecentMemoryItem] = Field(default_factory=list)
    recent_knowledge: List[RecentMemoryItem] = Field(default_factory=list)


class UserMemoryCacheResponse(BaseModel):
    """Response for GET /api/memory-cache"""

    success: bool
    user_id: str
    agent_id: str
    profile: Optional[ProfileResult] = None
    entities: List[EntityResult] = Field(default_factory=list)
    recently_accessed: List[RecentlyAccessedItem] = Field(default_factory=list)
    recent_memories: RecentMemories = Field(default_factory=RecentMemories)
    cache_metadata: Dict[str, Any] = Field(default_factory=dict)
