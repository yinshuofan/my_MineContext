# -*- coding: utf-8 -*-

"""
Unified Search API - Base Search Strategy
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from opencontext.server.search.models import TimeRange, TypedResults


class BaseSearchStrategy(ABC):
    """Abstract base class for search strategies"""

    @abstractmethod
    async def search(
        self,
        query: str,
        context_types: List[str],
        top_k: int,
        time_range: Optional[TimeRange],
        user_id: Optional[str],
        device_id: Optional[str],
        agent_id: Optional[str],
    ) -> TypedResults:
        """
        Execute search and return typed results.

        Args:
            query: Natural language search query
            context_types: List of context types to search
            top_k: Maximum results per type
            time_range: Optional time range filter
            user_id: User identifier for multi-user filtering
            device_id: Device identifier for multi-user filtering
            agent_id: Agent identifier for multi-user filtering

        Returns:
            TypedResults with results grouped by context type
        """
        pass
