# -*- coding: utf-8 -*-

"""
Unified Search API
"""

from opencontext.server.search.base_strategy import BaseSearchStrategy
from opencontext.server.search.fast_strategy import FastSearchStrategy
from opencontext.server.search.intelligent_strategy import IntelligentSearchStrategy
from opencontext.server.search.models import (
    EntityResult,
    ProfileResult,
    SearchMetadata,
    SearchStrategy,
    TimeRange,
    TypedResults,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
    VectorResult,
)

__all__ = [
    "BaseSearchStrategy",
    "FastSearchStrategy",
    "IntelligentSearchStrategy",
    "SearchStrategy",
    "TimeRange",
    "UnifiedSearchRequest",
    "UnifiedSearchResponse",
    "TypedResults",
    "ProfileResult",
    "EntityResult",
    "VectorResult",
    "SearchMetadata",
]
