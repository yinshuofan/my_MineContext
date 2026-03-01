#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge merge strategy — the only remaining merge strategy after refactor.

Only the KNOWLEDGE context type uses vector-based similarity merging.
- profile/entity: overwrite via relational DB (handled in context_operations)
- document: delete-old + insert-new (handled in context_operations)
- event: immutable append (no merging)
- knowledge: append + merge similar (this module)
"""

import math
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from opencontext.models.context import ExtractedData, ProcessedContext
from opencontext.models.enums import ContextType
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ContextTypeAwareStrategy(ABC):
    """
    Context type-aware merge strategy base class.
    Retained as base class for KnowledgeMergeStrategy.
    """

    def __init__(self, config: dict):
        self.config = config
        self.context_type = self.get_context_type()
        self.similarity_threshold = config.get(
            f"{self.context_type.value}_similarity_threshold", 0.8
        )
        self.retention_days = config.get(f"{self.context_type.value}_retention_days", 30)
        self.max_merge_count = config.get(f"{self.context_type.value}_max_merge_count", 3)

    @abstractmethod
    def get_context_type(self) -> ContextType:
        """Return the context type handled by this strategy"""
        pass

    @abstractmethod
    def can_merge(self, target: ProcessedContext, source: ProcessedContext) -> Tuple[bool, float]:
        """
        Determine if two contexts can be merged, return (can_merge, similarity_score)
        """
        pass

    @abstractmethod
    def merge_contexts(
        self, target: ProcessedContext, sources: List[ProcessedContext]
    ) -> Optional[ProcessedContext]:
        """
        Execute specific merge logic, return merged context
        """
        pass

    def calculate_forgetting_probability(self, context: ProcessedContext) -> float:
        """
        Calculate forgetting probability based on forgetting curve
        """
        if not context.properties.update_time:
            return 0.0

        age_days = (datetime.now() - context.properties.update_time).days

        tau = self.retention_days / 3
        base_forgetting = 1.0 - math.exp(-age_days / tau)

        importance_factor = (10 - context.extracted_data.importance) / 10.0
        access_factor = 1.0 / (1 + context.properties.merge_count)

        forgetting_prob = base_forgetting * importance_factor * access_factor
        return min(forgetting_prob, 0.95)

    def should_cleanup(self, context: ProcessedContext) -> bool:
        """
        Determine if this context should be cleaned up
        """
        if not context.properties.update_time:
            return False

        age_days = (datetime.now() - context.properties.update_time).days

        if age_days > self.retention_days and context.extracted_data.importance < 5:
            return True

        import random

        forgetting_prob = self.calculate_forgetting_probability(context)
        return random.random() < forgetting_prob


class KnowledgeMergeStrategy(ContextTypeAwareStrategy):
    """
    Knowledge context merge strategy.

    Merges similar knowledge entries based on:
    1. Vector similarity > threshold (default 0.8)
    2. Keyword overlap > 0.3
    """

    def get_context_type(self) -> ContextType:
        return ContextType.KNOWLEDGE

    def can_merge(self, target: ProcessedContext, source: ProcessedContext) -> Tuple[bool, float]:
        """
        Knowledge merge criteria:
        1. High keyword overlap (same domain)
        2. Vector similarity above threshold
        """
        # Keyword overlap check
        target_keywords = set(target.extracted_data.keywords)
        source_keywords = set(source.extracted_data.keywords)

        keyword_overlap = 0.0
        if target_keywords and source_keywords:
            keyword_overlap = len(target_keywords.intersection(source_keywords)) / len(
                target_keywords.union(source_keywords)
            )

        # Entity overlap check
        target_entities = set(target.extracted_data.entities)
        source_entities = set(source.extracted_data.entities)

        entity_overlap = 0.0
        if target_entities and source_entities:
            entity_overlap = len(target_entities.intersection(source_entities)) / len(
                target_entities.union(source_entities)
            )

        # Need some keyword or entity overlap
        if keyword_overlap < 0.3 and entity_overlap < 0.3:
            return False, 0.0

        # Vector similarity check
        if target.vectorize and source.vectorize:
            vector_sim = self._calculate_cosine_similarity(
                target.vectorize.vector, source.vectorize.vector
            )

            if vector_sim > self.similarity_threshold:
                final_score = (entity_overlap * 0.3) + (keyword_overlap * 0.3) + (vector_sim * 0.4)
                return True, final_score

        return False, 0.0

    def merge_contexts(
        self, target: ProcessedContext, sources: List[ProcessedContext]
    ) -> Optional[ProcessedContext]:
        """
        Knowledge merge logic: deduplicate and combine knowledge entries.
        """
        from collections import Counter

        all_contexts = [target] + sources

        # Merge entities and keywords by frequency
        all_entities = [e for ctx in all_contexts for e in ctx.extracted_data.entities]
        entity_counter = Counter(all_entities)
        merged_entities = [e for e, _ in entity_counter.most_common(10)]

        all_keywords = [k for ctx in all_contexts for k in ctx.extracted_data.keywords]
        keyword_counter = Counter(all_keywords)
        merged_keywords = [k for k, _ in keyword_counter.most_common(10)]

        # Title and summary: use the most important context's title, combine summaries
        sorted_contexts = sorted(
            all_contexts, key=lambda x: x.extracted_data.importance, reverse=True
        )
        merged_title = sorted_contexts[0].extracted_data.title

        if len(all_contexts) == 1:
            merged_summary = all_contexts[0].extracted_data.summary
        else:
            # Keep the highest-importance summary as the base
            merged_summary = sorted_contexts[0].extracted_data.summary

        merged_importance = min(max(ctx.extracted_data.importance for ctx in all_contexts), 10)
        merged_confidence = min(
            sum(ctx.extracted_data.confidence for ctx in all_contexts) // len(all_contexts), 10
        )

        return self._create_merged_context(
            target,
            sources,
            {
                "title": merged_title,
                "summary": merged_summary,
                "entities": merged_entities,
                "keywords": merged_keywords,
                "importance": merged_importance,
                "confidence": merged_confidence,
            },
        )

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _create_merged_context(
        self, target: ProcessedContext, sources: List[ProcessedContext], merged_data: Dict[str, Any]
    ) -> ProcessedContext:
        """Create merged context object"""
        from opencontext.models.context import ContextProperties, Vectorize

        extracted_data = ExtractedData(
            title=merged_data["title"],
            summary=merged_data["summary"],
            keywords=merged_data["keywords"],
            entities=merged_data["entities"],
            context_type=ContextType.KNOWLEDGE,
            confidence=merged_data["confidence"],
            importance=merged_data["importance"],
        )

        properties = ContextProperties(
            raw_properties=target.properties.raw_properties,
            create_time=target.properties.create_time,
            event_time=target.properties.event_time,
            is_processed=True,
            has_compression=True,
            update_time=datetime.now(),
            merge_count=target.properties.merge_count + len(sources),
            duration_count=target.properties.duration_count
            + sum(s.properties.duration_count for s in sources),
            enable_merge=True,
        )

        return ProcessedContext(
            extracted_data=extracted_data,
            properties=properties,
            vectorize=Vectorize(text=extracted_data.title + " " + extracted_data.summary),
        )


class StrategyFactory:
    """Merge strategy factory — only knowledge strategy remains"""

    def __init__(self, config: dict):
        self.config = config
        self._strategies = {
            ContextType.KNOWLEDGE: KnowledgeMergeStrategy(config),
        }

    def get_supported_types(self) -> List[ContextType]:
        """Get all supported context types"""
        return list(self._strategies.keys())

    def get_strategy(self, context_type: ContextType) -> Optional[ContextTypeAwareStrategy]:
        """Get the merge strategy for a specific context type"""
        return self._strategies.get(context_type)
