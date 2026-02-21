#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Context merge processor — simplified to only handle KNOWLEDGE type merging.

After the type system refactor:
- profile/entity: overwrite via relational DB (no merging needed)
- document: delete-old + insert-new (no merging needed)
- event: immutable append (no merging needed)
- knowledge: similarity-based merge (this module)
"""
import datetime
import json
import math
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from opencontext.config import GlobalConfig
from opencontext.context_processing.merger.merge_strategies import (
    ContextTypeAwareStrategy,
    StrategyFactory,
)
from opencontext.context_processing.processor.base_processor import BaseContextProcessor
from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.context import *
from opencontext.models.enums import ContextType
from opencontext.storage.global_storage import get_storage
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ContextMerger(BaseContextProcessor):
    def __init__(self):
        from opencontext.config.global_config import get_config, get_prompt_manager

        config = get_config("processing.context_merger") or {}
        super().__init__(config)
        self.enable_memory_management = config.get("enable_memory_management", False)
        self.prompt_manager = get_prompt_manager()
        self._similarity_threshold = config.get("similarity_threshold", 0.85)
        self._statistics = {"merges_attempted": 0, "merges_succeeded": 0, "errors": 0}

        # Initialize strategy management (only knowledge strategy)
        self.strategies: Dict[ContextType, ContextTypeAwareStrategy] = {}
        self._initialize_strategies()

        # Intelligent merging switch
        self.use_intelligent_merging = config.get("use_intelligent_merging", True)

    @property
    def storage(self):
        """Get storage from global singleton"""
        return get_storage()

    def get_name(self) -> str:
        return "merger"

    def get_description(self) -> str:
        return "Merges similar knowledge contexts using intelligent strategies."

    def _initialize_strategies(self):
        """Initialize merge strategies — only knowledge type"""
        strategy_factory = StrategyFactory(self.config)

        supported_types = strategy_factory.get_supported_types()
        for context_type in supported_types:
            strategy = strategy_factory.get_strategy(context_type)
            if strategy:
                self.strategies[context_type] = strategy
                logger.info(f"Initialized merge strategy for {context_type.value}")

        logger.info(f"Initialized {len(self.strategies)} context merge strategies")

    def process(self, context: ProcessedContext) -> List[ProcessedContext]:
        return []

    def can_process(self, context: ProcessedContext) -> bool:
        """Only knowledge type contexts can be merged"""
        return (
            context.extracted_data.summary is not None
            and context.extracted_data.context_type == ContextType.KNOWLEDGE
        )

    def find_merge_target(self, context: ProcessedContext) -> Optional[ProcessedContext]:
        """
        Find merge target for knowledge contexts using intelligent strategies.
        Only processes KNOWLEDGE type — other types skip merging entirely.
        """
        if not context.properties.enable_merge:
            return None

        # Only knowledge type goes through merge
        if context.extracted_data.context_type != ContextType.KNOWLEDGE:
            return None

        # Check for vectorization
        if not context.vectorize:
            return None
        do_vectorize(context.vectorize)

        context_type = context.extracted_data.context_type

        # Use intelligent merging strategy
        if self.use_intelligent_merging and context_type in self.strategies:
            return self._find_intelligent_merge_target(context)

        # Fallback to legacy logic
        return self._find_legacy_merge_target(context)

    def _find_intelligent_merge_target(
        self, context: ProcessedContext
    ) -> Optional[ProcessedContext]:
        """Find merge target using intelligent strategies"""
        context_type = context.extracted_data.context_type
        strategy = self.strategies.get(context_type)

        if not strategy:
            logger.warning(f"No strategy found for context type {context_type.value}")
            return None

        try:
            candidates = self._get_merge_candidates(context)

            best_target = None
            best_score = 0.0

            for candidate in candidates:
                if candidate.id == context.id:
                    continue

                can_merge, score = strategy.can_merge(candidate, context)
                if can_merge and score > best_score:
                    best_target = candidate
                    best_score = score

            if best_target:
                logger.info(
                    f"Found intelligent merge target {best_target.id} for {context.id} "
                    f"with score {best_score:.3f}"
                )

            return best_target

        except Exception as e:
            logger.error(f"Error in intelligent merge target finding: {e}", exc_info=True)
            return None

    def _get_merge_candidates(
        self, context: ProcessedContext, max_candidates: int = 10
    ) -> List[ProcessedContext]:
        """Get merge candidate contexts"""
        try:
            backend = self.storage._get_or_create_backend(context.extracted_data.context_type.value)
            if not backend:
                return []

            similar_results = backend.query(
                query=Vectorize(vector=context.vectorize.vector),
                top_k=max_candidates + 1,
                filters={},
            )

            candidates = [result[0] for result in similar_results if result[0].id != context.id]
            return candidates[:max_candidates]

        except Exception as e:
            logger.error(f"Error getting merge candidates: {e}", exc_info=True)
            return []

    def _find_legacy_merge_target(self, context: ProcessedContext) -> Optional[ProcessedContext]:
        """Find merge target using legacy similarity logic"""
        sim_target, _ = self._find_similarity_merge_target(context)
        return sim_target

    def _find_similarity_merge_target(
        self, context: ProcessedContext
    ) -> tuple[Optional[ProcessedContext], float]:
        try:
            backend = self.storage._get_or_create_backend(context.extracted_data.context_type.value)
            similar_results = backend.query(
                query=Vectorize(vector=context.vectorize.vector), top_k=2, filters={}
            )
            similar_results = [res for res in similar_results if res[0].id != context.id]

            if similar_results:
                most_similar_context, score = similar_results[0]
                logger.info(
                    f"Most similar context to {context.id} is {most_similar_context.id} "
                    f"with score: {score}"
                )
                if score > self._similarity_threshold:
                    return most_similar_context, score
        except Exception as e:
            logger.error(
                f"Error querying for similar contexts for target {context.id}: {e}", exc_info=True
            )

        return None, 0.0

    def merge_multiple(
        self, target: ProcessedContext, sources: List[ProcessedContext]
    ) -> Optional[ProcessedContext]:
        """
        Merges multiple source contexts into a target context.
        """
        if not sources:
            return target

        self._statistics["merges_attempted"] += 1

        try:
            context_type = target.extracted_data.context_type

            # Use intelligent merging strategy
            if self.use_intelligent_merging and context_type in self.strategies:
                merged_context = self._merge_with_intelligent_strategy(target, sources)
            else:
                merged_context = self._merge_with_llm(target, sources)

            if merged_context:
                self._statistics["merges_succeeded"] += 1
                return merged_context
            else:
                self._statistics["errors"] += 1
                logger.warning(f"Merging failed for target {target.id}. Returning None.")
                return None

        except Exception as e:
            logger.error(
                f"An exception occurred during merging for target {target.id}: {e}", exc_info=True
            )
            self._statistics["errors"] += 1
            return None

    def _merge_with_intelligent_strategy(
        self, target: ProcessedContext, sources: List[ProcessedContext]
    ) -> Optional[ProcessedContext]:
        """Merge using intelligent strategies"""
        context_type = target.extracted_data.context_type
        strategy = self.strategies.get(context_type)

        if not strategy:
            logger.warning(
                f"No strategy found for context type {context_type.value}, "
                f"falling back to LLM merge"
            )
            return self._merge_with_llm(target, sources)

        try:
            merged_context = strategy.merge_contexts(target, sources)
            if merged_context:
                logger.info(
                    f"Successfully merged using {context_type.value} strategy: "
                    f"{len(sources)} sources into target {target.id}"
                )
            return merged_context

        except Exception as e:
            logger.error(f"Error in intelligent strategy merge: {e}", exc_info=True)
            logger.info("Falling back to LLM-based merge")
            return self._merge_with_llm(target, sources)

    def _merge_with_llm(
        self, target: ProcessedContext, sources: List[ProcessedContext]
    ) -> Optional[ProcessedContext]:
        """
        Uses an LLM to merge a list of source contexts into a target context.
        """
        try:
            context_type = target.extracted_data.context_type
            type_specific_prompt = f"merging.{context_type.value}_merging"

            prompt_group = self.prompt_manager.get_prompt_group(type_specific_prompt)
            if prompt_group and "user" in prompt_group:
                prompt_name = type_specific_prompt
                logger.info(f"Using type-specific prompt: {prompt_name}")
            else:
                prompt_name = "merging.context_merging_multiple"
                prompt_group = self.prompt_manager.get_prompt_group(prompt_name)
                logger.info(f"Using generic prompt: {prompt_name}")

            if not prompt_group or "user" not in prompt_group:
                logger.error(f"User prompt for '{prompt_name}' not found.")
                return None

            user_template = prompt_group["user"]
            system_prompt = prompt_group.get("system")

            target_context_json = target.extracted_data.model_dump_json(
                exclude={"context_type"}, indent=2
            )
            source_contexts_json = json.dumps(
                [s.extracted_data.model_dump(exclude={"context_type"}) for s in sources],
                indent=2,
                ensure_ascii=False,
            )
            user_prompt = user_template.format(
                target_context_json=target_context_json, source_contexts_json=source_contexts_json
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = generate_with_messages(messages)
            if response:
                if "无需合并" in response:
                    logger.info(f"LLM indicated no merge needed for target {target.id}.")
                    return None

                try:
                    response_data = parse_json_from_response(response)
                    required_fields = [
                        "title",
                        "summary",
                        "keywords",
                        "entities",
                        "tags",
                        "importance",
                        "confidence",
                    ]
                    if not all(field in response_data for field in required_fields):
                        logger.warning(
                            f"LLM response is missing one or more required fields. "
                            f"Got: {response_data.keys()}"
                        )
                        return None
                    extracted_data = ExtractedData(
                        title=response_data["title"],
                        summary=response_data["summary"],
                        keywords=response_data["keywords"],
                        entities=response_data["entities"],
                        context_type=target.extracted_data.context_type,
                        confidence=int(response_data["confidence"]),
                        importance=int(response_data["importance"]),
                    )
                    now = datetime.datetime.now()
                    properties = ContextProperties(
                        raw_properties=target.properties.raw_properties,
                        create_time=target.properties.create_time,
                        event_time=target.properties.event_time,
                        is_processed=True,
                        has_compression=True,
                        update_time=now,
                        merge_count=target.properties.merge_count,
                        duration_count=target.properties.duration_count
                        + sum(s.properties.duration_count for s in sources),
                        enable_merge=True,
                    )
                    if (
                        response_data.get("event_time", "null") != "null"
                        and response_data.get("event_time") is not None
                    ):
                        try:
                            event_time_str = response_data.get("event_time")
                            if len(event_time_str) == 8 and ":" in event_time_str:
                                today = datetime.date.today().isoformat()
                                full_event_time_str = f"{today}T{event_time_str}"
                                properties.event_time = datetime.datetime.fromisoformat(
                                    full_event_time_str
                                )
                            else:
                                properties.event_time = datetime.datetime.fromisoformat(
                                    event_time_str
                                )
                        except ValueError:
                            logger.warning(f"Cannot parse event_time format: {event_time_str}")

                    merged_context = ProcessedContext(
                        extracted_data=extracted_data,
                        properties=properties,
                        vectorize=Vectorize(
                            text=extracted_data.title + " " + extracted_data.summary
                        ),
                    )
                    logger.info(
                        f"Successfully merged {len(sources)} sources into context {target.id}"
                    )
                    logger.debug(
                        f"sources:\n{json.dumps([s.extracted_data.summary for s in sources], ensure_ascii=False, indent=2)}"
                    )
                    logger.debug(f"target:\n  {target.extracted_data.summary}")
                    logger.debug(f"merged_context:\n  {merged_context.extracted_data.summary}")
                    return merged_context
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode LLM response as JSON: {response}")
                    return None
            else:
                logger.warning("LLM returned no response for merging.")
                return None
        except Exception as e:
            logger.exception(f"Error during LLM-based merging: {e}")
            return None

    def get_statistics(self):
        return self._statistics

    def periodic_memory_compression(self, interval_seconds: int):
        """
        Periodic memory compression — only for KNOWLEDGE type.
        """
        if interval_seconds <= 0:
            logger.warning("interval_seconds must be greater than 0.")
            return
        logger.info("Starting periodic memory compression...")
        try:
            filter = {
                "update_time_ts": {
                    "$gte": int(
                        (
                            datetime.datetime.now()
                            - timedelta(seconds=interval_seconds)
                            - timedelta(minutes=5)
                        ).timestamp()
                    ),
                    "$lte": int((datetime.datetime.now() - timedelta(minutes=5)).timestamp()),
                },
                "has_compression": False,
                "enable_merge": True,
            }
            limit = 1000
            offset = 0
            while True:
                contexts_by_backend = self.storage.get_all_processed_contexts(
                    limit=limit,
                    offset=offset,
                    filter=filter,
                    need_vector=True,
                    context_types=[ContextType.KNOWLEDGE.value],
                )

                if not any(contexts_by_backend.values()):
                    logger.info("No more recent knowledge contexts to process in this iteration.")
                    break
                has_merge = False
                for backend_name, backend_contexts in contexts_by_backend.items():
                    if len(backend_contexts) < 2:
                        continue
                    has_merge = True
                    logger.info(
                        f"Processing {len(backend_contexts)} knowledge contexts "
                        f"from backend '{backend_name}'."
                    )

                    groups = self._group_contexts_by_similarity(
                        backend_contexts, self._similarity_threshold
                    )

                    for group in groups:
                        if len(group) > 1:
                            group.sort(key=lambda c: c.properties.create_time)
                            target_candidate = group[-1]
                            sources = group[:-1]

                            merged_context = self.merge_multiple(target_candidate, sources)
                            if merged_context:
                                self.storage.upsert_processed_context(merged_context)
                                self.storage.delete_processed_context(
                                    target_candidate.id,
                                    target_candidate.extracted_data.context_type.value,
                                )
                                for ctx in sources:
                                    self.storage.delete_processed_context(
                                        ctx.id, ctx.extracted_data.context_type.value
                                    )
                                logger.info(
                                    f"Successfully merged within group and cleaned up "
                                    f"{len(sources)} source contexts."
                                )

                if not has_merge:
                    break
                offset += limit

            logger.info("Periodic memory compression finished.")
        except Exception as e:
            logger.exception(f"Error during periodic memory compression: {e}")

    def periodic_memory_compression_for_user(
        self,
        user_id: str,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        interval_seconds: int = 1800,
    ):
        """
        Periodic memory compression for a specific user — only KNOWLEDGE type.
        """
        if interval_seconds <= 0:
            logger.warning("interval_seconds must be greater than 0.")
            return

        logger.info(
            f"Starting periodic memory compression for user={user_id}, "
            f"device={device_id}, agent={agent_id}..."
        )

        try:
            filter = {
                "update_time_ts": {
                    "$gte": int(
                        (
                            datetime.datetime.now()
                            - timedelta(seconds=interval_seconds)
                            - timedelta(minutes=5)
                        ).timestamp()
                    ),
                    "$lte": int((datetime.datetime.now() - timedelta(minutes=5)).timestamp()),
                },
                "enable_merge": True,
                "user_id": user_id,
            }

            if device_id:
                filter["device_id"] = device_id
            if agent_id:
                filter["agent_id"] = agent_id

            limit = 1000
            offset = 0

            while True:
                contexts_by_backend = self.storage.get_all_processed_contexts(
                    limit=limit,
                    offset=offset,
                    filter=filter,
                    need_vector=True,
                    context_types=[ContextType.KNOWLEDGE.value],
                )

                if not any(contexts_by_backend.values()):
                    logger.info(
                        f"No more recent knowledge contexts to process "
                        f"for user={user_id} in this iteration."
                    )
                    break

                has_merge = False
                for backend_name, backend_contexts in contexts_by_backend.items():
                    if len(backend_contexts) < 2:
                        continue
                    has_merge = True
                    logger.info(
                        f"Processing {len(backend_contexts)} knowledge contexts "
                        f"from backend '{backend_name}' for user={user_id}."
                    )

                    groups = self._group_contexts_by_similarity(
                        backend_contexts, self._similarity_threshold
                    )

                    for group in groups:
                        if len(group) > 1:
                            group.sort(key=lambda c: c.properties.create_time)
                            target_candidate = group[-1]
                            sources = group[:-1]

                            merged_context = self.merge_multiple(target_candidate, sources)
                            if merged_context:
                                self.storage.upsert_processed_context(merged_context)
                                self.storage.delete_processed_context(
                                    target_candidate.id,
                                    target_candidate.extracted_data.context_type.value,
                                )
                                for ctx in sources:
                                    self.storage.delete_processed_context(
                                        ctx.id, ctx.extracted_data.context_type.value
                                    )
                                logger.info(
                                    f"Successfully merged within group and cleaned up "
                                    f"{len(sources)} source contexts for user={user_id}."
                                )

                if not has_merge:
                    break
                offset += limit

            logger.info(f"Periodic memory compression finished for user={user_id}.")
        except Exception as e:
            logger.exception(f"Error during periodic memory compression for user={user_id}: {e}")

    def _group_contexts_by_similarity(
        self, contexts: List[ProcessedContext], threshold: float
    ) -> List[List[ProcessedContext]]:
        """Greedily groups contexts by similarity based on their embeddings."""
        if not contexts:
            return []

        groups = []
        remaining_contexts = list(contexts)

        while remaining_contexts:
            seed = remaining_contexts.pop(0)
            new_group = [seed]

            to_remove = []
            seed_embedding = seed.vectorize.vector
            for i, ctx in enumerate(remaining_contexts):
                ctx_embedding = ctx.vectorize.vector
                if self._calculate_similarity(seed_embedding, ctx_embedding) > threshold:
                    new_group.append(ctx)
                    to_remove.append(i)

            for i in sorted(to_remove, reverse=True):
                remaining_contexts.pop(i)

            groups.append(new_group)

        return groups

    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculates cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None or not emb1 or not emb2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm_emb1 = math.sqrt(sum(a * a for a in emb1))
        norm_emb2 = math.sqrt(sum(b * b for b in emb2))

        if norm_emb1 == 0 or norm_emb2 == 0:
            return 0.0

        return dot_product / (norm_emb1 * norm_emb2)

    def intelligent_memory_cleanup(self):
        """
        Intelligent memory cleanup: based on forgetting curve and importance.
        Only applies to knowledge type.
        """
        if not self.enable_memory_management:
            return

        logger.info("Starting intelligent memory cleanup...")

        try:
            cleanup_stats = {"total_checked": 0, "cleaned_up": 0, "errors": 0}

            for context_type, strategy in self.strategies.items():
                type_stats = self._cleanup_contexts_by_type(context_type, strategy)
                cleanup_stats["total_checked"] += type_stats["checked"]
                cleanup_stats["cleaned_up"] += type_stats["cleaned"]
                cleanup_stats["errors"] += type_stats["errors"]

            logger.info(
                f"Memory cleanup completed. Checked: {cleanup_stats['total_checked']}, "
                f"Cleaned: {cleanup_stats['cleaned_up']}, Errors: {cleanup_stats['errors']}"
            )

        except Exception as e:
            logger.error(f"Error during intelligent memory cleanup: {e}", exc_info=True)

    def _cleanup_contexts_by_type(
        self, context_type: ContextType, strategy: ContextTypeAwareStrategy
    ) -> Dict[str, int]:
        """Clean up contexts by type"""
        stats = {"checked": 0, "cleaned": 0, "errors": 0}

        try:
            limit = 100
            offset = 0

            ids_to_delete: List[str] = []
            while True:
                contexts = self._get_contexts_for_cleanup(context_type.value, limit, offset)
                if not contexts:
                    break

                for context in contexts:
                    stats["checked"] += 1
                    try:
                        if strategy.should_cleanup(context):
                            ids_to_delete.append(context.id)
                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Error evaluating cleanup for context {context.id}: {e}")

                if len(contexts) < limit:
                    break
                offset += limit

            if ids_to_delete:
                try:
                    if self.storage.delete_batch_processed_contexts(
                        ids_to_delete, context_type.value
                    ):
                        stats["cleaned"] += len(ids_to_delete)
                        logger.info(
                            f"Batch cleaned {len(ids_to_delete)} contexts "
                            f"of type {context_type.value}"
                        )
                    else:
                        logger.error(
                            f"Batch delete failed for {len(ids_to_delete)} contexts "
                            f"of type {context_type.value}"
                        )
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Error during batch delete: {e}")

            logger.info(
                f"Cleanup for {context_type.value}: "
                f"checked {stats['checked']}, cleaned {stats['cleaned']}"
            )

        except Exception as e:
            logger.error(f"Error in cleanup for {context_type.value}: {e}", exc_info=True)
            stats["errors"] += 1

        return stats

    def _get_contexts_for_cleanup(
        self, context_type_value: str, limit: int, offset: int
    ) -> List[ProcessedContext]:
        """Get contexts that need cleanup checking"""
        try:
            contexts_dict = self.storage.get_all_processed_contexts(
                limit=limit, offset=offset, context_types=[context_type_value]
            )
            return contexts_dict.get(context_type_value, [])

        except Exception as e:
            logger.error(f"Error getting contexts for cleanup: {e}")
            return []

    def memory_reinforcement(self, context_ids: List[str]):
        """
        Memory reinforcement: reset forgetting state and boost importance.
        """
        if not context_ids:
            return

        logger.info(f"Starting memory reinforcement for {len(context_ids)} contexts")

        reinforced_count = 0

        for context_id in context_ids:
            try:
                context = self._find_context_by_id(context_id)
                if not context:
                    logger.warning(f"Context {context_id} not found for reinforcement")
                    continue

                reinforced_context = self._apply_memory_reinforcement(context)
                if reinforced_context:
                    self.storage.upsert_processed_context(reinforced_context)
                    reinforced_count += 1
                    logger.debug(f"Reinforced context {context_id}")

            except Exception as e:
                logger.error(f"Error reinforcing context {context_id}: {e}", exc_info=True)

        logger.info(
            f"Memory reinforcement completed: "
            f"{reinforced_count}/{len(context_ids)} contexts reinforced"
        )

    def _find_context_by_id(self, context_id: str) -> Optional[ProcessedContext]:
        """Find a context by ID across all backends"""
        try:
            for context_type in ContextType:
                backend = self.storage._get_or_create_backend(context_type.value)
                if backend:
                    contexts_dict = self.storage.get_all_processed_contexts(
                        limit=1000, offset=0, filter={}
                    )
                    all_contexts = []
                    for contexts_list in contexts_dict.values():
                        all_contexts.extend(contexts_list)

                    for context in all_contexts:
                        if context.id == context_id:
                            return context

            return None

        except Exception as e:
            logger.error(f"Error finding context {context_id}: {e}", exc_info=True)
            return None

    def _apply_memory_reinforcement(self, context: ProcessedContext) -> Optional[ProcessedContext]:
        """Apply memory reinforcement logic"""
        from datetime import datetime as dt

        try:
            reinforced_properties = ContextProperties(
                raw_properties=context.properties.raw_properties,
                create_time=context.properties.create_time,
                event_time=context.properties.event_time,
                is_processed=context.properties.is_processed,
                has_compression=context.properties.has_compression,
                update_time=dt.now(),
                merge_count=context.properties.merge_count + 1,
                duration_count=context.properties.duration_count,
                enable_merge=context.properties.enable_merge,
            )

            reinforced_importance = min(context.extracted_data.importance + 1, 9)

            reinforced_extracted_data = ExtractedData(
                title=context.extracted_data.title,
                summary=context.extracted_data.summary,
                keywords=context.extracted_data.keywords,
                entities=context.extracted_data.entities,
                context_type=context.extracted_data.context_type,
                confidence=context.extracted_data.confidence,
                importance=reinforced_importance,
            )

            return ProcessedContext(
                id=context.id,
                extracted_data=reinforced_extracted_data,
                properties=reinforced_properties,
                vectorize=context.vectorize,
            )

        except Exception as e:
            logger.error(f"Error applying memory reinforcement: {e}", exc_info=True)
            return None

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory management statistics"""
        stats = {
            "merge_statistics": self._statistics,
            "strategy_count": len(self.strategies),
            "supported_types": [ct.value for ct in self.strategies.keys()],
            "config": {
                "use_intelligent_merging": self.use_intelligent_merging,
                "enable_memory_management": self.enable_memory_management,
            },
        }

        strategy_stats = {}
        for context_type, strategy in self.strategies.items():
            try:
                strategy_stats[context_type.value] = {
                    "similarity_threshold": strategy.similarity_threshold,
                    "retention_days": strategy.retention_days,
                    "max_merge_count": strategy.max_merge_count,
                }
            except Exception as e:
                logger.error(f"Error getting stats for {context_type.value}: {e}")

        stats["strategy_configurations"] = strategy_stats
        return stats
