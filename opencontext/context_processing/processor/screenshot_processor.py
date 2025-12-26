#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Screenshot processor - Stateless version with Redis state externalization.
Supports high concurrency with async processing.
"""
import asyncio
import base64
import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from opencontext.context_processing.processor.base_processor import BaseContextProcessor
from opencontext.context_processing.processor.entity_processor import (
    refresh_entities,
    validate_and_clean_entities,
)
from opencontext.llm.global_embedding_client import do_vectorize_async
from opencontext.llm.global_vlm_client import generate_with_messages_async
from opencontext.models.context import *
from opencontext.models.enums import get_context_type_descriptions_for_extraction
from opencontext.monitoring.monitor import record_processing_error
from opencontext.storage.global_storage import get_storage
from opencontext.tools.tool_definitions import ALL_TOOL_DEFINITIONS
from opencontext.utils.image import calculate_phash, resize_image
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger
from opencontext.config.global_config import get_prompt_group
from opencontext.monitoring import (
    increment_data_count,
    increment_recording_stat,
    record_processing_metrics,
)

logger = get_logger(__name__)

# Redis key prefixes for state externalization
REDIS_PHASH_KEY_PREFIX = "screenshot:phash"
REDIS_PROCESSED_CACHE_PREFIX = "processed_cache"
REDIS_LOCK_PREFIX = "lock:screenshot"


class ScreenshotProcessor(BaseContextProcessor):
    """
    Stateless processor for processing and analyzing screenshots.
    
    All state is externalized to Redis:
    - Screenshot phash deduplication queue -> Redis List
    - Processed context cache -> Redis Hash
    
    Supports high concurrency with async processing and distributed locks.
    """

    def __init__(self):
        """Initialize ScreenshotProcessor."""
        from opencontext.config.global_config import get_config

        config = get_config("processing.screenshot_processor") or {}
        super().__init__(config)

        self._similarity_hash_threshold = self.config.get("similarity_hash_threshold", 2)
        self._batch_size = self.config.get("batch_size", 10)
        self._batch_timeout = self.config.get("batch_timeout", 20)
        self._max_raw_properties = self.config.get("max_raw_properties", 5)
        self._max_image_size = self.config.get("max_image_size", 0)
        self._resize_quality = self.config.get("resize_quality", 95)
        self._enabled_delete = self.config.get("enabled_delete", False)
        
        # Redis cache settings
        self._phash_cache_size = self.config.get("phash_cache_size", 100)
        self._phash_cache_ttl = self.config.get("phash_cache_ttl", 3600)  # 1 hour
        self._processed_cache_ttl = self.config.get("processed_cache_ttl", 3600)  # 1 hour
        self._lock_timeout = self.config.get("lock_timeout", 30)  # 30 seconds
        
        # Get Redis cache instance
        self._redis_cache = None
        self._init_redis_cache()

    def _init_redis_cache(self):
        """Initialize Redis cache connection."""
        try:
            from opencontext.storage.redis_cache import get_redis_cache
            self._redis_cache = get_redis_cache()
            if self._redis_cache and self._redis_cache.is_connected():
                logger.info("ScreenshotProcessor: Redis cache connected for state externalization")
            else:
                logger.warning("ScreenshotProcessor: Redis not available, using degraded mode")
                self._redis_cache = None
        except Exception as e:
            logger.warning(f"ScreenshotProcessor: Failed to init Redis cache: {e}")
            self._redis_cache = None

    def _get_phash_key(self, user_id: str = "default", device_id: str = "default") -> str:
        """Generate Redis key for phash cache."""
        return f"{REDIS_PHASH_KEY_PREFIX}:{user_id}:{device_id}"

    def _get_processed_cache_key(self, context_type: str, user_id: str = "default", device_id: str = "default") -> str:
        """Generate Redis key for processed context cache."""
        return f"{REDIS_PROCESSED_CACHE_PREFIX}:{context_type}:{user_id}:{device_id}"

    def _get_lock_key(self, user_id: str = "default", device_id: str = "default") -> str:
        """Generate Redis key for distributed lock."""
        return f"{REDIS_LOCK_PREFIX}:{user_id}:{device_id}"

    def shutdown(self, graceful: bool = False):
        """Gracefully shut down processor (no-op for stateless processor)."""
        logger.info("ScreenshotProcessor shutdown (stateless, no cleanup needed)")

    def get_name(self) -> str:
        """Return the processor name."""
        return "screenshot_processor"

    def get_description(self) -> str:
        """Return the processor description."""
        return "Stateless screenshot processor with Redis state externalization and async processing."

    def can_process(self, context: RawContextProperties) -> bool:
        """Check if this processor can handle the given context."""
        return (
            isinstance(context, RawContextProperties) and context.source == ContextSource.SCREENSHOT
        )

    def _is_duplicate_redis(self, phash: str, user_id: str = "default", device_id: str = "default") -> bool:
        """
        Check if screenshot is duplicate using Redis cache.
        
        Args:
            phash: Perceptual hash of the screenshot
            user_id: User identifier
            device_id: Device identifier
            
        Returns:
            True if duplicate, False if new
        """
        if not self._redis_cache or not self._redis_cache.is_connected():
            # Degraded mode: no deduplication
            logger.warning("Redis not available, skipping deduplication")
            return False
        
        key = self._get_phash_key(user_id, device_id)
        
        try:
            # Get recent phashes from Redis
            recent_hashes = self._redis_cache.lrange(key, 0, self._phash_cache_size - 1)
            
            for cached_hash in recent_hashes:
                try:
                    diff = bin(int(str(phash), 16) ^ int(str(cached_hash), 16)).count("1")
                    if diff <= self._similarity_hash_threshold:
                        # Found duplicate
                        return True
                except (ValueError, TypeError):
                    continue
            
            # Not duplicate, add to cache
            self._redis_cache.lpush(key, phash)
            self._redis_cache.ltrim(key, 0, self._phash_cache_size - 1)
            self._redis_cache.expire(key, self._phash_cache_ttl)
            
            return False
            
        except Exception as e:
            logger.error(f"Redis phash check error: {e}")
            return False

    def _get_cached_contexts(self, context_type: str, user_id: str = "default", device_id: str = "default") -> Dict[str, ProcessedContext]:
        """
        Get cached processed contexts from Redis.
        
        Args:
            context_type: Type of context
            user_id: User identifier
            device_id: Device identifier
            
        Returns:
            Dictionary of context_id -> ProcessedContext
        """
        if not self._redis_cache or not self._redis_cache.is_connected():
            return {}
        
        key = self._get_processed_cache_key(context_type, user_id, device_id)
        
        try:
            cached_data = self._redis_cache.hgetall_json(key)
            result = {}
            for ctx_id, ctx_data in cached_data.items():
                try:
                    if isinstance(ctx_data, dict):
                        result[ctx_id] = ProcessedContext.from_dict(ctx_data)
                    elif isinstance(ctx_data, ProcessedContext):
                        result[ctx_id] = ctx_data
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached context {ctx_id}: {e}")
            return result
        except Exception as e:
            logger.error(f"Redis get cached contexts error: {e}")
            return {}

    def _set_cached_contexts(self, context_type: str, contexts: Dict[str, ProcessedContext], 
                             user_id: str = "default", device_id: str = "default"):
        """
        Set cached processed contexts in Redis.
        
        Args:
            context_type: Type of context
            contexts: Dictionary of context_id -> ProcessedContext
            user_id: User identifier
            device_id: Device identifier
        """
        if not self._redis_cache or not self._redis_cache.is_connected():
            return
        
        key = self._get_processed_cache_key(context_type, user_id, device_id)
        
        try:
            # Convert ProcessedContext to dict for storage
            data = {}
            for ctx_id, ctx in contexts.items():
                if hasattr(ctx, 'to_dict'):
                    data[ctx_id] = ctx.to_dict()
                else:
                    data[ctx_id] = ctx
            
            if data:
                self._redis_cache.hmset_json(key, data)
                self._redis_cache.expire(key, self._processed_cache_ttl)
        except Exception as e:
            logger.error(f"Redis set cached contexts error: {e}")

    def _delete_cached_context(self, context_type: str, context_id: str,
                               user_id: str = "default", device_id: str = "default"):
        """Delete a specific context from Redis cache."""
        if not self._redis_cache or not self._redis_cache.is_connected():
            return
        
        key = self._get_processed_cache_key(context_type, user_id, device_id)
        
        try:
            self._redis_cache.hdel(key, context_id)
        except Exception as e:
            logger.error(f"Redis delete cached context error: {e}")

    async def process_async(self, context: RawContextProperties, 
                           user_id: str = "default", device_id: str = "default") -> List[ProcessedContext]:
        """
        Process a single screenshot asynchronously.
        
        This is the main entry point for async processing.
        Supports concurrent requests without blocking.
        
        Args:
            context: Raw screenshot context
            user_id: User identifier for state isolation
            device_id: Device identifier for state isolation
            
        Returns:
            List of processed contexts
        """
        if not self.can_process(context):
            return []
        
        try:
            # Resize image if needed
            if self._max_image_size > 0:
                resize_image(context.content_path, self._max_image_size, self._resize_quality)
            
            # Calculate phash for deduplication
            phash = calculate_phash(context.content_path)
            if phash is None:
                logger.error(f"Failed to calculate phash for {context.content_path}")
                return []
            
            # Check for duplicate using Redis
            if self._is_duplicate_redis(phash, user_id, device_id):
                logger.debug(f"Duplicate screenshot detected, skipping")
                if self._enabled_delete:
                    try:
                        os.remove(context.content_path)
                    except Exception as e:
                        logger.error(f"Failed to delete duplicate screenshot: {e}")
                return []
            
            # Record screenshot path for UI display
            try:
                from opencontext.monitoring import record_screenshot_path
                if context.content_path:
                    record_screenshot_path(context.content_path)
            except ImportError:
                pass
            
            # Process with VLM
            increment_data_count("screenshot", count=1)
            
            import time
            start_time = time.time()
            
            processed_contexts = await self._process_single_screenshot(context, user_id, device_id)
            
            # Record metrics
            try:
                duration_ms = int((time.time() - start_time) * 1000)
                record_processing_metrics(
                    processor_name=self.get_name(),
                    operation="screenshot_process",
                    duration_ms=duration_ms,
                    context_count=len(processed_contexts),
                )
                
                for ctx in processed_contexts:
                    increment_data_count("context", count=1, context_type=ctx.extracted_data.context_type.value)
                
                increment_recording_stat("processed", len(processed_contexts))
            except Exception:
                pass
            
            return processed_contexts
            
        except Exception as e:
            logger.exception(f"Error processing screenshot {context.content_path}: {e}")
            record_processing_error(str(e), processor_name=self.get_name(), context_count=1)
            increment_recording_stat("failed", 1)
            return []

    def process(self, context: RawContextProperties) -> bool:
        """
        Synchronous wrapper for process_async.
        For backward compatibility.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create task
                asyncio.create_task(self._process_and_store(context))
                return True
            else:
                # Run in new event loop
                results = asyncio.run(self.process_async(context))
                if results:
                    get_storage().batch_upsert_processed_context(results)
                return bool(results)
        except Exception as e:
            logger.error(f"Error in synchronous process: {e}")
            return False

    async def _process_and_store(self, context: RawContextProperties):
        """Process and store results asynchronously."""
        results = await self.process_async(context)
        if results:
            get_storage().batch_upsert_processed_context(results)

    async def _process_single_screenshot(self, raw_context: RawContextProperties,
                                         user_id: str = "default", 
                                         device_id: str = "default") -> List[ProcessedContext]:
        """
        Process a single screenshot with VLM and merge with cached contexts.
        
        Args:
            raw_context: Raw screenshot context
            user_id: User identifier
            device_id: Device identifier
            
        Returns:
            List of processed contexts ready for storage
        """
        # Step 1: Extract information with VLM
        vlm_items = await self._process_vlm_single(raw_context)
        
        if not vlm_items:
            return []
        
        # Step 2: Merge with cached contexts
        merged_contexts = await self._merge_contexts(vlm_items, user_id, device_id)
        
        return merged_contexts

    async def _process_vlm_single(self, raw_context: RawContextProperties) -> List[ProcessedContext]:
        """Process a single screenshot with VLM."""
        prompt_group = get_prompt_group("processing.extraction.screenshot_analyze")
        system_prompt = prompt_group.get("system")
        user_prompt_template = prompt_group.get("user")
        
        if not system_prompt or not user_prompt_template:
            logger.error("Failed to get complete prompt for screenshot_analyze.")
            raise ValueError("Missing prompt configuration for screenshot_analyze")

        image_path = raw_context.content_path
        if not image_path or not os.path.exists(image_path):
            logger.error(f"Screenshot path is invalid or does not exist: {image_path}")
            raise ValueError(f"Screenshot path is invalid or does not exist: {image_path}")

        base64_image = self._encode_image_to_base64(image_path)
        if not base64_image:
            logger.warning(f"Failed to encode image: {image_path}")
            raise ValueError(f"Failed to encode image: {image_path}")

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                },
            }
        ]

        time_now = datetime.datetime.now()
        user_prompt = user_prompt_template.format(
            current_date=time_now.isoformat(),
            current_timestamp=int(time_now.timestamp()),
            current_timezone=time_now.tzname(),
        )
        content.insert(0, {"type": "text", "text": user_prompt})
        system_prompt = system_prompt.format(
            context_type_descriptions=get_context_type_descriptions_for_extraction()
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        try:
            raw_llm_response = await generate_with_messages_async(messages)
        except Exception as e:
            logger.error(f"Failed to get VLM response. Error: {e}")
            raise ValueError(f"Failed to get VLM response. Error: {e}")

        raw_resp = parse_json_from_response(raw_llm_response)
        if not raw_resp:
            logger.error("Empty VLM response.")
            raise ValueError("Empty VLM response.")
        
        items = raw_resp.get("items", [])
        processed_items = []
        for item in items:
            ctx = self._create_processed_context(item, raw_context)
            if ctx:
                processed_items.append(ctx)
        
        return processed_items

    async def _merge_contexts(self, processed_items: List[ProcessedContext],
                             user_id: str = "default", 
                             device_id: str = "default") -> List[ProcessedContext]:
        """
        Merge newly processed items with cached items based on context_type semantics.
        Uses Redis for state storage.
        """
        if not processed_items:
            return []

        # Group by context_type
        items_by_type: Dict[ContextType, List[ProcessedContext]] = {}
        for item in processed_items:
            context_type = item.extracted_data.context_type
            items_by_type.setdefault(context_type, []).append(item)

        # Process each context type concurrently
        tasks = []
        for context_type, new_items in items_by_type.items():
            # Get cached items from Redis
            cached_items_dict = self._get_cached_contexts(context_type.value, user_id, device_id)
            cached_items = list(cached_items_dict.values())
            tasks.append(self._merge_items_with_llm(context_type, new_items, cached_items, user_id, device_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_newly_created = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Merge task {idx} failed with error: {result}")
                continue
            if result:
                context_type = result.get("context_type")
                all_newly_created.extend(result.get("processed_contexts", []))
                
                # Update Redis cache
                new_ctxs = result.get("new_ctxs", {})
                if new_ctxs:
                    self._set_cached_contexts(context_type, new_ctxs, user_id, device_id)
                
                # Delete old contexts from storage
                for item_id in result.get("need_to_del_ids", []):
                    get_storage().delete_processed_context(item_id, context_type)
                    self._delete_cached_context(context_type, item_id, user_id, device_id)
        
        return all_newly_created

    async def _merge_items_with_llm(self, context_type: ContextType, 
                                    new_items: List[ProcessedContext], 
                                    cached_items: List[ProcessedContext],
                                    user_id: str = "default",
                                    device_id: str = "default") -> Dict[str, Any]:
        """
        Call LLM to merge items and directly return ProcessedContext objects.
        """
        prompt_group = get_prompt_group("merging.screenshot_batch_merging")
        all_items_map = {item.id: item for item in new_items + cached_items}
        items_json = json.dumps([self._item_to_dict(item) for item in new_items + cached_items], ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": prompt_group["system"]},
            {"role": "user", "content": prompt_group["user"].format(
                context_type=context_type.value,
                items_json=items_json
            )},
        ]
        response = await generate_with_messages_async(messages)

        if not response:
            raise ValueError(f"Empty LLM response when merge items for context type: {context_type.value}")

        response_data = parse_json_from_response(response)
        if not isinstance(response_data, dict) or "items" not in response_data:
            logger.error(f"merge_items_with_llm, Invalid response format: {response_data}")
            raise ValueError(f"Invalid response format when merge items for context type: {context_type.value}")

        # Process results and build ProcessedContext objects
        result_contexts = []
        now = datetime.datetime.now()
        need_to_del_ids = []
        final_context = None
        new_ctxs = {}
        entity_refresh_items = []
        
        for result in response_data.get("items", []):
            merge_type = result.get("merge_type")
            data = result.get("data", {})

            if merge_type == "merged":
                merged_ids = result.get("merged_ids", [])
                if not merged_ids:
                    logger.error("merged type but no merged_ids, skipping")
                    continue
                items_to_merge = [all_items_map[id] for id in merged_ids if id in all_items_map]
                if not items_to_merge:
                    logger.error(f"No valid items for merged_ids: {merged_ids}")
                    continue

                min_create_time = min((i.properties.create_time for i in items_to_merge if i.properties.create_time), default=now)
                event_time = self._parse_event_time_str(
                    data.get("event_time"),
                    max((i.properties.event_time for i in items_to_merge if i.properties.event_time), default=now)
                )

                all_raw_props = []
                for item in items_to_merge:
                    all_raw_props.extend(item.properties.raw_properties)

                merged_ctx = ProcessedContext(
                    properties=ContextProperties(
                        raw_properties=all_raw_props,
                        create_time=min_create_time,
                        update_time=now,
                        event_time=event_time,
                        enable_merge=True,
                        is_happend=event_time <= now if event_time else False,
                        duration_count=sum(i.properties.duration_count for i in items_to_merge),
                        merge_count=sum(i.properties.merge_count for i in items_to_merge) + 1,
                    ),
                    extracted_data=ExtractedData(
                        title=data.get("title", ""),
                        summary=data.get("summary", ""),
                        keywords=sorted(set(data.get("keywords", []))),
                        entities=[],
                        context_type=context_type,
                        importance=self._safe_int(data.get("importance")),
                        confidence=self._safe_int(data.get("confidence")),
                    ),
                    vectorize=Vectorize(
                        content_format=ContentFormat.TEXT,
                        text=f"{data.get('title', '')} {data.get('summary', '')}",
                    ),
                )

                final_context = merged_ctx
                # Mark old items for deletion from cache
                for item in items_to_merge:
                    if item.id in all_items_map:
                        need_to_del_ids.append(item.id)
                logger.debug(f"Merged {len(merged_ids)} items for context type: {context_type.value}")
                
            elif merge_type == "new":
                merged_ids = result.get("merged_ids", [])
                if not merged_ids or merged_ids[0] not in all_items_map:
                    logger.error("new type but no merged_ids or merged_ids[0] not in all_items_map, skipping")
                    continue
                # Check if already in cache (from Redis)
                cached = self._get_cached_contexts(context_type.value, user_id, device_id)
                if merged_ids[0] in cached:
                    continue
                final_context = all_items_map[merged_ids[0]]
            
            if final_context:
                new_ctxs[final_context.id] = final_context
                entity_refresh_items.append((final_context, data.get("entities", [])))

        # Second pass: parallel refresh entities
        entity_tasks = [
            self._parse_single_context(item, entities)
            for item, entities in entity_refresh_items
        ]
        
        if entity_tasks:
            entities_results = await asyncio.gather(*entity_tasks, return_exceptions=True)
            for entities_result in entities_results:
                if isinstance(entities_result, Exception):
                    logger.error(f"Entity refresh failed: {entities_result}")
                elif entities_result:
                    result_contexts.append(entities_result)

        return {
            "processed_contexts": result_contexts, 
            "need_to_del_ids": need_to_del_ids, 
            "new_ctxs": new_ctxs, 
            "context_type": context_type.value
        }

    async def _parse_single_context(self, item: ProcessedContext, entities: List[Dict[str, Any]]) -> ProcessedContext:
        """Parse a single context item."""
        entities_info = validate_and_clean_entities(entities)
        vectorize_task = do_vectorize_async(item.vectorize)
        entities_task = refresh_entities(entities_info, item.vectorize.text)
        _, entities_results = await asyncio.gather(vectorize_task, entities_task)
        item.extracted_data.entities = entities_results
        return item

    def _parse_event_time_str(self, time_str: Optional[str], default: datetime.datetime) -> datetime.datetime:
        """Parse ISO time string, return default if invalid."""
        if not time_str or time_str == "null":
            return default
        try:
            if any(
                invalid_char in time_str
                for invalid_char in ["xxxx", "XXXX", "TZ:TZ", "TZ", "????"]
            ):
                return default
            elif time_str.endswith("Z"):
                time_str = time_str[:-1] + "+00:00"
                return datetime.datetime.fromisoformat(time_str)
            return default
        except (ValueError, TypeError):
            return default

    def _safe_int(self, value, default=0) -> int:
        """Safely convert to int."""
        if value is None or value == "" or value == "null":
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _item_to_dict(self, item: ProcessedContext) -> Dict[str, Any]:
        """Convert a ProcessedContext item to a dictionary for LLM."""
        return {
            **item.extracted_data.to_dict(),
            "id": item.id,
            "event_time": item.properties.event_time.isoformat()
            if item.properties.event_time
            else None,
        }

    async def batch_process(self, raw_contexts: List[RawContextProperties],
                           user_id: str = "default",
                           device_id: str = "default") -> List[ProcessedContext]:
        """
        Batch process screenshots using Vision LLM with concurrent processing.
        
        Args:
            raw_contexts: List of raw screenshot contexts
            user_id: User identifier
            device_id: Device identifier
            
        Returns:
            List of processed contexts
        """
        logger.info(f"Processing {len(raw_contexts)} screenshots concurrently")

        # Step 1: Process all VLM tasks concurrently
        vlm_results = await asyncio.gather(
            *[self._process_vlm_single(raw_context) for raw_context in raw_contexts],
            return_exceptions=True
        )

        all_vlm_items = []
        for idx, result in enumerate(vlm_results):
            if isinstance(result, Exception):
                logger.error(f"Screenshot {idx} failed with error: {result}")
                increment_recording_stat("failed", 1)
                record_processing_error(str(result), processor_name=self.get_name(), context_count=1)
                continue
            if result:
                all_vlm_items.extend(result)

        if not all_vlm_items:
            return []

        logger.info(f"VLM parsing completed, got {len(all_vlm_items)} items")

        # Step 2: Merge contexts concurrently
        newly_processed_contexts = await self._merge_contexts(all_vlm_items, user_id, device_id)
        return newly_processed_contexts

    def _create_processed_context(self, analysis: Dict[str, Any], raw_context: RawContextProperties = None) -> ProcessedContext:
        """Create a ProcessedContext from VLM analysis."""
        now = datetime.datetime.now()
        if not analysis:
            logger.warning(f"Skipping incomplete item: {analysis}")
            return None
        
        context_type = None
        try:
            context_type_str = analysis.get("context_type", "semantic_context")
            from opencontext.models.enums import get_context_type_for_analysis
            context_type = get_context_type_for_analysis(context_type_str)
        except Exception as e:
            logger.warning(f"Error processing context_type: {e}, using default activity_context.")
            from opencontext.models.enums import ContextType
            context_type = ContextType.ACTIVITY_CONTEXT

        event_time = self._parse_event_time_str(analysis.get("event_time"), now)

        entities = []
        raw_keywords = analysis.get("keywords", [])
        extracted_data = ExtractedData(
            title=analysis.get("title", ""),
            summary=analysis.get("summary", ""),
            keywords=sorted(list(set(raw_keywords))),
            entities=entities,
            context_type=context_type,
            importance=self._safe_int(analysis.get("importance"), 0),
            confidence=self._safe_int(analysis.get("confidence"), 0),
        )

        new_context = ProcessedContext(
            properties=ContextProperties(
                raw_properties=[raw_context] if raw_context else [],
                source=ContextSource.SCREENSHOT,
                create_time=raw_context.create_time if raw_context else now,
                update_time=now,
                event_time=event_time,
                enable_merge=True,
                is_happend=event_time <= now,
            ),
            extracted_data=extracted_data,
            vectorize=Vectorize(
                content_format=ContentFormat.TEXT,
                text=f"{extracted_data.title} {extracted_data.summary}",
            ),
        )
        return new_context

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image file to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path} to base64: {e}")
            return None
