# -*- coding: utf-8 -*-

"""
UserMemoryCacheManager

Builds and maintains per-user memory cache snapshots in Redis.
Three sections: user profile (+ entities), recently accessed memories, recent memories (hierarchical).

Snapshot (profile + entities + recent memories) is cached in Redis with a short TTL.
Recently accessed items are stored in a separate Redis Hash and always read in real-time.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from opencontext.config.global_config import get_config
from opencontext.models.context import ProcessedContext
from opencontext.models.enums import ContextType
from opencontext.server.cache.models import (
    DailySummaryItem,
    RecentlyAccessedItem,
    RecentMemories,
    RecentMemoryItem,
    UserMemoryCacheResponse,
)
from opencontext.server.search.models import EntityResult, ProfileResult
from opencontext.storage.global_storage import get_storage
from opencontext.storage.redis_cache import get_cache
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class UserMemoryCacheManager:
    """Manages per-user memory cache snapshots."""

    def __init__(self):
        self._config = self._load_config()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._build_semaphore: Optional[asyncio.Semaphore] = None

    def _capture_loop(self) -> None:
        """Capture the event loop reference when called from async context."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def _get_build_semaphore(self) -> asyncio.Semaphore:
        """Lazily create the build semaphore."""
        if self._build_semaphore is None:
            self._build_semaphore = asyncio.Semaphore(3)
        return self._build_semaphore

    def _load_config(self) -> Dict[str, Any]:
        raw = get_config("memory_cache") or {}
        return {
            "snapshot_ttl": raw.get("snapshot_ttl", 300),
            "recent_days": raw.get("recent_days", 7),
            "max_recently_accessed": raw.get("max_recently_accessed", 50),
            "max_today_events": raw.get("max_today_events", 30),
            "max_recent_documents": raw.get("max_recent_documents", 10),
            "max_recent_knowledge": raw.get("max_recent_knowledge", 10),
            "accessed_ttl": raw.get("accessed_ttl", 604800),  # 7 days
            "max_entities": raw.get("max_entities", 20),
        }

    # ─── Key helpers ───

    @staticmethod
    def _accessed_key(user_id: str, device_id: str = "default", agent_id: str = "default") -> str:
        return f"memory_cache:accessed:{user_id}:{device_id}:{agent_id}"

    @staticmethod
    def _snapshot_key(user_id: str, device_id: str, agent_id: str) -> str:
        return f"memory_cache:snapshot:{user_id}:{device_id}:{agent_id}"

    # ─── Access Tracking (called after every search) ───

    async def track_accessed(
        self,
        user_id: str,
        items: List[Dict[str, Any]],
        device_id: str = "default",
        agent_id: str = "default",
    ) -> None:
        """Record context IDs returned in search results for a user.

        Uses hmset_json for single-RTT batch write. Lazy trimming when hash
        grows beyond max * 2 to avoid per-call overhead.
        """
        if not user_id or not items:
            return

        self._capture_loop()
        cache = await get_cache()
        key = self._accessed_key(user_id, device_id, agent_id)
        now = time.time()

        # Build mapping: {context_type}:{id} → metadata JSON
        mapping: Dict[str, Any] = {}
        for item in items:
            field = f"{item.get('context_type', '')}:{item.get('id', '')}"
            mapping[field] = {
                "id": item.get("id", ""),
                "context_type": item.get("context_type", ""),
                "title": item.get("title"),
                "summary": item.get("summary"),
                "keywords": item.get("keywords", []),
                "score": item.get("score"),
                "event_time": item.get("event_time"),
                "create_time": item.get("create_time"),
                "accessed_ts": now,
            }

        await cache.hmset_json(key, mapping)
        await cache.expire(key, self._config["accessed_ttl"])

        # Lazy trimming: only when significantly oversized
        max_size = self._config["max_recently_accessed"]
        current_size = await cache.hlen(key)
        if current_size > max_size * 2:
            await self._trim_accessed(cache, key, max_size)

    async def _trim_accessed(self, cache, key: str, max_size: int) -> None:
        """Remove oldest entries from the accessed hash."""
        all_data = await cache.hgetall(key)
        if not all_data:
            return

        # Parse and sort by accessed_ts
        items_with_ts: List[Tuple[str, float]] = []
        for field, value in all_data.items():
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
                ts = parsed.get("accessed_ts", 0) if isinstance(parsed, dict) else 0
                items_with_ts.append((field, ts))
            except (json.JSONDecodeError, TypeError):
                items_with_ts.append((field, 0))

        # Sort ascending (oldest first), delete excess
        items_with_ts.sort(key=lambda x: x[1])
        to_delete = len(items_with_ts) - max_size
        if to_delete > 0:
            fields_to_remove = [f for f, _ in items_with_ts[:to_delete]]
            await cache.hdel(key, *fields_to_remove)

    # ─── Snapshot Invalidation ───

    async def invalidate_snapshot(
        self, user_id: str, device_id: str = "default", agent_id: str = "default"
    ) -> None:
        """Invalidate the cached snapshot for a user."""
        cache = await get_cache()
        key = self._snapshot_key(user_id, device_id, agent_id)
        await cache.delete(key)
        logger.debug(
            f"Invalidated memory cache snapshot for user={user_id}, "
            f"device={device_id}, agent={agent_id}"
        )

    # ─── Main Entry Point ───

    async def get_user_memory_cache(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        recent_days: Optional[int] = None,
        max_recent_events_today: Optional[int] = None,
        max_accessed: int = 20,
        force_refresh: bool = False,
    ) -> UserMemoryCacheResponse:
        """Get the user's memory cache with stampede prevention."""
        self._capture_loop()
        cache = await get_cache()
        snapshot_key = self._snapshot_key(user_id, device_id, agent_id)

        # 1. Recently accessed — always real-time from Redis
        accessed = await self._get_recently_accessed(
            cache, user_id, max_accessed, device_id, agent_id
        )

        # 2. Try cached snapshot
        if not force_refresh:
            snapshot = await cache.get_json(snapshot_key)
            if snapshot:
                remaining_ttl = await cache.ttl(snapshot_key)
                return self._merge_response(
                    snapshot, accessed, cache_hit=True, ttl_remaining=max(remaining_ttl, 0)
                )

        # 3. Cache miss — acquire distributed lock to prevent stampede
        lock_key = f"memory_cache:build:{user_id}:{device_id}:{agent_id}"
        lock_token = await cache.acquire_lock(
            lock_key, timeout=30, blocking=True, blocking_timeout=5
        )

        if lock_token:
            try:
                # Double-check: another worker may have built it while we waited
                snapshot = await cache.get_json(snapshot_key)
                if snapshot and not force_refresh:
                    return self._merge_response(
                        snapshot,
                        accessed,
                        cache_hit=True,
                        ttl_remaining=await cache.ttl(snapshot_key),
                    )

                # Actually build snapshot
                snapshot_data = await self._build_snapshot(
                    user_id, device_id, agent_id, recent_days, max_recent_events_today
                )
                ttl = self._config["snapshot_ttl"]
                await cache.set_json(snapshot_key, snapshot_data, ttl=ttl)
                return self._merge_response(
                    snapshot_data, accessed, cache_hit=False, ttl_remaining=ttl
                )
            finally:
                await cache.release_lock(lock_key, lock_token)
        else:
            # Lock timeout — use semaphore to limit concurrent builds
            snapshot = await cache.get_json(snapshot_key)
            if snapshot:
                return self._merge_response(
                    snapshot,
                    accessed,
                    cache_hit=True,
                    ttl_remaining=await cache.ttl(snapshot_key),
                )

            sem = self._get_build_semaphore()
            try:
                await asyncio.wait_for(sem.acquire(), timeout=5)
                try:
                    # Double-check after acquiring semaphore
                    snapshot = await cache.get_json(snapshot_key)
                    if snapshot:
                        return self._merge_response(
                            snapshot,
                            accessed,
                            cache_hit=True,
                            ttl_remaining=await cache.ttl(snapshot_key),
                        )
                    snapshot_data = await self._build_snapshot(
                        user_id,
                        device_id,
                        agent_id,
                        recent_days,
                        max_recent_events_today,
                    )
                    ttl = self._config["snapshot_ttl"]
                    await cache.set_json(snapshot_key, snapshot_data, ttl=ttl)
                    return self._merge_response(
                        snapshot_data, accessed, cache_hit=False, ttl_remaining=ttl
                    )
                finally:
                    sem.release()
            except asyncio.TimeoutError:
                # Semaphore timeout — last resort
                snapshot = await cache.get_json(snapshot_key)
                if snapshot:
                    return self._merge_response(
                        snapshot,
                        accessed,
                        cache_hit=True,
                        ttl_remaining=await cache.ttl(snapshot_key),
                    )
                snapshot_data = await self._build_snapshot(
                    user_id,
                    device_id,
                    agent_id,
                    recent_days,
                    max_recent_events_today,
                )
                return self._merge_response(
                    snapshot_data, accessed, cache_hit=False, ttl_remaining=0
                )

    # ─── Recently Accessed (real-time from Redis) ───

    async def _get_recently_accessed(
        self,
        cache,
        user_id: str,
        max_items: int,
        device_id: str = "default",
        agent_id: str = "default",
    ) -> List[RecentlyAccessedItem]:
        """Read recently-accessed items from Redis hash, sorted by accessed_ts desc."""
        key = self._accessed_key(user_id, device_id, agent_id)
        all_data = await cache.hgetall(key)
        if not all_data:
            return []

        items: List[RecentlyAccessedItem] = []
        for field, value in all_data.items():
            try:
                data = json.loads(value) if isinstance(value, str) else value
                if not isinstance(data, dict):
                    continue
                items.append(
                    RecentlyAccessedItem(
                        id=data.get("id", ""),
                        title=data.get("title"),
                        summary=data.get("summary"),
                        context_type=data.get("context_type", ""),
                        keywords=data.get("keywords", []),
                        accessed_ts=data.get("accessed_ts", 0),
                        score=data.get("score"),
                        event_time=data.get("event_time"),
                        create_time=data.get("create_time"),
                    )
                )
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        # Sort by accessed_ts descending (most recent first)
        items.sort(key=lambda x: x.accessed_ts, reverse=True)
        return items[:max_items]

    # ─── Snapshot Building ───

    async def _build_snapshot(
        self,
        user_id: str,
        device_id: str,
        agent_id: str,
        recent_days: Optional[int],
        max_today_events: Optional[int],
    ) -> Dict[str, Any]:
        """Build the snapshot from storage backends (profile + entities + recent memories)."""
        t0 = time.perf_counter()
        storage = get_storage()
        days = recent_days if recent_days is not None else self._config["recent_days"]
        max_events_today = max_today_events or self._config["max_today_events"]

        # Compute time boundaries
        now = datetime.now(tz=timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_ts = int(today_start.timestamp())
        yesterday = (today_start - timedelta(days=1)).strftime("%Y-%m-%d")
        week_start = today_start - timedelta(days=days)
        week_start_ts = int(week_start.timestamp())
        period_start = (today_start - timedelta(days=days - 1)).strftime("%Y-%m-%d")

        # Parallel queries
        tasks = {
            "profile": asyncio.to_thread(storage.get_profile, user_id, device_id, agent_id),
            "entities": asyncio.to_thread(
                storage.list_entities,
                user_id,
                device_id,
                agent_id,
                None,
                self._config["max_entities"],
                0,
            ),
            "today_events": asyncio.to_thread(
                storage.get_all_processed_contexts,
                [ContextType.EVENT.value],
                max_events_today,
                0,
                {
                    "hierarchy_level": {"$gte": 0, "$lte": 0},
                    "event_time_ts": {"$gte": today_start_ts},
                },
                False,
                user_id,
            ),
            "daily_summaries": asyncio.to_thread(
                storage.search_hierarchy,
                ContextType.EVENT.value,
                1,  # L1
                period_start,
                yesterday,
                user_id,
                days,
            ),
            "recent_docs": asyncio.to_thread(
                storage.get_all_processed_contexts,
                [ContextType.DOCUMENT.value],
                self._config["max_recent_documents"],
                0,
                {"created_at_ts": {"$gte": week_start_ts}},
                False,
                user_id,
            ),
            "recent_knowledge": asyncio.to_thread(
                storage.get_all_processed_contexts,
                [ContextType.KNOWLEDGE.value],
                self._config["max_recent_knowledge"],
                0,
                {"created_at_ts": {"$gte": week_start_ts}},
                False,
                user_id,
            ),
        }

        task_names = list(tasks.keys())
        raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results_map = dict(zip(task_names, raw_results))

        # Log errors
        for name, result in results_map.items():
            if isinstance(result, Exception):
                logger.error(f"Memory cache build error ({name}): {result}")

        t_queries = time.perf_counter()
        logger.info(f"[memory-cache] parallel queries: {(t_queries - t0)*1000:.0f}ms")

        # Assemble snapshot
        snapshot: Dict[str, Any] = {
            "user_id": user_id,
            "device_id": device_id,
            "agent_id": agent_id,
            "built_at": now.isoformat(),
            "recent_days": days,
        }

        # Profile
        profile_data = results_map.get("profile")
        if profile_data and not isinstance(profile_data, Exception):
            snapshot["profile"] = {
                "user_id": profile_data.get("user_id", user_id),
                "device_id": profile_data.get("device_id", device_id),
                "agent_id": profile_data.get("agent_id", agent_id),
                "content": profile_data.get("content", ""),
                "summary": profile_data.get("summary"),
                "keywords": profile_data.get("keywords", []),
                "metadata": profile_data.get("metadata", {}),
            }

        # Entities
        entity_data = results_map.get("entities")
        if entity_data and not isinstance(entity_data, Exception):
            snapshot["entities"] = [
                {
                    "id": e.get("id", ""),
                    "entity_name": e.get("entity_name", ""),
                    "entity_type": e.get("entity_type"),
                    "content": e.get("content", ""),
                    "summary": e.get("summary"),
                    "aliases": e.get("aliases", []),
                    "score": 1.0,
                }
                for e in entity_data
            ]

        # Recent memories — hierarchical
        recent_memories: Dict[str, Any] = {}

        # Today's L0 events
        today_data = results_map.get("today_events")
        if today_data and not isinstance(today_data, Exception):
            today_items = []
            for ctx_type, contexts in today_data.items():
                for ctx in contexts:
                    today_items.append(self._ctx_to_recent_item(ctx))
            today_items.sort(
                key=lambda x: x.get("event_time") or x.get("create_time") or "",
                reverse=True,
            )
            recent_memories["today_events"] = today_items[:max_events_today]

        # Daily summaries (L1)
        summaries_data = results_map.get("daily_summaries")
        if summaries_data and not isinstance(summaries_data, Exception):
            daily_items = []
            for ctx, score in summaries_data:
                daily_items.append(
                    {
                        "id": ctx.id,
                        "time_bucket": ctx.properties.time_bucket or "",
                        "summary": ctx.extracted_data.summary if ctx.extracted_data else None,
                        "children_count": len(ctx.properties.children_ids)
                        if ctx.properties.children_ids
                        else 0,
                    }
                )
            daily_items.sort(key=lambda x: x["time_bucket"], reverse=True)
            recent_memories["daily_summaries"] = daily_items

        # Recent documents
        docs_data = results_map.get("recent_docs")
        if docs_data and not isinstance(docs_data, Exception):
            doc_items = []
            for ctx_type, contexts in docs_data.items():
                for ctx in contexts:
                    doc_items.append(self._ctx_to_recent_item(ctx))
            doc_items.sort(key=lambda x: x.get("create_time") or "", reverse=True)
            recent_memories["recent_documents"] = doc_items

        # Recent knowledge
        knowledge_data = results_map.get("recent_knowledge")
        if knowledge_data and not isinstance(knowledge_data, Exception):
            knowledge_items = []
            for ctx_type, contexts in knowledge_data.items():
                for ctx in contexts:
                    knowledge_items.append(self._ctx_to_recent_item(ctx))
            knowledge_items.sort(key=lambda x: x.get("create_time") or "", reverse=True)
            recent_memories["recent_knowledge"] = knowledge_items

        snapshot["recent_memories"] = recent_memories

        t_end = time.perf_counter()
        logger.info(f"[memory-cache] snapshot built for user={user_id}: {(t_end - t0)*1000:.0f}ms")

        return snapshot

    @staticmethod
    def _ctx_to_recent_item(ctx: ProcessedContext) -> Dict[str, Any]:
        """Convert a ProcessedContext to a dict for RecentMemoryItem."""
        ed = ctx.extracted_data
        props = ctx.properties

        create_time = None
        if props.create_time:
            create_time = (
                props.create_time.isoformat()
                if hasattr(props.create_time, "isoformat")
                else str(props.create_time)
            )

        event_time = None
        if props.event_time:
            event_time = (
                props.event_time.isoformat()
                if hasattr(props.event_time, "isoformat")
                else str(props.event_time)
            )

        return {
            "id": ctx.id,
            "title": ed.title if ed else None,
            "summary": ed.summary if ed else None,
            "context_type": ed.context_type.value if ed and ed.context_type else "",
            "keywords": ed.keywords if ed else [],
            "entities": ed.entities if ed else [],
            "importance": ed.importance if ed else 0,
            "create_time": create_time,
            "event_time": event_time,
        }

    # ─── Response Assembly ───

    def _merge_response(
        self,
        snapshot_data: Dict[str, Any],
        accessed: List[RecentlyAccessedItem],
        cache_hit: bool,
        ttl_remaining: int = 0,
    ) -> UserMemoryCacheResponse:
        """Merge snapshot data with real-time accessed items into final response."""
        # Profile
        profile = None
        profile_data = snapshot_data.get("profile")
        if profile_data:
            profile = ProfileResult(**profile_data)

        # Entities
        entities = []
        for e in snapshot_data.get("entities", []):
            entities.append(EntityResult(**e))

        # Recent memories
        rm_data = snapshot_data.get("recent_memories", {})
        recent_memories = RecentMemories(
            today_events=[RecentMemoryItem(**item) for item in rm_data.get("today_events", [])],
            daily_summaries=[
                DailySummaryItem(**item) for item in rm_data.get("daily_summaries", [])
            ],
            recent_documents=[
                RecentMemoryItem(**item) for item in rm_data.get("recent_documents", [])
            ],
            recent_knowledge=[
                RecentMemoryItem(**item) for item in rm_data.get("recent_knowledge", [])
            ],
        )

        return UserMemoryCacheResponse(
            success=True,
            user_id=snapshot_data.get("user_id", ""),
            device_id=snapshot_data.get("device_id", "default"),
            agent_id=snapshot_data.get("agent_id", "default"),
            profile=profile,
            entities=entities,
            recently_accessed=accessed,
            recent_memories=recent_memories,
            cache_metadata={
                "cache_hit": cache_hit,
                "built_at": snapshot_data.get("built_at", ""),
                "ttl_remaining_s": max(ttl_remaining, 0),
                "recent_days": snapshot_data.get("recent_days", self._config["recent_days"]),
            },
        )


# ─── Module-level singleton ───

_manager: Optional[UserMemoryCacheManager] = None


def get_memory_cache_manager() -> UserMemoryCacheManager:
    """Get or create the singleton UserMemoryCacheManager."""
    global _manager
    if _manager is None:
        _manager = UserMemoryCacheManager()
    return _manager
