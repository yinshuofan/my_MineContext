# -*- coding: utf-8 -*-

"""
MemoryCacheManager

Builds and maintains per-user (or per-agent) memory cache snapshots in Redis.
Four response sections: profile, recently accessed, today events, daily summaries.

Snapshot is cached in Redis with a short TTL (internally stores full data for all types).
Response assembly (_merge_response) simplifies to: SimpleProfile, SimpleTodayEvent, SimpleDailySummary.
Recently accessed items are stored in a separate Redis Hash and always read in real-time.

Snapshots automatically include both user interaction events and agent base memories.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from opencontext.config.global_config import get_config
from opencontext.models.context import ProcessedContext
from opencontext.models.enums import ContextType
from opencontext.server.cache.models import (
    DailySummaryItem,
    RecentlyAccessedItem,
    RecentMemoryItem,
    SimpleDailySummary,
    SimpleProfile,
    SimpleTodayEvent,
    UserMemoryCacheResponse,
)
from opencontext.storage.global_storage import get_storage
from opencontext.storage.redis_cache import get_cache
from opencontext.utils.media_refs import normalize_media_refs
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.time_utils import now as tz_now, today_start as tz_today_start

logger = get_logger(__name__)


class MemoryCacheManager:
    """Manages per-user (or per-agent) memory cache snapshots."""

    def __init__(self):
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        raw = get_config("memory_cache") or {}
        return {
            "snapshot_ttl": raw.get("snapshot_ttl", 3600),
            "recent_days": raw.get("recent_days", 3),
            "max_recently_accessed": raw.get("max_recently_accessed", 25),
            "max_today_events": raw.get("max_today_events", 5),
            "max_recent_documents": raw.get("max_recent_documents", 10),
            "max_recent_knowledge": raw.get("max_recent_knowledge", 10),
            "accessed_ttl": raw.get("accessed_ttl", 604800),  # 7 days
        }

    def reload_config(self):
        """Re-read config values from GlobalConfig."""
        self._config = self._load_config()

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
                "event_time_start": item.get("event_time_start"),
                "create_time": item.get("create_time"),
                "media_refs": normalize_media_refs(item.get("media_refs")),
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
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
    ) -> None:
        """Invalidate the cached snapshot for a user."""
        cache = await get_cache()
        key = self._snapshot_key(user_id, device_id, agent_id)
        await cache.delete(key)
        logger.debug(
            f"Invalidated memory cache snapshot for user={user_id}, "
            f"device={device_id}, agent={agent_id}"
        )

    async def refresh_snapshot(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
    ) -> bool:
        """Invalidate and proactively rebuild the snapshot cache.

        Called after new contexts are stored to ensure the cache reflects latest data.
        Uses the same distributed lock as get_user_memory_cache() for stampede prevention.

        Returns True if snapshot was rebuilt, False if skipped (lock held by another worker).
        """
        cache = await get_cache()
        snapshot_key = self._snapshot_key(user_id, device_id, agent_id)
        lock_key = f"memory_cache:build:{user_id}:{device_id}:{agent_id}"

        lock_token = await cache.acquire_lock(
            lock_key, timeout=30, blocking=True, blocking_timeout=5
        )
        if lock_token:
            try:
                await cache.delete(snapshot_key)
                snapshot_data = await self._build_snapshot(
                    user_id, device_id, agent_id, None, None
                )
                ttl = self._config["snapshot_ttl"]
                await cache.set_json(snapshot_key, snapshot_data, ttl=ttl)
                logger.info(
                    f"Proactively refreshed memory cache for "
                    f"user={user_id}, device={device_id}, agent={agent_id}"
                )
                return True
            finally:
                await cache.release_lock(lock_key, lock_token)
        else:
            # Another worker is building — just invalidate, next read will rebuild
            await cache.delete(snapshot_key)
            logger.debug(
                f"Proactive refresh skipped (lock held) for "
                f"user={user_id}, falling back to invalidation"
            )
            return False

    # ─── Main Entry Point ───

    async def get_user_memory_cache(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        recent_days: Optional[int] = None,
        max_recent_events_today: Optional[int] = None,
        max_accessed: int = 5,
        force_refresh: bool = False,
        include_sections: Optional[Set[str]] = None,
    ) -> UserMemoryCacheResponse:
        """Get the user's memory cache with stampede prevention.

        Args:
            include_sections: Set of section names to include in response.
                Valid values: "profile", "events", "accessed".
                Default (None): {"profile", "events", "accessed"}.
                Snapshot is always built fully for caching; filtering is response-level only.
        """
        sections = include_sections or {"profile", "events", "accessed"}
        cache = await get_cache()

        # 1. Recently accessed — only if requested, always real-time from Redis
        accessed = []
        if "accessed" in sections:
            accessed = await self._get_recently_accessed(
                cache, user_id, max_accessed, device_id, agent_id
            )

        # 2. If only accessed is requested, skip snapshot entirely
        need_snapshot = bool(sections - {"accessed"})
        if not need_snapshot:
            return self._merge_response(
                {"user_id": user_id, "device_id": device_id, "agent_id": agent_id,
                 "recent_memories": {}},
                accessed, cache_hit=False, ttl_remaining=0, include_sections=sections,
            )

        snapshot_key = self._snapshot_key(user_id, device_id, agent_id)

        # 3. Try cached snapshot
        if not force_refresh:
            snapshot = await cache.get_json(snapshot_key)
            if snapshot:
                remaining_ttl = await cache.ttl(snapshot_key)
                return self._merge_response(
                    snapshot, accessed, cache_hit=True,
                    ttl_remaining=max(remaining_ttl, 0),
                    include_sections=sections,
                )

        # 4. Cache miss — acquire distributed lock to prevent stampede
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
                        snapshot, accessed, cache_hit=True,
                        ttl_remaining=await cache.ttl(snapshot_key),
                        include_sections=sections,
                    )

                # Actually build snapshot (always full, for caching)
                snapshot_data = await self._build_snapshot(
                    user_id, device_id, agent_id, recent_days,
                    max_recent_events_today,
                )
                ttl = self._config["snapshot_ttl"]
                await cache.set_json(snapshot_key, snapshot_data, ttl=ttl)
                return self._merge_response(
                    snapshot_data, accessed, cache_hit=False, ttl_remaining=ttl,
                    include_sections=sections,
                )
            finally:
                await cache.release_lock(lock_key, lock_token)
        else:
            # Lock acquisition timed out — try cache once more, else build directly
            snapshot = await cache.get_json(snapshot_key)
            if snapshot:
                return self._merge_response(
                    snapshot, accessed, cache_hit=True,
                    ttl_remaining=await cache.ttl(snapshot_key),
                    include_sections=sections,
                )
            # Another instance likely holds the lock and is building; build uncached
            snapshot_data = await self._build_snapshot(
                user_id, device_id, agent_id, recent_days,
                max_recent_events_today,
            )
            return self._merge_response(
                snapshot_data, accessed, cache_hit=False, ttl_remaining=0,
                include_sections=sections,
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
                        event_time_start=data.get("event_time_start"),
                        create_time=data.get("create_time"),
                        media_refs=normalize_media_refs(data.get("media_refs")),
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
        """Build the snapshot from storage backends (profile + recent memories).

        Automatically includes both user events and agent base memories.
        Profile type is determined by agent_id (agent_profile if non-default).
        """
        t0 = time.perf_counter()
        storage = get_storage()
        days = recent_days if recent_days is not None else self._config["recent_days"]
        max_events_today = max_today_events or self._config["max_today_events"]

        # Resolve context types — always user events
        l0_type = ContextType.EVENT.value
        l1_type = ContextType.DAILY_SUMMARY.value
        summary_type_values = {ContextType.DAILY_SUMMARY.value, ContextType.WEEKLY_SUMMARY.value, ContextType.MONTHLY_SUMMARY.value}

        # Profile — use agent_profile if agent_id is not default
        if agent_id and agent_id != "default":
            profile_context_type = ContextType.AGENT_PROFILE.value
        else:
            profile_context_type = ContextType.PROFILE.value

        # Compute time boundaries
        now = tz_now()
        today_start_val = tz_today_start()
        today_start_ts = int(today_start_val.timestamp())
        yesterday_start = today_start_val - timedelta(days=1)
        yesterday_end_ts = float(today_start_ts - 1)  # end of yesterday
        week_start = today_start_val - timedelta(days=days)
        week_start_ts = int(week_start.timestamp())
        period_start_dt = today_start_val - timedelta(days=days - 1)
        period_start_ts = float(int(period_start_dt.timestamp()))

        # Parallel queries — always include docs/knowledge and base events
        tasks = {
            "profile": storage.get_profile(
                user_id, device_id, agent_id, context_type=profile_context_type
            ),
            "today_events": storage.get_all_processed_contexts(
                context_types=[l0_type],
                limit=max_events_today,
                offset=0,
                filter={
                    "hierarchy_level": 0,
                    "event_time_start_ts": {"$gte": today_start_ts},
                },
                need_vector=False,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            ),
            "daily_summaries": storage.search_hierarchy(
                context_type=l1_type,
                hierarchy_level=1,  # L1
                time_start=period_start_ts,
                time_end=yesterday_end_ts,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
                top_k=days,
            ),
            "recent_docs": storage.get_all_processed_contexts(
                context_types=[ContextType.DOCUMENT.value],
                limit=self._config["max_recent_documents"],
                offset=0,
                filter={"create_time_ts": {"$gte": week_start_ts}},
                need_vector=False,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            ),
            "recent_knowledge": storage.get_all_processed_contexts(
                context_types=[ContextType.KNOWLEDGE.value],
                limit=self._config["max_recent_knowledge"],
                offset=0,
                filter={"create_time_ts": {"$gte": week_start_ts}},
                need_vector=False,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            ),
            "base_events": storage.get_all_processed_contexts(
                context_types=[ContextType.AGENT_BASE_EVENT.value],
                limit=20,
                user_id="__base__",
                device_id=device_id,
                agent_id=agent_id,
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

        # Profile — with agent_base_profile fallback for non-default agent_id
        profile_data = results_map.get("profile")
        if isinstance(profile_data, Exception):
            profile_data = None
        if not profile_data and agent_id and agent_id != "default":
            profile_data = await storage.get_profile(
                "__base__", device_id, agent_id, context_type="agent_base_profile"
            )
            if profile_data:
                logger.debug(
                    f"[memory-cache] Using __base__ fallback profile "
                    f"for user={user_id}, agent={agent_id}"
                )
        if profile_data:
            snapshot["profile"] = {
                "user_id": user_id,
                "device_id": device_id,
                "agent_id": agent_id,
                "factual_profile": profile_data.get("factual_profile", ""),
                "behavioral_profile": profile_data.get("behavioral_profile"),
                "metadata": profile_data.get("metadata", {}),
            }

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
                key=lambda x: x.get("event_time_start") or x.get("create_time") or "",
            )
            recent_memories["today_events"] = today_items[:max_events_today]

        # Daily summaries (L1)
        summaries_data = results_map.get("daily_summaries")
        if summaries_data and not isinstance(summaries_data, Exception):
            daily_items = []
            for ctx, score in summaries_data:
                # Derive event_time_start string from properties
                ets = ""
                if ctx.properties and ctx.properties.event_time_start:
                    ets_val = ctx.properties.event_time_start
                    if hasattr(ets_val, "isoformat"):
                        ets = ets_val.isoformat()
                    else:
                        ets = str(ets_val)
                daily_items.append(
                    {
                        "id": ctx.id,
                        "event_time_start": ets,
                        "title": ctx.extracted_data.title if ctx.extracted_data else None,
                        "summary": ctx.extracted_data.summary if ctx.extracted_data else None,
                        "children_count": self._count_refs(ctx, summary_type_values),
                    }
                )
            daily_items.sort(key=lambda x: x["event_time_start"], reverse=True)
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

        # Base events (agent base memories)
        base_data = results_map.get("base_events")
        if base_data and not isinstance(base_data, Exception):
            base_items = []
            for ctx_type, contexts in base_data.items():
                for ctx in contexts:
                    base_items.append(self._ctx_to_recent_item(ctx))
            base_items.sort(key=lambda x: x.get("create_time") or "", reverse=True)
            recent_memories["base_memories"] = base_items

        snapshot["recent_memories"] = recent_memories

        t_end = time.perf_counter()
        logger.info(f"[memory-cache] snapshot built for user={user_id}: {(t_end - t0)*1000:.0f}ms")

        return snapshot

    @staticmethod
    def _count_refs(ctx: ProcessedContext, summary_type_values: set = None) -> int:
        """Count child IDs from refs, excluding upward (parent/summary) refs."""
        if ctx.properties.refs:
            count = 0
            for key, ids in ctx.properties.refs.items():
                if summary_type_values and key in summary_type_values:
                    continue  # skip upward refs to parent summaries
                count += len(ids)
            return count
        return 0

    @staticmethod
    def _ctx_to_recent_item(ctx: ProcessedContext) -> Dict[str, Any]:
        """Convert a ProcessedContext to a dict for RecentMemoryItem."""
        ed = ctx.extracted_data
        props = ctx.properties

        create_time = None
        if props.create_time:
            if hasattr(props.create_time, "isoformat"):
                create_time = props.create_time.isoformat()
            else:
                create_time = str(props.create_time)

        event_time_start = None
        if props.event_time_start:
            if hasattr(props.event_time_start, "isoformat"):
                event_time_start = props.event_time_start.isoformat()
            else:
                event_time_start = str(props.event_time_start)

        result = {
            "id": ctx.id,
            "title": ed.title if ed else None,
            "summary": ed.summary if ed else None,
            "context_type": ed.context_type.value if ed and ed.context_type else "",
            "keywords": ed.keywords if ed else [],
            "entities": ed.entities if ed else [],
            "importance": ed.importance if ed else 0,
            "create_time": create_time,
            "event_time_start": event_time_start,
        }

        # Include media_refs from metadata if present (multimodal content)
        media_refs = normalize_media_refs(ctx.metadata.get("media_refs") if ctx.metadata else None)
        if media_refs:
            result["media_refs"] = media_refs

        return result

    # ─── Response Assembly ───

    def _merge_response(
        self,
        snapshot_data: Dict[str, Any],
        accessed: List[RecentlyAccessedItem],
        cache_hit: bool,
        ttl_remaining: int = 0,
        include_sections: Optional[Set[str]] = None,
    ) -> UserMemoryCacheResponse:
        """Merge snapshot data with real-time accessed items into final response.

        Only populates sections listed in include_sections. Unrequested sections
        remain None in the response (= not requested). Requested but empty sections
        are set to [] (= no data).
        """
        sections = include_sections or {"profile", "events", "accessed"}
        rm_data = snapshot_data.get("recent_memories", {})

        # Profile
        profile = None
        if "profile" in sections:
            profile_data = snapshot_data.get("profile")
            if profile_data:
                profile = SimpleProfile(
                    factual_profile=profile_data.get("factual_profile", ""),
                    behavioral_profile=profile_data.get("behavioral_profile"),
                    metadata=profile_data.get("metadata", {}),
                )

        # Collect IDs from shown sections (for dedup against accessed)
        snapshot_ids: set = set()

        # Events (today_events + daily_summaries)
        today_events = None
        daily_summaries = None
        if "events" in sections:
            for item in rm_data.get("today_events", []):
                if item.get("id"):
                    snapshot_ids.add(item["id"])
            for item in rm_data.get("daily_summaries", []):
                if item.get("id"):
                    snapshot_ids.add(item["id"])

            daily_summaries = [
                SimpleDailySummary(
                    event_time_start=item.get("event_time_start", ""),
                    title=item.get("title"),
                    summary=item.get("summary"),
                )
                for item in rm_data.get("daily_summaries", [])
            ]
            today_events = [
                SimpleTodayEvent(
                    title=item.get("title"),
                    summary=item.get("summary"),
                    event_time_start=item.get("event_time_start"),
                )
                for item in rm_data.get("today_events", [])
            ]

        # Recently accessed
        filtered_accessed = None
        if "accessed" in sections:
            filtered_accessed = [
                item
                for item in accessed
                if item.context_type != "profile" and item.id not in snapshot_ids
            ]

        return UserMemoryCacheResponse(
            success=True,
            user_id=snapshot_data.get("user_id", ""),
            device_id=snapshot_data.get("device_id", "default"),
            agent_id=snapshot_data.get("agent_id", "default"),
            profile=profile,
            recently_accessed=filtered_accessed,
            daily_summaries=daily_summaries,
            today_events=today_events,
        )


# Backward compatibility alias
UserMemoryCacheManager = MemoryCacheManager


# ─── Module-level singleton ───

_manager: Optional[MemoryCacheManager] = None


def get_memory_cache_manager() -> MemoryCacheManager:
    """Get or create the singleton MemoryCacheManager."""
    global _manager
    if _manager is None:
        _manager = MemoryCacheManager()
    return _manager
