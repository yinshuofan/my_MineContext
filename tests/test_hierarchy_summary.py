#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hierarchy Summary End-to-End Integration Test
层级摘要端到端集成测试

Tests the full L0 → L1 → L2 → L3 hierarchy summary generation pipeline
using real services (VikingDB, MySQL, LLM API).

Usage:
    uv run python tests/test_hierarchy_summary.py --config config/config.yaml
    uv run python tests/test_hierarchy_summary.py --config config/config.yaml --user-id test_user
    uv run python tests/test_hierarchy_summary.py --config config/config.yaml --cleanup
"""

import argparse
import asyncio
import datetime
import os
import sys
import traceback
import uuid

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    Vectorize,
)
from opencontext.models.enums import ContentFormat, ContextType


# ── Test event data ──

TEST_EVENTS = [
    {
        "date": "2026-02-17",
        "hour": 10,
        "title": "团队周会讨论 Q1 目标",
        "summary": "上午十点召开团队周会，讨论了 Q1 季度的核心目标和关键成果指标。产品侧确认了搜索优化和用户体验改进两个重点方向，技术侧需要完成向量检索性能提升和存储层重构。各组负责人汇报了上周进展并更新了风险事项。",
        "keywords": ["周会", "Q1目标", "OKR", "搜索优化", "向量检索"],
        "entities": ["产品组", "技术组"],
    },
    {
        "date": "2026-02-18",
        "hour": 14,
        "title": "代码评审与技术方案讨论",
        "summary": "下午进行了搜索模块的代码评审，主要审查了新的向量索引策略实现。讨论了 HNSW 参数调优方案，确定了 ef_construction=200, M=16 的配置。同时评审了 Redis 缓存层的连接池优化 PR，发现了一个潜在的连接泄漏问题并安排修复。",
        "keywords": ["代码评审", "HNSW", "Redis", "连接池", "向量索引"],
        "entities": ["搜索模块", "Redis"],
    },
    {
        "date": "2026-02-19",
        "hour": 11,
        "title": "与客户沟通产品需求",
        "summary": "上午与客户A进行了产品需求沟通会。客户希望增加多语言支持和自定义知识库导入功能。讨论了技术可行性，多语言嵌入模型可以支持中英日韩四种语言，知识库导入需要新增文档解析管道。预计需要两个迭代周期完成。",
        "keywords": ["客户沟通", "多语言", "知识库导入", "需求分析"],
        "entities": ["客户A"],
    },
    {
        "date": "2026-02-20",
        "hour": 15,
        "title": "完成搜索模块性能优化",
        "summary": "完成了搜索模块的性能优化工作。通过引入查询缓存和向量预计算，将 P99 延迟从 350ms 降低到 120ms。优化了批量嵌入生成流程，支持最大 32 条并发请求。更新了性能监控面板，添加了延迟分位数和吞吐量指标。",
        "keywords": ["性能优化", "P99延迟", "查询缓存", "批量嵌入", "监控"],
        "entities": ["搜索模块"],
    },
    {
        "date": "2026-02-21",
        "hour": 9,
        "title": "部署生产环境并监控",
        "summary": "上午九点开始生产环境部署，采用蓝绿部署策略。先部署到预发环境验证通过后切换流量。部署过程中发现一个数据库连接超时问题，通过调整连接池最大空闲时间解决。部署完成后持续监控两小时，各项指标正常，错误率低于 0.01%。",
        "keywords": ["生产部署", "蓝绿部署", "监控", "连接超时", "错误率"],
        "entities": ["生产环境"],
    },
    {
        "date": "2026-02-28",
        "hour": 16,
        "title": "月度复盘与下月规划",
        "summary": "下午组织了二月份的月度复盘会议。回顾了本月完成的主要工作：搜索性能优化上线、Redis连接池问题修复、客户需求调研完成。识别了需要改进的方面：文档更新滞后、跨团队协作效率待提升。讨论了三月份的重点计划，包括多语言支持开发和知识库导入功能。",
        "keywords": ["月度复盘", "规划", "多语言", "知识库", "改进"],
        "entities": ["产品组", "技术组"],
    },
]


def _make_event_time(date_str: str, hour: int) -> datetime.datetime:
    """Create a timezone-aware datetime from date string and hour."""
    d = datetime.date.fromisoformat(date_str)
    return datetime.datetime(d.year, d.month, d.day, hour, 0, 0, tzinfo=datetime.timezone.utc)


def create_l0_event(event_data: dict, user_id: str) -> ProcessedContext:
    """Create a single L0 ProcessedContext from test event data."""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    event_time = _make_event_time(event_data["date"], event_data["hour"])

    return ProcessedContext(
        id=str(uuid.uuid4()),
        properties=ContextProperties(
            raw_properties=[],
            create_time=now,
            event_time=event_time,
            update_time=now,
            is_processed=True,
            enable_merge=False,
            is_happend=True,
            user_id=user_id,
            device_id="default",
            agent_id="default",
            hierarchy_level=0,
        ),
        extracted_data=ExtractedData(
            title=event_data["title"],
            summary=event_data["summary"],
            keywords=event_data["keywords"],
            entities=event_data.get("entities", []),
            context_type=ContextType.EVENT,
            importance=5,
            confidence=90,
        ),
        vectorize=Vectorize(
            content_format=ContentFormat.TEXT,
            text=f"{event_data['title']} {event_data['summary']}",
        ),
    )


async def init_system(config_path: str):
    """Initialize config and storage."""
    os.environ["OPENCONTEXT_CONFIG_PATH"] = config_path

    from opencontext.config.global_config import GlobalConfig

    GlobalConfig.get_instance()
    print("  Config loaded")

    from opencontext.storage.global_storage import GlobalStorage

    await GlobalStorage.get_instance().ensure_initialized()

    from opencontext.storage.global_storage import get_storage

    storage = get_storage()
    if not storage:
        raise RuntimeError("Storage initialization failed — check config and service connectivity")
    print("  Storage initialized")
    return storage


async def store_test_events(storage, user_id: str) -> list[ProcessedContext]:
    """Create and store L0 test events. Returns the list of stored events."""
    events = [create_l0_event(data, user_id) for data in TEST_EVENTS]
    ok = await storage.batch_upsert_processed_context(events)
    if not ok:
        raise RuntimeError("Failed to store test events")
    return events


async def generate_summaries(user_id: str, dates: list[str], week: str, month: str):
    """Generate L1 → L2 → L3 summaries step by step."""
    from opencontext.periodic_task.hierarchy_summary import HierarchySummaryTask

    task = HierarchySummaryTask()
    results = {"l1": {}, "l2": None, "l3": None}

    # L1: daily summaries for each date
    for date_str in dates:
        print(f"  L1 daily {date_str}...", end=" ", flush=True)
        try:
            ctx = await task._generate_daily_summary(user_id, date_str)
            if ctx:
                results["l1"][date_str] = ctx
                print(f"OK (id={ctx.id[:8]}...)")
            else:
                print("SKIPPED (no data or already exists)")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()

    # L2: weekly summary
    print(f"  L2 weekly {week}...", end=" ", flush=True)
    try:
        ctx = await task._generate_weekly_summary(user_id, week)
        if ctx:
            results["l2"] = ctx
            children_count = len(ctx.properties.children_ids)
            print(f"OK (id={ctx.id[:8]}..., children={children_count})")
        else:
            print("SKIPPED (no data or already exists)")
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()

    # L3: monthly summary
    print(f"  L3 monthly {month}...", end=" ", flush=True)
    try:
        ctx = await task._generate_monthly_summary(user_id, month)
        if ctx:
            results["l3"] = ctx
            children_count = len(ctx.properties.children_ids)
            print(f"OK (id={ctx.id[:8]}..., children={children_count})")
        else:
            print("SKIPPED (no data or already exists)")
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()

    return results


async def verify_summaries(storage, user_id: str, dates: list[str], week: str, month: str):
    """Verify that summaries exist in storage."""
    print()
    all_passed = True

    # Verify L1
    l1_found = 0
    for date_str in dates:
        hits = await storage.search_hierarchy(
            context_type=ContextType.EVENT.value,
            hierarchy_level=1,
            time_bucket_start=date_str,
            time_bucket_end=date_str,
            user_id=user_id,
            top_k=1,
        )
        if hits:
            l1_found += 1
        else:
            print(f"  WARN: L1 summary missing for {date_str}")
    status = "PASS" if l1_found == len(dates) else "PARTIAL"
    if l1_found < len(dates):
        all_passed = False
    print(f"  L1 summaries: {l1_found}/{len(dates)} ({status})")

    # Verify L2
    l2_hits = await storage.search_hierarchy(
        context_type=ContextType.EVENT.value,
        hierarchy_level=2,
        time_bucket_start=week,
        time_bucket_end=week,
        user_id=user_id,
        top_k=1,
    )
    if l2_hits:
        ctx, score = l2_hits[0]
        print(f"  L2 weekly {week}: PASS (id={ctx.id[:8]}...)")
    else:
        print(f"  L2 weekly {week}: FAIL (not found)")
        all_passed = False

    # Verify L3
    l3_hits = await storage.search_hierarchy(
        context_type=ContextType.EVENT.value,
        hierarchy_level=3,
        time_bucket_start=month,
        time_bucket_end=month,
        user_id=user_id,
        top_k=1,
    )
    if l3_hits:
        ctx, score = l3_hits[0]
        print(f"  L3 monthly {month}: PASS (id={ctx.id[:8]}...)")
    else:
        print(f"  L3 monthly {month}: FAIL (not found)")
        all_passed = False

    return all_passed


async def print_summary_content(storage, user_id: str, dates: list[str], week: str, month: str):
    """Print the content of generated summaries for inspection."""
    print("\n--- Generated Summary Content ---\n")

    for level, label, tb_start, tb_end in [
        *[(1, f"L1 Daily {d}", d, d) for d in dates],
        (2, f"L2 Weekly {week}", week, week),
        (3, f"L3 Monthly {month}", month, month),
    ]:
        hits = await storage.search_hierarchy(
            context_type=ContextType.EVENT.value,
            hierarchy_level=level,
            time_bucket_start=tb_start,
            time_bucket_end=tb_end,
            user_id=user_id,
            top_k=1,
        )
        if hits:
            ctx, _ = hits[0]
            print(f"[{label}] {ctx.extracted_data.title}")
            summary = ctx.extracted_data.summary or ""
            # Print first 200 chars of summary
            if len(summary) > 200:
                print(f"  {summary[:200]}...")
            else:
                print(f"  {summary}")
            print(f"  keywords: {ctx.extracted_data.keywords}")
            print()


async def cleanup_test_data(storage, user_id: str, dates: list[str], week: str, month: str):
    """Delete test events and summaries from storage."""
    print("\n[Cleanup] Deleting test data...")

    # Delete L0 events
    l0_filters = {"hierarchy_level": {"$gte": 0, "$lte": 0}}
    l0_dict = await storage.get_all_processed_contexts(
        context_types=[ContextType.EVENT.value],
        limit=500,
        filter=l0_filters,
        user_id=user_id,
    )
    l0_events = l0_dict.get(ContextType.EVENT.value, [])
    deleted = 0
    for ctx in l0_events:
        try:
            await storage.delete_processed_context(ctx.id, ContextType.EVENT)
            deleted += 1
        except Exception:
            pass
    print(f"  Deleted {deleted} L0 events")

    # Delete L1/L2/L3 summaries
    for level in [1, 2, 3]:
        level_name = {1: "L1", 2: "L2", 3: "L3"}[level]
        # Use a broad time range to catch all test summaries
        hits = await storage.search_hierarchy(
            context_type=ContextType.EVENT.value,
            hierarchy_level=level,
            time_bucket_start="2026-02-01",
            time_bucket_end="2026-02-28",
            user_id=user_id,
            top_k=50,
        )
        count = 0
        for ctx, _ in hits:
            try:
                await storage.delete_processed_context(ctx.id, ContextType.EVENT)
                count += 1
            except Exception:
                pass
        print(f"  Deleted {count} {level_name} summaries")


async def main():
    parser = argparse.ArgumentParser(description="Hierarchy Summary Integration Test")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--user-id", default="test_hierarchy_user", help="Test user ID")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test data after test")
    parser.add_argument(
        "--cleanup-only", action="store_true", help="Only clean up existing test data, skip test"
    )
    parser.add_argument(
        "--show-content", action="store_true", help="Print generated summary content"
    )
    args = parser.parse_args()

    user_id = args.user_id

    # Target dates and periods
    dates = ["2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-28"]
    week = "2026-W08"  # ISO week 8: Feb 16-22
    month = "2026-02"

    print("=== Hierarchy Summary Integration Test ===\n")

    # Step 1: Initialize
    print("[1/4] Initializing system...")
    try:
        storage = await init_system(args.config)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # Cleanup-only mode
    if args.cleanup_only:
        await cleanup_test_data(storage, user_id, dates, week, month)
        print("\nCleanup complete.")
        return

    # Step 2: Create test events
    print(f"[2/4] Creating {len(TEST_EVENTS)} test L0 events for user={user_id}...")
    try:
        events = await store_test_events(storage, user_id)
        print(f"  OK ({len(events)} events stored)")
        for evt in events:
            print(f"    {evt.properties.event_time.strftime('%Y-%m-%d %H:%M')} - {evt.extracted_data.title}")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # Step 3: Generate summaries
    print("[3/4] Generating hierarchy summaries...")
    try:
        results = await generate_summaries(user_id, dates, week, month)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # Step 4: Verify
    print("[4/4] Verifying summaries in storage...")
    all_passed = await verify_summaries(storage, user_id, dates, week, month)

    # Optional: print content
    if args.show_content:
        await print_summary_content(storage, user_id, dates, week, month)

    # Summary
    print()
    if all_passed:
        print("=== ALL TESTS PASSED ===")
    else:
        print("=== SOME TESTS FAILED ===")

    # Optional cleanup
    if args.cleanup:
        await cleanup_test_data(storage, user_id, dates, week, month)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
