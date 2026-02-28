#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct hierarchy summary test — bypasses scheduler, calls HierarchySummaryTask directly.
Uses the same config as the running server to connect to VikingDB/MySQL.

Usage:
    uv run python tests/run_hierarchy_direct.py --config config/config.yaml
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main():
    parser = argparse.ArgumentParser(description="Run hierarchy summary directly")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--user-id", default="test_hierarchy_user")
    args = parser.parse_args()

    user_id = args.user_id
    os.environ["OPENCONTEXT_CONFIG_PATH"] = args.config

    # Step 1: Config
    print("[1/5] Loading config...", flush=True)
    t0 = time.time()
    from opencontext.config.global_config import GlobalConfig

    GlobalConfig.get_instance()
    print(f"  OK ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Storage
    print("[2/5] Initializing storage...", flush=True)
    t0 = time.time()

    from opencontext.storage.global_storage import GlobalStorage

    gs = GlobalStorage.get_instance()

    print("  Calling ensure_initialized()...", flush=True)
    try:
        await asyncio.wait_for(gs.ensure_initialized(), timeout=30)
    except asyncio.TimeoutError:
        print("  TIMEOUT after 30s! VikingDB init may be slow.", flush=True)
        print("  Retrying with 60s timeout...", flush=True)
        try:
            await asyncio.wait_for(gs.ensure_initialized(), timeout=60)
        except asyncio.TimeoutError:
            print("  FAILED: Storage init timed out after 60s", flush=True)
            sys.exit(1)

    from opencontext.storage.global_storage import get_storage

    storage = get_storage()
    if not storage:
        print("  FAILED: storage is None after init", flush=True)
        sys.exit(1)
    print(f"  OK ({time.time()-t0:.1f}s)", flush=True)

    # Step 3: Check existing L0 events for this user
    print(f"[3/5] Checking existing L0 events for user={user_id}...", flush=True)
    from opencontext.models.enums import ContextType

    l0_dict = await storage.get_all_processed_contexts(
        context_types=[ContextType.EVENT.value],
        limit=100,
        filter={"hierarchy_level": {"$gte": 0, "$lte": 0}},
        user_id=user_id,
    )
    l0_events = l0_dict.get(ContextType.EVENT.value, [])
    print(f"  Found {len(l0_events)} L0 events", flush=True)
    for evt in l0_events[:10]:
        et = evt.properties.event_time
        title = evt.extracted_data.title or "(no title)"
        print(f"    {et} - {title[:60]}", flush=True)

    if not l0_events:
        print("  WARNING: No L0 events found. Hierarchy summary needs events to summarize.", flush=True)
        print("  Push some chat data first via: curl -X POST http://localhost:1733/api/push/chat ...", flush=True)

    # Step 4: Run HierarchySummaryTask
    print(f"[4/5] Running HierarchySummaryTask.execute()...", flush=True)
    t0 = time.time()

    from opencontext.periodic_task.base import TaskContext
    from opencontext.periodic_task.hierarchy_summary import HierarchySummaryTask

    task = HierarchySummaryTask()
    context = TaskContext(
        user_id=user_id,
        device_id="default",
        agent_id="default",
        task_type="hierarchy_summary",
    )

    result = await task.execute(context)
    elapsed = time.time() - t0
    print(f"  Result: success={result.success}, message={result.message}", flush=True)
    if result.data:
        print(f"  Data: {result.data}", flush=True)
    if result.error:
        print(f"  Error: {result.error}", flush=True)
    print(f"  Time: {elapsed:.1f}s", flush=True)

    # Step 5: Verify summaries
    print(f"[5/5] Verifying generated summaries...", flush=True)
    for level, label in [(1, "L1 daily"), (2, "L2 weekly"), (3, "L3 monthly")]:
        hits = await storage.search_hierarchy(
            context_type=ContextType.EVENT.value,
            hierarchy_level=level,
            time_bucket_start="2026-02-01",
            time_bucket_end="2026-03-01",
            user_id=user_id,
            top_k=10,
        )
        if hits:
            for ctx, score in hits:
                tb = ctx.properties.time_bucket
                title = ctx.extracted_data.title or "(no title)"
                print(f"  {label} [{tb}]: {title[:80]}", flush=True)
        else:
            print(f"  {label}: none found", flush=True)

    print(f"\n=== Done ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
