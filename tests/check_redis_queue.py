#!/usr/bin/env python
"""Check Redis scheduler queue state for hierarchy_summary."""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main():
    os.environ["OPENCONTEXT_CONFIG_PATH"] = os.environ.get(
        "OPENCONTEXT_CONFIG_PATH", "config/config.yaml"
    )
    from opencontext.config.global_config import GlobalConfig

    GlobalConfig.get_instance()
    from opencontext.config.global_config import get_config

    redis_config = get_config("redis")

    import redis.asyncio as aioredis

    r = aioredis.Redis(
        host=redis_config.get("host", "localhost"),
        port=int(redis_config.get("port", 6379)),
        password=redis_config.get("password", "") or None,
        db=int(redis_config.get("db", 0)),
        decode_responses=True,
    )

    now = int(time.time())
    queue_key = "scheduler:queue:hierarchy_summary"

    # Count total
    total = await r.zcard(queue_key)
    print(f"Total tasks in queue: {total}")

    # Count overdue (score <= now)
    overdue = await r.zcount(queue_key, "-inf", now)
    print(f"Overdue tasks (due now or earlier): {overdue}")

    # Count future (score > now)
    future = await r.zcount(queue_key, now + 1, "+inf")
    print(f"Future tasks: {future}")

    # Show first 5 entries (lowest scores = most overdue)
    earliest = await r.zrange(queue_key, 0, 4, withscores=True)
    print(f"\nEarliest 5 tasks:")
    for member, score in earliest:
        ago = now - int(score)
        print(f"  {member} -> due {ago}s ago ({int(score)})")

    # Show our test user task
    test_score = await r.zscore(queue_key, "test_hierarchy_user:default:default")
    if test_score:
        ago = now - int(test_score)
        print(f"\nTest user task: due {ago}s ago (score={int(test_score)})")
    else:
        print(f"\nTest user task: NOT in queue")

    # Check task type config
    config_key = "scheduler:task_type:hierarchy_summary"
    config = await r.hgetall(config_key)
    print(f"\nTask type config: {config}")

    # Check if any task hashes exist for the earliest entries
    print(f"\nChecking task hashes for first 3 entries:")
    for member, score in earliest[:3]:
        task_key = f"scheduler:task:hierarchy_summary:{member}"
        exists = await r.exists(task_key)
        if exists:
            data = await r.hgetall(task_key)
            print(f"  {member}: {data}")
        else:
            print(f"  {member}: NO HASH (orphaned queue entry)")

    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
