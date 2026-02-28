#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trigger hierarchy summary task immediately via Redis.
Injects a task with due_time=now so the running server's scheduler picks it up.

Usage:
    uv run python tests/trigger_hierarchy.py --user-id test_hierarchy_user
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main():
    parser = argparse.ArgumentParser(description="Trigger hierarchy summary via Redis")
    parser.add_argument("--user-id", default="test_hierarchy_user")
    parser.add_argument("--device-id", default="default")
    parser.add_argument("--agent-id", default="default")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    os.environ["OPENCONTEXT_CONFIG_PATH"] = args.config

    from opencontext.config.global_config import GlobalConfig

    GlobalConfig.get_instance()

    from opencontext.config.global_config import get_config

    redis_config = get_config("redis")
    if not redis_config or not redis_config.get("enabled"):
        print("ERROR: Redis is not enabled in config")
        sys.exit(1)

    # Connect to Redis directly
    import redis.asyncio as aioredis

    host = redis_config.get("host", "localhost")
    port = int(redis_config.get("port", 6379))
    password = redis_config.get("password", "") or None
    db = int(redis_config.get("db", 0))

    r = aioredis.Redis(host=host, port=port, password=password, db=db, decode_responses=True)

    try:
        await r.ping()
        print(f"Redis connected: {host}:{port}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Redis: {e}")
        sys.exit(1)

    user_key = f"{args.user_id}:{args.device_id}:{args.agent_id}"
    task_type = "hierarchy_summary"

    # Key names (matching RedisTaskScheduler conventions)
    task_key = f"scheduler:task:{task_type}:{user_key}"
    queue_key = f"scheduler:queue:{task_type}"
    last_exec_key = f"scheduler:last_exec:{task_type}:{user_key}"
    fail_count_key = f"scheduler:fail_count:{task_type}:{user_key}"

    # Check current state
    print(f"\n--- Current state for {user_key} ---")
    existing_task = await r.hgetall(task_key)
    if existing_task:
        print(f"  Existing task: {existing_task}")
    else:
        print("  No existing task")

    last_exec = await r.get(last_exec_key)
    if last_exec:
        ago = int(time.time()) - int(last_exec)
        print(f"  Last executed: {ago}s ago")
    else:
        print("  Never executed")

    queue_score = await r.zscore(queue_key, user_key)
    if queue_score:
        due_in = int(queue_score) - int(time.time())
        print(f"  Queue due in: {due_in}s")
    else:
        print("  Not in queue")

    fail_count = await r.get(fail_count_key)
    if fail_count:
        print(f"  Fail count: {fail_count}")

    # Clear blockers
    print("\n--- Clearing blockers ---")
    await r.delete(last_exec_key)
    print(f"  Deleted last_exec key")
    await r.delete(fail_count_key)
    print(f"  Deleted fail_count key")
    await r.delete(task_key)
    print(f"  Deleted existing task")

    # Create task with due_time = now (immediate execution)
    now = int(time.time())
    task_info = {
        "task_type": task_type,
        "user_key": user_key,
        "user_id": args.user_id,
        "device_id": args.device_id,
        "agent_id": args.agent_id,
        "status": "pending",
        "created_at": str(now),
        "scheduled_at": str(now),  # Due NOW
        "last_activity": str(now),
        "retry_count": "0",
    }

    pipe = r.pipeline()
    pipe.hset(task_key, mapping=task_info)
    pipe.expire(task_key, 172800)  # 48h TTL
    pipe.zadd(queue_key, {user_key: now})  # Due immediately
    await pipe.execute()

    print(f"\n--- Task injected ---")
    print(f"  Task key: {task_key}")
    print(f"  Queue key: {queue_key}")
    print(f"  Due at: now (score={now})")
    print(f"\nThe running server's scheduler will pick this up on next check cycle.")
    print(f"Check scheduler interval in config (default: 10s).")

    # Monitor for completion
    print(f"\n--- Monitoring task status (30s timeout) ---")
    for i in range(30):
        await asyncio.sleep(1)
        task_state = await r.hgetall(task_key)
        if not task_state:
            print(f"  [{i+1}s] Task completed and cleaned up")
            break
        status = task_state.get("status", "unknown")
        if status == "completed":
            print(f"  [{i+1}s] Task COMPLETED!")
            break
        elif status == "failed":
            print(f"  [{i+1}s] Task FAILED: {task_state}")
            break
        elif status == "running":
            print(f"  [{i+1}s] Task running...")
        else:
            if i % 5 == 0:
                print(f"  [{i+1}s] Status: {status}")
    else:
        print("  Timeout waiting for task completion")
        final = await r.hgetall(task_key)
        print(f"  Final state: {final}")

    # Check logs for results
    print("\n--- Check server logs for hierarchy summary output ---")
    print(f"  grep 'hierarchy' logs/opencontext_*.log | tail -20")

    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
