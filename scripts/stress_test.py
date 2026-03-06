"""
MineContext Stress Test Script
Usage:
    uv run python scripts/stress_test.py [OPTIONS]

Options:
    --base-url      Base URL (default: http://localhost:8088)
    --concurrency   Concurrent requests (default: 50)
    --duration      Test duration in seconds (default: 30)
    --endpoints     Comma-separated endpoints to test: health,search,push_chat,memory_cache,vector_search
                    (default: all)
    --ramp-up       Ramp-up seconds to gradually increase concurrency (default: 5)
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field

import aiohttp


# ---------------------------------------------------------------------------
# Test payloads
# ---------------------------------------------------------------------------

SEARCH_QUERIES = [
    "project timeline and budget",
    "meeting notes from last week",
    "architecture design decisions",
    "deployment pipeline configuration",
    "user authentication flow",
    "database migration strategy",
    "API performance optimization",
    "microservice communication patterns",
    "error handling best practices",
    "code review feedback",
]

CHAT_MESSAGES = [
    "I prefer using Python for data analysis",
    "The meeting with John is scheduled for next Tuesday at 3 PM",
    "We decided to use Redis for caching instead of Memcached",
    "The Q3 budget review shows a 15% increase in cloud costs",
    "New team member Alice will join the backend team next week",
    "The CI/CD pipeline needs to be updated for the new microservice",
    "Customer feedback suggests we need better search functionality",
    "The database migration to PostgreSQL is planned for next month",
    "Security audit found two medium-severity issues in the API layer",
    "Sprint retrospective: we need to improve our testing coverage",
]


def make_search_payload():
    return {
        "query": random.choice(SEARCH_QUERIES),
        "top_k": 10,
        "drill_up": False,
        "user_id": f"stress_user_{random.randint(1, 10)}",
        "device_id": "default",
        "agent_id": "default",
    }


def make_vector_search_payload():
    return {
        "query": random.choice(SEARCH_QUERIES),
        "top_k": 10,
        "context_types": ["knowledge", "document"],
        "filters": {},
        "user_id": f"stress_user_{random.randint(1, 10)}",
        "device_id": "default",
        "agent_id": "default",
    }


def make_push_chat_payload():
    msg = random.choice(CHAT_MESSAGES)
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": msg}]},
        ],
        "user_id": f"stress_user_{random.randint(1, 10)}",
        "device_id": "default",
        "agent_id": "default",
    }


# ---------------------------------------------------------------------------
# Endpoint definitions
# ---------------------------------------------------------------------------

ENDPOINTS = {
    "health": {
        "method": "GET",
        "path": "/api/health",
        "payload": None,
        "description": "Health check (lightweight read)",
    },
    "search": {
        "method": "POST",
        "path": "/api/search",
        "payload": make_search_payload,
        "description": "Event search with semantic query",
    },
    "vector_search": {
        "method": "POST",
        "path": "/api/vector_search",
        "payload": make_vector_search_payload,
        "description": "Direct vector search",
    },
    "push_chat": {
        "method": "POST",
        "path": "/api/push/chat",
        "payload": make_push_chat_payload,
        "description": "Push chat message (buffer mode)",
    },
    "memory_cache": {
        "method": "GET",
        "path": "/api/memory-cache",
        "payload": None,
        "description": "Memory cache snapshot",
        "params": {
            "user_id": "stress_user_1",
            "device_id": "default",
            "agent_id": "default",
            "recent_days": "7",
            "force_refresh": "false",
        },
    },
}


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class EndpointStats:
    name: str
    latencies: list = field(default_factory=list)
    errors: int = 0
    status_codes: dict = field(default_factory=dict)
    total: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

async def worker(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint_name: str,
    endpoint_cfg: dict,
    stats: EndpointStats,
    stop_event: asyncio.Event,
    semaphore: asyncio.Semaphore,
):
    url = base_url + endpoint_cfg["path"]
    method = endpoint_cfg["method"]
    payload_fn = endpoint_cfg.get("payload")
    params = endpoint_cfg.get("params")

    while not stop_event.is_set():
        async with semaphore:
            if stop_event.is_set():
                break
            body = payload_fn() if callable(payload_fn) else None
            # randomize params for memory_cache
            if params and endpoint_name == "memory_cache":
                p = dict(params)
                p["user_id"] = f"stress_user_{random.randint(1, 10)}"
            else:
                p = params

            t0 = time.perf_counter()
            try:
                if method == "GET":
                    async with session.get(url, params=p, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        await resp.read()
                        status = resp.status
                else:
                    async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        await resp.read()
                        status = resp.status
                latency = (time.perf_counter() - t0) * 1000  # ms
                stats.latencies.append(latency)
                stats.status_codes[status] = stats.status_codes.get(status, 0) + 1
                stats.total += 1
            except Exception as e:
                stats.errors += 1
                stats.total += 1


# ---------------------------------------------------------------------------
# Run test for one endpoint
# ---------------------------------------------------------------------------

async def run_endpoint_test(
    base_url: str,
    endpoint_name: str,
    endpoint_cfg: dict,
    concurrency: int,
    duration: int,
    ramp_up: int,
) -> EndpointStats:
    stats = EndpointStats(name=endpoint_name)
    stop_event = asyncio.Event()
    semaphore = asyncio.Semaphore(concurrency)

    headers = {"X-API-Key": "your-secure-api-key"}
    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        # Spawn more workers than concurrency to keep the pipeline full
        worker_count = concurrency * 2
        stats.start_time = time.perf_counter()

        tasks = [
            asyncio.create_task(
                worker(session, base_url, endpoint_name, endpoint_cfg, stats, stop_event, semaphore)
            )
            for _ in range(worker_count)
        ]

        await asyncio.sleep(duration)
        stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        stats.end_time = time.perf_counter()

    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(all_stats: list[EndpointStats]):
    print("\n" + "=" * 80)
    print("  STRESS TEST RESULTS")
    print("=" * 80)

    for stats in all_stats:
        elapsed = stats.end_time - stats.start_time
        rps = stats.total / elapsed if elapsed > 0 else 0
        error_rate = (stats.errors / stats.total * 100) if stats.total > 0 else 0

        cfg = ENDPOINTS[stats.name]
        print(f"\n{'─' * 80}")
        print(f"  {stats.name.upper()}  ({cfg['method']} {cfg['path']})")
        print(f"  {cfg['description']}")
        print(f"{'─' * 80}")
        print(f"  Total requests : {stats.total}")
        print(f"  Duration       : {elapsed:.1f}s")
        print(f"  Throughput     : {rps:.1f} req/s")
        print(f"  Errors         : {stats.errors}  ({error_rate:.1f}%)")

        if stats.status_codes:
            codes_str = ", ".join(f"{k}: {v}" for k, v in sorted(stats.status_codes.items()))
            print(f"  Status codes   : {codes_str}")

        if stats.latencies:
            lats = sorted(stats.latencies)
            print(f"  Latency (ms):")
            print(f"    min    : {lats[0]:.1f}")
            print(f"    p50    : {percentile(lats, 50):.1f}")
            print(f"    p90    : {percentile(lats, 90):.1f}")
            print(f"    p95    : {percentile(lats, 95):.1f}")
            print(f"    p99    : {percentile(lats, 99):.1f}")
            print(f"    max    : {lats[-1]:.1f}")
            print(f"    avg    : {statistics.mean(lats):.1f}")
        else:
            print(f"  Latency: N/A (all requests failed)")

    print(f"\n{'=' * 80}\n")


def percentile(sorted_data, p):
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="MineContext Stress Test")
    parser.add_argument("--base-url", default="http://localhost:8088")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--endpoints", default="health,search,push_chat,memory_cache,vector_search")
    parser.add_argument("--ramp-up", type=int, default=5)
    args = parser.parse_args()

    selected = [e.strip() for e in args.endpoints.split(",") if e.strip() in ENDPOINTS]
    if not selected:
        print("No valid endpoints selected. Available:", ", ".join(ENDPOINTS.keys()))
        return

    print(f"MineContext Stress Test")
    print(f"  Base URL    : {args.base_url}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Duration    : {args.duration}s per endpoint")
    print(f"  Endpoints   : {', '.join(selected)}")
    print()

    # Warm up: single request per endpoint
    headers = {"X-API-Key": "your-secure-api-key"}
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        for ep_name in selected:
            cfg = ENDPOINTS[ep_name]
            url = args.base_url + cfg["path"]
            try:
                if cfg["method"] == "GET":
                    async with session.get(url, params=cfg.get("params"), timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        await resp.read()
                        print(f"  Warmup {ep_name}: {resp.status}")
                else:
                    body = cfg["payload"]() if callable(cfg.get("payload")) else None
                    async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        await resp.read()
                        print(f"  Warmup {ep_name}: {resp.status}")
            except Exception as e:
                print(f"  Warmup {ep_name}: FAILED ({e})")

    print()
    all_stats = []

    for ep_name in selected:
        cfg = ENDPOINTS[ep_name]
        print(f"Testing {ep_name} ({cfg['method']} {cfg['path']}) for {args.duration}s ...")
        stats = await run_endpoint_test(
            base_url=args.base_url,
            endpoint_name=ep_name,
            endpoint_cfg=cfg,
            concurrency=args.concurrency,
            duration=args.duration,
            ramp_up=args.ramp_up,
        )
        all_stats.append(stats)
        print(f"  -> {stats.total} requests, {stats.errors} errors")

    print_report(all_stats)


if __name__ == "__main__":
    asyncio.run(main())
