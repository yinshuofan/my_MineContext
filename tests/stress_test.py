"""
MineContext Stress Test Script

Stress tests the three core API endpoints:
- POST /api/push/chat
- POST /api/search
- GET /api/memory-cache

Usage:
    python tests/stress_test.py                                    # defaults: 200 concurrent, 60s
    python tests/stress_test.py --concurrency 5 --duration 10      # light smoke test
    python tests/stress_test.py --push-mode direct --concurrency 50 # full pipeline test
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class StressTestConfig:
    base_url: str = "http://localhost:1733"
    concurrency: int = 200
    duration: int = 60
    warmup_time: int = 10
    cooldown_time: int = 5
    report_interval: int = 5
    api_key: Optional[str] = None

    weight_push: int = 50
    weight_search: int = 30
    weight_cache: int = 20

    push_process_mode: str = "buffer"
    push_flush: bool = False

    search_strategy: str = "fast"
    search_top_k: int = 20

    connector_limit: int = 300
    request_timeout: int = 65


# ---------------------------------------------------------------------------
# Test data pools
# ---------------------------------------------------------------------------

# Realistic chat messages covering all 5 context types
_CHAT_TEXTS = [
    # Profile-trigger
    "I prefer using Python for backend development and TypeScript for frontend.",
    "My timezone is UTC+8 and I usually work from 9am to 6pm.",
    "I like concise, technical responses without unnecessary explanations.",
    "Please always respond in Chinese when I ask in Chinese.",
    "I'm a senior engineer focused on distributed systems.",
    "I prefer dark mode and vim keybindings in my IDE.",
    "My communication style is direct and to the point.",
    "I usually review PRs in the morning before standup.",
    # Entity-trigger
    "John from the engineering team is leading the database migration project.",
    "The marketing team consists of Sarah, Mike, and Lisa.",
    "Talk to Alex about the Kubernetes cluster configuration.",
    "Our client Acme Corp wants the deliverables by March 15th.",
    "Professor Wang recommended using the actor model for concurrency.",
    "The DevOps team led by Chen Wei is handling the CI/CD pipeline.",
    "Emily from design shared the new mockups for the dashboard.",
    "The backend squad (Tom, Jerry, Alice) owns the payment service.",
    # Event-trigger
    "Had a meeting with the design team about the new dashboard UI this morning.",
    "Deployed version 2.3.1 to production at 3:45 PM today.",
    "Completed the code review for PR #421 on the authentication module.",
    "Sprint retrospective concluded with 3 action items for next sprint.",
    "The weekly all-hands meeting covered Q1 OKR progress.",
    "Database migration completed successfully after 2 hours of downtime.",
    "Onboarding session for 3 new hires happened this afternoon.",
    "Incident postmortem for the outage on Feb 20 was documented.",
    # Knowledge-trigger
    "React hooks allow functional components to use state and lifecycle methods.",
    "The circuit breaker pattern prevents cascading failures in microservices.",
    "In Python, asyncio.to_thread() bridges sync code into an async event loop.",
    "Vector databases store embeddings and support semantic similarity search.",
    "The CAP theorem states you can only have two of consistency, availability, and partition tolerance.",
    "Redis Streams provide a log-based data structure similar to Kafka topics.",
    "SQLAlchemy's QueuePool manages database connection pooling with configurable limits.",
    "FastAPI uses Pydantic for request validation and automatic OpenAPI documentation.",
    # Mixed/casual
    "Can you help me with the project timeline?",
    "What was discussed in yesterday's standup?",
    "Remind me what decisions we made about the API versioning strategy.",
    "How does our authentication flow work?",
    "What are the main risks for the upcoming release?",
    "Summarize the key points from last week's planning session.",
    "Who is responsible for the monitoring infrastructure?",
    "What patterns do we use for error handling in the backend?",
]

_SEARCH_QUERIES = [
    "project timeline and budget",
    "what are John's responsibilities",
    "microservice architecture patterns",
    "recent meeting summaries",
    "my communication preferences",
    "database migration progress",
    "sprint retrospective action items",
    "how does the authentication flow work",
    "Kubernetes cluster configuration",
    "API versioning strategy decisions",
    "error handling patterns in backend",
    "deployment procedures and checklist",
    "team structure and responsibilities",
    "performance optimization techniques",
    "monitoring and alerting setup",
    "code review best practices",
    "CI/CD pipeline configuration",
    "incident response procedures",
    "data backup and recovery plan",
    "security audit findings",
    "recent production deployments",
    "user feedback from last release",
    "technical debt priorities",
    "onboarding process for new engineers",
    "quarterly OKR progress",
    "payment service architecture",
    "Redis caching strategy",
    "vector search implementation details",
    "weekly all-hands meeting notes",
    "Python asyncio best practices",
]

_CONTEXT_TYPES = ["profile", "entity", "document", "event", "knowledge"]

_ASSISTANT_RESPONSES = [
    "I've noted that information.",
    "Got it, I'll keep that in mind for future interactions.",
    "That's been recorded. Anything else you'd like to add?",
    "Understood. I'll incorporate that into our context.",
    "Thanks for sharing. This helps me provide better assistance.",
]


class TestDataPool:
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.user_ids = [f"stress_user_{i:03d}" for i in range(1, 51)]
        self.device_ids = [f"stress_device_{i}" for i in range(1, 11)]
        self.agent_ids = [f"stress_agent_{i}" for i in range(1, 6)]

        self._chat_bodies = self._generate_chat_bodies(100)
        self._search_bodies = self._generate_search_bodies(80)
        self._cache_params = self._generate_cache_params(40)

    def _random_identity(self) -> dict:
        return {
            "user_id": random.choice(self.user_ids),
            "device_id": random.choice(self.device_ids),
            "agent_id": random.choice(self.agent_ids),
        }

    def _generate_chat_bodies(self, count: int) -> list:
        bodies = []
        for _ in range(count):
            num_messages = random.randint(1, 5)
            messages = []
            for j in range(num_messages):
                if j % 2 == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": random.choice(_CHAT_TEXTS)}
                            ],
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": random.choice(_ASSISTANT_RESPONSES),
                                }
                            ],
                        }
                    )
            body = {
                **self._random_identity(),
                "messages": messages,
                "process_mode": self.config.push_process_mode,
                "flush_immediately": self.config.push_flush,
            }
            bodies.append(body)
        return bodies

    def _generate_search_bodies(self, count: int) -> list:
        bodies = []
        for _ in range(count):
            body = {
                **self._random_identity(),
                "query": random.choice(_SEARCH_QUERIES),
                "strategy": self.config.search_strategy,
                "top_k": self.config.search_top_k,
            }
            # 30% of queries use a random subset of context types
            if random.random() < 0.3:
                k = random.randint(1, 4)
                body["context_types"] = random.sample(_CONTEXT_TYPES, k)
            bodies.append(body)
        return bodies

    def _generate_cache_params(self, count: int) -> list:
        params_list = []
        for _ in range(count):
            params = {
                **self._random_identity(),
                "max_accessed": random.choice([5, 10, 20, 50]),
            }
            if random.random() < 0.05:
                params["force_refresh"] = "true"
            params_list.append(params)
        return params_list

    def random_push_body(self) -> dict:
        return random.choice(self._chat_bodies)

    def random_search_body(self) -> dict:
        return random.choice(self._search_bodies)

    def random_cache_params(self) -> dict:
        return random.choice(self._cache_params)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class RequestResult:
    endpoint: str
    status_code: int
    latency_ms: float
    error: Optional[str]
    timestamp: float


class MetricsCollector:
    def __init__(self):
        self.results: list[RequestResult] = []
        self._last_report_idx: int = 0
        self._start_time: float = 0.0

    def set_start_time(self, t: float) -> None:
        self._start_time = t

    def record(self, result: RequestResult) -> None:
        self.results.append(result)

    def get_interval_stats(self) -> dict:
        current_idx = len(self.results)
        interval_results = self.results[self._last_report_idx : current_idx]
        self._last_report_idx = current_idx

        if not interval_results:
            return {
                "rps": 0.0,
                "avg_ms": 0.0,
                "err_pct": 0.0,
                "counts": defaultdict(int),
            }

        elapsed = interval_results[-1].timestamp - interval_results[0].timestamp
        rps = len(interval_results) / max(elapsed, 0.001)
        avg_ms = statistics.mean(r.latency_ms for r in interval_results)
        errors = sum(1 for r in interval_results if r.error is not None)
        err_pct = (errors / len(interval_results)) * 100

        counts: dict[str, int] = defaultdict(int)
        for r in interval_results:
            counts[r.endpoint] += 1

        return {"rps": rps, "avg_ms": avg_ms, "err_pct": err_pct, "counts": counts}

    def get_final_report(self) -> dict:
        if not self.results:
            return {}

        total_elapsed = self.results[-1].timestamp - self._start_time
        total = len(self.results)
        successful = sum(1 for r in self.results if r.error is None)
        failed = total - successful

        # Per-endpoint breakdown
        by_endpoint: dict[str, list[RequestResult]] = defaultdict(list)
        for r in self.results:
            by_endpoint[r.endpoint].append(r)

        endpoint_stats = {}
        for ep, results in sorted(by_endpoint.items()):
            latencies = sorted(r.latency_ms for r in results)
            ep_errors = [r for r in results if r.error is not None]

            # Error categories
            error_cats: dict[str, int] = defaultdict(int)
            for r in ep_errors:
                error_cats[self._categorize_error(r)] += 1

            endpoint_stats[ep] = {
                "count": len(results),
                "rps": len(results) / max(total_elapsed, 0.001),
                "p50": self._percentile(latencies, 0.50),
                "p95": self._percentile(latencies, 0.95),
                "p99": self._percentile(latencies, 0.99),
                "max": latencies[-1] if latencies else 0,
                "errors": len(ep_errors),
                "error_pct": (len(ep_errors) / len(results)) * 100 if results else 0,
                "error_categories": dict(error_cats),
            }

        # Overall latency
        all_latencies = sorted(r.latency_ms for r in self.results)
        return {
            "duration_s": total_elapsed,
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_pct": (successful / total) * 100 if total else 0,
            "overall_rps": total / max(total_elapsed, 0.001),
            "overall_p50": self._percentile(all_latencies, 0.50),
            "overall_p95": self._percentile(all_latencies, 0.95),
            "overall_p99": self._percentile(all_latencies, 0.99),
            "overall_max": all_latencies[-1] if all_latencies else 0,
            "endpoints": endpoint_stats,
        }

    @staticmethod
    def _categorize_error(r: RequestResult) -> str:
        if r.error and "timeout" in r.error.lower():
            return "timeout"
        if r.error and ("connect" in r.error.lower() or "refused" in r.error.lower()):
            return "connection_error"
        if 400 <= r.status_code < 500:
            return "http_4xx"
        if 500 <= r.status_code < 600:
            return "http_5xx"
        return "other"

    @staticmethod
    def _percentile(sorted_data: list, pct: float) -> float:
        if not sorted_data:
            return 0.0
        idx = int(len(sorted_data) * pct)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]


# ---------------------------------------------------------------------------
# Request executor
# ---------------------------------------------------------------------------
class RequestExecutor:
    ENDPOINTS = ["push_chat", "search", "memory_cache"]

    def __init__(
        self,
        session: aiohttp.ClientSession,
        config: StressTestConfig,
        data_pool: TestDataPool,
    ):
        self.session = session
        self.config = config
        self.data = data_pool
        self._weights = [config.weight_push, config.weight_search, config.weight_cache]

    async def execute_random(self) -> RequestResult:
        (endpoint,) = random.choices(self.ENDPOINTS, weights=self._weights, k=1)
        if endpoint == "push_chat":
            return await self._push_chat()
        elif endpoint == "search":
            return await self._search()
        else:
            return await self._memory_cache()

    async def _push_chat(self) -> RequestResult:
        url = f"{self.config.base_url}/api/push/chat"
        body = self.data.random_push_body()
        t0 = time.monotonic()
        try:
            async with self.session.post(url, json=body) as resp:
                data = await resp.json()
                latency = (time.monotonic() - t0) * 1000
                if resp.status == 200 and data.get("code") == 0:
                    return RequestResult("push_chat", resp.status, latency, None, t0)
                else:
                    msg = data.get("message", f"status={resp.status}")
                    return RequestResult("push_chat", resp.status, latency, msg, t0)
        except asyncio.TimeoutError:
            latency = (time.monotonic() - t0) * 1000
            return RequestResult("push_chat", 0, latency, "timeout", t0)
        except aiohttp.ClientConnectorError as e:
            latency = (time.monotonic() - t0) * 1000
            return RequestResult("push_chat", 0, latency, f"connection_error: {e}", t0)
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            return RequestResult("push_chat", 0, latency, str(e), t0)

    async def _search(self) -> RequestResult:
        url = f"{self.config.base_url}/api/search"
        body = self.data.random_search_body()
        t0 = time.monotonic()
        try:
            async with self.session.post(url, json=body) as resp:
                data = await resp.json()
                latency = (time.monotonic() - t0) * 1000
                if resp.status == 200 and data.get("success") is True:
                    return RequestResult("search", resp.status, latency, None, t0)
                else:
                    msg = data.get("error", data.get("message", f"status={resp.status}"))
                    return RequestResult("search", resp.status, latency, msg, t0)
        except asyncio.TimeoutError:
            latency = (time.monotonic() - t0) * 1000
            return RequestResult("search", 0, latency, "timeout", t0)
        except aiohttp.ClientConnectorError as e:
            latency = (time.monotonic() - t0) * 1000
            return RequestResult("search", 0, latency, f"connection_error: {e}", t0)
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            return RequestResult("search", 0, latency, str(e), t0)

    async def _memory_cache(self) -> RequestResult:
        url = f"{self.config.base_url}/api/memory-cache"
        params = self.data.random_cache_params()
        t0 = time.monotonic()
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                latency = (time.monotonic() - t0) * 1000
                if resp.status == 200 and data.get("success") is True:
                    return RequestResult("memory_cache", resp.status, latency, None, t0)
                else:
                    msg = data.get("error", data.get("message", f"status={resp.status}"))
                    return RequestResult(
                        "memory_cache", resp.status, latency, msg, t0
                    )
        except asyncio.TimeoutError:
            latency = (time.monotonic() - t0) * 1000
            return RequestResult("memory_cache", 0, latency, "timeout", t0)
        except aiohttp.ClientConnectorError as e:
            latency = (time.monotonic() - t0) * 1000
            return RequestResult(
                "memory_cache", 0, latency, f"connection_error: {e}", t0
            )
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            return RequestResult("memory_cache", 0, latency, str(e), t0)


# ---------------------------------------------------------------------------
# Stress test runner
# ---------------------------------------------------------------------------
class StressTestRunner:
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.data_pool = TestDataPool(config)
        self.metrics = MetricsCollector()
        self._stop_event = asyncio.Event()
        self._test_start: float = 0.0

    def _elapsed_str(self) -> str:
        elapsed = time.monotonic() - self._test_start
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        return f"{mins:02d}:{secs:02d}"

    async def run(self) -> None:
        self._print_banner()

        connector = aiohttp.TCPConnector(
            limit=self.config.connector_limit,
            limit_per_host=self.config.connector_limit,
        )
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=headers
        ) as session:
            # Health check
            if not await self._health_check(session):
                print("\nHealth check failed. Is the server running?")
                sys.exit(1)

            self._test_start = time.monotonic()
            self.metrics.set_start_time(self._test_start)

            executor = RequestExecutor(session, self.config, self.data_pool)

            # Warmup phase
            print(
                f"[{self._elapsed_str()}] Starting warmup phase ({self.config.warmup_time}s)..."
            )
            workers = await self._warmup_phase(executor)

            # Sustained phase
            print(
                f"[{self._elapsed_str()}] Warmup complete. Sustained load phase "
                f"({self.config.duration}s, {self.config.concurrency} workers)..."
            )
            reporter_task = asyncio.create_task(self._reporter_loop())

            await asyncio.sleep(self.config.duration)

            # Cooldown
            print(f"\n[{self._elapsed_str()}] Stopping workers (cooldown {self.config.cooldown_time}s)...")
            self._stop_event.set()
            reporter_task.cancel()

            done, pending = await asyncio.wait(
                workers, timeout=self.config.cooldown_time
            )
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        self._print_final_report()

    async def _health_check(self, session: aiohttp.ClientSession) -> bool:
        url = f"{self.config.base_url}/health"
        print(f"[--:--] Health check ({url})...", end=" ", flush=True)
        t0 = time.monotonic()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                latency = (time.monotonic() - t0) * 1000
                if resp.status == 200:
                    print(f"OK ({latency:.0f}ms)")
                    return True
                else:
                    print(f"FAILED (status={resp.status})")
                    return False
        except Exception as e:
            print(f"FAILED ({e})")
            return False

    async def _warmup_phase(self, executor: RequestExecutor) -> list:
        workers = []
        total = self.config.concurrency
        warmup = self.config.warmup_time

        if warmup <= 0:
            # No warmup, start all workers at once
            for _ in range(total):
                workers.append(asyncio.create_task(self._worker(executor)))
            return workers

        batch_interval = warmup / max(total, 1)
        # Spawn at least 1 per interval, adjust for large concurrency
        if batch_interval < 0.05:
            # Spawn in bigger batches
            batches = warmup * 20  # 20 batches per second
            per_batch = max(1, total // batches)
            batch_interval = warmup / batches
        else:
            per_batch = 1

        spawned = 0
        while spawned < total and not self._stop_event.is_set():
            count = min(per_batch, total - spawned)
            for _ in range(count):
                workers.append(asyncio.create_task(self._worker(executor)))
            spawned += count
            if spawned < total:
                await asyncio.sleep(batch_interval)

        return workers

    async def _worker(self, executor: RequestExecutor) -> None:
        while not self._stop_event.is_set():
            try:
                result = await executor.execute_random()
                self.metrics.record(result)
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.metrics.record(
                    RequestResult(
                        endpoint="unknown",
                        status_code=0,
                        latency_ms=0,
                        error=str(e),
                        timestamp=time.monotonic(),
                    )
                )

    async def _reporter_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.config.report_interval)
                stats = self.metrics.get_interval_stats()
                counts = stats["counts"]
                counts_str = " ".join(
                    f"{ep}:{counts.get(ep, 0)}" for ep in RequestExecutor.ENDPOINTS
                )
                print(
                    f"[{self._elapsed_str()}] "
                    f"RPS: {stats['rps']:>7.1f} | "
                    f"Avg: {stats['avg_ms']:>6.0f}ms | "
                    f"Err: {stats['err_pct']:>5.1f}% | "
                    f"{counts_str}"
                )
        except asyncio.CancelledError:
            return

    def _print_banner(self) -> None:
        weights = (
            f"push={self.config.weight_push}% "
            f"search={self.config.weight_search}% "
            f"cache={self.config.weight_cache}%"
        )
        total_time = (
            self.config.warmup_time + self.config.duration + self.config.cooldown_time
        )
        print("=" * 64)
        print("  MineContext Stress Test")
        print("=" * 64)
        print(f"  Target:          {self.config.base_url}")
        print(f"  Concurrency:     {self.config.concurrency}")
        print(
            f"  Duration:        {self.config.duration}s "
            f"(+ {self.config.warmup_time}s warmup + {self.config.cooldown_time}s cooldown)"
        )
        print(f"  Max test time:   ~{total_time}s")
        print(f"  Endpoint weights: {weights}")
        print(f"  Push mode:       {self.config.push_process_mode}")
        print(f"  Search strategy: {self.config.search_strategy}")
        if self.config.api_key:
            print(f"  API Key:         {'*' * 8}...{self.config.api_key[-4:]}")
        print("=" * 64)

    def _print_final_report(self) -> None:
        report = self.metrics.get_final_report()
        if not report:
            print("\nNo requests completed.")
            return

        print()
        print("=" * 64)
        print("  STRESS TEST RESULTS")
        print("=" * 64)
        print(f"  Duration:        {report['duration_s']:.1f}s")
        print(f"  Total requests:  {report['total']:,}")
        print(
            f"  Successful:      {report['successful']:,} ({report['success_pct']:.2f}%)"
        )
        print(
            f"  Failed:          {report['failed']:,} "
            f"({100 - report['success_pct']:.2f}%)"
        )
        print(f"  Overall RPS:     {report['overall_rps']:.1f}")
        print()
        print("  Per-Endpoint Breakdown:")
        print("  " + "-" * 60)

        for ep, stats in report["endpoints"].items():
            print(f"  {ep} ({stats['count']:,} requests)")
            print(f"    RPS:     {stats['rps']:.1f}")
            print(
                f"    Latency: p50={stats['p50']:.0f}ms  "
                f"p95={stats['p95']:.0f}ms  "
                f"p99={stats['p99']:.0f}ms  "
                f"max={stats['max']:.0f}ms"
            )
            print(f"    Errors:  {stats['errors']} ({stats['error_pct']:.2f}%)")
            if stats["error_categories"]:
                for cat, count in sorted(stats["error_categories"].items()):
                    print(f"      {cat}: {count}")
            print()

        print("  " + "-" * 60)
        print("  Latency Distribution (all endpoints):")
        print(f"    p50:  {report['overall_p50']:.0f}ms")
        print(f"    p95:  {report['overall_p95']:.0f}ms")
        print(f"    p99:  {report['overall_p99']:.0f}ms")
        print(f"    max:  {report['overall_max']:.0f}ms")
        print("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> StressTestConfig:
    parser = argparse.ArgumentParser(
        description="MineContext API Stress Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/stress_test.py                                     # defaults
  python tests/stress_test.py --concurrency 5 --duration 10       # smoke test
  python tests/stress_test.py --push-mode direct --concurrency 50 # full pipeline
  python tests/stress_test.py --weight-push 100 --weight-search 0 --weight-cache 0
        """,
    )

    parser.add_argument(
        "--base-url", default="http://localhost:1733", help="Target server URL"
    )
    parser.add_argument(
        "--concurrency", type=int, default=200, help="Number of concurrent workers"
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Sustained load duration (seconds)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup ramp-up time (seconds)"
    )
    parser.add_argument(
        "--cooldown", type=int, default=5, help="Cooldown drain time (seconds)"
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=5,
        help="Seconds between live stats output",
    )
    parser.add_argument("--api-key", default=None, help="X-API-Key header value")

    parser.add_argument(
        "--weight-push", type=int, default=50, help="Push endpoint weight (%%)"
    )
    parser.add_argument(
        "--weight-search", type=int, default=30, help="Search endpoint weight (%%)"
    )
    parser.add_argument(
        "--weight-cache", type=int, default=20, help="Cache endpoint weight (%%)"
    )

    parser.add_argument(
        "--push-mode",
        choices=["buffer", "direct"],
        default="buffer",
        help="Push process_mode",
    )
    parser.add_argument(
        "--push-flush", action="store_true", help="Set flush_immediately=true"
    )
    parser.add_argument(
        "--search-strategy",
        choices=["fast", "intelligent"],
        default="fast",
        help="Search strategy",
    )
    parser.add_argument(
        "--search-top-k", type=int, default=20, help="Search top_k parameter"
    )

    parser.add_argument(
        "--connector-limit",
        type=int,
        default=300,
        help="aiohttp TCPConnector connection limit",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=65,
        help="Per-request timeout (seconds)",
    )

    args = parser.parse_args()

    # Validate weights
    total_weight = args.weight_push + args.weight_search + args.weight_cache
    if total_weight != 100:
        parser.error(
            f"Endpoint weights must sum to 100 (got {total_weight}: "
            f"push={args.weight_push} + search={args.weight_search} + cache={args.weight_cache})"
        )

    return StressTestConfig(
        base_url=args.base_url.rstrip("/"),
        concurrency=args.concurrency,
        duration=args.duration,
        warmup_time=args.warmup,
        cooldown_time=args.cooldown,
        report_interval=args.report_interval,
        api_key=args.api_key,
        weight_push=args.weight_push,
        weight_search=args.weight_search,
        weight_cache=args.weight_cache,
        push_process_mode=args.push_mode,
        push_flush=args.push_flush,
        search_strategy=args.search_strategy,
        search_top_k=args.search_top_k,
        connector_limit=args.connector_limit,
        request_timeout=args.request_timeout,
    )


async def async_main() -> None:
    config = parse_args()
    runner = StressTestRunner(config)
    try:
        await runner.run()
    except KeyboardInterrupt:
        print(f"\nInterrupted. Printing results collected so far...")
        runner._print_final_report()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nForce exit.")
        sys.exit(1)


if __name__ == "__main__":
    main()
