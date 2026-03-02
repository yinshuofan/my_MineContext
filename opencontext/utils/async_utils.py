"""Async utility functions."""

import asyncio

from opencontext.utils.logging_utils import get_logger

logger = get_logger("async_utils")


def fire_and_forget(coro):
    """Schedule a coroutine fire-and-forget, works from sync or async context.

    If a running event loop exists, schedules via run_coroutine_threadsafe.
    Otherwise, falls back to asyncio.run().
    """
    try:
        loop = asyncio.get_running_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        coro_name = getattr(coro, "__qualname__", None) or getattr(coro, "__name__", "unknown")

        def _on_done(f):
            if f.cancelled():
                return
            exc = f.exception()
            if exc:
                logger.opt(exception=exc).error("fire_and_forget task failed: {}", coro_name)

        future.add_done_callback(_on_done)
    except RuntimeError:
        asyncio.run(coro)
