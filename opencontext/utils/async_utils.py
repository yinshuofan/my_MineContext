"""Async utility functions."""

import asyncio


def fire_and_forget(coro):
    """Schedule a coroutine fire-and-forget, works from sync or async context.

    If a running event loop exists, schedules via run_coroutine_threadsafe.
    Otherwise, falls back to asyncio.run().
    """
    try:
        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        asyncio.run(coro)
