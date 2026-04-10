"""
Config Reload Manager

Uses Redis Pub/Sub to propagate config reload signals across workers.
Each worker subscribes to a channel; when a reload is published,
the registered callback (OpenContext.reload_components) is invoked.

Follows the StreamInterruptManager pattern.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Optional

from opencontext.storage.redis_cache import get_redis_cache
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

RELOAD_CHANNEL = "config:reload"

_manager: Optional["ConfigReloadManager"] = None


class ConfigReloadManager:
    """Listens for config reload signals via Redis Pub/Sub and executes a reload callback."""

    def __init__(self):
        self._subscriber_task: asyncio.Task | None = None
        self._reload_fn: Callable[[], Coroutine] | None = None

    async def start(self, reload_fn: Callable[[], Coroutine]) -> None:
        """Start the Pub/Sub subscriber background task."""
        self._reload_fn = reload_fn
        if self._subscriber_task is not None and not self._subscriber_task.done():
            return
        self._subscriber_task = asyncio.create_task(self._subscribe_loop())
        logger.info("ConfigReloadManager started")

    async def stop(self) -> None:
        """Cancel the subscriber task."""
        if self._subscriber_task is not None and not self._subscriber_task.done():
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except asyncio.CancelledError:
                pass
        logger.info("ConfigReloadManager stopped")

    async def trigger_reload(self) -> int:
        """Publish a reload signal to all workers. Returns number of receivers."""
        try:
            cache = get_redis_cache()
            if cache:
                return await cache.publish(RELOAD_CHANNEL, "reload")
        except Exception as e:
            logger.warning(f"Failed to publish reload signal: {e}")
        return 0

    async def _subscribe_loop(self) -> None:
        """Persistent background coroutine. Auto-reconnects on Redis errors."""
        while True:
            pubsub = None
            try:
                cache = get_redis_cache()
                if not cache:
                    logger.warning("Redis unavailable, config reload subscriber stopping")
                    return

                pubsub = await cache.create_pubsub()
                if pubsub is None:
                    return

                channel = f"{cache.config.key_prefix}{RELOAD_CHANNEL}"
                await pubsub.subscribe(channel)
                logger.debug(f"Subscribed to config reload channel: {channel}")

                while True:
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if msg is None:
                        continue
                    if msg["type"] == "message" and self._reload_fn:
                        logger.info("Received config reload signal, reloading components...")
                        try:
                            await self._reload_fn()
                        except Exception as e:
                            logger.error(f"Component reload failed: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Config reload subscriber error, reconnecting in 2s: {e}")
                await asyncio.sleep(2)
            finally:
                if pubsub is not None:
                    try:
                        await pubsub.unsubscribe()
                        await pubsub.close()
                    except Exception:
                        pass


def get_config_reload_manager() -> ConfigReloadManager:
    """Get or create the singleton ConfigReloadManager."""
    global _manager
    if _manager is None:
        _manager = ConfigReloadManager()
    return _manager
