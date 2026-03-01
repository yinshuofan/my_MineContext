"""
Cross-Worker Stream Interrupt Manager

Uses Redis Pub/Sub to propagate interrupt signals across workers.
Falls back to local dict when Redis is unavailable.
"""

import asyncio
from typing import Dict, Optional

from opencontext.storage.redis_cache import get_redis_cache
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

_instance: Optional["StreamInterruptManager"] = None


class StreamInterruptManager:
    """Manages stream interrupt flags with cross-worker propagation via Redis Pub/Sub.

    One persistent PSUBSCRIBE pattern subscriber per worker handles all active streams.
    The local flags dict is the single source of truth for the streaming loop.
    """

    CHANNEL_PREFIX = "stream:interrupt:"

    def __init__(self):
        self._local_flags: Dict[int, bool] = {}
        self._subscriber_task: Optional[asyncio.Task] = None

    async def register(self, msg_id: int) -> None:
        """Register a new active stream. Starts the pattern subscriber if needed."""
        self._local_flags[msg_id] = False
        await self._ensure_subscriber()

    def is_interrupted(self, msg_id: int) -> bool:
        """Check if a stream has been interrupted. Sync, zero I/O."""
        return self._local_flags.get(msg_id, False)

    async def interrupt(self, msg_id: int) -> None:
        """Interrupt a stream: set local flag + publish to Redis for other workers."""
        # Always set local flag (handles single-worker and same-worker case)
        if msg_id in self._local_flags:
            self._local_flags[msg_id] = True

        # Publish to Redis for cross-worker delivery
        try:
            cache = get_redis_cache()
            channel = f"{self.CHANNEL_PREFIX}{msg_id}"
            receivers = await cache.publish(channel, "1")
            logger.info(
                f"Published interrupt for message {msg_id}, "
                f"received by {receivers} subscriber(s)"
            )
        except Exception as e:
            logger.warning(f"Redis publish failed for interrupt {msg_id}: {e}")

    async def unregister(self, msg_id: int) -> None:
        """Remove a stream from tracking."""
        self._local_flags.pop(msg_id, None)

    async def close(self) -> None:
        """Cancel the subscriber task and clean up."""
        if self._subscriber_task is not None and not self._subscriber_task.done():
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except asyncio.CancelledError:
                pass
        self._local_flags.clear()

    async def _ensure_subscriber(self) -> None:
        """Start the pattern subscriber task if it's not running."""
        if self._subscriber_task is not None and not self._subscriber_task.done():
            return

        try:
            cache = get_redis_cache()
            pubsub = await cache.create_pubsub()
            if pubsub is None:
                logger.debug("Redis pubsub unavailable, using local-only interrupt")
                return

            self._subscriber_task = asyncio.create_task(
                self._pattern_subscribe_loop(cache.config.key_prefix)
            )
            logger.info("Started stream interrupt pattern subscriber")
        except Exception as e:
            logger.warning(f"Failed to start interrupt subscriber: {e}")

    async def _pattern_subscribe_loop(self, key_prefix: str) -> None:
        """Persistent background coroutine that listens for interrupt messages.
        Auto-reconnects after transient Redis errors."""
        pattern = f"{key_prefix}{self.CHANNEL_PREFIX}*"
        prefix_len = len(f"{key_prefix}{self.CHANNEL_PREFIX}")

        while True:
            pubsub = None
            try:
                cache = get_redis_cache()
                pubsub = await cache.create_pubsub()
                if pubsub is None:
                    logger.warning("Redis unavailable, interrupt subscriber stopping")
                    return

                await pubsub.psubscribe(pattern)
                logger.debug(f"Subscribed to pattern: {pattern}")

                while True:
                    msg = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )
                    if msg is None:
                        continue

                    if msg["type"] == "pmessage":
                        try:
                            channel = msg["channel"]
                            # Channel may be bytes if decode_responses is ever disabled
                            if isinstance(channel, bytes):
                                channel = channel.decode("utf-8")
                            msg_id = int(channel[prefix_len:])

                            if msg_id in self._local_flags:
                                self._local_flags[msg_id] = True
                                logger.info(
                                    f"Received cross-worker interrupt for message {msg_id}"
                                )
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse interrupt channel: {e}")

            except asyncio.CancelledError:
                logger.debug("Interrupt subscriber cancelled")
                break
            except Exception as e:
                logger.warning(f"Interrupt subscriber error, reconnecting in 2s: {e}")
                await asyncio.sleep(2)
            finally:
                if pubsub is not None:
                    try:
                        await pubsub.punsubscribe(pattern)
                        await pubsub.close()
                    except Exception:
                        pass

        logger.debug("Interrupt subscriber stopped")


def get_stream_interrupt_manager() -> StreamInterruptManager:
    """Get or create the singleton StreamInterruptManager."""
    global _instance
    if _instance is None:
        _instance = StreamInterruptManager()
    return _instance
