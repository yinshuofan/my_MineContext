# Settings Apply Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an "Apply Settings" button that broadcasts a Redis Pub/Sub signal to all workers, triggering each to gracefully restart its Capture, Scheduler, and Memory Cache components with the latest config.

**Architecture:** Settings save already persists to `user_setting.yaml`. A new `/api/settings/apply` endpoint publishes a reload signal via Redis Pub/Sub. Each worker runs a `ConfigReloadManager` background coroutine (following the existing `StreamInterruptManager` pattern) that subscribes to this channel and executes component-level graceful restart when signaled.

**Tech Stack:** Python 3.10+, FastAPI, Redis Pub/Sub (`redis.asyncio`), Bootstrap 5

---

### Task 1: Add `reload_config()` to small components

These are trivial methods that refresh cached config. Do them first since they have no dependencies.

**Files:**
- Modify: `opencontext/server/component_initializer.py:46-50`
- Modify: `opencontext/server/cache/memory_cache_manager.py:42-56`
- Modify: `opencontext/managers/capture_manager.py:366-390`

**Step 1: Add `reload_config()` to `ComponentInitializer`**

In `opencontext/server/component_initializer.py`, add after `__init__` (after line 51):

```python
def reload_config(self):
    """Refresh the cached config reference from GlobalConfig."""
    self.config = GlobalConfig.get_instance().get_config()
```

**Step 2: Add `reload_config()` to `UserMemoryCacheManager`**

In `opencontext/server/cache/memory_cache_manager.py`, add a method to the class:

```python
def reload_config(self):
    """Re-read config values from GlobalConfig."""
    self._config = self._load_config()
```

**Step 3: Add `clear_components()` to `ContextCaptureManager`**

In `opencontext/managers/capture_manager.py`, add before `shutdown()`:

```python
def clear_components(self):
    """Stop and remove all components for re-initialization."""
    self.stop_all_components()
    self._components.clear()
    self._component_configs.clear()
    self._running_components.clear()
```

**Step 4: Compile-check all three files**

Run:
```bash
python -m py_compile opencontext/server/component_initializer.py
python -m py_compile opencontext/server/cache/memory_cache_manager.py
python -m py_compile opencontext/managers/capture_manager.py
```

**Step 5: Commit**

```bash
git add opencontext/server/component_initializer.py opencontext/server/cache/memory_cache_manager.py opencontext/managers/capture_manager.py
git commit -m "feat(settings): add reload_config/clear_components to ComponentInitializer, MemoryCacheManager, CaptureManager"
```

---

### Task 2: Add `reload_components()` to `OpenContext`

This is the central reload logic that orchestrates all component restarts.

**Files:**
- Modify: `opencontext/server/opencontext.py:254` (add before `shutdown()`)

**Step 1: Add the method**

Add to `OpenContext` class, before the `shutdown()` method:

```python
async def reload_components(self) -> None:
    """Reload config and gracefully restart Capture, Scheduler, and MemoryCache components."""
    logger.info("Reloading components with updated configuration...")

    # 1. Reload GlobalConfig from YAML
    config_mgr = GlobalConfig.get_instance().get_config_manager()
    if config_mgr:
        config_mgr.load_config(config_mgr.get_config_path())

    # 2. Refresh ComponentInitializer's cached config reference
    self.component_initializer.reload_config()

    # 3. Restart Capture: stop → clear → reinit → start
    try:
        self.capture_manager.clear_components()
        self.component_initializer.initialize_capture_components(self.capture_manager)
        self.capture_manager.start_all_components()
        logger.info("Capture components reloaded")
    except Exception as e:
        logger.error(f"Failed to reload capture components: {e}")

    # 4. Restart Scheduler: stop → reinit → start
    try:
        await self.component_initializer.stop_task_scheduler()
        self.component_initializer.initialize_task_scheduler(self.processor_manager)
        await self.component_initializer.start_task_scheduler()
        logger.info("Task scheduler reloaded")
    except Exception as e:
        logger.error(f"Failed to reload task scheduler: {e}")

    # 5. Reload Memory Cache Manager config
    try:
        from opencontext.server.cache.memory_cache_manager import get_memory_cache_manager
        get_memory_cache_manager().reload_config()
        logger.info("Memory cache manager config reloaded")
    except Exception as e:
        logger.error(f"Failed to reload memory cache config: {e}")

    logger.info("Component reload complete")
```

**Step 2: Compile-check**

Run:
```bash
python -m py_compile opencontext/server/opencontext.py
```

**Step 3: Commit**

```bash
git add opencontext/server/opencontext.py
git commit -m "feat(settings): add reload_components() to OpenContext for graceful component restart"
```

---

### Task 3: Create `ConfigReloadManager`

The Pub/Sub subscriber singleton, following `StreamInterruptManager` pattern.

**Files:**
- Create: `opencontext/server/config_reload_manager.py`

**Step 1: Write the module**

```python
"""
Config Reload Manager

Uses Redis Pub/Sub to propagate config reload signals across workers.
Each worker subscribes to a channel; when a reload is published,
the registered callback (OpenContext.reload_components) is invoked.

Follows the StreamInterruptManager pattern.
"""

import asyncio
from typing import Callable, Coroutine, Optional

from opencontext.storage.redis_cache import get_redis_cache
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

RELOAD_CHANNEL = "config:reload"

_manager: Optional["ConfigReloadManager"] = None


class ConfigReloadManager:
    """Listens for config reload signals via Redis Pub/Sub and executes a reload callback."""

    def __init__(self):
        self._subscriber_task: Optional[asyncio.Task] = None
        self._reload_fn: Optional[Callable[[], Coroutine]] = None

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
                    msg = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )
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
```

**Step 2: Compile-check**

Run:
```bash
python -m py_compile opencontext/server/config_reload_manager.py
```

**Step 3: Commit**

```bash
git add opencontext/server/config_reload_manager.py
git commit -m "feat(settings): add ConfigReloadManager with Redis Pub/Sub for cross-worker reload"
```

---

### Task 4: Wire into FastAPI lifespan

Start the subscriber on app startup, stop on shutdown.

**Files:**
- Modify: `opencontext/cli.py:72-96` (lifespan function)

**Step 1: Add startup code**

In the `lifespan()` function, after the scheduler start block (after line 78) and before `yield`, add:

```python
    # Start config reload manager (listens for cross-worker reload signals)
    try:
        from opencontext.server.config_reload_manager import get_config_reload_manager

        await get_config_reload_manager().start(context_lab.reload_components)
    except Exception as e:
        logger.warning(f"Failed to start config reload manager: {e}")
```

**Step 2: Add shutdown code**

In the shutdown section, after the stream interrupt manager close block (after line 96) and before the executor shutdown, add:

```python
    # Stop config reload manager
    try:
        from opencontext.server.config_reload_manager import get_config_reload_manager

        await get_config_reload_manager().stop()
    except Exception as e:
        logger.warning(f"Error stopping config reload manager: {e}")
```

**Step 3: Compile-check**

Run:
```bash
python -m py_compile opencontext/cli.py
```

**Step 4: Commit**

```bash
git add opencontext/cli.py
git commit -m "feat(settings): wire ConfigReloadManager into FastAPI lifespan"
```

---

### Task 5: Add `/api/settings/apply` endpoint

**Files:**
- Modify: `opencontext/server/routes/settings.py:535` (after reset endpoint)

**Step 1: Add the endpoint**

After the `reset_settings` function (end of file), add:

```python
@router.post("/api/settings/apply")
async def apply_settings(_auth: str = auth_dependency):
    """Apply saved settings by broadcasting a reload signal to all workers."""
    try:
        from opencontext.server.config_reload_manager import get_config_reload_manager

        receivers = await get_config_reload_manager().trigger_reload()
        logger.info(f"Config reload signal sent, received by {receivers} subscriber(s)")
        return convert_resp(
            message=f"Settings apply signal sent to {receivers} worker(s). "
            "Components will reload within a few seconds."
        )
    except Exception as e:
        logger.exception(f"Failed to apply settings: {e}")
        return convert_resp(code=500, status=500, message=f"Failed to apply settings: {str(e)}")
```

**Step 2: Compile-check**

Run:
```bash
python -m py_compile opencontext/server/routes/settings.py
```

**Step 3: Commit**

```bash
git add opencontext/server/routes/settings.py
git commit -m "feat(settings): add POST /api/settings/apply endpoint"
```

---

### Task 6: Add "Apply Settings" button to frontend

**Files:**
- Modify: `opencontext/web/templates/settings.html:5-7` (page header area)
- Modify: `opencontext/web/static/js/settings.js` (end of file)

**Step 1: Add the button to the HTML header**

In `settings.html`, replace the existing header div (lines 5-7):

```html
    <div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
        <h1 class="h2">System Settings</h1>
        <button type="button" class="btn btn-warning" onclick="applySettings()">
            <i class="bi bi-arrow-repeat"></i> Apply Settings
        </button>
    </div>
```

**Step 2: Add the JS function**

In `settings.js`, add before the `// ==================== Page Init ====================` section:

```javascript
// ==================== Apply Settings ====================

async function applySettings() {
    if (!confirm('Apply current settings? This will briefly restart some service components.')) {
        return;
    }

    try {
        showToast('Applying settings...');
        const response = await fetch('/api/settings/apply', { method: 'POST' });
        const data = await response.json();

        if (data.code === 0) {
            showToast(data.message || 'Settings applied successfully');
        } else {
            showToast('Apply failed: ' + (data.message || 'Unknown error'), true);
        }
    } catch (error) {
        console.error('Failed to apply settings:', error);
        showToast('Apply failed', true);
    }
}
```

**Step 3: Commit**

```bash
git add opencontext/web/templates/settings.html opencontext/web/static/js/settings.js
git commit -m "feat(settings): add Apply Settings button to frontend"
```

---

### Task 7: Update docs

**Files:**
- Modify: `docs/curls.sh` (add new endpoint)
- Modify: `opencontext/server/MODULE.md` (add new endpoint to route table)

**Step 1: Add curl example to `docs/curls.sh`**

Add under the settings section:

```bash
# Apply settings (restart components with latest config)
curl -X POST http://localhost:1733/api/settings/apply
```

**Step 2: Update MODULE.md route table if it lists settings endpoints**

Add the `/api/settings/apply` endpoint entry.

**Step 3: Commit**

```bash
git add docs/curls.sh opencontext/server/MODULE.md
git commit -m "docs: add /api/settings/apply endpoint to curls.sh and MODULE.md"
```
