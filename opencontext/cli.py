#
# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Command-line interface - provides the entry point for command-line tools
"""

import argparse
import asyncio
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from opencontext.server.api import router as api_router
from opencontext.server.opencontext import OpenContext
from opencontext.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

# Global variable for multi-process support
_context_lab_instance = None


class StartupDependencyError(RuntimeError):
    """Raised when a critical runtime dependency fails during startup."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage} startup failed: {message}")
        self.stage = stage


def _clear_context_lab_instance() -> None:
    global _context_lab_instance
    _context_lab_instance = None


def _create_context_lab() -> OpenContext:
    """Create or return the global OpenContext instance for the current process."""
    global _context_lab_instance
    if _context_lab_instance is None:
        try:
            _context_lab_instance = OpenContext()
            _context_lab_instance.initialize()
        except Exception as e:
            _context_lab_instance = None
            logger.exception(f"Failed to initialize OpenContext runtime: {e}")
            raise StartupDependencyError("opencontext", str(e)) from e
    return _context_lab_instance


def get_or_create_context_lab() -> OpenContext:
    """Backward-compatible wrapper for the process-local OpenContext singleton."""
    return _create_context_lab()


async def _ensure_storage_ready(max_retries: int = 3, delays: tuple[int, ...] = (3, 6)) -> None:
    """Initialize storage and fail fast if it stays unavailable."""
    from opencontext.storage.global_storage import GlobalStorage

    storage_mgr = GlobalStorage.get_instance()
    for attempt in range(max_retries):
        logger.info(f"Initializing storage (attempt {attempt + 1}/{max_retries})")
        await storage_mgr.ensure_initialized()
        if storage_mgr.get_storage() is not None:
            logger.info("Storage ready")
            return

        if attempt < max_retries - 1:
            delay = delays[min(attempt, len(delays) - 1)]
            logger.warning(f"Storage init failed, retry in {delay}s ({attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)

    raise StartupDependencyError("storage", "storage initialization failed after retries")


async def _ensure_redis_ready() -> None:
    """Validate that Redis has been explicitly configured and is reachable."""
    from opencontext.config.global_config import GlobalConfig
    from opencontext.storage.redis_cache import peek_redis_cache

    redis_config = GlobalConfig.get_instance().get_config("redis") or {}
    if not redis_config.get("enabled", True):
        logger.info("Redis disabled in configuration, skipping Redis readiness check")
        return

    cache = peek_redis_cache()
    if cache is None:
        raise StartupDependencyError("redis", "redis cache not initialized")

    try:
        if not await cache.is_connected():
            raise StartupDependencyError("redis", "redis connectivity check failed")
    except StartupDependencyError:
        raise
    except Exception as e:
        raise StartupDependencyError("redis", str(e)) from e

    logger.info(f"Redis ready: {cache.config.host}:{cache.config.port}/{cache.config.db}")


async def _apply_db_backed_settings(context_lab: OpenContext) -> None:
    """Load DB-backed settings and refresh scheduler config when needed."""
    try:
        from opencontext.config.global_config import GlobalConfig

        config_mgr = GlobalConfig.get_instance().get_config_manager()
        if not config_mgr:
            return

        settings_changed = await config_mgr.init_db_settings()
        if settings_changed:
            logger.info("DB-backed settings loaded, reinitializing components")
            context_lab.component_initializer.reload_config()
            context_lab.component_initializer.initialize_task_scheduler(
                context_lab.processor_manager
            )
    except Exception as e:
        logger.warning(f"DB settings init failed, using file-based settings: {e}")


async def _close_object_storage() -> None:
    """Close object storage if it was initialized."""
    try:
        from opencontext.storage.object_storage.global_object_storage import GlobalObjectStorage

        obj_storage = GlobalObjectStorage._instance if GlobalObjectStorage._initialized else None
        if obj_storage:
            await obj_storage.close()
    except Exception as e:
        logger.warning(f"Error closing object storage: {e}")


async def _shutdown_runtime(
    context_lab: OpenContext | None,
    executor: ThreadPoolExecutor | None = None,
    graceful: bool = True,
) -> None:
    """Shutdown runtime components in a safe order."""
    try:
        if context_lab and hasattr(context_lab, "component_initializer"):
            await context_lab.component_initializer.stop_task_scheduler()
    except Exception as e:
        logger.warning(f"Error stopping task scheduler: {e}")

    try:
        from opencontext.server.stream_interrupt import get_stream_interrupt_manager

        await get_stream_interrupt_manager().close()
    except Exception as e:
        logger.warning(f"Error stopping stream interrupt manager: {e}")

    try:
        from opencontext.server.config_reload_manager import get_config_reload_manager

        await get_config_reload_manager().stop()
    except Exception as e:
        logger.warning(f"Error stopping config reload manager: {e}")

    if context_lab is not None:
        try:
            await asyncio.to_thread(context_lab.shutdown, graceful)
        except Exception as e:
            logger.warning(f"Error shutting down OpenContext runtime: {e}")

    await _close_object_storage()

    try:
        from opencontext.storage.redis_cache import close_redis_cache

        await close_redis_cache()
    except Exception as e:
        logger.warning(f"Error closing Redis cache: {e}")

    try:
        from opencontext.storage.global_storage import GlobalStorage

        await GlobalStorage.get_instance().close()
    except Exception as e:
        logger.warning(f"Error closing global storage: {e}")

    _clear_context_lab_instance()

    if executor is not None:
        try:
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
        except Exception as e:
            logger.warning(f"Error shutting down executor: {e}")


async def _cleanup_failed_startup(
    context_lab: OpenContext | None,
    executor: ThreadPoolExecutor | None = None,
) -> None:
    """Cleanup partial startup state without masking the original failure."""
    await _shutdown_runtime(context_lab, executor=executor, graceful=False)


async def _bootstrap_runtime(
    *,
    enable_capture: bool,
    enable_scheduler: bool,
    enable_reload: bool,
    executor: ThreadPoolExecutor | None = None,
) -> OpenContext:
    """Bootstrap the runtime in a single, explicit order."""
    context_lab: OpenContext | None = None
    try:
        logger.info("Bootstrapping OpenContext runtime")
        context_lab = _create_context_lab()

        await _ensure_storage_ready()
        await _ensure_redis_ready()
        await _apply_db_backed_settings(context_lab)

        if enable_capture:
            logger.info("Starting capture components")
            context_lab.start_capture()

        if enable_scheduler and hasattr(context_lab, "component_initializer"):
            logger.info("Starting task scheduler")
            await context_lab.component_initializer.start_task_scheduler()

        if enable_reload:
            from opencontext.server.config_reload_manager import get_config_reload_manager

            logger.info("Starting config reload manager")
            await get_config_reload_manager().start(context_lab.reload_components)

        logger.info("OpenContext runtime bootstrap completed")
        return context_lab
    except Exception:
        await _cleanup_failed_startup(context_lab, executor=executor)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    executor: ThreadPoolExecutor | None = None
    context_lab: OpenContext | None = None

    try:
        # In multi-worker mode, uvicorn imports `opencontext.cli:app` in worker
        # subprocesses without running `main()`. Reconfigure logging here so
        # worker-side startup failures are emitted.
        _setup_logging(os.environ.get("OPENCONTEXT_CONFIG_PATH"))
        logger.info(
            f"Worker lifespan startup pid={os.getpid()} "
            f"config_path={os.environ.get('OPENCONTEXT_CONFIG_PATH', 'config/config.yaml')}"
        )

        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=10)
        loop.set_default_executor(executor)

        context_lab = await _bootstrap_runtime(
            enable_capture=True,
            enable_scheduler=True,
            enable_reload=True,
            executor=executor,
        )
        app.state.context_lab_instance = context_lab

        yield
    finally:
        if context_lab is not None:
            await _shutdown_runtime(context_lab, executor=executor)
            if hasattr(app.state, "context_lab_instance"):
                delattr(app.state, "context_lab_instance")
        elif executor is not None:
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)


app = FastAPI(title="OpenContext", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Request ID middleware (added after CORS so it runs first due to LIFO ordering)
from opencontext.server.middleware.request_id import RequestIDMiddleware

app.add_middleware(RequestIDMiddleware)

# Project root
if hasattr(sys, "_MEIPASS"):
    project_root = Path(sys._MEIPASS)
else:
    project_root = Path(__file__).parent.parent.parent.resolve()


def _get_project_root() -> Path:
    """Get the project root directory."""
    return project_root


def _setup_static_files() -> None:
    """Setup static file mounts for the FastAPI app."""
    # Mount static files
    if hasattr(sys, "_MEIPASS"):
        static_path = Path(sys._MEIPASS) / "opencontext/web/static"
    else:
        static_path = Path(__file__).parent / "web/static"

    print(f"Static path: {static_path}")
    print(f"Static path exists: {static_path.exists()}")
    print(f"Static path absolute: {static_path.resolve()}")

    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        print(f"Mounted static files from: {static_path}")
    else:
        print(f"Static path does not exist: {static_path}")


_setup_static_files()

app.include_router(api_router)


def start_web_server(host: str, port: int, workers: int = 1) -> None:
    """Start the web server."""
    if workers > 1:
        logger.info(f"Starting with {workers} worker processes")
        uvicorn.run("opencontext.cli:app", host=host, port=port, log_level="info", workers=workers)
    else:
        uvicorn.run(app, host=host, port=port, log_level="info")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenContext - Context capture, processing, storage and consumption system"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    start_parser = subparsers.add_parser("start", help="Start OpenContext server")
    start_parser.add_argument("--config", type=str, help="Configuration file path")
    start_parser.add_argument("--host", type=str, help="Host address (overrides config file)")
    start_parser.add_argument("--port", type=int, help="Port number (overrides config file)")
    start_parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )
    start_parser.add_argument(
        "--scheduler-only",
        action="store_true",
        help="Run only the task scheduler without web server or capture",
    )

    return parser.parse_args()


def _run_headless_mode(*, enable_capture: bool, enable_scheduler: bool) -> int:
    """Run in headless mode without a web server."""

    async def _run_async() -> None:
        context_lab = await _bootstrap_runtime(
            enable_capture=enable_capture,
            enable_scheduler=enable_scheduler,
            enable_reload=False,
        )
        try:
            logger.info("Running in headless mode. Waiting for shutdown signal...")
            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            try:
                loop.add_signal_handler(signal.SIGTERM, stop_event.set)
            except NotImplementedError:
                pass  # Windows: rely on KeyboardInterrupt via asyncio.run()

            await stop_event.wait()
        finally:
            await _shutdown_runtime(context_lab)

    try:
        asyncio.run(_run_async())
        return 0
    except StartupDependencyError as e:
        logger.error(str(e))
        return 1
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        return 0


def handle_start(args: argparse.Namespace) -> int:
    """Handle the start command."""
    from opencontext.config.global_config import get_config

    web_config = get_config("web")
    workers = getattr(args, "workers", 1)

    if getattr(args, "scheduler_only", False):
        return _run_headless_mode(enable_capture=False, enable_scheduler=True)

    if not web_config.get("enabled", True):
        return _run_headless_mode(enable_capture=True, enable_scheduler=True)

    host = args.host if args.host else web_config.get("host", "localhost")
    port = args.port if args.port else web_config.get("port", 1733)

    logger.info(f"Starting web server on {host}:{port} with {workers} worker(s)")
    start_web_server(host, port, workers)
    return 0


def _setup_logging(config_path: str | None) -> None:
    """Setup logging configuration."""
    # Propagate config path via env var for multi-worker subprocess inheritance
    if config_path:
        os.environ["OPENCONTEXT_CONFIG_PATH"] = config_path

    from opencontext.config.global_config import GlobalConfig

    GlobalConfig.get_instance().initialize(config_path)

    setup_logging(GlobalConfig.get_instance().get_config("logging"))

    from opencontext.utils.time_utils import init_timezone

    tz_name = GlobalConfig.get_instance().get_config("timezone")
    init_timezone(tz_name)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    _setup_logging(getattr(args, "config", None))

    logger.debug(f"Command line arguments: {args}")

    if not args.command:
        logger.error(
            "No command specified. Use 'opencontext start' or 'opencontext --help' for usage."
        )
        return 1

    if args.command == "start":
        return handle_start(args)

    logger.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
