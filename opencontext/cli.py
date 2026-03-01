# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Command-line interface - provides the entry point for command-line tools
"""

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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


def get_or_create_context_lab():
    """Get or create the global OpenContext instance for the current process."""
    global _context_lab_instance
    if _context_lab_instance is None:
        _context_lab_instance = _initialize_context_lab()
        _context_lab_instance.start_capture()
    return _context_lab_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Increase default thread pool for asyncio.to_thread() calls
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=10)
    loop.set_default_executor(executor)

    # Startup
    if not hasattr(app.state, "context_lab_instance"):
        app.state.context_lab_instance = get_or_create_context_lab()

    # Initialize async storage (must happen inside event loop)
    from opencontext.storage.global_storage import GlobalStorage, get_storage

    await GlobalStorage.get_instance().ensure_initialized()

    # Update OpenContext's storage reference now that async init is done
    context_lab = app.state.context_lab_instance
    if context_lab and context_lab.storage is None:
        context_lab.storage = get_storage()

    # Start task scheduler after event loop is running
    try:
        if context_lab and hasattr(context_lab, "component_initializer"):
            await context_lab.component_initializer.start_task_scheduler()
    except Exception as e:
        logger.warning(f"Failed to start task scheduler: {e}")

    yield

    # Shutdown - stop scheduler first (while event loop is still running)
    try:
        context_lab = getattr(app.state, "context_lab_instance", None)
        if context_lab and hasattr(context_lab, "component_initializer"):
            await context_lab.component_initializer.stop_task_scheduler()
    except Exception as e:
        logger.warning(f"Error stopping task scheduler: {e}")

    # Stop stream interrupt subscriber
    try:
        from opencontext.server.stream_interrupt import get_stream_interrupt_manager

        await get_stream_interrupt_manager().close()
    except Exception as e:
        logger.warning(f"Error stopping stream interrupt manager: {e}")

    # Shutdown executor, waiting for in-flight thread pool tasks
    executor.shutdown(wait=True, cancel_futures=True)


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


def start_web_server(
    context_lab_instance: Optional[OpenContext],
    host: str,
    port: int,
    workers: int = 1,
) -> None:
    """Start the web server with the given opencontext instance.

    Args:
        context_lab_instance: The opencontext instance to attach to the app (None in multi-worker mode)
        host: Host address to bind to
        port: Port number to bind to
        workers: Number of worker processes
    """
    if workers > 1:
        logger.info(f"Starting with {workers} worker processes")
        # For multi-process mode, use import string to avoid the warning
        uvicorn.run("opencontext.cli:app", host=host, port=port, log_level="info", workers=workers)
    else:
        # For single process mode, use the existing instance
        app.state.context_lab_instance = context_lab_instance
        uvicorn.run(app, host=host, port=port, log_level="info")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="OpenContext - Context capture, processing, storage and consumption system"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start OpenContext server")
    start_parser.add_argument("--config", type=str, help="Configuration file path")
    start_parser.add_argument("--host", type=str, help="Host address (overrides config file)")
    start_parser.add_argument("--port", type=int, help="Port number (overrides config file)")
    start_parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )

    return parser.parse_args()


def _initialize_context_lab() -> OpenContext:
    """Initialize the OpenContext instance.

    Returns:
        Initialized OpenContext instance

    Raises:
        RuntimeError: If initialization fails
    """
    try:
        lab_instance = OpenContext()
        lab_instance.initialize()
        return lab_instance
    except Exception as e:
        logger.error(f"Failed to initialize OpenContext: {e}")
        raise RuntimeError(f"OpenContext initialization failed: {e}") from e


def _run_headless_mode(lab_instance: OpenContext) -> None:
    """Run in headless mode without web server.

    Args:
        lab_instance: The opencontext instance
    """
    import asyncio

    async def _run_async():
        try:
            # Start task scheduler
            if hasattr(lab_instance, "component_initializer"):
                await lab_instance.component_initializer.start_task_scheduler()

            logger.info("Running in headless mode. Press Ctrl+C to exit.")
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            # Stop task scheduler
            if hasattr(lab_instance, "component_initializer"):
                await lab_instance.component_initializer.stop_task_scheduler()

    try:
        asyncio.run(_run_async())
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        lab_instance.shutdown()


def handle_start(args: argparse.Namespace) -> int:
    """Handle the start command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from opencontext.config.global_config import get_config

    web_config = get_config("web")
    workers = getattr(args, "workers", 1)

    if not web_config.get("enabled", True):
        # Headless mode
        try:
            lab_instance = _initialize_context_lab()
        except RuntimeError:
            return 1
        logger.info("Starting all modules")
        lab_instance.start_capture()
        _run_headless_mode(lab_instance)
        return 0

    # Command line arguments override config file
    host = args.host if args.host else web_config.get("host", "localhost")
    port = args.port if args.port else web_config.get("port", 1733)

    if workers > 1:
        # Multi-worker: main process is just a supervisor, workers self-initialize in lifespan
        logger.info(f"Starting web server on {host}:{port} with {workers} workers")
        start_web_server(None, host, port, workers)
    else:
        # Single-worker: initialize here and pass to app directly
        try:
            lab_instance = _initialize_context_lab()
        except RuntimeError:
            return 1
        logger.info("Starting all modules")
        lab_instance.start_capture()
        try:
            logger.info(f"Starting web server on {host}:{port}")
            start_web_server(lab_instance, host, port)
        finally:
            logger.info("Web server closed, shutting down capture modules...")
            lab_instance.shutdown()

    return 0


def _setup_logging(config_path: Optional[str]) -> None:
    """Setup logging configuration.

    Args:
        config_path: Optional path to configuration file
    """
    import os

    # Propagate config path via env var for multi-worker subprocess inheritance
    if config_path:
        os.environ["OPENCONTEXT_CONFIG_PATH"] = config_path

    from opencontext.config.global_config import GlobalConfig

    GlobalConfig.get_instance().initialize(config_path)

    setup_logging(GlobalConfig.get_instance().get_config("logging"))


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = parse_args()

    # Setup logging first
    _setup_logging(getattr(args, "config", None))

    logger.debug(f"Command line arguments: {args}")

    if not args.command:
        logger.error(
            "No command specified. Use 'opencontext start' or 'opencontext --help' for usage."
        )
        return 1

    if args.command == "start":
        return handle_start(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
