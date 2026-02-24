#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Component initialization utilities for OpenContext.
Separated from main OpenContext class for better maintainability.
"""

import importlib
from typing import Any, Dict, Optional

from opencontext.config.config_manager import ConfigManager
from opencontext.config.global_config import GlobalConfig
from opencontext.config.prompt_manager import PromptManager
from opencontext.context_capture.text_chat import TextChatCapture

# Import capture components
from opencontext.context_capture.vault_document_monitor import VaultDocumentMonitor
from opencontext.context_capture.web_link_capture import WebLinkCapture
from opencontext.context_processing.processor.processor_factory import ProcessorFactory
from opencontext.managers.capture_manager import ContextCaptureManager
from opencontext.managers.processor_manager import ContextProcessorManager
from opencontext.storage.global_storage import get_storage
from opencontext.storage.unified_storage import UnifiedStorage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Component mappings
CAPTURE_COMPONENTS = {
    "vault_document_monitor": VaultDocumentMonitor,
    "web_link_capture": WebLinkCapture,
    "text_chat": TextChatCapture,
}


class ComponentInitializer:
    """Handles initialization of various OpenContext components."""

    def __init__(self):
        # Use global config
        global_config = GlobalConfig.get_instance()
        self.config_manager = global_config.get_config_manager()

        self.config = (
            self.config_manager.get_config()
            if self.config_manager
            else GlobalConfig.get_instance().get_config()
        )
        self.global_config = GlobalConfig.get_instance()

    def _to_camel_case(self, name: str) -> str:
        """Convert snake_case to CamelCase."""
        return "".join(word.capitalize() for word in name.split("_"))

    def initialize_capture_components(self, capture_manager: ContextCaptureManager) -> None:
        """Initialize context capture components."""
        capture_config = self.config.get("capture", {})
        if not capture_config or not capture_config.get("enabled", False):
            logger.info("Capture modules not found or not enabled in configuration")
            return

        for name, config in capture_config.items():
            if (
                name == "enabled"
                or not isinstance(config, dict)
                or not config.get("enabled", False)
            ):
                continue

            try:
                component_instance = self._create_capture_component(name, config)
                capture_manager.register_component(name, component_instance)
                capture_manager.initialize_component(name, config)
                logger.info(f"Capture component '{name}' initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize capture component '{name}': {e}")
                raise RuntimeError(f"Failed to initialize capture component '{name}'") from e

        logger.info("Context capture modules initialization complete")

    def _create_capture_component(self, name: str, config: Dict[str, Any]):
        """Create a capture component instance."""
        if name in CAPTURE_COMPONENTS:
            component_class = CAPTURE_COMPONENTS[name]
            return component_class()

        # Fallback to dynamic import
        module_path = config.get("module")
        class_name = config.get("class")

        if not module_path or not class_name:
            module_path = f"opencontext.context_capture.{name}"
            class_name = f"{self._to_camel_case(name)}Capture"
            logger.info(
                f"Auto-inferred capture component '{name}' module='{module_path}' and class='{class_name}'"
            )

        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
        return component_class()

    def initialize_processors(
        self, processor_manager: ContextProcessorManager, processed_context_callback
    ) -> None:
        """
        Initialize context processors.

        Processors now automatically obtain required dependencies from global configuration.
        """
        processing_config = self.config.get("processing", {})
        if not processing_config or not processing_config.get("enabled", False):
            logger.info("Processing modules not found or not enabled in configuration")
            return

        processor_factory = ProcessorFactory()

        # Now config.yaml structure is flattened, directly under processing
        # Create various processors
        processor_types = ["document_processor", "text_chat_processor"]

        for processor_type in processor_types:
            processor_config = processing_config.get(processor_type, {})
            if processor_config.get("enabled", False):
                try:
                    # Now processors use parameterless constructors
                    processor = processor_factory.create_processor(processor_type)
                    if processor:
                        processor.set_callback(processed_context_callback)
                        processor_manager.register_processor(processor)
                        logger.info(
                            f"Processor component '{processor_type}' created and registered successfully"
                        )
                    else:
                        logger.error(f"Failed to create processor component '{processor_type}'")
                        raise RuntimeError(
                            f"Failed to create processor component '{processor_type}'"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to initialize processor component '{processor_type}': {e}",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Failed to initialize processor component '{processor_type}'"
                    ) from e

        # Initialize context merger if enabled
        if processing_config.get("context_merger", {}).get("enabled", False):
            # Direct instantiation instead of using factory to avoid circular import
            from opencontext.context_processing.merger.context_merger import ContextMerger

            merger = ContextMerger()
            processor_manager.set_merger(merger)
            logger.info("Context merger initialized")
            # Note: Periodic compression is now managed by the task scheduler,
            # not by processor_manager. Call initialize_task_scheduler() separately.

        logger.info("Context processors initialization complete")

    def initialize_task_scheduler(
        self, processor_manager: Optional[ContextProcessorManager] = None
    ) -> None:
        """
        Initialize the task scheduler for periodic tasks.

        The scheduler handles:
        - Memory compression (user_activity triggered)
        - Data cleanup (periodic)
        - Other scheduled tasks

        Args:
            processor_manager: Optional processor manager to get the merger instance.
                              If provided and merger is set, it will be reused.
        """
        scheduler_config = self.config.get("scheduler", {})
        if not scheduler_config.get("enabled", False):
            logger.info("Task scheduler not enabled in configuration")
            return

        try:
            from opencontext.periodic_task import (
                create_cleanup_handler,
                create_compression_handler,
                create_hierarchy_handler,
            )
            from opencontext.scheduler import get_scheduler, init_scheduler
            from opencontext.storage.redis_cache import get_redis_cache

            # Get Redis cache (singleton already initialized by OpenContext.initialize())
            redis_cache = get_redis_cache()
            if not redis_cache:
                logger.warning("Redis cache not available, task scheduler requires Redis")
                return

            # Initialize scheduler
            scheduler = init_scheduler(redis_cache, scheduler_config)

            # Register task handlers
            tasks_config = scheduler_config.get("tasks", {})

            # Memory compression handler
            if tasks_config.get("memory_compression", {}).get("enabled", False):
                # Try to get merger from processor_manager first
                merger = None
                if processor_manager:
                    merger = processor_manager.get_merger()

                # If not available, create a new one
                if not merger:
                    from opencontext.context_processing.merger.context_merger import ContextMerger

                    merger = ContextMerger()
                    logger.info("Created new ContextMerger for compression task")
                else:
                    logger.info("Reusing merger from processor_manager for compression task")

                compression_handler = create_compression_handler(merger)
                scheduler.register_handler("memory_compression", compression_handler)
                logger.info("Registered memory_compression task handler")

            # Data cleanup handler
            if tasks_config.get("data_cleanup", {}).get("enabled", False):
                storage = get_storage()
                retention_days = tasks_config.get("data_cleanup", {}).get("retention_days", 30)

                # Get or create merger for intelligent cleanup
                cleanup_merger = None
                if processor_manager:
                    cleanup_merger = processor_manager.get_merger()

                if not cleanup_merger:
                    from opencontext.context_processing.merger.context_merger import ContextMerger

                    cleanup_merger = ContextMerger()
                    logger.info("Created new ContextMerger for cleanup task")
                else:
                    logger.info("Reusing merger from processor_manager for cleanup task")

                cleanup_handler = create_cleanup_handler(
                    context_merger=cleanup_merger, storage=storage, retention_days=retention_days
                )
                scheduler.register_handler("data_cleanup", cleanup_handler)
                logger.info("Registered data_cleanup task handler with intelligent cleanup")

            # Hierarchy summary handler (event L0→L1→L2→L3 summaries)
            if tasks_config.get("hierarchy_summary", {}).get("enabled", False):
                hierarchy_handler = create_hierarchy_handler()
                scheduler.register_handler("hierarchy_summary", hierarchy_handler)
                logger.info("Registered hierarchy_summary task handler")

            logger.info("Task scheduler initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize task scheduler: {e}")

    async def start_task_scheduler(self) -> None:
        """
        Start the task scheduler background executor.
        This should be called after the event loop is running.
        """
        try:
            from opencontext.scheduler import get_scheduler

            scheduler = get_scheduler()
            if scheduler:
                await scheduler.start()
                logger.info("Task scheduler started")
        except Exception as e:
            logger.exception(f"Failed to start task scheduler: {e}")

    def stop_task_scheduler(self) -> None:
        """
        Stop the task scheduler.
        """
        try:
            from opencontext.scheduler import get_scheduler

            scheduler = get_scheduler()
            if scheduler:
                scheduler.stop()
                logger.info("Task scheduler stopped")
        except Exception as e:
            logger.exception(f"Failed to stop task scheduler: {e}")
