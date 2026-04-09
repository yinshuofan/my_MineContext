#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


"""
Context processing manager for managing and coordinating context processing components
"""

import asyncio
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from opencontext.interfaces import IContextProcessor
from opencontext.models import ContextSource, ProcessedContext, RawContextProperties

BATCH_PROCESSOR_MAP = {
    "user_memory": "text_chat_processor",
    "agent_memory": "agent_memory_processor",
}

# DAG dependency configuration: processor_name -> list of processor_names it depends on.
# Processors not listed here have no dependencies and run in the first layer.
PROCESSOR_DEPENDENCIES = {
    "agent_memory": ["user_memory"],
}


class ContextProcessorManager:
    """
    Context processing manager

    Manages and coordinates multiple context processing components, providing unified interface for context processing

    Note: Periodic compression tasks are now managed by the task scheduler (opencontext.scheduler),
    not by this manager. The merger is still set here for use by other components.
    """

    def __init__(self, max_workers: int = 5):
        """Initialize context processing manager"""
        self._processors: Dict[str, IContextProcessor] = {}
        self._callback: Optional[Callable[[List[Any]], None]] = None
        self._merger: Optional[IContextProcessor] = None
        self._statistics: Dict[str, Any] = {
            "total_processed_inputs": 0,
            "total_contexts_generated": 0,
            "processors": {},
            "errors": 0,
        }
        self._routing_table: Dict[ContextSource, List[str]] = {}
        self._define_routing()
        self._lock = Lock()
        self._max_workers = max_workers

    def _define_routing(self):
        """
        Define processing chain routing rules in code
        Users can modify here to customize routing
        """
        self._routing_table = {
            ContextSource.LOCAL_FILE: "document_processor",
            ContextSource.VAULT: "document_processor",
            ContextSource.WEB_LINK: "document_processor",
            ContextSource.CHAT_LOG: "text_chat_processor",
            ContextSource.INPUT: "text_chat_processor",
        }

    def register_processor(self, processor: IContextProcessor) -> bool:
        """
        Register processing component
        """
        processor_name = processor.get_name()

        if processor_name in self._processors:
            logger.warning(
                f"Processing component '{processor_name}' already registered, will be overwritten"
            )

        self._processors[processor_name] = processor
        self._statistics["processors"][processor_name] = processor.get_statistics()

        logger.info(f"Processing component '{processor_name}' registered successfully")
        return True

    def set_merger(self, merger: IContextProcessor) -> None:
        """
        Set merger component

        Note: The merger is set here for use by other components (e.g., Push API triggered compression).
        Periodic compression is now managed by the task scheduler.
        """
        self._merger = merger
        logger.info(f"Merger component '{merger.get_name()}' has been set")

    def get_merger(self) -> Optional[IContextProcessor]:
        """
        Get the merger component

        Returns:
            The merger component if set, None otherwise
        """
        return self._merger

    def get_processor(self, processor_name: str) -> Optional[IContextProcessor]:
        return self._processors.get(processor_name)

    def get_all_processors(self) -> Dict[str, IContextProcessor]:
        return self._processors.copy()

    def set_callback(self, callback: Callable[[List[Any]], None]) -> None:
        self._callback = callback

    async def process(self, initial_input: RawContextProperties) -> bool:
        """
        Process single input through processing chain
        """
        # 1. Dynamically select preprocessing chain based on input type (excluding merger and embedding)
        processor_name = self._routing_table.get(initial_input.source)
        if not processor_name:
            logger.error(
                f"No processing component defined for source_type='{initial_input.source}' or content_format='{initial_input.content_format}', no processing will be performed"
            )
            return False

        processor = self._processors.get(processor_name)
        if not processor or not processor.can_process(initial_input):
            logger.error(
                f"Processor '{processor_name}' in processing chain not registered or does not support processing input type {initial_input.source}"
            )
            return False

        try:
            processed_contexts = await processor.process(initial_input)
            if processed_contexts and self._callback:
                await self._callback(processed_contexts)
            return bool(processed_contexts)
        except Exception as e:
            logger.exception(
                f"Processing component '{processor_name}' encountered exception while processing data: {e}"
            )
            return False

    async def process_batch(
        self, raw_context: RawContextProperties, processor_names: List[str]
    ) -> List[ProcessedContext]:
        """
        Process a raw context through multiple processors with DAG-based ordering.

        Processors are grouped into layers via topological sort on PROCESSOR_DEPENDENCIES.
        Each layer runs in parallel; layers execute sequentially.
        Later layers receive accumulated results from all prior layers via `prior_results`.
        If a returned context has the same ID as an existing one, it replaces it.
        """
        if not processor_names:
            return []

        # Resolve external names to (external_name, processor_instance) pairs
        resolved = []
        for ext_name in processor_names:
            internal_name = BATCH_PROCESSOR_MAP.get(ext_name, ext_name)
            processor = self._processors.get(internal_name)
            if not processor:
                logger.warning(f"Processor '{internal_name}' not found, skipping")
                continue
            if not processor.can_process(raw_context):
                logger.debug(f"Processor '{internal_name}' cannot process this context, skipping")
                continue
            resolved.append((ext_name, processor))

        if not resolved:
            return []

        # Build layers via topological sort
        layers = self._topological_sort([name for name, _ in resolved])
        proc_map = {name: proc for name, proc in resolved}

        # Execute layer by layer
        accumulated: Dict[str, ProcessedContext] = {}  # id -> context

        for layer in layers:
            layer_processors = [(name, proc_map[name]) for name in layer if name in proc_map]
            if not layer_processors:
                continue

            prior = list(accumulated.values()) if accumulated else None

            tasks = []
            task_names = []
            for name, proc in layer_processors:
                tasks.append(proc.process(raw_context, prior_results=prior))
                task_names.append(name)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Processor '{name}' failed: {result}")
                    continue
                if not result:
                    continue
                logger.debug(f"Processor '{name}' produced {len(result)} contexts")
                for ctx in result:
                    accumulated[ctx.id] = ctx  # replace if same ID, append if new

        all_contexts = list(accumulated.values())

        if all_contexts:
            type_summary: Dict[str, int] = {}
            for ctx in all_contexts:
                t = ctx.extracted_data.context_type.value
                type_summary[t] = type_summary.get(t, 0) + 1
            logger.info(f"process_batch produced {len(all_contexts)} contexts: {type_summary}")

        if all_contexts and self._callback:
            await self._callback(all_contexts)

        return all_contexts

    @staticmethod
    def _topological_sort(processor_names: List[str]) -> List[List[str]]:
        """
        Sort processor names into execution layers based on PROCESSOR_DEPENDENCIES.

        Returns a list of layers, where each layer is a list of processor names
        that can run in parallel. Earlier layers complete before later ones start.
        """
        name_set = set(processor_names)
        # Filter dependencies to only include processors in this batch
        deps = {}
        for name in processor_names:
            raw_deps = PROCESSOR_DEPENDENCIES.get(name, [])
            deps[name] = [d for d in raw_deps if d in name_set]

        layers = []
        placed: set = set()

        while len(placed) < len(processor_names):
            # Find all processors whose dependencies are satisfied
            layer = [
                name
                for name in processor_names
                if name not in placed and all(d in placed for d in deps[name])
            ]
            if not layer:
                # Circular dependency — break by placing remaining
                remaining = [n for n in processor_names if n not in placed]
                logger.error(
                    f"Circular dependency detected among: {remaining}. Running them together."
                )
                layers.append(remaining)
                break
            layers.append(layer)
            placed.update(layer)

        return layers

    async def batch_process(self, initial_inputs: List[RawContextProperties]) -> Dict[str, bool]:
        """Batch process raw context data"""
        tasks = [self.process(initial_input) for initial_input in initial_inputs]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for initial_input, result in zip(initial_inputs, raw_results):
            if isinstance(result, Exception):
                logger.exception(f"'{initial_input.object_id}' generated an exception: {result}")
                results[initial_input.object_id] = False
            else:
                results[initial_input.object_id] = result
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all processors and managers
        """
        with self._lock:
            # Update latest processor statistics
            for name, processor in self._processors.items():
                self._statistics["processors"][name] = processor.get_statistics()
            return self._statistics.copy()

    def shutdown(self, graceful: bool = False) -> None:
        """
        Close manager and all processors

        Note: Task scheduler shutdown is handled separately by OpenContext.
        """
        logger.info("Shutting down context processing manager...")
        for processor in self._processors.values():
            processor.shutdown()
        logger.info("Context processing manager has been shut down")

    def reset_statistics(self) -> None:
        """
        Reset statistics for manager and all processors
        """
        with self._lock:
            for processor in self._processors.values():
                processor.reset_statistics()

            self._statistics["total_processed_inputs"] = 0
            self._statistics["total_contexts_generated"] = 0
            self._statistics["errors"] = 0
            logger.info("All statistics have been reset")
