#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Global embedding client singleton wrapper.

Supports multimodal embedding via the Ark HTTP API (doubao-embedding-vision).
Uses role-based instructions (corpus vs query) to control embedding behavior.
"""

import asyncio
import threading
from collections.abc import Sequence

from opencontext.config.global_config import get_config
from opencontext.llm.llm_client import LLMClient, LLMType
from opencontext.models.context import Vectorize
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default configuration values
_DEFAULT_DIMENSIONS = 2048
_DEFAULT_MAX_CONCURRENCY = 15
_DEFAULT_SEARCH_INSTRUCTION = "根据这个查询，找到最相关的记忆内容"
_DEFAULT_TARGET_MODALITY = "text/image/video"
_DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"


class GlobalEmbeddingClient:
    """
    Global embedding client (singleton pattern).

    Uses the Ark multimodal embedding API for vectorization.
    Supports corpus-side and query-side instructions per the
    doubao-embedding-vision model specification.
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize global embedding client"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._embedding_client: LLMClient | None = None
                    self._auto_initialized = False
                    # Multimodal config (populated during auto-init)
                    self._dimensions: int = _DEFAULT_DIMENSIONS
                    self._max_concurrency: int = _DEFAULT_MAX_CONCURRENCY
                    self._search_instruction: str = _DEFAULT_SEARCH_INSTRUCTION
                    self._target_modality: str = _DEFAULT_TARGET_MODALITY
                    GlobalEmbeddingClient._initialized = True

    @classmethod
    def get_instance(cls) -> "GlobalEmbeddingClient":
        """
        Get global embedding client instance
        """
        instance = cls()
        # If not initialized yet, try auto-initialization
        if not instance._auto_initialized and instance._embedding_client is None:
            instance._auto_initialize()
        return instance

    def _auto_initialize(self):
        """Auto-initialize embedding client"""
        if self._auto_initialized:
            return
        try:
            embedding_config = get_config("embedding_model")
            if not embedding_config:
                logger.warning("No embedding config found in embedding_model")
                self._auto_initialized = True
                return

            self._embedding_client = LLMClient(llm_type=LLMType.EMBEDDING, config=embedding_config)

            # Read multimodal-specific config
            self._dimensions = int(embedding_config.get("dimensions", _DEFAULT_DIMENSIONS))
            self._max_concurrency = int(
                embedding_config.get("max_concurrency", _DEFAULT_MAX_CONCURRENCY)
            )
            self._search_instruction = embedding_config.get(
                "search_instruction", _DEFAULT_SEARCH_INSTRUCTION
            )
            self._target_modality = embedding_config.get(
                "target_modality", _DEFAULT_TARGET_MODALITY
            )

            logger.info(
                f"GlobalEmbeddingClient auto-initialized successfully "
                f"(dimensions={self._dimensions}, max_concurrency={self._max_concurrency})"
            )
            self._auto_initialized = True
        except Exception as e:
            logger.error(f"GlobalEmbeddingClient auto-initialization failed: {e}")
            self._auto_initialized = True

    def is_initialized(self) -> bool:
        return self._embedding_client is not None

    def reinitialize(self, new_config: dict | None = None):
        """
        Reinitialize embedding client (thread-safe)
        """
        with self._lock:
            try:
                embedding_config = get_config("embedding_model")
                if not embedding_config:
                    logger.error("No embedding config found during reinitialize")
                    raise ValueError("No embedding config found")
                logger.info("Reinitializing embedding client...")
                new_client = LLMClient(llm_type=LLMType.EMBEDDING, config=embedding_config)
                old_client = self._embedding_client
                self._embedding_client = new_client

                # Update multimodal config
                self._dimensions = int(embedding_config.get("dimensions", _DEFAULT_DIMENSIONS))
                self._max_concurrency = int(
                    embedding_config.get("max_concurrency", _DEFAULT_MAX_CONCURRENCY)
                )
                self._search_instruction = embedding_config.get(
                    "search_instruction", _DEFAULT_SEARCH_INSTRUCTION
                )
                self._target_modality = embedding_config.get(
                    "target_modality", _DEFAULT_TARGET_MODALITY
                )

                # Close old client's HTTP session if it exists
                if old_client is not None:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(old_client.close_http_session())
                    except RuntimeError:
                        pass  # No running loop, session will be GC'd

                logger.info("Embedding client reinitialization completed")
            except Exception as e:
                logger.error(f"Failed to reinitialize embedding client: {e}")
                return False
            return True

    # ------------------------------------------------------------------
    # Instruction building
    # ------------------------------------------------------------------

    def _build_instruction(self, vectorize: Vectorize, role: str) -> str:
        """Build the instruction string based on role and content modality.

        Args:
            vectorize: The Vectorize object containing content to embed.
            role: "corpus" for indexing or "query" for retrieval.

        Returns:
            Instruction string for the Ark multimodal embedding API.
        """
        if role == "corpus":
            modality = vectorize.get_modality_string()
            return f"Instruction:Compress the {modality} into one word.\nQuery:"
        else:
            # Query-side instruction
            return (
                f"Target_modality: {self._target_modality}.\n"
                f"Instruction:{self._search_instruction}\n"
                f"Query:"
            )

    # ------------------------------------------------------------------
    # Core vectorization methods
    # ------------------------------------------------------------------

    async def do_vectorize(self, vectorize: Vectorize, role: str = "corpus", **kwargs):
        """Vectorize a single Vectorize object using the Ark multimodal API.

        Args:
            vectorize: The Vectorize object to embed. Its ``vector`` field
                will be set in-place.
            role: ``"corpus"`` (default) for indexing, ``"query"`` for retrieval.
                Determines which instruction template is used.
            **kwargs: Reserved for forward compatibility.
        """
        if vectorize.vector is not None:
            return

        ark_input = vectorize.build_ark_input()
        if not ark_input:
            logger.warning("Vectorize object has no content to embed, skipping")
            return

        instruction = self._build_instruction(vectorize, role)
        embedding = await self._embedding_client.generate_multimodal_embedding(  # type: ignore[union-attr]
            input_data=ark_input,
            instruction=instruction,
            dimensions=self._dimensions,
        )
        vectorize.vector = embedding

    async def do_vectorize_batch(
        self, vectorizes: Sequence[Vectorize], role: str = "corpus", **kwargs
    ):
        """Vectorize multiple Vectorize objects using concurrent Ark API calls.

        The Ark multimodal embedding API returns one embedding per request,
        so batch processing is achieved through concurrent requests controlled
        by an asyncio.Semaphore.

        Args:
            vectorizes: Sequence of Vectorize objects. Items with an existing
                ``vector`` are skipped.
            role: ``"corpus"`` (default) or ``"query"``.
            **kwargs: Reserved for forward compatibility.
        """
        pending = [
            (i, v) for i, v in enumerate(vectorizes) if v.vector is None and v.build_ark_input()
        ]

        if not pending:
            return

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _vectorize_one(idx: int, v: Vectorize) -> None:
            async with semaphore:
                await self.do_vectorize(v, role=role)

        await asyncio.gather(
            *[_vectorize_one(i, v) for i, v in pending],
            return_exceptions=True,
        )

        # Log any failures
        for i, v in pending:
            if v.vector is None:
                logger.warning(
                    f"Failed to vectorize item {i} in batch "
                    f"(text={v.get_text()[:50] if v.get_text() else 'N/A'}...)"  # type: ignore[index]
                )


def is_initialized() -> bool:
    return GlobalEmbeddingClient.get_instance().is_initialized()


async def do_vectorize(vectorize_obj: Vectorize, **kwargs):
    return await GlobalEmbeddingClient.get_instance().do_vectorize(vectorize_obj, **kwargs)


async def do_vectorize_batch(vectorize_objs: Sequence[Vectorize], **kwargs):
    return await GlobalEmbeddingClient.get_instance().do_vectorize_batch(vectorize_objs, **kwargs)
