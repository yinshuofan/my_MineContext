# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
OpenContext module: llm_client
"""

import asyncio
import time as _time
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp
from openai import APIError, AsyncOpenAI

from opencontext.models.context import Vectorize
from opencontext.monitoring import record_processing_stage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMProvider(Enum):
    OPENAI = "openai"
    DOUBAO = "doubao"
    DASHSCOPE = "dashscope"


class LLMType(Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"


class LLMClient:
    def __init__(self, llm_type: LLMType, config: Dict[str, Any]):
        self.llm_type = llm_type
        self.config = config
        self.model = config.get("model")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout", 300)
        self.max_retries = config.get("max_retries", 3)
        self.provider = config.get("provider", LLMProvider.OPENAI.value)
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("API key, base URL, and model must be provided")
        self._max_concurrent = int(config.get("max_concurrent", 10))
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        # aiohttp session for multimodal embedding (lazy-initialized)
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._http_session_lock = asyncio.Lock()

    @property
    def _sem(self) -> asyncio.Semaphore:
        """Lazy-init semaphore to ensure creation inside an event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def generate_with_messages(self, messages: List[Dict[str, Any]], **kwargs):
        if self.llm_type == LLMType.CHAT:
            return await self._openai_chat_completion(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type for message generation: {self.llm_type}")

    async def generate_with_messages_stream(self, messages: List[Dict[str, Any]], **kwargs):
        """Async stream generate response"""
        if self.llm_type == LLMType.CHAT:
            async for chunk in self._openai_chat_completion_stream(messages, **kwargs):
                yield chunk
        else:
            raise ValueError(f"Unsupported LLM type for stream generation: {self.llm_type}")

    async def generate_embedding(self, text: str, **kwargs) -> List[float]:
        if self.llm_type == LLMType.EMBEDDING:
            return await self._openai_embedding(text, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type for embedding generation: {self.llm_type}")

    async def _openai_chat_completion(self, messages: List[Dict[str, Any]], **kwargs):
        """Async chat completion"""
        import time

        await self._sem.acquire()
        try:
            request_start = time.time()
            try:
                tools = kwargs.get("tools", None)
                thinking = kwargs.get("thinking", None)

                create_params = {
                    "model": self.model,
                    "messages": messages,
                }
                if tools:
                    create_params["tools"] = tools
                    create_params["tool_choice"] = "auto"

                if thinking:
                    if self.provider == LLMProvider.DOUBAO.value:
                        if thinking == "disabled":
                            create_params["reasoning_effort"] = "minimal"
                    elif self.provider == LLMProvider.DASHSCOPE.value:
                        create_params["extra_body"] = {"thinking": {"type": thinking}}
                # Stage: LLM API call
                api_start = time.time()
                response = await self.client.chat.completions.create(**create_params)

                await record_processing_stage(
                    "chat_cost", int((time.time() - api_start) * 1000), status="success"
                )

                # Record token usage
                if hasattr(response, "usage") and response.usage:
                    try:
                        from opencontext.monitoring import record_token_usage

                        await record_token_usage(
                            model=self.model,
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                        )
                    except ImportError:
                        pass  # Monitoring module not installed or initialized

                return response
            except APIError as e:
                logger.exception(f"OpenAI API async error: {e}")
                # Record failure
                try:
                    await record_processing_stage(
                        "chat_cost", int((time.time() - request_start) * 1000), status="failure"
                    )
                except ImportError:
                    pass
                raise
        finally:
            self._sem.release()

    async def _openai_chat_completion_stream(self, messages: List[Dict[str, Any]], **kwargs):
        """Async stream chat completion - async generator"""
        await self._sem.acquire()
        try:
            try:
                tools = kwargs.get("tools", None)
                thinking = kwargs.get("thinking", None)

                create_params = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                }
                if tools:
                    create_params["tools"] = tools
                    create_params["tool_choice"] = "auto"

                if thinking:
                    if self.provider == LLMProvider.DOUBAO.value:
                        if thinking == "disabled":
                            create_params["reasoning_effort"] = "minimal"
                    elif self.provider == LLMProvider.DASHSCOPE.value:
                        create_params["extra_body"] = {"thinking": {"type": thinking}}

                stream = await self.client.chat.completions.create(**create_params)

                # Return stream object directly, it's already an async iterator
                async for chunk in stream:
                    yield chunk
            except APIError as e:
                logger.error(f"OpenAI API async stream error: {e}")
                raise
        finally:
            self._sem.release()

    async def _openai_embedding(self, text: str, **kwargs) -> List[float]:
        await self._sem.acquire()
        try:
            response = await self.client.embeddings.create(model=self.model, input=[text])
            embedding = response.data[0].embedding

            # Record token usage
            if hasattr(response, "usage") and response.usage:
                try:
                    from opencontext.monitoring import record_token_usage

                    await record_token_usage(
                        model=self.model,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=0,  # embedding has no completion tokens
                        total_tokens=response.usage.total_tokens,
                    )
                except ImportError:
                    pass  # Monitoring module not installed or initialized

            output_dim = int(kwargs.get("output_dim", self.config.get("output_dim", 0)))
            if output_dim and len(embedding) > output_dim:
                import math

                embedding = embedding[:output_dim]
                norm = math.sqrt(sum(x**2 for x in embedding))
                if norm > 0:
                    embedding = [x / norm for x in embedding]

            return embedding
        except APIError as e:
            logger.error(f"OpenAI API error during embedding: {e}")
            raise
        finally:
            self._sem.release()

    async def vectorize(self, vectorize: Vectorize, **kwargs):
        if vectorize.vector:
            return
        content = vectorize.get_vectorize_content()
        if not content:
            return
        vectorize.vector = await self.generate_embedding(content, **kwargs)

    async def generate_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single API call (with internal chunking)."""
        if not texts:
            return []
        if self.llm_type != LLMType.EMBEDDING:
            raise ValueError(f"Unsupported LLM type for embedding generation: {self.llm_type}")

        max_batch = int(kwargs.pop("max_batch_size", self.config.get("max_batch_size", 64)))
        output_dim = int(kwargs.get("output_dim", self.config.get("output_dim", 0)))
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), max_batch):
            chunk = texts[i : i + max_batch]
            try:
                async with self._sem:
                    response = await self.client.embeddings.create(model=self.model, input=chunk)
                sorted_data = sorted(response.data, key=lambda d: d.index)
                chunk_embeddings = [d.embedding for d in sorted_data]

                if output_dim:
                    chunk_embeddings = self._truncate_embeddings(chunk_embeddings, output_dim)

                if hasattr(response, "usage") and response.usage:
                    try:
                        from opencontext.monitoring import record_token_usage

                        await record_token_usage(
                            model=self.model,
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=0,
                            total_tokens=response.usage.total_tokens,
                        )
                    except ImportError:
                        pass

                all_embeddings.extend(chunk_embeddings)
            except APIError as e:
                logger.warning(
                    f"Batch embedding async failed for chunk ({len(chunk)} texts), "
                    f"falling back to individual calls: {e}"
                )
                for text in chunk:
                    all_embeddings.append(await self.generate_embedding(text, **kwargs))

        return all_embeddings

    def _truncate_embeddings(
        self, embeddings: List[List[float]], output_dim: int
    ) -> List[List[float]]:
        """Truncate embeddings to output_dim and re-normalize with L2 norm."""
        import math

        result = []
        for embedding in embeddings:
            if len(embedding) > output_dim:
                embedding = embedding[:output_dim]
                norm = math.sqrt(sum(x**2 for x in embedding))
                if norm > 0:
                    embedding = [x / norm for x in embedding]
            result.append(embedding)
        return result

    # ------------------------------------------------------------------
    # Multimodal embedding via Ark HTTP API (aiohttp, not SDK)
    # ------------------------------------------------------------------

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp session for multimodal embedding."""
        if self._http_session is None or self._http_session.closed:
            async with self._http_session_lock:
                if self._http_session is None or self._http_session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=20,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True,
                    )
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    self._http_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                    )
        return self._http_session

    async def generate_multimodal_embedding(
        self,
        input_data: List[Dict[str, Any]],
        instruction: str,
        dimensions: int = 2048,
    ) -> List[float]:
        """Call Ark multimodal embedding API via HTTP (not SDK).

        Args:
            input_data: List of input items matching Ark API format
                (e.g. [{"type": "text", "text": "..."}, {"type": "image_url", ...}]).
            instruction: The instruction string for corpus or query side.
            dimensions: Output vector dimension (default 2048).

        Returns:
            Embedding vector as a list of floats.

        Raises:
            Exception: On HTTP or API errors after exhausting retries.
        """
        payload = {
            "model": self.model,
            "encoding_format": "float",
            "dimensions": dimensions,
            "instructions": instruction,
            "input": input_data,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        session = await self._get_http_session()

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._sem:
                    request_start = _time.time()
                    async with session.post(
                        self.base_url, json=payload, headers=headers
                    ) as resp:
                        body = await resp.json()
                        if resp.status != 200:
                            error_msg = (
                                body.get("error", {}).get("message", resp.reason)
                                if isinstance(body, dict)
                                else str(resp.reason)
                            )
                            raise Exception(
                                f"Ark multimodal embedding API returned HTTP {resp.status}: "
                                f"{error_msg}"
                            )

                        embedding = body["data"]["embedding"]

                        # Record token usage
                        usage = body.get("usage", {})
                        if usage:
                            try:
                                from opencontext.monitoring import record_token_usage

                                await record_token_usage(
                                    model=self.model,
                                    prompt_tokens=usage.get("prompt_tokens", 0),
                                    completion_tokens=0,
                                    total_tokens=usage.get("total_tokens", 0),
                                )
                            except ImportError:
                                pass

                        await record_processing_stage(
                            "multimodal_embedding_cost",
                            int((_time.time() - request_start) * 1000),
                            status="success",
                        )

                        return embedding

            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = min(2**attempt, 8)
                    logger.warning(
                        f"Multimodal embedding attempt {attempt + 1}/{self.max_retries + 1} "
                        f"failed: {e}. Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"Multimodal embedding failed after {self.max_retries + 1} attempts: {e}"
                    )

        raise last_exc  # type: ignore[misc]

    async def close_http_session(self) -> None:
        """Close the aiohttp session used for multimodal embedding."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    async def validate(self) -> tuple[bool, str]:
        """
        Validate LLM configuration by making a simple API call.

        Returns:
            tuple[bool, str]: (success, message)
        """

        def _extract_error_summary(error: Any) -> str:
            """
            Extract a concise error summary from API error messages.
            Removes verbose API error details and keeps only the essential information.
            """
            error_msg = str(error)
            if not error_msg:
                return "Unknown error"

            # 1. Check for specific Volcengine/Doubao error codes
            volcengine_errors = {
                "AccessDenied": "Access denied. Please ensure the model is enabled in the Volcengine console.",
                "QuotaExceeded": "Quota exceeded. Please check your Volcengine account balance.",
                "ModelAccountIpmRateLimitExceeded": "Model rate limit (IPM) exceeded.",
                "AccountRateLimitExceeded": "Account rate limit exceeded.",
                "RateLimitExceeded": "Rate limit exceeded.",
                "InternalServiceError": "Volcengine internal service error.",
                "ServiceUnavailable": "Service unavailable.",
                "MethodNotAllowed": "Method not allowed. Check your configuration.",
            }

            for code, msg in volcengine_errors.items():
                if code in error_msg:
                    return msg

            # 2. Check for OpenAI specific errors
            openai_errors = {
                "insufficient_quota": "Insufficient quota. Check your plan and billing details.",
                "invalid_api_key": "Invalid API key provided.",
                "model_not_found": "The model does not exist or you do not have access to it.",
                "context_length_exceeded": "Context length exceeded.",
            }

            for code, msg in openai_errors.items():
                if code in error_msg:
                    return msg

            # If it's an API error with detailed JSON response, extract key info
            if "Error code:" in error_msg:
                parts = error_msg.split("Error code:", 1)
                if len(parts) > 1:
                    code_part = parts[1].strip()
                    # Get just the code number and basic message
                    if "-" in code_part:
                        code = code_part.split("-", 1)[0].strip()
                        # Try to extract the error type/message from the dict
                        if "'message':" in code_part:
                            try:
                                msg_start = code_part.find("'message':") + len("'message':")
                                msg_part = code_part[msg_start:].strip()
                                if msg_part.startswith("'") or msg_part.startswith('"'):
                                    quote_char = msg_part[0]
                                    msg_end = msg_part.find(quote_char, 1)
                                    if msg_end > 0:
                                        actual_msg = msg_part[1:msg_end]
                                        # Remove Request id and everything after it
                                        if ". Request id:" in actual_msg:
                                            actual_msg = actual_msg.split(". Request id:")[0]
                                        return actual_msg
                            except Exception as e:
                                logger.debug(f"Failed to parse error message details: {e}")
                                pass
                        return f"Error {code}"

            # If the message is already concise (< 150 chars), return as-is
            if len(error_msg) < 150:
                return error_msg

            # Otherwise, truncate with ellipsis
            return error_msg[:147] + "..."

        try:
            if self.llm_type == LLMType.CHAT:
                messages = [{"role": "user", "content": "Hi"}]
                response = await self.client.chat.completions.create(
                    model=self.model, messages=messages
                )
                if response.choices and len(response.choices) > 0:
                    return True, "Chat model validation successful"
                else:
                    return False, "Chat model returned empty response"

            elif self.llm_type == LLMType.EMBEDDING:
                # Test with a simple multimodal embedding call
                instruction = "Instruction:Compress the text into one word.\nQuery:"
                embedding = await self.generate_multimodal_embedding(
                    input_data=[{"type": "text", "text": "test"}],
                    instruction=instruction,
                )
                if embedding and len(embedding) > 0:
                    return True, "Embedding model validation successful"
                else:
                    return False, "Embedding model returned empty response"
            else:
                return False, f"Unsupported LLM type: {self.llm_type}"

        except APIError as e:
            logger.error(f"LLM validation failed with API error: {e}")
            # Extract concise error summary before returning
            concise_error = _extract_error_summary(e)
            return False, concise_error
        except Exception as e:
            logger.error(f"LLM validation failed with unexpected error: {e}")
            # Extract concise error summary before returning
            concise_error = _extract_error_summary(e)
            return False, concise_error
