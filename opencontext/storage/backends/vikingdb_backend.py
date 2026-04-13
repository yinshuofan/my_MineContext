# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
VikingDB vector storage backend - Volcengine VikingDB Service (HTTP API)
https://www.volcengine.com/docs/84313/1254458

This implementation uses HTTP API with Volcengine V4 signature for authentication.
Uses a single collection with context_type field filtering to reduce operational costs.
"""

import asyncio
import contextlib
import datetime
import hashlib
import hmac
import json
import os
from enum import Enum
from typing import Any
from urllib.parse import quote, urlencode

import aiohttp

from opencontext.llm.global_embedding_client import do_vectorize, do_vectorize_batch
from opencontext.models.context import ContextProperties, ExtractedData, ProcessedContext, Vectorize
from opencontext.models.enums import ContextType
from opencontext.storage.base_storage import IVectorStorageBackend, StorageType
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.media_refs import normalize_media_refs
from opencontext.utils.time_utils import utc_now

logger = get_logger(__name__)

# Single collection name for all context types
DEFAULT_COLLECTION_NAME = "opencontext"
DEFAULT_INDEX_NAME = "opencontext_index"

# Field constants
FIELD_DOCUMENT = "document"

# Time fields
FIELD_CREATE_TIME = "create_time"
FIELD_CREATE_TIME_TS = "create_time_ts"
FIELD_EVENT_TIME_START = "event_time_start"
FIELD_EVENT_TIME_START_TS = "event_time_start_ts"
FIELD_EVENT_TIME_END = "event_time_end"
FIELD_EVENT_TIME_END_TS = "event_time_end_ts"
FIELD_UPDATE_TIME = "update_time"
FIELD_UPDATE_TIME_TS = "update_time_ts"
FIELD_LAST_CALL_TIME = "last_call_time"
FIELD_LAST_CALL_TIME_TS = "last_call_time_ts"

# User identification fields
FIELD_USER_ID = "user_id"
FIELD_DEVICE_ID = "device_id"
FIELD_AGENT_ID = "agent_id"

# Extracted data fields
FIELD_TITLE = "title"
FIELD_SUMMARY = "summary"
FIELD_KEYWORDS = "keywords"
FIELD_ENTITIES = "entities"
FIELD_CONTEXT_TYPE = "context_type"
FIELD_CONFIDENCE = "confidence"
FIELD_IMPORTANCE = "importance"
FIELD_AGENT_COMMENTARY = "agent_commentary"
# Properties fields
FIELD_IS_PROCESSED = "is_processed"
FIELD_HAS_COMPRESSION = "has_compression"
FIELD_ENABLE_MERGE = "enable_merge"
FIELD_CALL_COUNT = "call_count"
FIELD_MERGE_COUNT = "merge_count"
FIELD_DURATION_COUNT = "duration_count"

# Document tracking fields
FIELD_FILE_PATH = "file_path"
FIELD_RAW_TYPE = "raw_type"
FIELD_RAW_ID = "raw_id"

# Hierarchy and document overwrite fields
FIELD_HIERARCHY_LEVEL = "hierarchy_level"
FIELD_REFS = "refs"

# Multimodal fields
FIELD_CONTENT_MODALITIES = "content_modalities"
FIELD_MEDIA_REFS = "media_refs"

# Data type field
FIELD_DATA_TYPE = "data_type"
DATA_TYPE_CONTEXT = "context"

# HTTP API constants
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_MAX_CONNECTIONS_PER_HOST = 20

# VikingDB API constants
VIKINGDB_SERVICE = "vikingdb"
VIKINGDB_VERSION = "2025-06-09"


class VolcengineAuth:
    """
    Volcengine V4 signature authentication class.
    Implements the signature algorithm for Volcengine API requests.
    """

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str = "cn-beijing",
        service: str = VIKINGDB_SERVICE,
    ):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self.service = service

    def _get_canonical_uri(self, path: str) -> str:
        """Get canonical URI."""
        return quote(path, safe="/")

    def _get_canonical_query_string(self, params: dict[str, str]) -> str:
        """Get canonical query string."""
        if not params:
            return ""
        sorted_params = sorted(params.items())
        return "&".join([f"{quote(k, safe='')}={quote(str(v), safe='')}" for k, v in sorted_params])

    def _get_canonical_headers(self, headers: dict[str, str]) -> tuple[str, str]:
        """Get canonical headers and signed headers."""
        # Headers to sign (lowercase)
        headers_to_sign = {}
        for key, value in headers.items():
            lower_key = key.lower()
            if lower_key in ["host", "content-type", "x-date", "x-content-sha256"]:
                headers_to_sign[lower_key] = value.strip()

        # Sort headers
        sorted_headers = sorted(headers_to_sign.items())
        canonical_headers = "\n".join([f"{k}:{v}" for k, v in sorted_headers]) + "\n"
        signed_headers = ";".join([k for k, v in sorted_headers])

        return canonical_headers, signed_headers

    def _get_payload_hash(self, body: str) -> str:
        """Get SHA256 hash of request body."""
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def _hmac_sha256(self, key: bytes, msg: str) -> bytes:
        """HMAC-SHA256 signing."""
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    def _get_signing_key(self, date_stamp: str) -> bytes:
        """Get signing key using secret access key."""
        k_date = self._hmac_sha256(self.secret_access_key.encode("utf-8"), date_stamp)
        k_region = self._hmac_sha256(k_date, self.region)
        k_service = self._hmac_sha256(k_region, self.service)
        k_signing = self._hmac_sha256(k_service, "request")
        return k_signing

    def sign_request(
        self,
        method: str,
        host: str,
        path: str,
        headers: dict[str, str],
        body: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Sign a request with Volcengine V4 signature.

        Args:
            method: HTTP method
            host: API host
            path: Request path
            headers: Request headers
            body: Request body
            params: Query parameters

        Returns:
            Headers with Authorization added
        """
        # Get current time
        t = utc_now()  # Volcengine V4 signature requires UTC
        x_date = t.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = t.strftime("%Y%m%d")

        # Add required headers
        payload_hash = self._get_payload_hash(body)
        headers_to_sign = {
            "Host": host,
            "Content-Type": "application/json",
            "X-Date": x_date,
            "X-Content-Sha256": payload_hash,
        }
        headers_to_sign.update(headers)

        # Build canonical request
        canonical_uri = self._get_canonical_uri(path)
        canonical_query_string = self._get_canonical_query_string(params or {})
        canonical_headers, signed_headers = self._get_canonical_headers(headers_to_sign)

        canonical_request = "\n".join(
            [
                method.upper(),
                canonical_uri,
                canonical_query_string,
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )

        # Build string to sign
        credential_scope = f"{date_stamp}/{self.region}/{self.service}/request"
        hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        string_to_sign = "\n".join(
            [
                "HMAC-SHA256",
                x_date,
                credential_scope,
                hashed_canonical_request,
            ]
        )

        # Calculate signature
        signing_key = self._get_signing_key(date_stamp)
        signature = hmac.new(
            signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # Build authorization header
        authorization = (
            f"HMAC-SHA256 Credential={self.access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        # Return headers with authorization
        result_headers = dict(headers_to_sign)
        result_headers["Authorization"] = authorization

        return result_headers


class VikingDBHTTPClient:
    """
    HTTP client for VikingDB API.
    Supports both control plane (console) and data plane APIs.
    Uses aiohttp for async HTTP requests.
    """

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str = "cn-beijing",
        data_host: str | None = None,
        console_host: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_connections_per_host: int = DEFAULT_MAX_CONNECTIONS_PER_HOST,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._region = region
        self._timeout = timeout
        self._max_connections = max_connections
        self._max_connections_per_host = max_connections_per_host
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Set API hosts
        self._data_host = data_host or f"api-vikingdb.vikingdb.{region}.volces.com"
        self._console_host = console_host or f"vikingdb.{region}.volcengineapi.com"

        # Initialize auth
        self._auth = VolcengineAuth(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region,
        )

        # Initialize async session (lazy)
        self._async_session: aiohttp.ClientSession | None = None
        self._async_lock = asyncio.Lock()

    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async session."""
        if self._async_session is None or self._async_session.closed:
            async with self._async_lock:
                if self._async_session is None or self._async_session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=self._max_connections,
                        limit_per_host=self._max_connections_per_host,
                        ttl_dns_cache=300,
                        enable_cleanup_closed=True,
                    )
                    timeout = aiohttp.ClientTimeout(total=self._timeout)
                    self._async_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                    )
        return self._async_session

    async def _async_request(
        self,
        method: str,
        host: str,
        path: str,
        data: dict | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make an asynchronous HTTP request with retry.

        Args:
            method: HTTP method
            host: API host
            path: Request path
            data: Request body
            params: Query parameters

        Returns:
            Response JSON as dict
        """
        body = json.dumps(data) if data else ""

        # Sign request
        headers = self._auth.sign_request(
            method=method,
            host=host,
            path=path,
            headers={},
            body=body,
            params=params,
        )

        # Build URL
        url = f"https://{host}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"

        session = await self._get_async_session()

        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body,
                ) as response:
                    response_text = await response.text()

                    if response.status == 404:
                        try:
                            return json.loads(response_text)
                        except Exception:
                            error_msg = (
                                f"API request failed with status "
                                f"{response.status}: {response_text[:500]}"
                            )
                            logger.error(error_msg)
                            raise Exception(error_msg) from None

                    if response.status in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                        await asyncio.sleep(self._retry_delay * (attempt + 1))
                        # Re-sign for retry (time may have changed)
                        body_for_sign = json.dumps(data) if data else ""
                        headers = self._auth.sign_request(
                            method=method,
                            host=host,
                            path=path,
                            headers={},
                            body=body_for_sign,
                            params=params,
                        )
                        continue

                    if response.status != 200:
                        error_msg = (
                            f"API request failed with status "
                            f"{response.status}: {response_text[:500]}"
                        )
                        logger.error(error_msg)
                        raise Exception(error_msg)

                    try:
                        return json.loads(response_text)
                    except Exception:
                        logger.error(f"Failed to parse JSON response: {response_text[:500]}")
                        raise

            except (TimeoutError, aiohttp.ClientError) as e:
                last_exc = e
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                    # Re-sign for retry
                    body_for_sign = json.dumps(data) if data else ""
                    headers = self._auth.sign_request(
                        method=method,
                        host=host,
                        path=path,
                        headers={},
                        body=body_for_sign,
                        params=params,
                    )
                    continue
                raise

        raise last_exc  # type: ignore[misc]

    async def async_console_request(
        self,
        action: str,
        data: dict | None = None,
        version: str = VIKINGDB_VERSION,
    ) -> dict[str, Any]:
        """
        Make async request to console (control plane) API.

        Args:
            action: API action name (e.g., CreateVikingdbCollection)
            data: Request body
            version: API version

        Returns:
            Response JSON as dict
        """
        params = {
            "Action": action,
            "Version": version,
        }

        return await self._async_request(
            method="POST",
            host=self._console_host,
            path="/",
            data=data,
            params=params,
        )

    async def async_data_request(
        self,
        path: str,
        data: dict | None = None,
    ) -> dict[str, Any]:
        """
        Make async request to data plane API.

        Args:
            path: API path (e.g., /api/collection/upsert_data)
            data: Request body

        Returns:
            Response JSON as dict
        """
        return await self._async_request(
            method="POST",
            host=self._data_host,
            path=path,
            data=data,
        )

    async def close(self):
        """Close the HTTP client and release resources."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()


class VikingDBBackend(IVectorStorageBackend):
    """
    VikingDB vector storage backend implementation.

    Uses Volcengine VikingDB service for vector storage and similarity search.
    Uses a single collection with context_type field filtering to reduce costs.
    Implements the IVectorStorageBackend interface for compatibility with
    the opencontext storage system.
    """

    def __init__(self):
        self._client: VikingDBHTTPClient | None = None
        self._collection_name: str = DEFAULT_COLLECTION_NAME
        self._index_name: str = DEFAULT_INDEX_NAME
        self._dimension: int = 0
        self._initialized: bool = False
        self._config: dict[str, Any] = {}
        self._collection_ready: bool = False
        self._index_ready: bool = False

    async def initialize(self, config: dict[str, Any]) -> bool:
        """
        Initialize VikingDB backend.

        Args:
            config: Configuration dictionary

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self._config = config
            vikingdb_config = config.get("config", {})

            # Get credentials from config or environment
            access_key_id = vikingdb_config.get("access_key_id") or os.environ.get(
                "VOLCENGINE_ACCESS_KEY_ID"
            )
            secret_access_key = vikingdb_config.get("secret_access_key") or os.environ.get(
                "VOLCENGINE_SECRET_ACCESS_KEY"
            )

            if not access_key_id or not secret_access_key:
                logger.error("VikingDB credentials not provided")
                return False

            region = vikingdb_config.get("region", "cn-beijing")
            data_host = vikingdb_config.get("data_host")
            console_host = vikingdb_config.get("console_host")

            self._dimension = vikingdb_config.get("dimension", 1024)
            self._collection_name = vikingdb_config.get("collection_name", DEFAULT_COLLECTION_NAME)
            self._index_name = vikingdb_config.get("index_name", DEFAULT_INDEX_NAME)

            # Initialize HTTP client
            self._client = VikingDBHTTPClient(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                region=region,
                data_host=data_host,
                console_host=console_host,
            )

            # Ensure single collection and index exist
            index_type = vikingdb_config.get("index_type", "hnsw")
            distance_type = vikingdb_config.get("distance_type", "cosine")

            await self._ensure_collection_and_index(
                dimension=self._dimension,
                index_type=index_type,
                distance_type=distance_type,
            )

            self._initialized = True
            logger.info(
                f"VikingDB backend initialized with single collection: {self._collection_name}"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize VikingDB backend: {e}")
            return False

    async def _ensure_collection_and_index(
        self,
        dimension: int,
        index_type: str = "hnsw",
        distance_type: str = "cosine",
    ) -> None:
        """Ensure single collection and index exist, create if not."""
        try:
            # Check if collection exists
            result = await self._client.async_console_request(  # type: ignore[union-attr]
                action="GetVikingdbCollection",
                data={"CollectionName": self._collection_name},
            )

            if result.get("ResponseMetadata", {}).get("Error"):
                error_code = result["ResponseMetadata"]["Error"].get("Code", "")
                if "NotExist" in error_code or "NotFound" in error_code:
                    await self._create_collection(dimension)
                else:
                    logger.error(f"Failed to get collection {self._collection_name}: {result}")
                    raise RuntimeError(f"Failed to get collection: {result}")
            else:
                logger.debug(f"VikingDB collection already exists: {self._collection_name}")

            self._collection_ready = True

            # Check if index exists
            result = await self._client.async_console_request(  # type: ignore[union-attr]
                action="GetVikingdbIndex",
                data={
                    "CollectionName": self._collection_name,
                    "IndexName": self._index_name,
                },
            )

            if result.get("ResponseMetadata", {}).get("Error"):
                error_code = result["ResponseMetadata"]["Error"].get("Code", "")
                if "NotExist" in error_code or "NotFound" in error_code:
                    await self._create_index(
                        index_type=index_type,
                        distance_type=distance_type,
                    )
                else:
                    logger.error(f"Failed to get index {self._index_name}: {result}")
                    raise RuntimeError(f"Failed to get index: {result}")
            else:
                logger.debug(f"VikingDB index already exists: {self._index_name}")

            self._index_ready = True

        except Exception as e:
            logger.exception(f"Error ensuring collection and index: {e}")
            raise

    async def _create_collection(self, dimension: int) -> None:
        """Create the single collection with all necessary fields."""
        fields = [
            {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
            {"FieldName": "vector", "FieldType": "vector", "Dim": dimension},
            {"FieldName": FIELD_DATA_TYPE, "FieldType": "string"},
            {"FieldName": FIELD_USER_ID, "FieldType": "string"},
            {"FieldName": FIELD_DEVICE_ID, "FieldType": "string"},
            {"FieldName": FIELD_AGENT_ID, "FieldType": "string"},
            {"FieldName": FIELD_CREATE_TIME, "FieldType": "string"},
            {"FieldName": FIELD_CREATE_TIME_TS, "FieldType": "float32"},
            {"FieldName": FIELD_EVENT_TIME_START, "FieldType": "string"},
            {"FieldName": FIELD_EVENT_TIME_START_TS, "FieldType": "float32"},
            {"FieldName": FIELD_EVENT_TIME_END, "FieldType": "string"},
            {"FieldName": FIELD_EVENT_TIME_END_TS, "FieldType": "float32"},
            {"FieldName": FIELD_UPDATE_TIME, "FieldType": "string"},
            {"FieldName": FIELD_UPDATE_TIME_TS, "FieldType": "float32"},
            {"FieldName": FIELD_LAST_CALL_TIME, "FieldType": "string"},
            {"FieldName": FIELD_LAST_CALL_TIME_TS, "FieldType": "float32"},
            {"FieldName": FIELD_TITLE, "FieldType": "string"},
            {"FieldName": FIELD_SUMMARY, "FieldType": "string"},
            {"FieldName": FIELD_KEYWORDS, "FieldType": "string"},
            {"FieldName": FIELD_ENTITIES, "FieldType": "string"},
            {"FieldName": FIELD_CONTEXT_TYPE, "FieldType": "string"},
            {"FieldName": FIELD_CONFIDENCE, "FieldType": "float32"},
            {"FieldName": FIELD_IMPORTANCE, "FieldType": "float32"},
            {"FieldName": FIELD_AGENT_COMMENTARY, "FieldType": "string"},
            {"FieldName": FIELD_IS_PROCESSED, "FieldType": "bool"},
            {"FieldName": FIELD_HAS_COMPRESSION, "FieldType": "bool"},
            {"FieldName": FIELD_ENABLE_MERGE, "FieldType": "bool"},
            {"FieldName": FIELD_CALL_COUNT, "FieldType": "float32"},
            {"FieldName": FIELD_MERGE_COUNT, "FieldType": "float32"},
            {"FieldName": FIELD_DURATION_COUNT, "FieldType": "float32"},
            {"FieldName": FIELD_FILE_PATH, "FieldType": "string"},
            {"FieldName": FIELD_RAW_TYPE, "FieldType": "string"},
            {"FieldName": FIELD_RAW_ID, "FieldType": "string"},
            {"FieldName": FIELD_HIERARCHY_LEVEL, "FieldType": "int64"},
            {"FieldName": FIELD_REFS, "FieldType": "string"},
            {"FieldName": FIELD_DOCUMENT, "FieldType": "string"},
            {"FieldName": FIELD_CONTENT_MODALITIES, "FieldType": "string"},
            {"FieldName": FIELD_MEDIA_REFS, "FieldType": "string"},
        ]

        data = {
            "ProjectName": "default",
            "CollectionName": self._collection_name,
            "Fields": fields,
            "Description": "OpenContext unified collection for all context types",
        }

        logger.info(f"Creating VikingDB collection: {self._collection_name}")
        result = await self._client.async_console_request(  # type: ignore[union-attr]
            action="CreateVikingdbCollection",
            data=data,
        )

        if result.get("ResponseMetadata", {}).get("Error"):
            error = result["ResponseMetadata"]["Error"]
            raise RuntimeError(f"Failed to create collection: {error}")

        logger.info(f"VikingDB collection created: {self._collection_name}")

    async def _create_index(
        self,
        index_type: str = "hnsw",
        distance_type: str = "cosine",
    ) -> None:
        """Create index for the collection."""
        vector_index = {
            "IndexType": index_type.upper(),
            "Distance": distance_type,
            "Quant": "int8",
        }

        if index_type.lower() == "hnsw":
            vector_index["HnswM"] = 32  # type: ignore[assignment]
            vector_index["HnswCef"] = 64  # type: ignore[assignment]
            vector_index["HnswSef"] = 800  # type: ignore[assignment]

        data = {
            "CollectionName": self._collection_name,
            "IndexName": self._index_name,
            "VectorIndex": vector_index,
            "ScalarIndex": [
                FIELD_DATA_TYPE,
                FIELD_CONTEXT_TYPE,
                FIELD_USER_ID,
                FIELD_DEVICE_ID,
                FIELD_AGENT_ID,
                FIELD_RAW_TYPE,
                FIELD_RAW_ID,
                FIELD_IS_PROCESSED,
                FIELD_HAS_COMPRESSION,
                FIELD_ENABLE_MERGE,
                FIELD_CREATE_TIME_TS,
                FIELD_EVENT_TIME_START_TS,
                FIELD_EVENT_TIME_END_TS,
                FIELD_UPDATE_TIME_TS,
                FIELD_LAST_CALL_TIME_TS,
                FIELD_CONFIDENCE,
                FIELD_IMPORTANCE,
                FIELD_CALL_COUNT,
                FIELD_MERGE_COUNT,
                FIELD_DURATION_COUNT,
                FIELD_HIERARCHY_LEVEL,
                FIELD_CONTENT_MODALITIES,
            ],
            "Description": f"Index for {self._collection_name}",
        }

        logger.info(f"Creating VikingDB index: {self._index_name}")
        result = await self._client.async_console_request(  # type: ignore[union-attr]
            action="CreateVikingdbIndex",
            data=data,
        )

        if result.get("ResponseMetadata", {}).get("Error"):
            error = result["ResponseMetadata"]["Error"]
            raise RuntimeError(f"Failed to create index: {error}")

        logger.info(f"VikingDB index created: {self._index_name}")

    def get_name(self) -> str:
        """Get storage backend name."""
        return "vikingdb"

    def get_storage_type(self) -> StorageType:
        """Get storage type."""
        return StorageType.VECTOR_DB

    async def get_collection_names(self) -> list[str]:
        """Get all collection names managed by this backend."""
        if self._collection_ready:
            return [ct for ct in ContextType]
        return []

    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized

    async def _ensure_vectorized(self, context: ProcessedContext) -> list[float]:
        """Ensure context has a vector, generate if needed."""
        if context.vectorize and context.vectorize.vector:
            return list(context.vectorize.vector)

        if context.vectorize:
            await do_vectorize(context.vectorize)
            if context.vectorize.vector:
                return list(context.vectorize.vector)

        raise ValueError(f"Unable to get or generate vector for context {context.id}")

    def _context_to_doc_format(self, context: ProcessedContext) -> dict[str, Any]:
        """Convert ProcessedContext to VikingDB data format."""
        doc = context.model_dump(
            exclude_none=True, exclude={"properties", "extracted_data", "vectorize", "metadata"}
        )

        if context.extracted_data:
            extracted_data_dict = context.extracted_data.model_dump(exclude_none=True)
            doc.update(extracted_data_dict)

        if context.metadata:
            doc.update(context.metadata)

        if context.vectorize:
            text = context.vectorize.get_text()
            if text:
                doc[FIELD_DOCUMENT] = text
            # Store modality string for multimodal filtering
            doc[FIELD_CONTENT_MODALITIES] = context.vectorize.get_modality_string()

        # Store media_refs from metadata (always set to avoid VikingDB "default" placeholder)
        doc[FIELD_MEDIA_REFS] = (
            json.dumps(context.metadata["media_refs"], ensure_ascii=False)
            if context.metadata and "media_refs" in context.metadata
            else "[]"
        )

        if context.properties:
            properties_dict = context.properties.model_dump(exclude_none=True)
            properties_dict.pop("raw_properties", None)
            doc.update(properties_dict)

        # Explicit refs serialization — always present for backward compatibility
        doc[FIELD_REFS] = (
            json.dumps(context.properties.refs)
            if context.properties and context.properties.refs
            else "{}"
        )

        def default_json_serializer(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            return str(obj)

        fields = {}
        for key, value in list(doc.items()):
            if key in ["id", "embedding", "document"]:
                if key == "document":
                    fields[FIELD_DOCUMENT] = value
                continue

            if value is None:
                continue

            if isinstance(value, datetime.datetime):
                fields[f"{key}_ts"] = value.timestamp()
                fields[key] = value.isoformat()
            elif isinstance(value, Enum):
                fields[key] = value.value
            elif isinstance(value, (dict, list)):
                try:
                    fields[key] = json.dumps(
                        value, ensure_ascii=False, default=default_json_serializer
                    )
                except (TypeError, ValueError):
                    fields[key] = str(value)
            elif isinstance(value, bool):
                fields[key] = value
            elif isinstance(value, (int, float)):
                fields[key] = int(value) if key == FIELD_HIERARCHY_LEVEL else float(value)
            elif isinstance(value, str):
                fields[key] = value
            else:
                fields[key] = str(value)

        fields[FIELD_DATA_TYPE] = DATA_TYPE_CONTEXT

        return fields

    async def upsert_processed_context(self, context: ProcessedContext) -> str:
        results = await self.batch_upsert_processed_context([context])
        return results[0] if results else ""

    async def batch_upsert_processed_context(self, contexts: list[ProcessedContext]) -> list[str]:
        if not self._initialized:
            raise RuntimeError("VikingDB backend not initialized")

        # Batch pre-vectorize all contexts (fewer API calls)
        vectorizes = [c.vectorize for c in contexts if c.vectorize and not c.vectorize.vector]
        if vectorizes:
            await do_vectorize_batch(vectorizes)

        data_list = []
        logger.debug(f"Upserting contexts:{contexts}")
        for context in contexts:
            try:
                vector = await self._ensure_vectorized(context)
                fields = self._context_to_doc_format(context)

                data_item = {
                    "id": context.id,
                    "vector": vector,
                }
                data_item.update(fields)
                data_list.append(data_item)

            except Exception as e:
                logger.exception(f"Failed to process context {context.id}: {e}")
                continue

        if not data_list:
            return []

        stored_ids = []

        try:
            result = await self._client.async_data_request(  # type: ignore[union-attr]
                path="/api/vikingdb/data/upsert",
                data={
                    "collection_name": self._collection_name,
                    "data": data_list,
                },
            )

            if result.get("code") == "Success":
                for item in data_list:
                    stored_ids.append(item["id"])
                logger.debug(
                    f"[VikingDB] Upserted {len(data_list)} docs to {self._collection_name}"
                )
            else:
                logger.error(
                    f"[VikingDB] Failed to upsert to "
                    f"{self._collection_name}: {result.get('message')}"
                )

        except Exception as e:
            logger.exception(f"Failed to upsert contexts: {e}")

        return stored_ids  # type: ignore[return-value]

    async def get_processed_context(  # type: ignore[override]
        self, id: str, context_type: str, need_vector: bool = False
    ) -> ProcessedContext | None:
        if not self._initialized:
            return None

        try:
            result = await self._client.async_data_request(  # type: ignore[union-attr]
                path="/api/vikingdb/data/fetch_in_collection",
                data={
                    "collection_name": self._collection_name,
                    "ids": [id],
                },
            )

            if result.get("code") == "Success":
                fetch_result = result.get("result", {}).get("fetch", [])
                if fetch_result and len(fetch_result) > 0:
                    item = fetch_result[0]
                    doc = {"id": item.get("id")}
                    doc.update(item.get("fields", {}))
                    if doc.get(FIELD_CONTEXT_TYPE) == context_type:
                        return self._doc_to_context(doc, need_vector)

        except Exception as e:
            logger.exception(f"Failed to get context {id}: {e}")

        return None

    async def get_all_processed_contexts(
        self,
        context_types: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
        filter: dict[str, Any] | None = None,
        need_vector: bool = False,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
        skip_slice: bool = False,
    ) -> dict[str, list[ProcessedContext]]:
        if not self._initialized:
            return {}

        # Diagnostic: log caller when limit < 1
        if limit < 1:
            import traceback

            logger.warning(
                f"get_all_processed_contexts called with limit={limit}, "
                f"context_types={context_types}\n"
                f"Caller traceback:\n{''.join(traceback.format_stack())}"
            )

        result = {}

        target_types = context_types or [ct.value for ct in ContextType]

        for ctx_type in target_types:
            try:
                filter_dict = self._build_filter_dict(
                    filters=filter,
                    user_id=user_id,
                    device_id=device_id,
                    agent_id=agent_id,
                    context_type=ctx_type,
                    data_type=DATA_TYPE_CONTEXT,
                )

                if skip_slice:
                    fetch_limit = limit + offset
                    fetch_offset = 0
                else:
                    fetch_limit = limit
                    fetch_offset = offset

                data = {
                    "collection_name": self._collection_name,
                    "index_name": self._index_name,
                    "limit": fetch_limit,
                    "offset": fetch_offset,
                    "field": FIELD_CREATE_TIME_TS,
                    "order": "desc",
                }
                if filter_dict:
                    data["filter"] = filter_dict

                query_result = await self._client.async_data_request(  # type: ignore[union-attr]
                    path="/api/vikingdb/data/search/scalar", data=data
                )

                if query_result.get("code") == "Success":
                    output = query_result.get("result", {}).get("data", [])
                    if not skip_slice and len(output) > limit:
                        output = output[:limit]

                    contexts = []
                    for item in output:
                        doc = {"id": item.get("id")}
                        doc.update(item.get("fields", {}))
                        context = self._doc_to_context(doc, need_vector)
                        if context:
                            contexts.append(context)

                    if contexts:
                        result[ctx_type] = contexts

            except Exception as e:
                logger.exception(f"Failed to get contexts for type {ctx_type}: {e}")
                continue

        return result

    async def delete_processed_context(self, id: str, context_type: str) -> bool:
        return await self.delete_contexts([id], context_type)

    async def delete_contexts(self, ids: list[str], context_type: str) -> bool:
        if not self._initialized:
            return False

        try:
            result = await self._client.async_data_request(  # type: ignore[union-attr]
                path="/api/vikingdb/data/delete",
                data={
                    "collection_name": self._collection_name,
                    "ids": ids,
                },
            )

            if result.get("code") == "Success":
                logger.debug(f"Deleted {len(ids)} contexts from {self._collection_name}")
                return True
            else:
                logger.error(f"Failed to delete contexts: {result.get('message')}")
                return False

        except Exception as e:
            logger.exception(f"Failed to delete contexts: {e}")
            return False

    async def search(  # type: ignore[override]
        self,
        query: Vectorize,
        top_k: int = 10,
        context_types: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        need_vector: bool = False,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[ProcessedContext, float]]:
        if not self._initialized:
            return []

        query_vector = None
        if query.vector and len(query.vector) > 0:
            query_vector = list(query.vector)
        else:
            if query.text:  # type: ignore[attr-defined]
                await do_vectorize(query, role="query")
                query_vector = list(query.vector) if query.vector else None

        if not query_vector:
            all_contexts = await self.get_all_processed_contexts(
                context_types=context_types,
                limit=top_k,
                filter=filters,
                need_vector=need_vector,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            )
            all_results = []
            for type_contexts in all_contexts.values():
                for context in type_contexts:
                    all_results.append((context, 1.0))
            return all_results[:top_k] if len(all_results) > top_k else all_results

        # Split context_types into user and base groups for separate filtering
        # (base types skip user_id/device_id, but VikingDB does single-collection search)
        user_types = [ct for ct in (context_types or []) if not ct.startswith("agent_base")]
        base_types = [ct for ct in (context_types or []) if ct.startswith("agent_base")]

        all_results = []

        # Search user types (with user_id/device_id)
        if user_types:
            filter_dict = self._build_filter_dict(
                filters=filters,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
                context_types=user_types,
                data_type=DATA_TYPE_CONTEXT,
            )
            results = await self._vector_search(query_vector, filter_dict, top_k, need_vector)
            all_results.extend(results)

        # Search base types (without user_id/device_id)
        if base_types:
            filter_dict = self._build_filter_dict(
                filters=filters,
                user_id=None,
                device_id=None,
                agent_id=agent_id,
                context_types=base_types,
                data_type=DATA_TYPE_CONTEXT,
            )
            results = await self._vector_search(query_vector, filter_dict, top_k, need_vector)
            all_results.extend(results)

        if score_threshold is not None:
            all_results = [(ctx, s) for ctx, s in all_results if s >= score_threshold]

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    async def _vector_search(
        self,
        query_vector: list,
        filter_dict: dict | None,
        top_k: int,
        need_vector: bool,
    ) -> list[tuple[ProcessedContext, float]]:
        """Execute a single vector search request against VikingDB."""
        results = []
        try:
            data = {
                "collection_name": self._collection_name,
                "index_name": self._index_name,
                "dense_vector": query_vector,
                "limit": top_k,
            }
            if filter_dict:
                data["filter"] = filter_dict

            result = await self._client.async_data_request(  # type: ignore[union-attr]
                path="/api/vikingdb/data/search/vector", data=data
            )

            if result.get("code") == "Success":
                output = result.get("result", {}).get("data", [])
                for item in output:
                    doc = {"id": item.get("id")}
                    doc.update(item.get("fields", {}))
                    context = self._doc_to_context(doc, need_vector)
                    if context:
                        score = item.get("score", 0.0)
                        results.append((context, score))
        except Exception as e:
            logger.exception(f"Vector search failed: {e}")
        return results

    def _doc_to_context(
        self, doc: dict[str, Any], need_vector: bool = False
    ) -> ProcessedContext | None:
        """Convert VikingDB data to ProcessedContext."""
        try:
            if not doc:
                return None

            fields = dict(doc)

            extracted_data_field_names = set(ExtractedData.model_fields.keys())
            properties_field_names = set(ContextProperties.model_fields.keys())
            vectorize_field_names = set(Vectorize.model_fields.keys())

            extracted_data_dict = {}
            properties_dict = {}
            vectorize_dict = {}
            metadata_dict = {}

            doc_text = fields.pop(FIELD_DOCUMENT, None)
            if doc_text:
                vectorize_dict["input"] = [{"type": "text", "text": doc_text}]

            if need_vector and doc.get("vector"):
                vectorize_dict["vector"] = doc["vector"]

            # Parse multimodal fields
            content_modalities = fields.pop(FIELD_CONTENT_MODALITIES, None)
            if content_modalities:
                metadata_dict[FIELD_CONTENT_MODALITIES] = content_modalities

            media_refs = normalize_media_refs(fields.pop(FIELD_MEDIA_REFS, None))
            if media_refs:
                metadata_dict[FIELD_MEDIA_REFS] = media_refs

            # Pop legacy fields to prevent them leaking into metadata
            fields.pop("images", None)
            fields.pop("videos", None)

            context_id = fields.pop("id", None)
            if not context_id:
                return None

            fields.pop("vector", None)
            fields.pop("score", None)
            fields.pop(FIELD_DATA_TYPE, None)

            TIME_FIELDS = {
                "create_time",
                "event_time_start",
                "event_time_end",
                "update_time",
                "last_call_time",
            }

            # Fields that should always remain as strings, never JSON-parsed
            STRING_ONLY_FIELDS = {"title", "summary", "document", "text"}

            for key, value in fields.items():
                if key.endswith("_ts"):
                    continue

                val = value
                if (
                    isinstance(value, str)
                    and value.startswith(("{", "["))
                    and key not in STRING_ONLY_FIELDS
                ):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        val = json.loads(value)

                if key in TIME_FIELDS and isinstance(val, str):
                    if val in ("default", "", "null", "None"):
                        continue
                    try:
                        val = datetime.datetime.fromisoformat(val.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid datetime value for field '{key}': {val}")
                        continue

                if key in extracted_data_field_names:
                    extracted_data_dict[key] = val
                elif key in properties_field_names:
                    properties_dict[key] = val
                elif key in vectorize_field_names:
                    vectorize_dict[key] = val
                else:
                    metadata_dict[key] = val

            context_dict = {
                "id": context_id,
                "extracted_data": ExtractedData.model_validate(extracted_data_dict),
                "properties": ContextProperties.model_validate(properties_dict),
                "vectorize": Vectorize.model_validate(vectorize_dict) if vectorize_dict else None,
                "metadata": metadata_dict if metadata_dict else None,
            }

            return ProcessedContext.model_validate(context_dict)

        except Exception as e:
            logger.exception(f"Failed to convert doc to ProcessedContext: {e}")
            return None

    def _parse_time_to_timestamp(self, time_value: Any) -> float | None:
        """Parse various time formats to Unix timestamp."""
        if time_value is None:
            return None

        try:
            if isinstance(time_value, (int, float)):
                if 946684800 < time_value < 4102444800:
                    return float(time_value)
                elif 946684800000 < time_value < 4102444800000:
                    return float(time_value) / 1000.0
                return float(time_value)

            if isinstance(time_value, str):
                time_str = time_value.replace("Z", "+00:00")
                if "." in time_str and "+" in time_str:
                    parts = time_str.split("+")
                    time_str = parts[0].split(".")[0] + "+" + parts[1]
                elif "." in time_str:
                    time_str = time_str.split(".")[0]

                dt = datetime.datetime.fromisoformat(time_str)
                return dt.timestamp()

            if isinstance(time_value, datetime.datetime):
                return time_value.timestamp()

        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to parse time value '{time_value}': {e}")

        return None

    def _build_filter_dict(
        self,
        filters: dict[str, Any] | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
        context_type: str | None = None,
        context_types: list[str] | None = None,
        data_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Build VikingDB filter dictionary."""
        conditions = []

        TIME_FIELD_MAPPING = {
            "create_time": FIELD_CREATE_TIME_TS,
        }

        RANGE_SUPPORTED_FIELDS = {
            FIELD_CREATE_TIME_TS,
            FIELD_EVENT_TIME_START_TS,
            FIELD_EVENT_TIME_END_TS,
            FIELD_UPDATE_TIME_TS,
            FIELD_LAST_CALL_TIME_TS,
            FIELD_CONFIDENCE,
            FIELD_IMPORTANCE,
            FIELD_CALL_COUNT,
            FIELD_MERGE_COUNT,
            FIELD_DURATION_COUNT,
        }

        if data_type:
            conditions.append({"op": "must", "field": FIELD_DATA_TYPE, "conds": [data_type]})

        if context_type:
            conditions.append({"op": "must", "field": FIELD_CONTEXT_TYPE, "conds": [context_type]})

        if context_types and len(context_types) > 0:
            conditions.append({"op": "must", "field": FIELD_CONTEXT_TYPE, "conds": context_types})

        # Skip user_id and device_id filters for agent_base_* types
        # (base memories are agent-level, not scoped to a specific user or device)
        # NOTE: This only works when context_type is a single value or context_types
        # has exactly one element. For mixed lists (e.g. ["event", "agent_base_event"]),
        # is_base falls through to False and user_id/device_id are applied to all types.
        # Callers with mixed types must split into separate calls (see search() above).
        _ctx = context_type or (
            context_types[0] if context_types and len(context_types) == 1 else None
        )
        is_base = _ctx and _ctx.startswith("agent_base")
        if user_id and not is_base:
            conditions.append({"op": "must", "field": FIELD_USER_ID, "conds": [user_id]})
        if device_id and not is_base:
            conditions.append({"op": "must", "field": FIELD_DEVICE_ID, "conds": [device_id]})
        if agent_id:
            conditions.append({"op": "must", "field": FIELD_AGENT_ID, "conds": [agent_id]})

        if filters:
            for key, value in filters.items():
                if key in ("context_type", "context_types", "data_type"):
                    continue
                # TODO: entities filter is skipped because entities are stored as a
                # JSON-serialized string (e.g. '["Alice","Bob"]') in a string field.
                # VikingDB's scalar filter cannot query individual elements inside a
                # JSON string. To enable entity filtering, change storage format to a
                # native list type and implement element-level matching.
                if key == "entities":
                    continue
                if value is None:
                    continue

                is_time_field = key in TIME_FIELD_MAPPING or key.endswith("_ts")
                filter_key = TIME_FIELD_MAPPING.get(key, key)

                if isinstance(value, dict):
                    supports_range = filter_key in RANGE_SUPPORTED_FIELDS

                    if supports_range:
                        range_filter = {
                            "op": "range",
                            "field": filter_key,
                        }
                        op_mapping = {
                            "$gte": "gte",
                            "$lte": "lte",
                            "$gt": "gt",
                            "$lt": "lt",
                        }
                        has_range = False
                        for op, range_key in op_mapping.items():
                            if op in value:
                                op_value = value[op]
                                if is_time_field:
                                    ts = self._parse_time_to_timestamp(op_value)
                                    if ts is not None:
                                        range_filter[range_key] = ts  # type: ignore[assignment]
                                        has_range = True
                                else:
                                    range_filter[range_key] = op_value
                                    has_range = True
                        if has_range:
                            conditions.append(range_filter)  # type: ignore[arg-type]
                    else:
                        range_ops = {"$gte", "$lte", "$gt", "$lt"}
                        if any(op in value for op in range_ops):
                            logger.warning(
                                f"Field '{filter_key}' does not support range operator. "
                                "Only fields in ScalarIndex with int64/float32 type support range. "
                                f"Skipping range filter for this field."
                            )

                    if "$eq" in value:
                        conditions.append(
                            {"op": "must", "field": filter_key, "conds": [value["$eq"]]}
                        )
                    if "$ne" in value:
                        conditions.append(
                            {"op": "must_not", "field": filter_key, "conds": [value["$ne"]]}
                        )
                elif isinstance(value, list):
                    conditions.append({"op": "must", "field": filter_key, "conds": value})
                elif isinstance(value, str):
                    if is_time_field and filter_key in RANGE_SUPPORTED_FIELDS:
                        ts = self._parse_time_to_timestamp(value)
                        if ts is not None:
                            conditions.append(
                                {
                                    "op": "range",
                                    "field": filter_key,
                                    "gte": ts - 0.5,  # type: ignore[dict-item]
                                    "lte": ts + 0.5,  # type: ignore[dict-item]
                                }
                            )
                        else:
                            conditions.append({"op": "must", "field": key, "conds": [value]})
                    else:
                        conditions.append({"op": "must", "field": filter_key, "conds": [value]})
                elif isinstance(value, bool):
                    conditions.append({"op": "must", "field": filter_key, "conds": [value]})  # type: ignore[list-item]
                elif isinstance(value, (int, float)) and filter_key in RANGE_SUPPORTED_FIELDS:
                    conditions.append(
                        {
                            "op": "range",
                            "field": filter_key,
                            "gte": float(value),  # type: ignore[dict-item]
                            "lte": float(value),  # type: ignore[dict-item]
                        }
                    )
                else:
                    conditions.append({"op": "must", "field": filter_key, "conds": [value]})

        if not conditions:
            return None

        filter_dict = conditions[0] if len(conditions) == 1 else {"op": "and", "conds": conditions}
        logger.debug(f"Built VikingDB filter: {filter_dict}")
        return filter_dict

    async def get_processed_context_count(
        self,
        context_type: str,
        filter: dict[str, Any] | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
    ) -> int:
        if not self._initialized:
            return 0

        try:
            filter_dict = self._build_filter_dict(
                filters=filter,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
                context_type=context_type,
                data_type=DATA_TYPE_CONTEXT,
            )

            data = {
                "collection_name": self._collection_name,
                "index_name": self._index_name,
                "limit": 100000,
                "output_fields": [],
                "field": FIELD_CREATE_TIME_TS,
                "order": "desc",
            }
            if filter_dict:
                data["filter"] = filter_dict

            result = await self._client.async_data_request(  # type: ignore[union-attr]
                path="/api/vikingdb/data/search/scalar", data=data
            )

            if result.get("code") == "Success":
                result_data = result.get("result", {})
                if "filter_matched_count" in result_data:
                    return result_data["filter_matched_count"]
                return result_data.get("total_return_count", 0)
            return 0
        except Exception as e:
            logger.error(f"Failed to get count for {context_type}: {e}")
            return 0

    async def get_all_processed_context_counts(self) -> dict[str, int]:
        counts = {}
        for ct in ContextType:
            counts[ct.value] = await self.get_processed_context_count(ct.value)
        return counts

    async def search_by_hierarchy(
        self,
        context_type: str,
        hierarchy_level: int,
        time_start: float | None = None,
        time_end: float | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
        top_k: int = 20,
    ) -> list[tuple[ProcessedContext, float]]:
        if not self._initialized:
            return []
        try:
            conditions = [
                {"op": "must", "field": FIELD_DATA_TYPE, "conds": [DATA_TYPE_CONTEXT]},
                {"op": "must", "field": FIELD_CONTEXT_TYPE, "conds": [context_type]},
                {"op": "must", "field": FIELD_HIERARCHY_LEVEL, "conds": [int(hierarchy_level)]},
            ]
            # Skip user_id and device_id for agent_base_* types
            is_base = context_type.startswith("agent_base")
            if user_id and not is_base:
                conditions.append({"op": "must", "field": FIELD_USER_ID, "conds": [user_id]})
            if device_id and not is_base:
                conditions.append({"op": "must", "field": FIELD_DEVICE_ID, "conds": [device_id]})
            if agent_id:
                conditions.append({"op": "must", "field": FIELD_AGENT_ID, "conds": [agent_id]})
            # Numeric range overlap
            if time_end is not None:
                conditions.append(
                    {
                        "op": "range",
                        "field": FIELD_EVENT_TIME_START_TS,
                        "lte": float(time_end),
                    }
                )
            if time_start is not None:
                conditions.append(
                    {
                        "op": "range",
                        "field": FIELD_EVENT_TIME_END_TS,
                        "gte": float(time_start),
                    }
                )
            filter_dict = (
                {"op": "and", "conds": conditions} if len(conditions) > 1 else conditions[0]
            )
            data = {
                "collection_name": self._collection_name,
                "index_name": self._index_name,
                "limit": top_k,
                "field": FIELD_CREATE_TIME_TS,
                "order": "desc",
                "filter": filter_dict,
            }
            result = await self._client.async_data_request(  # type: ignore[union-attr]
                path="/api/vikingdb/data/search/scalar", data=data
            )
            if result.get("code") != "Success":
                logger.error(f"Failed to search by hierarchy: {result.get('message')}")
                return []
            output = result.get("result", {}).get("data", [])
            results = []
            for item in output:
                doc = {"id": item.get("id")}
                doc.update(item.get("fields", {}))
                context = self._doc_to_context(doc, need_vector=False)
                if context:
                    results.append((context, 1.0))
            return results[:top_k]
        except Exception as e:
            logger.exception(f"Failed to search by hierarchy: {e}")
            return []

    async def get_hierarchy_map(
        self,
        owner_type: str,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
        l1_days: int = 7,
    ) -> dict[int, list]:
        """VikingDB optimized: single scalar search for all hierarchy levels."""
        from opencontext.models.enums import MEMORY_OWNER_TYPES
        from opencontext.utils.time_utils import now as tz_now

        types = MEMORY_OWNER_TYPES.get(owner_type)
        if not types or len(types) < 4:
            return {3: [], 2: [], 1: []}

        if not self._initialized:
            return {3: [], 2: [], 1: []}

        try:
            context_types = [types[1].value, types[2].value, types[3].value]

            # Build conditions — all 3 summary types in a single query
            conditions = [
                {"op": "must", "field": FIELD_DATA_TYPE, "conds": [DATA_TYPE_CONTEXT]},
                {"op": "must", "field": FIELD_CONTEXT_TYPE, "conds": context_types},
                {
                    "op": "must",
                    "field": FIELD_HIERARCHY_LEVEL,
                    "conds": [1, 2, 3],
                },
            ]

            # All types within one owner_type are consistently user or agent_base
            is_base = context_types[0].startswith("agent_base")
            if user_id and not is_base:
                conditions.append({"op": "must", "field": FIELD_USER_ID, "conds": [user_id]})
            if device_id and not is_base:
                conditions.append({"op": "must", "field": FIELD_DEVICE_ID, "conds": [device_id]})
            if agent_id:
                conditions.append({"op": "must", "field": FIELD_AGENT_ID, "conds": [agent_id]})

            filter_dict = (
                {"op": "and", "conds": conditions} if len(conditions) > 1 else conditions[0]
            )

            # L1 needs the tightest time window; use it as the query-level floor
            # so the DB doesn't return ancient L1 records we'd discard anyway.
            now = tz_now()
            from datetime import timedelta

            l1_cutoff = now - timedelta(days=l1_days)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            # Single scalar search call — generous limit covers all 3 levels
            data = {
                "collection_name": self._collection_name,
                "index_name": self._index_name,
                "limit": max(200, 100 + 52 + l1_days),
                "field": FIELD_CREATE_TIME_TS,
                "order": "desc",
                "filter": filter_dict,
            }
            result = await self._client.async_data_request(  # type: ignore[union-attr]
                path="/api/vikingdb/data/search/scalar", data=data
            )
            if result.get("code") != "Success":
                logger.error(f"Failed to get hierarchy map: {result.get('message')}")
                return {3: [], 2: [], 1: []}

            output = result.get("result", {}).get("data", [])

            # Parse results and group by hierarchy_level, filtering by time in memory
            grouped: dict[int, list] = {3: [], 2: [], 1: []}
            for item in output:
                doc = {"id": item.get("id")}
                doc.update(item.get("fields", {}))
                ctx = self._doc_to_context(doc, need_vector=False)
                if not ctx:
                    continue
                level = ctx.properties.hierarchy_level
                ts = ctx.properties.event_time_start
                if level == 3:
                    grouped[3].append(ctx)
                elif level == 2 and ts and ts >= month_start:
                    grouped[2].append(ctx)
                elif level == 1 and ts and ts >= l1_cutoff:
                    grouped[1].append(ctx)
            return grouped

        except Exception as e:
            logger.exception(f"Failed to get hierarchy map: {e}")
            return {3: [], 2: [], 1: []}

    async def get_by_ids(
        self,
        ids: list[str],
        context_type: str | None = None,
        need_vector: bool = False,
    ) -> list[ProcessedContext]:
        if not self._initialized:
            return []

        if not ids:
            return []

        try:
            result = await self._client.async_data_request(  # type: ignore[union-attr]
                path="/api/vikingdb/data/fetch_in_collection",
                data={
                    "collection_name": self._collection_name,
                    "ids": ids,
                },
            )

            if result.get("code") != "Success":
                logger.error(f"Failed to fetch by IDs: {result.get('message')}")
                return []

            fetch_result = result.get("result", {}).get("fetch", [])
            contexts = []
            for item in fetch_result:
                doc = {"id": item.get("id")}
                doc.update(item.get("fields", {}))

                if context_type and doc.get(FIELD_CONTEXT_TYPE) != context_type:
                    continue

                context = self._doc_to_context(doc, need_vector=need_vector)
                if context:
                    contexts.append(context)

            return contexts

        except Exception as e:
            logger.exception(f"Failed to get contexts by IDs: {e}")
            return []

    async def batch_update_refs(
        self,
        context_ids: list[str],
        ref_key: str,
        ref_value: str,
        context_type: str,
    ) -> int:
        """Add a ref entry to multiple contexts."""
        if not self._initialized or not context_ids:
            return 0

        # Fetch current refs for all context_ids
        try:
            fetch_result = await self._client.async_data_request(  # type: ignore[union-attr]
                path="/api/vikingdb/data/fetch_in_collection",
                data={
                    "collection_name": self._collection_name,
                    "ids": context_ids,
                },
            )
        except Exception as e:
            logger.warning(f"batch_update_refs fetch failed: {e}")
            return 0

        if fetch_result.get("code") != "Success":
            logger.warning(f"batch_update_refs fetch failed: {fetch_result.get('message')}")
            return 0

        fetched_items = fetch_result.get("result", {}).get("fetch", [])
        if not fetched_items:
            return 0

        # Build update data list with merged refs
        data_list = []
        for item in fetched_items:
            try:
                existing_refs_str = item.get("fields", {}).get(FIELD_REFS, "{}")
                existing_refs = json.loads(existing_refs_str) if existing_refs_str else {}
                if ref_key not in existing_refs:
                    existing_refs[ref_key] = []
                if ref_value not in existing_refs[ref_key]:
                    existing_refs[ref_key].append(ref_value)
                data_list.append(
                    {
                        "id": item.get("id"),
                        FIELD_REFS: json.dumps(existing_refs),
                    }
                )
            except Exception as e:
                logger.warning(f"batch_update_refs parse failed for {item.get('id')}: {e}")

        if not data_list:
            return 0

        updated = 0
        # VikingDB update API limit: 100 items per request
        for i in range(0, len(data_list), 100):
            batch = data_list[i : i + 100]
            try:
                result = await self._client.async_data_request(  # type: ignore[union-attr]
                    path="/api/vikingdb/data/update",
                    data={
                        "collection_name": self._collection_name,
                        "data": batch,
                    },
                )
                if result.get("code") == "Success":
                    updated += len(batch)
                else:
                    logger.warning(f"batch_update_refs update failed: {result.get('message')}")
            except Exception as e:
                logger.warning(f"batch_update_refs failed for batch: {e}")
        return updated

    async def close(self):
        """Close the backend and release resources."""
        if self._client:
            await self._client.close()
        self._initialized = False
        logger.info("VikingDB backend closed")
