# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
DashVector vector storage backend - Aliyun DashVector Service (HTTP API)
https://help.aliyun.com/product/2510217.html

This implementation uses HTTP API instead of SDK for better concurrency support.
"""

import asyncio
import datetime
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    Vectorize,
)
from opencontext.models.enums import ContentFormat, ContextType
from opencontext.storage.base_storage import IVectorStorageBackend, StorageType
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Collection and field constants
TODO_COLLECTION = "todo"
FIELD_DOCUMENT = "document"
FIELD_ORIGINAL_ID = "original_id"
FIELD_TODO_ID = "todo_id"
FIELD_CONTENT = "content"

# Time fields
FIELD_CREATED_AT = "created_at"
FIELD_CREATED_AT_TS = "created_at_ts"
FIELD_CREATE_TIME = "create_time"
FIELD_CREATE_TIME_TS = "create_time_ts"
FIELD_EVENT_TIME = "event_time"
FIELD_EVENT_TIME_TS = "event_time_ts"
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
FIELD_SOURCE = "source"

# Properties fields
FIELD_IS_PROCESSED = "is_processed"
FIELD_HAS_COMPRESSION = "has_compression"
FIELD_ENABLE_MERGE = "enable_merge"
FIELD_IS_HAPPEND = "is_happend"
FIELD_CALL_COUNT = "call_count"
FIELD_MERGE_COUNT = "merge_count"
FIELD_DURATION_COUNT = "duration_count"

# Document tracking fields
FIELD_FILE_PATH = "file_path"
FIELD_RAW_TYPE = "raw_type"
FIELD_RAW_ID = "raw_id"

# HTTP API constants
API_VERSION = "v1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_MAX_CONNECTIONS_PER_HOST = 20


class DashVectorHTTPClient:
    """
    Async HTTP client for DashVector API with connection pooling and retry support.
    Designed for high concurrency scenarios.
    """
    
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        timeout: float = DEFAULT_TIMEOUT,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_connections_per_host: int = DEFAULT_MAX_CONNECTIONS_PER_HOST,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._api_key = api_key
        self._endpoint = endpoint.rstrip('/')
        self._timeout = timeout
        self._max_connections = max_connections
        self._max_connections_per_host = max_connections_per_host
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        
        # Async session (lazy initialization)
        self._async_session: Optional[aiohttp.ClientSession] = None
        self._async_lock = asyncio.Lock() if AIOHTTP_AVAILABLE else None
        
        # Sync session with connection pooling
        self._sync_session = self._create_sync_session()
        
        # Thread pool for running async code in sync context
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # Event loop for async operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_lock = threading.Lock()
    
    def _create_sync_session(self) -> requests.Session:
        """Create a sync session with connection pooling and retry."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self._max_retries,
            backoff_factor=self._retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self._max_connections_per_host,
            pool_maxsize=self._max_connections,
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async session with connection pooling."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")
        
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
    
    def _get_headers(self) -> Dict[str, str]:
        """Get common headers for API requests."""
        return {
            "dashvector-auth-token": self._api_key,
            "Content-Type": "application/json",
        }
    
    def _build_url(self, path: str) -> str:
        """Build full URL for API endpoint."""
        return f"https://{self._endpoint}/{API_VERSION}/{path.lstrip('/')}"
    
    async def _async_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make async HTTP request with retry.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            path: API path
            data: Request body (for POST/PUT)
            params: Query parameters (for GET)
            
        Returns:
            Response JSON as dict
        """
        session = await self._get_async_session()
        url = self._build_url(path)
        headers = self._get_headers()
        
        last_error = None
        for attempt in range(self._max_retries):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                ) as response:
                    result = await response.json()
                    
                    # Check for rate limiting
                    if response.status == 429:
                        wait_time = self._retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    return result
                    
            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait_time = self._retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed: {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    
        raise last_error or RuntimeError("Request failed after retries")
    
    def _sync_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make sync HTTP request with retry (handled by session adapter).
        
        Args:
            method: HTTP method
            path: API path
            data: Request body
            params: Query parameters
            
        Returns:
            Response JSON as dict
        """
        url = self._build_url(path)
        headers = self._get_headers()
        
        response = self._sync_session.request(
            method=method,
            url=url,
            headers=headers,
            json=data,
            params=params,
            timeout=self._timeout,
        )
        
        return response.json()
    
    def request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        use_async: bool = False,
    ) -> Dict[str, Any]:
        """
        Make HTTP request (sync or async based on context).
        
        Args:
            method: HTTP method
            path: API path
            data: Request body
            params: Query parameters
            use_async: Force async mode
            
        Returns:
            Response JSON as dict
        """
        if use_async and AIOHTTP_AVAILABLE:
            # Try to use async if in async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, use async request
                return asyncio.ensure_future(
                    self._async_request(method, path, data, params)
                )
            except RuntimeError:
                # No running loop, use sync
                pass
        
        # Use sync request
        return self._sync_request(method, path, data, params)
    
    async def async_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Async request wrapper."""
        return await self._async_request(method, path, data, params)
    
    def close(self):
        """Close all sessions."""
        self._sync_session.close()
        if self._async_session and not self._async_session.closed:
            # Schedule async session close
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._async_session.close())
                else:
                    loop.run_until_complete(self._async_session.close())
            except Exception:
                pass
        self._executor.shutdown(wait=False)


class DashVectorBackend(IVectorStorageBackend):
    """
    DashVector vector storage backend using HTTP API.
    Designed for high concurrency multi-user scenarios.
    
    Features:
    - HTTP API based (no SDK dependency)
    - Connection pooling for high concurrency
    - Automatic retry with exponential backoff
    - SQL-style filter syntax
    """

    def __init__(self):
        self._client: Optional[DashVectorHTTPClient] = None
        self._collections: Dict[str, str] = {}  # context_type -> collection_name
        self._initialized = False
        self._config = None
        self._vector_size = None
        self._max_retry_count = 3
        self._retry_delay = 1.0

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the DashVector backend.
        
        Args:
            config: Configuration dictionary containing:
                - config.api_key: DashVector API key
                - config.endpoint: DashVector cluster endpoint
                - config.vector_size: Vector dimension (default: 1536)
                - config.timeout: Request timeout in seconds (default: 30.0)
                - config.max_connections: Max total connections (default: 100)
                - config.max_connections_per_host: Max connections per host (default: 20)
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self._config = config
            dashvector_config = config.get("config", {})

            # Get configuration
            api_key = dashvector_config.get("api_key")
            endpoint = dashvector_config.get("endpoint")
            self._vector_size = dashvector_config.get("vector_size", 1536)
            timeout = dashvector_config.get("timeout", DEFAULT_TIMEOUT)
            max_connections = dashvector_config.get("max_connections", DEFAULT_MAX_CONNECTIONS)
            max_connections_per_host = dashvector_config.get(
                "max_connections_per_host", DEFAULT_MAX_CONNECTIONS_PER_HOST
            )

            if not api_key or not endpoint:
                logger.error("DashVector API key and endpoint are required")
                return False

            # Create HTTP client
            self._client = DashVectorHTTPClient(
                api_key=api_key,
                endpoint=endpoint,
                timeout=timeout,
                max_connections=max_connections,
                max_connections_per_host=max_connections_per_host,
                max_retries=self._max_retry_count,
                retry_delay=self._retry_delay,
            )

            # Test connection and get existing collections
            if not self._check_connection():
                logger.error("Failed to connect to DashVector service")
                return False

            # Initialize collections for each context type
            for context_type in ContextType:
                collection_name = context_type.value
                self._ensure_collection(collection_name)
                self._collections[collection_name] = collection_name

            # Initialize todo collection
            self._ensure_collection(TODO_COLLECTION)
            self._collections[TODO_COLLECTION] = TODO_COLLECTION

            self._initialized = True
            logger.info(f"DashVector HTTP backend initialized with {len(self._collections)} collections")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize DashVector backend: {e}")
            return False

    def _check_connection(self) -> bool:
        """Check if connection to DashVector is working."""
        try:
            result = self._client.request("GET", "collections")
            return result.get("code") == 0
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    def _get_existing_collections(self) -> List[str]:
        """Get list of existing collection names."""
        try:
            result = self._client.request("GET", "collections")
            if result.get("code") == 0:
                return result.get("output", [])
            return []
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            return []

    def _ensure_collection(self, collection_name: str) -> None:
        """
        Ensure collection exists, create if not.
        
        Args:
            collection_name: Name of the collection to ensure
        """
        try:
            existing = self._get_existing_collections()
            
            if collection_name not in existing:
                # Create new collection
                data = {
                    "name": collection_name,
                    "dimension": self._vector_size,
                    "metric": "cosine",
                    "dtype": "FLOAT",
                    "fields_schema": {
                        # ID fields
                        FIELD_ORIGINAL_ID: "STRING",
                        
                        # Time fields (string format)
                        FIELD_CREATED_AT: "STRING",
                        FIELD_CREATE_TIME: "STRING",
                        FIELD_EVENT_TIME: "STRING",
                        FIELD_UPDATE_TIME: "STRING",
                        FIELD_LAST_CALL_TIME: "STRING",
                        
                        # Time fields (timestamp for filtering)
                        FIELD_CREATED_AT_TS: "FLOAT",
                        FIELD_CREATE_TIME_TS: "FLOAT",
                        FIELD_EVENT_TIME_TS: "FLOAT",
                        FIELD_UPDATE_TIME_TS: "FLOAT",
                        FIELD_LAST_CALL_TIME_TS: "FLOAT",
                        
                        # User identification fields
                        FIELD_USER_ID: "STRING",
                        FIELD_DEVICE_ID: "STRING",
                        FIELD_AGENT_ID: "STRING",
                        
                        # Extracted data fields
                        FIELD_TITLE: "STRING",
                        FIELD_SUMMARY: "STRING",
                        FIELD_KEYWORDS: "STRING",  # JSON array as string
                        FIELD_ENTITIES: "STRING",  # JSON array as string
                        FIELD_CONTEXT_TYPE: "STRING",
                        FIELD_CONFIDENCE: "FLOAT",
                        FIELD_IMPORTANCE: "FLOAT",
                        FIELD_SOURCE: "STRING",
                        
                        # Properties fields
                        FIELD_IS_PROCESSED: "BOOL",
                        FIELD_HAS_COMPRESSION: "BOOL",
                        FIELD_ENABLE_MERGE: "BOOL",
                        FIELD_IS_HAPPEND: "BOOL",
                        FIELD_CALL_COUNT: "FLOAT",
                        FIELD_MERGE_COUNT: "FLOAT",
                        FIELD_DURATION_COUNT: "FLOAT",
                        
                        # Document tracking fields
                        FIELD_FILE_PATH: "STRING",
                        FIELD_RAW_TYPE: "STRING",
                        FIELD_RAW_ID: "STRING",
                        
                        # Document content
                        FIELD_DOCUMENT: "STRING",
                    },
                }
                
                result = self._client.request("POST", "collections", data=data)
                
                if result.get("code") == 0:
                    logger.info(f"Created DashVector collection: {collection_name}")
                    # Wait for collection to be ready
                    self._wait_for_collection_ready(collection_name)
                else:
                    logger.warning(
                        f"Failed to create collection {collection_name}: {result.get('message')}"
                    )
            else:
                logger.debug(f"DashVector collection already exists: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name}: {e}")
            raise

    def _wait_for_collection_ready(
        self, 
        collection_name: str, 
        max_wait: int = 60
    ) -> bool:
        """
        Wait for collection to be ready after creation.
        
        Args:
            collection_name: Name of collection
            max_wait: Maximum wait time in seconds
            
        Returns:
            True if collection is ready, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                result = self._client.request(
                    "GET", 
                    f"collections/{collection_name}/stats"
                )
                if result.get("code") == 0:
                    return True
            except Exception:
                pass
            time.sleep(1)
        
        logger.warning(f"Timeout waiting for collection {collection_name} to be ready")
        return False

    def get_name(self) -> str:
        return "dashvector"

    def get_storage_type(self) -> StorageType:
        return StorageType.VECTOR_DB

    def get_collection_names(self) -> List[str]:
        """Get all collection names managed by this backend."""
        return list(self._collections.keys())

    def _ensure_vectorized(self, context: ProcessedContext) -> List[float]:
        """
        Ensure context has vector, generate if missing.
        
        Args:
            context: ProcessedContext to check
            
        Returns:
            Vector as list of floats
        """
        if context.vectorize and context.vectorize.vector:
            return list(context.vectorize.vector)
        
        # Generate vector
        if context.vectorize:
            do_vectorize(context.vectorize)
            if context.vectorize.vector:
                return list(context.vectorize.vector)
        
        raise ValueError(f"Unable to get or generate vector for context {context.id}")

    def _context_to_doc_format(self, context: ProcessedContext) -> Dict[str, Any]:
        """
        Convert ProcessedContext to DashVector Doc format.
        
        This method mirrors chromadb_backend._context_to_chroma_format to ensure
        all fields are properly stored with both ISO format and timestamp versions
        for datetime fields.
        
        Args:
            context: ProcessedContext to convert
            
        Returns:
            Dictionary in DashVector Doc format with all fields properly serialized
        """
        # Start with basic context fields (excluding nested objects)
        doc = context.model_dump(
            exclude_none=True, 
            exclude={"properties", "extracted_data", "vectorize", "metadata"}
        )
        
        # Add extracted_data fields
        if context.extracted_data:
            extracted_data_dict = context.extracted_data.model_dump(exclude_none=True)
            doc.update(extracted_data_dict)
        
        # Add metadata fields
        if context.metadata:
            doc.update(context.metadata)
        
        # Add document text
        if context.vectorize:
            if context.vectorize.content_format == ContentFormat.TEXT:
                doc[FIELD_DOCUMENT] = context.vectorize.text or ""
        
        # Add properties fields (excluding raw_properties which is a nested list)
        if context.properties:
            properties_dict = context.properties.model_dump(exclude_none=True)
            properties_dict.pop("raw_properties", None)
            doc.update(properties_dict)
        
        # JSON serializer for datetime and enum
        def default_json_serializer(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            return str(obj)
        
        # Process all fields for proper serialization
        fields = {}
        for key, value in list(doc.items()):
            # Skip special fields
            if key in ["id", "embedding", "document"]:
                if key == "id":
                    fields[FIELD_ORIGINAL_ID] = value
                elif key == "document":
                    fields[FIELD_DOCUMENT] = value
                continue
            
            if value is None:
                continue
            
            # Handle datetime fields - store both ISO string and timestamp
            if isinstance(value, datetime.datetime):
                fields[f"{key}_ts"] = value.timestamp()
                fields[key] = value.isoformat()
            # Handle enum fields
            elif isinstance(value, Enum):
                fields[key] = value.value
            # Handle dict and list fields - serialize to JSON string
            elif isinstance(value, (dict, list)):
                try:
                    fields[key] = json.dumps(
                        value, ensure_ascii=False, default=default_json_serializer
                    )
                except (TypeError, ValueError):
                    fields[key] = str(value)
            # Handle boolean fields
            elif isinstance(value, bool):
                fields[key] = value
            # Handle numeric fields
            elif isinstance(value, (int, float)):
                fields[key] = float(value)  # DashVector uses FLOAT for numbers
            # Handle string fields
            elif isinstance(value, str):
                fields[key] = value
            else:
                fields[key] = str(value)
        
        # Ensure original_id is set
        fields[FIELD_ORIGINAL_ID] = context.id
        
        # Add created_at timestamp (storage time)
        now = datetime.datetime.now()
        fields[FIELD_CREATED_AT] = now.isoformat()
        fields[FIELD_CREATED_AT_TS] = now.timestamp()
        
        return fields

    def _serialize_value(self, value: Any) -> Any:
        """
        Serialize value for DashVector storage.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized value
        """
        if value is None:
            return ""
        elif isinstance(value, datetime.datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return str(value)
        elif isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)

    def upsert_processed_context(self, context: ProcessedContext) -> str:
        """
        Store a single ProcessedContext.
        
        Args:
            context: ProcessedContext to store
            
        Returns:
            ID of stored context
        """
        results = self.batch_upsert_processed_context([context])
        return results[0] if results else ""

    def batch_upsert_processed_context(
        self, 
        contexts: List[ProcessedContext]
    ) -> List[str]:
        """
        Batch store ProcessedContexts.
        
        Args:
            contexts: List of ProcessedContext to store
            
        Returns:
            List of stored context IDs
        """
        if not self._initialized:
            raise RuntimeError("DashVector backend not initialized")

        # Group contexts by type
        contexts_by_type: Dict[str, List[ProcessedContext]] = {}
        for context in contexts:
            context_type = context.extracted_data.context_type.value
            if context_type not in contexts_by_type:
                contexts_by_type[context_type] = []
            contexts_by_type[context_type].append(context)

        stored_ids = []

        for context_type, type_contexts in contexts_by_type.items():
            if context_type not in self._collections:
                logger.warning(
                    f"No collection found for context_type '{context_type}', "
                    f"skipping storage"
                )
                continue

            docs = []
            for context in type_contexts:
                try:
                    vector = self._ensure_vectorized(context)
                    fields = self._context_to_doc_format(context)

                    doc = {
                        "id": context.id,
                        "vector": vector,
                        "fields": fields,
                    }
                    docs.append(doc)

                except Exception as e:
                    logger.exception(f"Failed to process context {context.id}: {e}")
                    continue

            if not docs:
                continue

            # Batch upsert via HTTP API
            try:
                result = self._client.request(
                    "POST",
                    f"collections/{context_type}/docs/upsert",
                    data={"docs": docs}
                )

                if result.get("code") == 0:
                    output = result.get("output", [])
                    for item in output:
                        if item.get("code") == 0:
                            stored_ids.append(item.get("id"))
                    logger.debug(f"Upserted {len(output)} docs to {context_type}")
                else:
                    logger.error(
                        f"Failed to upsert to {context_type}: {result.get('message')}"
                    )

            except Exception as e:
                logger.exception(f"Failed to upsert contexts to {context_type}: {e}")
                continue

        return stored_ids

    def get_processed_context(
        self, 
        id: str, 
        context_type: str,
        need_vector: bool = False
    ) -> Optional[ProcessedContext]:
        """
        Get a specific ProcessedContext by ID.
        
        Args:
            id: Context ID
            context_type: Type of context
            need_vector: Whether to include vector in result
            
        Returns:
            ProcessedContext if found, None otherwise
        """
        if not self._initialized:
            return None

        if context_type not in self._collections:
            return None

        try:
            # Fetch doc by ID via HTTP API
            result = self._client.request(
                "GET",
                f"collections/{context_type}/docs",
                params={"ids": id}
            )

            if result.get("code") == 0:
                output = result.get("output", {})
                if isinstance(output, dict) and id in output:
                    doc = output[id]
                    if doc:
                        return self._doc_to_context(doc, need_vector)

        except Exception as e:
            logger.exception(f"Failed to get context {id}: {e}")

        return None

    def get_all_processed_contexts(
        self,
        context_types: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        filter: Optional[Dict[str, Any]] = None,
        need_vector: bool = False,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, List[ProcessedContext]]:
        """
        Get all ProcessedContexts, optionally filtered.
        
        Args:
            context_types: List of context types to filter by
            limit: Maximum number of results per type
            offset: Offset for pagination
            filter: Additional filter conditions
            need_vector: Whether to include vectors
            user_id: Filter by user ID
            device_id: Filter by device ID
            agent_id: Filter by agent ID
            
        Returns:
            Dictionary mapping context_types to list of ProcessedContext
        """
        if not self._initialized:
            return {}

        result = {}
        target_types = context_types if context_types else [
            ct for ct in self._collections.keys() if ct != TODO_COLLECTION
        ]

        for ctx_type in target_types:
            if ctx_type not in self._collections:
                continue

            try:
                # Build filter string
                filter_str = self._build_filter_string(filter, user_id, device_id, agent_id)
                
                # Use query API to get all docs (with empty vector for filter-only)
                data = {
                    "topk": limit + offset,
                    "include_vector": need_vector,
                }
                if filter_str:
                    data["filter"] = filter_str

                query_result = self._client.request(
                    "POST",
                    f"collections/{ctx_type}/query",
                    data=data
                )

                if query_result.get("code") == 0:
                    output = query_result.get("output", [])
                    # Apply offset
                    if offset > 0:
                        output = output[offset:]
                    if len(output) > limit:
                        output = output[:limit]

                    contexts = []
                    for doc in output:
                        context = self._doc_to_context(doc, need_vector)
                        if context:
                            contexts.append(context)

                    if contexts:
                        result[ctx_type] = contexts

            except Exception as e:
                logger.exception(f"Failed to get contexts from {ctx_type}: {e}")
                continue

        return result

    def delete_processed_context(self, id: str, context_type: str) -> bool:
        """
        Delete a specific ProcessedContext.
        
        Args:
            id: Context ID to delete
            context_type: Type of context
            
        Returns:
            True if successful, False otherwise
        """
        return self.delete_contexts([id], context_type)

    def delete_contexts(self, ids: List[str], context_type: str) -> bool:
        """
        Delete multiple contexts.
        
        Args:
            ids: List of context IDs to delete
            context_type: Type of context
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False

        if context_type not in self._collections:
            return False

        try:
            result = self._client.request(
                "DELETE",
                f"collections/{context_type}/docs",
                data={"ids": ids}
            )
            
            if result.get("code") == 0:
                logger.debug(f"Deleted {len(ids)} contexts from {context_type}")
                return True
            else:
                logger.error(f"Failed to delete contexts: {result.get('message')}")
                return False
                
        except Exception as e:
            logger.exception(f"Failed to delete contexts: {e}")
            return False

    def search(
        self,
        query: Vectorize,
        top_k: int = 10,
        context_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        need_vector: bool = False,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Tuple[ProcessedContext, float]]:
        """
        Vector similarity search.
        
        Args:
            query: Query vectorize object
            top_k: Maximum number of results
            context_types: List of context types to search
            filters: Additional filter conditions
            need_vector: Whether to include vectors in results
            user_id: Filter by user ID
            device_id: Filter by device ID
            agent_id: Filter by agent ID
            
        Returns:
            List of (ProcessedContext, score) tuples
        """
        if not self._initialized:
            return []

        # Determine target collections
        if context_types:
            target_types = [ct for ct in context_types if ct in self._collections]
        else:
            target_types = [ct for ct in self._collections.keys() if ct != TODO_COLLECTION]

        # Get query vector
        query_vector = None
        if query.vector and len(query.vector) > 0:
            query_vector = list(query.vector)
        else:
            do_vectorize(query)
            query_vector = list(query.vector) if query.vector else None

        if not query_vector:
            logger.warning("Unable to get query vector, search failed")
            return []

        # Build filter string
        filter_str = self._build_filter_string(filters, user_id, device_id, agent_id)

        all_results = []

        for ctx_type in target_types:
            try:
                # Build query request
                data = {
                    "vector": query_vector,
                    "topk": top_k,
                    "include_vector": need_vector,
                }
                if filter_str:
                    data["filter"] = filter_str

                result = self._client.request(
                    "POST",
                    f"collections/{ctx_type}/query",
                    data=data
                )

                if result.get("code") == 0:
                    output = result.get("output", [])
                    for doc in output:
                        context = self._doc_to_context(doc, need_vector)
                        if context:
                            score = doc.get("score", 0.0)
                            all_results.append((context, score))

            except Exception as e:
                logger.exception(f"Vector search failed in {ctx_type}: {e}")
                continue

        # Sort by score descending
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def _doc_to_context(
        self, 
        doc: Dict[str, Any],
        need_vector: bool = False
    ) -> Optional[ProcessedContext]:
        """
        Convert DashVector Doc to ProcessedContext.
        
        Args:
            doc: DashVector Doc dict
            need_vector: Whether to include vector
            
        Returns:
            ProcessedContext if conversion successful, None otherwise
        """
        try:
            if not doc or not doc.get("id"):
                return None

            # Get fields from doc
            fields = doc.get("fields", {})
            
            # Separate fields into different categories
            extracted_data_field_names = set(ExtractedData.model_fields.keys())
            properties_field_names = set(ContextProperties.model_fields.keys())
            vectorize_field_names = set(Vectorize.model_fields.keys())

            extracted_data_dict = {}
            properties_dict = {}
            vectorize_dict = {}
            metadata_dict = {}

            # Get document text
            document = fields.pop(FIELD_DOCUMENT, None)
            if document:
                vectorize_dict["text"] = document

            # Get vector if needed
            if need_vector and doc.get("vector"):
                vectorize_dict["vector"] = doc["vector"]

            # Get original ID
            original_id = fields.pop(FIELD_ORIGINAL_ID, doc["id"])

            # Categorize fields
            for key, value in fields.items():
                if key.endswith("_ts"):
                    continue

                # Try to deserialize JSON strings
                val = value
                if isinstance(value, str) and value.startswith(("{", "[")):
                    try:
                        val = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if key in extracted_data_field_names:
                    extracted_data_dict[key] = val
                elif key in properties_field_names:
                    properties_dict[key] = val
                elif key in vectorize_field_names:
                    vectorize_dict[key] = val
                else:
                    metadata_dict[key] = val

            # Build context
            context_dict = {
                "id": original_id,
                "extracted_data": ExtractedData.model_validate(extracted_data_dict),
                "properties": ContextProperties.model_validate(properties_dict),
                "vectorize": Vectorize.model_validate(vectorize_dict) if vectorize_dict else None,
                "metadata": metadata_dict if metadata_dict else None,
            }

            return ProcessedContext.model_validate(context_dict)

        except Exception as e:
            logger.exception(f"Failed to convert doc to ProcessedContext: {e}")
            return None

    def _parse_time_to_timestamp(self, time_value: Any) -> Optional[float]:
        """
        Parse various time formats to Unix timestamp.
        
        Args:
            time_value: Time value in various formats
            
        Returns:
            Unix timestamp as float, or None if parsing failed
        """
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
                time_str = time_value.replace('Z', '+00:00')
                if '.' in time_str and '+' in time_str:
                    parts = time_str.split('+')
                    time_str = parts[0].split('.')[0] + '+' + parts[1]
                elif '.' in time_str:
                    time_str = time_str.split('.')[0]
                
                dt = datetime.datetime.fromisoformat(time_str)
                return dt.timestamp()
            
            if isinstance(time_value, datetime.datetime):
                return time_value.timestamp()
                
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to parse time value '{time_value}': {e}")
            
        return None

    def _build_filter_string(
        self,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Build DashVector filter string (SQL WHERE clause style).
        
        Args:
            filters: Additional filter conditions
            user_id: User ID filter
            device_id: Device ID filter
            agent_id: Agent ID filter
            
        Returns:
            Filter string or None if no filters
        """
        conditions = []
        
        # Time-related field mappings
        TIME_FIELD_MAPPING = {
            'created_at': FIELD_CREATED_AT_TS,
            'create_time': FIELD_CREATE_TIME_TS,
            FIELD_CREATED_AT: FIELD_CREATED_AT_TS,
        }

        # Add user identity filters
        if user_id:
            conditions.append(f"{FIELD_USER_ID} = '{user_id}'")
        if device_id:
            conditions.append(f"{FIELD_DEVICE_ID} = '{device_id}'")
        if agent_id:
            conditions.append(f"{FIELD_AGENT_ID} = '{agent_id}'")

        # Add custom filters
        if filters:
            for key, value in filters.items():
                if key in ('context_type', 'entities'):
                    continue
                if value is None:
                    continue
                
                is_time_field = key in TIME_FIELD_MAPPING or key.endswith('_ts')
                filter_key = TIME_FIELD_MAPPING.get(key, key)

                if isinstance(value, dict):
                    for op, op_symbol in [('$gte', '>='), ('$lte', '<='), ('$gt', '>'), ('$lt', '<')]:
                        if op in value:
                            op_value = value[op]
                            if is_time_field:
                                ts = self._parse_time_to_timestamp(op_value)
                                if ts is not None:
                                    conditions.append(f"{filter_key} {op_symbol} {ts}")
                            else:
                                conditions.append(f"{filter_key} {op_symbol} {op_value}")
                elif isinstance(value, list):
                    values_str = ', '.join([
                        f"'{v}'" if isinstance(v, str) else str(v) 
                        for v in value
                    ])
                    conditions.append(f"{filter_key} IN ({values_str})")
                elif isinstance(value, str):
                    if is_time_field:
                        ts = self._parse_time_to_timestamp(value)
                        if ts is not None:
                            conditions.append(f"{filter_key} >= {ts - 0.5}")
                            conditions.append(f"{filter_key} <= {ts + 0.5}")
                        else:
                            escaped_value = value.replace("'", "''")
                            conditions.append(f"{key} = '{escaped_value}'")
                    else:
                        escaped_value = value.replace("'", "''")
                        conditions.append(f"{filter_key} = '{escaped_value}'")
                elif isinstance(value, bool):
                    conditions.append(f"{filter_key} = {str(value).lower()}")
                else:
                    conditions.append(f"{filter_key} = {value}")

        if not conditions:
            return None

        filter_str = ' AND '.join(conditions)
        logger.debug(f"Built DashVector filter: {filter_str}")
        return filter_str

    def get_processed_context_count(self, context_type: str) -> int:
        """
        Get count of records for a context type.
        
        Args:
            context_type: Type of context
            
        Returns:
            Number of records
        """
        if not self._initialized:
            return 0

        if context_type not in self._collections:
            return 0

        try:
            result = self._client.request(
                "GET",
                f"collections/{context_type}/stats"
            )
            if result.get("code") == 0:
                output = result.get("output", {})
                return int(output.get("total_doc_count", 0))
            return 0
        except Exception as e:
            logger.error(f"Failed to get count for {context_type}: {e}")
            return 0

    def get_all_processed_context_counts(self) -> Dict[str, int]:
        """
        Get counts for all context types.
        
        Returns:
            Dictionary mapping context_type to count
        """
        counts = {}
        for context_type in self._collections.keys():
            if context_type != TODO_COLLECTION:
                counts[context_type] = self.get_processed_context_count(context_type)
        return counts

    # Todo-related methods
    def upsert_todo_embedding(
        self,
        todo_id: str,
        content: str,
        embedding: List[float],
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Store a todo embedding.
        
        Args:
            todo_id: Todo ID
            content: Todo content text
            embedding: Vector embedding
            user_id: User ID
            device_id: Device ID
            agent_id: Agent ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False

        try:
            doc = {
                "id": todo_id,
                "vector": list(embedding),
                "fields": {
                    FIELD_TODO_ID: todo_id,
                    FIELD_CONTENT: content,
                    FIELD_CREATED_AT: datetime.datetime.now().isoformat(),
                    FIELD_CREATED_AT_TS: datetime.datetime.now().timestamp(),
                    FIELD_USER_ID: user_id or "",
                    FIELD_DEVICE_ID: device_id or "",
                    FIELD_AGENT_ID: agent_id or "",
                },
            }

            result = self._client.request(
                "POST",
                f"collections/{TODO_COLLECTION}/docs/upsert",
                data={"docs": [doc]}
            )

            return result.get("code") == 0

        except Exception as e:
            logger.exception(f"Failed to upsert todo embedding: {e}")
            return False

    def search_similar_todos(
        self,
        embedding: List[float],
        top_k: int = 5,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Search for similar todos.
        
        Args:
            embedding: Query embedding
            top_k: Number of results
            user_id: Filter by user ID
            device_id: Filter by device ID
            agent_id: Filter by agent ID
            
        Returns:
            List of (todo_id, content, score) tuples
        """
        if not self._initialized:
            return []

        try:
            filter_str = self._build_filter_string(None, user_id, device_id, agent_id)
            
            data = {
                "vector": list(embedding),
                "topk": top_k,
                "include_vector": False,
            }
            if filter_str:
                data["filter"] = filter_str

            result = self._client.request(
                "POST",
                f"collections/{TODO_COLLECTION}/query",
                data=data
            )

            if result.get("code") == 0:
                output = result.get("output", [])
                results = []
                for doc in output:
                    fields = doc.get("fields", {})
                    todo_id = fields.get(FIELD_TODO_ID, doc.get("id"))
                    content = fields.get(FIELD_CONTENT, "")
                    score = doc.get("score", 0.0)
                    results.append((todo_id, content, score))
                return results

        except Exception as e:
            logger.exception(f"Failed to search similar todos: {e}")

        return []

    def delete_todo_embedding(self, todo_id: str) -> bool:
        """
        Delete a todo embedding.
        
        Args:
            todo_id: Todo ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False

        try:
            result = self._client.request(
                "DELETE",
                f"collections/{TODO_COLLECTION}/docs",
                data={"ids": [todo_id]}
            )
            return result.get("code") == 0

        except Exception as e:
            logger.exception(f"Failed to delete todo embedding: {e}")
            return False

    def close(self):
        """Close the backend and release resources."""
        if self._client:
            self._client.close()
        self._initialized = False
        logger.info("DashVector HTTP backend closed")
