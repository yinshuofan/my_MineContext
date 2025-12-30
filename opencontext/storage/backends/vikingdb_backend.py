# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
VikingDB vector storage backend - Volcengine VikingDB Service (HTTP API)
https://www.volcengine.com/docs/84313/1254458

This implementation uses HTTP API with Volcengine V4 signature for authentication.
"""

import asyncio
import datetime
import hashlib
import hmac
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

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
        return quote(path, safe='/')
    
    def _get_canonical_query_string(self, params: Dict[str, str]) -> str:
        """Get canonical query string."""
        if not params:
            return ""
        sorted_params = sorted(params.items())
        return '&'.join([
            f"{quote(k, safe='')}={quote(str(v), safe='')}"
            for k, v in sorted_params
        ])
    
    def _get_canonical_headers(self, headers: Dict[str, str]) -> Tuple[str, str]:
        """Get canonical headers and signed headers."""
        # Headers to sign (lowercase)
        headers_to_sign = {}
        for key, value in headers.items():
            lower_key = key.lower()
            if lower_key in ['host', 'content-type', 'x-date', 'x-content-sha256']:
                headers_to_sign[lower_key] = value.strip()
        
        # Sort headers
        sorted_headers = sorted(headers_to_sign.items())
        canonical_headers = '\n'.join([f"{k}:{v}" for k, v in sorted_headers]) + '\n'
        signed_headers = ';'.join([k for k, v in sorted_headers])
        
        return canonical_headers, signed_headers
    
    def _get_payload_hash(self, body: str) -> str:
        """Get SHA256 hash of request body."""
        return hashlib.sha256(body.encode('utf-8')).hexdigest()
    
    def _hmac_sha256(self, key: bytes, msg: str) -> bytes:
        """HMAC-SHA256 signing."""
        return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()
    
    def _get_signing_key(self, date_stamp: str) -> bytes:
        """Get signing key using secret access key."""
        k_date = self._hmac_sha256(self.secret_access_key.encode('utf-8'), date_stamp)
        k_region = self._hmac_sha256(k_date, self.region)
        k_service = self._hmac_sha256(k_region, self.service)
        k_signing = self._hmac_sha256(k_service, "request")
        return k_signing
    
    def sign_request(
        self,
        method: str,
        host: str,
        path: str,
        headers: Dict[str, str],
        body: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
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
        t = datetime.datetime.utcnow()
        x_date = t.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = t.strftime('%Y%m%d')
        
        # Add required headers
        headers = dict(headers)
        headers['Host'] = host
        headers['X-Date'] = x_date
        
        # Calculate payload hash
        payload_hash = self._get_payload_hash(body)
        headers['X-Content-Sha256'] = payload_hash
        
        # Build canonical request
        canonical_uri = self._get_canonical_uri(path)
        canonical_query_string = self._get_canonical_query_string(params or {})
        canonical_headers, signed_headers = self._get_canonical_headers(headers)
        
        canonical_request = '\n'.join([
            method.upper(),
            canonical_uri,
            canonical_query_string,
            canonical_headers,
            signed_headers,
            payload_hash
        ])
        
        # Build string to sign
        algorithm = "HMAC-SHA256"
        credential_scope = f"{date_stamp}/{self.region}/{self.service}/request"
        hashed_canonical_request = hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        
        string_to_sign = '\n'.join([
            algorithm,
            x_date,
            credential_scope,
            hashed_canonical_request
        ])
        
        # Calculate signature
        signing_key = self._get_signing_key(date_stamp)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        # Build authorization header
        authorization = (
            f"{algorithm} "
            f"Credential={self.access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )
        
        headers['Authorization'] = authorization
        
        return headers


class VikingDBHTTPClient:
    """
    HTTP client for VikingDB API with connection pooling and retry support.
    Supports both control plane (console) and data plane APIs.
    """
    
    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str = "cn-beijing",
        data_host: Optional[str] = None,
        console_host: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_connections_per_host: int = DEFAULT_MAX_CONNECTIONS_PER_HOST,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._region = region
        
        # Set default hosts based on region
        self._data_host = data_host or f"api-vikingdb.vikingdb.{region}.volces.com"
        self._console_host = console_host or f"vikingdb.{region}.volcengineapi.com"
        
        self._timeout = timeout
        self._max_connections = max_connections
        self._max_connections_per_host = max_connections_per_host
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        
        # Initialize auth
        self._auth = VolcengineAuth(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region,
            service=VIKINGDB_SERVICE,
        )
        
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
    
    def _get_base_headers(self) -> Dict[str, str]:
        """Get common headers for API requests."""
        return {
            "Content-Type": "application/json",
        }
    
    def _sync_request(
        self,
        method: str,
        host: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make sync HTTP request with V4 signature.
        
        Args:
            method: HTTP method
            host: API host
            path: API path
            data: Request body
            params: Query parameters
            
        Returns:
            Response JSON as dict
        """
        url = f"https://{host}{path}"
        body = json.dumps(data) if data else ""
        
        # Get base headers and sign request
        headers = self._get_base_headers()
        signed_headers = self._auth.sign_request(
            method=method,
            host=host,
            path=path,
            headers=headers,
            body=body,
            params=params,
        )
        
        response = self._sync_session.request(
            method=method,
            url=url,
            headers=signed_headers,
            data=body if body else None,
            params=params,
            timeout=self._timeout,
        )
        
        return response.json()
    
    def console_request(
        self,
        action: str,
        data: Optional[Dict] = None,
        version: str = VIKINGDB_VERSION,
    ) -> Dict[str, Any]:
        """
        Make request to console (control plane) API.
        
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
        
        return self._sync_request(
            method="POST",
            host=self._console_host,
            path="/",
            data=data,
            params=params,
        )
    
    def data_request(
        self,
        path: str,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make request to data plane API.
        
        Args:
            path: API path (e.g., /api/collection/upsert_data)
            data: Request body
            
        Returns:
            Response JSON as dict
        """
        return self._sync_request(
            method="POST",
            host=self._data_host,
            path=path,
            data=data,
        )
    
    def close(self):
        """Close the HTTP client and release resources."""
        if self._sync_session:
            self._sync_session.close()
        if self._async_session and not self._async_session.closed:
            # Close async session in executor
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._async_session.close())
                else:
                    loop.run_until_complete(self._async_session.close())
            except Exception:
                pass
        if self._executor:
            self._executor.shutdown(wait=False)


class VikingDBBackend(IVectorStorageBackend):
    """
    VikingDB vector storage backend implementation.
    
    Uses Volcengine VikingDB service for vector storage and similarity search.
    Implements the IVectorStorageBackend interface for compatibility with
    the opencontext storage system.
    """
    
    def __init__(self):
        self._client: Optional[VikingDBHTTPClient] = None
        self._collections: Dict[str, bool] = {}
        self._indexes: Dict[str, str] = {}  # collection_name -> index_name
        self._dimension: int = 0
        self._initialized: bool = False
        self._config: Dict[str, Any] = {}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize VikingDB backend.
        
        Args:
            config: Configuration dictionary with:
                - access_key_id: Volcengine Access Key ID
                - secret_access_key: Volcengine Secret Access Key
                - region: Region (default: cn-beijing)
                - data_host: Data plane API host (optional)
                - console_host: Console API host (optional)
                - dimension: Vector dimension
                - context_types: List of context types to create collections for
                - index_type: Index type (default: hnsw)
                - distance_type: Distance metric (default: ip)
                
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self._config = config
            vikingdb_config = config.get("config", {})
            
            # Get credentials from config or environment
            access_key_id = vikingdb_config.get("access_key_id") or os.environ.get("VOLCENGINE_ACCESS_KEY_ID")
            secret_access_key = vikingdb_config.get("secret_access_key") or os.environ.get("VOLCENGINE_SECRET_ACCESS_KEY")
            
            if not access_key_id or not secret_access_key:
                logger.error("VikingDB credentials not provided")
                return False
            
            region = vikingdb_config.get("region", "cn-beijing")
            data_host = vikingdb_config.get("data_host")
            console_host = vikingdb_config.get("console_host")
            
            self._dimension = config.get("dimension", 1024)
            
            # Initialize HTTP client
            self._client = VikingDBHTTPClient(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                region=region,
                data_host=data_host,
                console_host=console_host,
            )
            
            # Get context types to create collections for
            context_types = config.get("context_types", [])
            if not context_types:
                # Default context types
                context_types = [ct.value for ct in ContextType]
            
            # Add todo collection
            # if TODO_COLLECTION not in context_types:
            #     context_types.append(TODO_COLLECTION)
            
            # Ensure collections and indexes exist
            index_type = config.get("index_type", "hnsw")
            distance_type = config.get("distance_type", "ip")
            
            for ctx_type in context_types:
                self._ensure_collection_and_index(
                    collection_name=ctx_type,
                    dimension=self._dimension,
                    index_type=index_type,
                    distance_type=distance_type,
                )
            
            self._initialized = True
            logger.info(f"VikingDB backend initialized with {len(self._collections)} collections")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize VikingDB backend: {e}")
            return False
    
    def _ensure_collection_and_index(
        self,
        collection_name: str,
        dimension: int,
        index_type: str = "hnsw",
        distance_type: str = "ip",
    ) -> None:
        """
        Ensure collection and index exist, create if not.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            index_type: Index type (hnsw, flat, ivf)
            distance_type: Distance metric (ip, l2, cosine)
        """
        try:
            # Check if collection exists
            result = self._client.console_request(
                action="GetVikingdbCollection",
                data={"CollectionName": collection_name},
            )
            
            if result.get("ResponseMetadata", {}).get("Error"):
                error_code = result["ResponseMetadata"]["Error"].get("Code", "")
                if "NotExist" in error_code or "NotFound" in error_code:
                    # Create collection
                    self._create_collection(collection_name, dimension)
                else:
                    logger.error(f"Failed to get collection {collection_name}: {result}")
                    return
            else:
                logger.debug(f"VikingDB collection already exists: {collection_name}")
            
            self._collections[collection_name] = True
            
            # Check if index exists
            index_name = f"{collection_name}_index"
            result = self._client.console_request(
                action="GetVikingdbIndex",
                data={
                    "CollectionName": collection_name,
                    "IndexName": index_name,
                },
            )
            
            if result.get("ResponseMetadata", {}).get("Error"):
                error_code = result["ResponseMetadata"]["Error"].get("Code", "")
                if "NotExist" in error_code or "NotFound" in error_code:
                    # Create index
                    self._create_index(
                        collection_name=collection_name,
                        index_name=index_name,
                        index_type=index_type,
                        distance_type=distance_type,
                    )
                else:
                    logger.error(f"Failed to get index {index_name}: {result}")
                    return
            else:
                logger.debug(f"VikingDB index already exists: {index_name}")
            
            self._indexes[collection_name] = index_name
            
        except Exception as e:
            logger.exception(f"Error ensuring collection {collection_name}: {e}")
            raise
    
    def _create_collection(self, collection_name: str, dimension: int) -> None:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
        """
        # Define fields for the collection
        fields = [
            {
                "FieldName": "id",
                "FieldType": "string",
                "IsPrimaryKey": True,
            },
            {
                "FieldName": "vector",
                "FieldType": "vector",
                "Dim": dimension,
            },
            # User identification fields
            {"FieldName": FIELD_USER_ID, "FieldType": "string"},
            {"FieldName": FIELD_DEVICE_ID, "FieldType": "string"},
            {"FieldName": FIELD_AGENT_ID, "FieldType": "string"},
            # Time fields
            {"FieldName": FIELD_CREATED_AT, "FieldType": "string"},
            {"FieldName": FIELD_CREATED_AT_TS, "FieldType": "float32"},
            {"FieldName": FIELD_CREATE_TIME, "FieldType": "string"},
            {"FieldName": FIELD_CREATE_TIME_TS, "FieldType": "float32"},
            {"FieldName": FIELD_EVENT_TIME, "FieldType": "string"},
            {"FieldName": FIELD_EVENT_TIME_TS, "FieldType": "float32"},
            {"FieldName": FIELD_UPDATE_TIME, "FieldType": "string"},
            {"FieldName": FIELD_UPDATE_TIME_TS, "FieldType": "float32"},
            {"FieldName": FIELD_LAST_CALL_TIME, "FieldType": "string"},
            {"FieldName": FIELD_LAST_CALL_TIME_TS, "FieldType": "float32"},
            # Extracted data fields
            {"FieldName": FIELD_ORIGINAL_ID, "FieldType": "string"},
            {"FieldName": FIELD_TITLE, "FieldType": "string"},
            {"FieldName": FIELD_SUMMARY, "FieldType": "string"},
            {"FieldName": FIELD_KEYWORDS, "FieldType": "string"},
            {"FieldName": FIELD_ENTITIES, "FieldType": "string"},
            {"FieldName": FIELD_CONTEXT_TYPE, "FieldType": "string"},
            {"FieldName": FIELD_CONFIDENCE, "FieldType": "float32"},
            {"FieldName": FIELD_IMPORTANCE, "FieldType": "float32"},
            {"FieldName": FIELD_SOURCE, "FieldType": "string"},
            # Properties fields
            {"FieldName": FIELD_IS_PROCESSED, "FieldType": "bool"},
            {"FieldName": FIELD_HAS_COMPRESSION, "FieldType": "bool"},
            {"FieldName": FIELD_ENABLE_MERGE, "FieldType": "bool"},
            {"FieldName": FIELD_IS_HAPPEND, "FieldType": "bool"},
            {"FieldName": FIELD_CALL_COUNT, "FieldType": "float32"},
            {"FieldName": FIELD_MERGE_COUNT, "FieldType": "float32"},
            {"FieldName": FIELD_DURATION_COUNT, "FieldType": "float32"},
            # Document tracking fields
            {"FieldName": FIELD_FILE_PATH, "FieldType": "string"},
            {"FieldName": FIELD_RAW_TYPE, "FieldType": "string"},
            {"FieldName": FIELD_RAW_ID, "FieldType": "string"},
            # Document content
            {"FieldName": FIELD_DOCUMENT, "FieldType": "string"},
            # Todo fields
            {"FieldName": FIELD_TODO_ID, "FieldType": "string"},
            {"FieldName": FIELD_CONTENT, "FieldType": "string"},
        ]
        
        data = {
            "CollectionName": collection_name,
            "Fields": fields,
            "PrimaryKey": "id",
            "Description": f"OpenContext {collection_name} collection",
        }
        
        result = self._client.console_request(
            action="CreateVikingdbCollection",
            data=data,
        )
        
        if result.get("ResponseMetadata", {}).get("Error"):
            logger.error(f"Failed to create collection {collection_name}: {result}")
            raise RuntimeError(f"Failed to create collection: {result}")
        
        logger.info(f"Created VikingDB collection: {collection_name}")
        
        # Wait for collection to be ready
        self._wait_for_collection_ready(collection_name)
    
    def _create_index(
        self,
        collection_name: str,
        index_name: str,
        index_type: str = "hnsw",
        distance_type: str = "ip",
    ) -> None:
        """
        Create a new index on a collection.
        
        Args:
            collection_name: Name of the collection
            index_name: Name of the index
            index_type: Index type (hnsw, flat, ivf)
            distance_type: Distance metric (ip, l2, cosine)
        """
        data = {
            "CollectionName": collection_name,
            "IndexName": index_name,
            "VectorIndex": {
                "IndexType": index_type,
                "Distance": distance_type,
                "Quant": "int8",
            },
            "CpuQuota": 2,
            "ShardCount": 1,
            "Description": f"Index for {collection_name}",
        }
        
        result = self._client.console_request(
            action="CreateVikingdbIndex",
            data=data,
        )
        
        if result.get("ResponseMetadata", {}).get("Error"):
            logger.error(f"Failed to create index {index_name}: {result}")
            raise RuntimeError(f"Failed to create index: {result}")
        
        logger.info(f"Created VikingDB index: {index_name}")
        
        # Wait for index to be ready
        self._wait_for_index_ready(collection_name, index_name)
    
    def _wait_for_collection_ready(
        self,
        collection_name: str,
        max_wait: int = 60,
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
                result = self._client.console_request(
                    action="GetVikingdbCollection",
                    data={"CollectionName": collection_name},
                )
                if not result.get("ResponseMetadata", {}).get("Error"):
                    status = result.get("Result", {}).get("Status", "")
                    if status == "RUNNING":
                        return True
            except Exception:
                pass
            time.sleep(2)
        
        logger.warning(f"Timeout waiting for collection {collection_name} to be ready")
        return False
    
    def _wait_for_index_ready(
        self,
        collection_name: str,
        index_name: str,
        max_wait: int = 120,
    ) -> bool:
        """
        Wait for index to be ready after creation.
        
        Args:
            collection_name: Name of collection
            index_name: Name of index
            max_wait: Maximum wait time in seconds
            
        Returns:
            True if index is ready, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                result = self._client.console_request(
                    action="GetVikingdbIndex",
                    data={
                        "CollectionName": collection_name,
                        "IndexName": index_name,
                    },
                )
                if not result.get("ResponseMetadata", {}).get("Error"):
                    status = result.get("Result", {}).get("Status", "")
                    if status == "RUNNING":
                        return True
            except Exception:
                pass
            time.sleep(2)
        
        logger.warning(f"Timeout waiting for index {index_name} to be ready")
        return False
    
    def get_name(self) -> str:
        return "vikingdb"
    
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
        Convert ProcessedContext to VikingDB data format.
        
        Args:
            context: ProcessedContext to convert
            
        Returns:
            Dictionary in VikingDB data format with all fields properly serialized
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
                fields[key] = float(value)
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
            raise RuntimeError("VikingDB backend not initialized")
        
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
            
            data_list = []
            for context in type_contexts:
                try:
                    vector = self._ensure_vectorized(context)
                    fields = self._context_to_doc_format(context)
                    
                    # Build data item for VikingDB
                    data_item = {
                        "id": context.id,
                        "vector": vector,
                    }
                    # Add all fields
                    data_item.update(fields)
                    data_list.append(data_item)
                    
                except Exception as e:
                    logger.exception(f"Failed to process context {context.id}: {e}")
                    continue
            
            if not data_list:
                continue
            
            # Batch upsert via data plane API
            try:
                result = self._client.data_request(
                    path="/api/collection/upsert_data",
                    data={
                        "collection_name": context_type,
                        "data": data_list,
                    }
                )
                
                if result.get("code") == 0:
                    for item in data_list:
                        stored_ids.append(item["id"])
                    logger.debug(f"Upserted {len(data_list)} docs to {context_type}")
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
            # Fetch data by ID via data plane API
            result = self._client.data_request(
                path="/api/collection/fetch_data",
                data={
                    "collection_name": context_type,
                    "ids": [id],
                }
            )
            
            if result.get("code") == 0:
                data = result.get("data", [])
                if data and len(data) > 0:
                    return self._doc_to_context(data[0], need_vector)
                    
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
            
            if ctx_type not in self._indexes:
                continue
            
            try:
                # Build filter
                filter_dict = self._build_filter_dict(filter, user_id, device_id, agent_id)
                
                # Use search API with empty vector for filter-only query
                # VikingDB requires using SearchByScalar for non-vector queries
                data = {
                    "collection_name": ctx_type,
                    "index_name": self._indexes[ctx_type],
                    "limit": limit + offset,
                }
                if filter_dict:
                    data["filter"] = filter_dict
                
                query_result = self._client.data_request(
                    path="/api/index/search_by_scalar",
                    data=data
                )
                
                if query_result.get("code") == 0:
                    output = query_result.get("data", [])
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
            result = self._client.data_request(
                path="/api/collection/del_data",
                data={
                    "collection_name": context_type,
                    "ids": ids,
                }
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
        
        # Build filter
        filter_dict = self._build_filter_dict(filters, user_id, device_id, agent_id)
        
        all_results = []
        
        for ctx_type in target_types:
            if ctx_type not in self._indexes:
                continue
            
            try:
                # Build search request
                data = {
                    "collection_name": ctx_type,
                    "index_name": self._indexes[ctx_type],
                    "search": {
                        "dense_vectors": [query_vector],
                        "limit": top_k,
                    }
                }
                if filter_dict:
                    data["search"]["filter"] = filter_dict
                
                result = self._client.data_request(
                    path="/api/index/search",
                    data=data
                )
                
                if result.get("code") == 0:
                    output = result.get("data", [])
                    # VikingDB returns results in nested structure
                    if output and len(output) > 0:
                        for item in output[0]:  # First query result
                            context = self._doc_to_context(item, need_vector)
                            if context:
                                score = item.get("score", 0.0)
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
        Convert VikingDB data to ProcessedContext.
        
        Args:
            doc: VikingDB data dict
            need_vector: Whether to include vector
            
        Returns:
            ProcessedContext if conversion successful, None otherwise
        """
        try:
            if not doc:
                return None
            
            # Get fields from doc (VikingDB returns flat structure)
            fields = dict(doc)
            
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
            original_id = fields.pop(FIELD_ORIGINAL_ID, None) or fields.pop("id", None)
            if not original_id:
                return None
            
            # Remove internal fields
            fields.pop("vector", None)
            fields.pop("id", None)
            fields.pop("score", None)
            
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
    
    def _build_filter_dict(
        self,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build VikingDB filter dictionary.
        
        VikingDB uses a structured filter format:
        {
            "op": "must",  # must, must_not, should
            "conditions": [
                {"field_name": "field", "op": "=", "value": "value"}
            ]
        }
        
        Args:
            filters: Additional filter conditions
            user_id: User ID filter
            device_id: Device ID filter
            agent_id: Agent ID filter
            
        Returns:
            Filter dictionary or None if no filters
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
            conditions.append({
                "field_name": FIELD_USER_ID,
                "op": "=",
                "value": user_id
            })
        if device_id:
            conditions.append({
                "field_name": FIELD_DEVICE_ID,
                "op": "=",
                "value": device_id
            })
        if agent_id:
            conditions.append({
                "field_name": FIELD_AGENT_ID,
                "op": "=",
                "value": agent_id
            })
        
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
                    # Handle comparison operators
                    op_mapping = {
                        '$gte': '>=',
                        '$lte': '<=',
                        '$gt': '>',
                        '$lt': '<',
                        '$eq': '=',
                        '$ne': '!=',
                    }
                    for op, op_symbol in op_mapping.items():
                        if op in value:
                            op_value = value[op]
                            if is_time_field:
                                ts = self._parse_time_to_timestamp(op_value)
                                if ts is not None:
                                    conditions.append({
                                        "field_name": filter_key,
                                        "op": op_symbol,
                                        "value": ts
                                    })
                            else:
                                conditions.append({
                                    "field_name": filter_key,
                                    "op": op_symbol,
                                    "value": op_value
                                })
                elif isinstance(value, list):
                    # Handle IN operator
                    conditions.append({
                        "field_name": filter_key,
                        "op": "in",
                        "value": value
                    })
                elif isinstance(value, str):
                    if is_time_field:
                        ts = self._parse_time_to_timestamp(value)
                        if ts is not None:
                            # For exact time match, use range
                            conditions.append({
                                "field_name": filter_key,
                                "op": ">=",
                                "value": ts - 0.5
                            })
                            conditions.append({
                                "field_name": filter_key,
                                "op": "<=",
                                "value": ts + 0.5
                            })
                        else:
                            conditions.append({
                                "field_name": key,
                                "op": "=",
                                "value": value
                            })
                    else:
                        conditions.append({
                            "field_name": filter_key,
                            "op": "=",
                            "value": value
                        })
                elif isinstance(value, bool):
                    conditions.append({
                        "field_name": filter_key,
                        "op": "=",
                        "value": value
                    })
                else:
                    conditions.append({
                        "field_name": filter_key,
                        "op": "=",
                        "value": value
                    })
        
        if not conditions:
            return None
        
        filter_dict = {
            "op": "must",
            "conditions": conditions
        }
        logger.debug(f"Built VikingDB filter: {filter_dict}")
        return filter_dict
    
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
            result = self._client.console_request(
                action="GetVikingdbCollection",
                data={"CollectionName": context_type}
            )
            if not result.get("ResponseMetadata", {}).get("Error"):
                return int(result.get("Result", {}).get("Stat", {}).get("DataCount", 0))
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
            data_item = {
                "id": str(todo_id),
                "vector": list(embedding),
                FIELD_TODO_ID: str(todo_id),
                FIELD_CONTENT: content,
                FIELD_CREATED_AT: datetime.datetime.now().isoformat(),
                FIELD_CREATED_AT_TS: datetime.datetime.now().timestamp(),
                FIELD_USER_ID: user_id or "",
                FIELD_DEVICE_ID: device_id or "",
                FIELD_AGENT_ID: agent_id or "",
            }
            
            result = self._client.data_request(
                path="/api/collection/upsert_data",
                data={
                    "collection_name": TODO_COLLECTION,
                    "data": [data_item],
                }
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
        
        if TODO_COLLECTION not in self._indexes:
            return []
        
        try:
            filter_dict = self._build_filter_dict(None, user_id, device_id, agent_id)
            
            data = {
                "collection_name": TODO_COLLECTION,
                "index_name": self._indexes[TODO_COLLECTION],
                "search": {
                    "dense_vectors": [list(embedding)],
                    "limit": top_k,
                }
            }
            if filter_dict:
                data["search"]["filter"] = filter_dict
            
            result = self._client.data_request(
                path="/api/index/search",
                data=data
            )
            
            if result.get("code") == 0:
                output = result.get("data", [])
                results = []
                if output and len(output) > 0:
                    for item in output[0]:
                        todo_id = item.get(FIELD_TODO_ID) or item.get("id")
                        content = item.get(FIELD_CONTENT, "")
                        score = item.get("score", 0.0)
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
            result = self._client.data_request(
                path="/api/collection/del_data",
                data={
                    "collection_name": TODO_COLLECTION,
                    "ids": [str(todo_id)],
                }
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
        logger.info("VikingDB backend closed")
