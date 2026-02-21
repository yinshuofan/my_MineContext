
# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
VikingDB vector storage backend - Volcengine VikingDB Service (HTTP API)
https://www.volcengine.com/docs/84313/1254458

This implementation uses HTTP API with Volcengine V4 signature for authentication.
Uses a single collection with context_type field filtering to reduce operational costs.
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

# Single collection name for all context types
DEFAULT_COLLECTION_NAME = "opencontext"
DEFAULT_INDEX_NAME = "opencontext_index"

# Field constants
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

# Hierarchy and document overwrite fields
FIELD_HIERARCHY_LEVEL = "hierarchy_level"
FIELD_PARENT_ID = "parent_id"
FIELD_TIME_BUCKET = "time_bucket"
FIELD_SOURCE_FILE_KEY = "source_file_key"
FIELD_CHILDREN_IDS = "children_ids"

# Data type field (to distinguish between context and todo)
FIELD_DATA_TYPE = "data_type"
DATA_TYPE_CONTEXT = "context"
DATA_TYPE_TODO = "todo"

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
        payload_hash = self._get_payload_hash(body)
        headers_to_sign = {
            'Host': host,
            'Content-Type': 'application/json',
            'X-Date': x_date,
            'X-Content-Sha256': payload_hash,
        }
        headers_to_sign.update(headers)
        
        # Build canonical request
        canonical_uri = self._get_canonical_uri(path)
        canonical_query_string = self._get_canonical_query_string(params or {})
        canonical_headers, signed_headers = self._get_canonical_headers(headers_to_sign)
        
        canonical_request = '\n'.join([
            method.upper(),
            canonical_uri,
            canonical_query_string,
            canonical_headers,
            signed_headers,
            payload_hash,
        ])
        
        # Build string to sign
        credential_scope = f"{date_stamp}/{self.region}/{self.service}/request"
        hashed_canonical_request = hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        string_to_sign = '\n'.join([
            'HMAC-SHA256',
            x_date,
            credential_scope,
            hashed_canonical_request,
        ])
        
        # Calculate signature
        signing_key = self._get_signing_key(date_stamp)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        # Build authorization header
        authorization = (
            f"HMAC-SHA256 Credential={self.access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        
        # Return headers with authorization
        result_headers = dict(headers_to_sign)
        result_headers['Authorization'] = authorization
        
        return result_headers


class VikingDBHTTPClient:
    """
    HTTP client for VikingDB API.
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
        
        # Initialize sync session
        self._sync_session: Optional[requests.Session] = None
        self._session_lock = threading.Lock()
        
        # Initialize async session (lazy)
        self._async_session: Optional[aiohttp.ClientSession] = None
        self._async_lock = asyncio.Lock() if asyncio else None
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=10)
    
    def _create_sync_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self._max_retries,
            backoff_factor=self._retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self._max_connections_per_host,
            pool_maxsize=self._max_connections,
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_sync_session(self) -> requests.Session:
        """Get or create sync session."""
        if self._sync_session is None:
            with self._session_lock:
                if self._sync_session is None:
                    self._sync_session = self._create_sync_session()
        return self._sync_session
    
    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async session."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is not installed")
        
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
    
    def _sync_request(
        self,
        method: str,
        host: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a synchronous HTTP request.
        
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
        
        session = self._get_sync_session()
        response = session.request(
            method=method,
            url=url,
            headers=headers,
            data=body,
            timeout=self._timeout,
        )

        # For 404 status, return the response JSON so caller can handle NotFound errors
        # This is needed for GetVikingdbCollection/GetVikingdbIndex to check if resource exists
        if response.status_code == 404:
            try:
                return response.json()
            except Exception:
                error_msg = f"API request failed with status {response.status_code}: {response.text[:500]}"
                logger.error(error_msg)
                raise Exception(error_msg)
        
        if response.status_code != 200:
            error_msg = f"API request failed with status {response.status_code}: {response.text[:500]}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {response.text[:500]}")
            raise
    
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
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    loop.create_task(self._async_session.close())
                else:
                    try:
                        asyncio.run(self._async_session.close())
                    except RuntimeError:
                        # If we can't get a loop and can't run a new one (e.g. nested), just ignore
                        pass
            except Exception:
                pass
        if self._executor:
            self._executor.shutdown(wait=False)


class VikingDBBackend(IVectorStorageBackend):
    """
    VikingDB vector storage backend implementation.
    
    Uses Volcengine VikingDB service for vector storage and similarity search.
    Uses a single collection with context_type field filtering to reduce costs.
    Implements the IVectorStorageBackend interface for compatibility with
    the opencontext storage system.
    """
    
    def __init__(self):
        self._client: Optional[VikingDBHTTPClient] = None
        self._collection_name: str = DEFAULT_COLLECTION_NAME
        self._index_name: str = DEFAULT_INDEX_NAME
        self._dimension: int = 0
        self._initialized: bool = False
        self._config: Dict[str, Any] = {}
        self._collection_ready: bool = False
        self._index_ready: bool = False
    
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
                - collection_name: Collection name (optional, default: opencontext)
                - index_name: Index name (optional, default: opencontext_index)
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
            
            # Get context types to create collections for
            context_types = config.get("context_types", [])
            if not context_types:
                # Default context types
                context_types = [ct.value for ct in ContextType]
            
            # Add todo collection
            # if TODO_COLLECTION not in context_types:
            #     context_types.append(TODO_COLLECTION)
            
            # Ensure single collection and index exist
            index_type = vikingdb_config.get("index_type", "hnsw")
            distance_type = vikingdb_config.get("distance_type", "ip")
            
            self._ensure_collection_and_index(
                dimension=self._dimension,
                index_type=index_type,
                distance_type=distance_type,
            )
            
            self._initialized = True
            logger.info(f"VikingDB backend initialized with single collection: {self._collection_name}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize VikingDB backend: {e}")
            return False
    
    def _ensure_collection_and_index(
        self,
        dimension: int,
        index_type: str = "hnsw",
        distance_type: str = "cosine",
    ) -> None:
        """
        Ensure single collection and index exist, create if not.
        
        Args:
            dimension: Vector dimension
            index_type: Index type (hnsw, flat, ivf)
            distance_type: Distance metric (ip, l2, cosine)
        """
        try:
            # Check if collection exists
            result = self._client.console_request(
                action="GetVikingdbCollection",
                data={"CollectionName": self._collection_name},
            )
            
            if result.get("ResponseMetadata", {}).get("Error"):
                error_code = result["ResponseMetadata"]["Error"].get("Code", "")
                if "NotExist" in error_code or "NotFound" in error_code:
                    # Create collection
                    self._create_collection(dimension)
                else:
                    logger.error(f"Failed to get collection {self._collection_name}: {result}")
                    raise RuntimeError(f"Failed to get collection: {result}")
            else:
                logger.debug(f"VikingDB collection already exists: {self._collection_name}")
            
            self._collection_ready = True
            
            # Check if index exists
            result = self._client.console_request(
                action="GetVikingdbIndex",
                data={
                    "CollectionName": self._collection_name,
                    "IndexName": self._index_name,
                },
            )
            
            if result.get("ResponseMetadata", {}).get("Error"):
                error_code = result["ResponseMetadata"]["Error"].get("Code", "")
                if "NotExist" in error_code or "NotFound" in error_code:
                    # Create index
                    self._create_index(
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
    
    def _create_collection(self, dimension: int) -> None:
        """
        Create the single collection with all necessary fields.
        
        Args:
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
            # Data type field (context or todo)
            {"FieldName": FIELD_DATA_TYPE, "FieldType": "string"},
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
            # Hierarchy and document overwrite fields
            {"FieldName": FIELD_HIERARCHY_LEVEL, "FieldType": "float32"},
            {"FieldName": FIELD_PARENT_ID, "FieldType": "string"},
            {"FieldName": FIELD_TIME_BUCKET, "FieldType": "string"},
            {"FieldName": FIELD_SOURCE_FILE_KEY, "FieldType": "string"},
            {"FieldName": FIELD_CHILDREN_IDS, "FieldType": "string"},
            # Document content
            {"FieldName": FIELD_DOCUMENT, "FieldType": "string"},
            # Todo fields
            {"FieldName": FIELD_TODO_ID, "FieldType": "string"},
            {"FieldName": FIELD_CONTENT, "FieldType": "string"},
        ]
        
        data = {
            "CollectionName": self._collection_name,
            "Fields": fields,
            "Description": "OpenContext unified collection for all context types and todos",
        }
        
        logger.info(f"Creating VikingDB collection: {self._collection_name}")
        result = self._client.console_request(
            action="CreateVikingdbCollection",
            data=data,
        )
        
        if result.get("ResponseMetadata", {}).get("Error"):
            error = result["ResponseMetadata"]["Error"]
            raise RuntimeError(f"Failed to create collection: {error}")
        
        logger.info(f"VikingDB collection created: {self._collection_name}")
    
    def _create_index(
        self,
        index_type: str = "hnsw",
        distance_type: str = "ip",
    ) -> None:
        """
        Create index for the collection.
        
        Args:
            index_type: Index type (hnsw, flat, ivf)
            distance_type: Distance metric (ip, l2, cosine)
        """
        # Build index config based on type
        vector_index = {
            "IndexType": index_type.upper(),
            "Distance": distance_type,
        }
        
        # Add HNSW specific params
        if index_type.lower() == "hnsw":
            vector_index["HnswM"] = 32
            vector_index["HnswCef"] = 64
            vector_index["HnswSef"] = 800
        
        data = {
            "CollectionName": self._collection_name,
            "IndexName": self._index_name,
            "VectorIndex": vector_index,
            # Add scalar index for filtering fields
            # Note: All fields that need range/enumeration filtering must be included here
            # - Range filtering (int64, float32): time timestamp fields
            # - Enumeration filtering (string, int64, bool, list): identity and type fields
            "ScalarIndex": [
                # Identity and type fields (string - enumeration filtering)
                FIELD_DATA_TYPE,
                FIELD_CONTEXT_TYPE,
                FIELD_USER_ID,
                FIELD_DEVICE_ID,
                FIELD_AGENT_ID,
                FIELD_SOURCE,
                FIELD_RAW_TYPE,
                FIELD_RAW_ID,
                FIELD_ORIGINAL_ID,
                FIELD_TODO_ID,
                # Hierarchy and document overwrite fields
                FIELD_SOURCE_FILE_KEY,
                FIELD_TIME_BUCKET,
                FIELD_PARENT_ID,
                # Boolean fields (bool - enumeration filtering)
                FIELD_IS_PROCESSED,
                FIELD_HAS_COMPRESSION,
                FIELD_ENABLE_MERGE,
                FIELD_IS_HAPPEND,
                # Time timestamp fields (float32 - range filtering)
                FIELD_CREATED_AT_TS,
                FIELD_CREATE_TIME_TS,
                FIELD_EVENT_TIME_TS,
                FIELD_UPDATE_TIME_TS,
                FIELD_LAST_CALL_TIME_TS,
                # Numeric fields (float32 - range filtering)
                FIELD_CONFIDENCE,
                FIELD_IMPORTANCE,
                FIELD_CALL_COUNT,
                FIELD_MERGE_COUNT,
                FIELD_DURATION_COUNT,
                FIELD_HIERARCHY_LEVEL,
            ],
            "Description": f"Index for {self._collection_name}",
        }
        
        logger.info(f"Creating VikingDB index: {self._index_name}")
        result = self._client.console_request(
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
    
    def get_collection_names(self) -> List[str]:
        """Get all collection names managed by this backend."""
        if self._collection_ready:
            return [ct for ct in ContextType]
        return []
    
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized
    
    def _ensure_vectorized(self, context: ProcessedContext) -> List[float]:
        """
        Ensure context has a vector, generate if needed.
        
        Args:
            context: ProcessedContext to vectorize
            
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
        
        # Add data type field
        fields[FIELD_DATA_TYPE] = DATA_TYPE_CONTEXT
        
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
        Batch store ProcessedContexts to single collection.
        
        Args:
            contexts: List of ProcessedContext to store
            
        Returns:
            List of stored context IDs
        """
        if not self._initialized:
            raise RuntimeError("VikingDB backend not initialized")
        
        data_list = []
        logger.debug(f"Upserting contexts:{contexts}")
        for context in contexts:
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
            return []
        
        stored_ids = []
        
        # Batch upsert via data plane API
        try:
            result = self._client.data_request(
                path="/api/vikingdb/data/upsert",
                data={
                    "collection_name": self._collection_name,
                    "data": data_list,
                }
            )
            
            if result.get("code") == "Success":
                for item in data_list:
                    stored_ids.append(item["id"])
                logger.debug(f"[VikingDB] Upserted {len(data_list)} docs to {self._collection_name}")
            else:
                logger.error(
                    f"[VikingDB] Failed to upsert to {self._collection_name}: {result.get('message')}"
                )
                
        except Exception as e:
            logger.exception(f"Failed to upsert contexts: {e}")
        
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
            context_type: Type of context (used for filtering)
            need_vector: Whether to include vector in result
            
        Returns:
            ProcessedContext if found, None otherwise
        """
        if not self._initialized:
            return None
        
        try:
            # Fetch data by ID via data plane API
            result = self._client.data_request(
                path="/api/vikingdb/data/fetch_in_collection",
                data={
                    "collection_name": self._collection_name,
                    "ids": [id],
                }
            )
            
            if result.get("code") == "Success":
                fetch_result = result.get("result", {}).get("fetch", [])
                if fetch_result and len(fetch_result) > 0:
                    item = fetch_result[0]
                    # Reconstruct doc from id and fields
                    doc = {"id": item.get("id")}
                    doc.update(item.get("fields", {}))
                    # Verify context_type matches
                    if doc.get(FIELD_CONTEXT_TYPE) == context_type:
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
        Get all ProcessedContexts, optionally filtered by context_type.
        
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
        
        # Determine target context types
        if context_types:
            target_types = context_types
        else:
            # Get all context types
            target_types = [ct.value for ct in ContextType]
        
        for ctx_type in target_types:
            try:
                # Build filter with context_type
                filter_dict = self._build_filter_dict(
                    filters=filter,
                    user_id=user_id,
                    device_id=device_id,
                    agent_id=agent_id,
                    context_type=ctx_type,
                    data_type=DATA_TYPE_CONTEXT,
                )
                
                # Use scalar search API with filter
                # Note: search_by_scalar requires a numeric field for sorting
                # Using created_at_ts as the sort field
                data = {
                    "collection_name": self._collection_name,
                    "index_name": self._index_name,
                    "limit": limit,
                    "offset": offset,
                    "field": FIELD_CREATED_AT_TS,
                    "order": "desc",
                }
                if filter_dict:
                    data["filter"] = filter_dict
                
                query_result = self._client.data_request(
                    path="/api/vikingdb/data/search/scalar",
                    data=data
                )
                
                if query_result.get("code") == "Success":
                    output = query_result.get("result", {}).get("data", [])
                    # Apply offset
                    # VikingDB API already handles offset and limit, so we don't need to slice by offset again
                    # if offset > 0:
                    #     output = output[offset:]
                    if len(output) > limit:
                        output = output[:limit]
                    
                    contexts = []
                    for item in output:
                        # Reconstruct doc from id and fields
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
    
    def delete_processed_context(self, id: str, context_type: str) -> bool:
        """
        Delete a specific ProcessedContext.
        
        Args:
            id: Context ID to delete
            context_type: Type of context (not used in single collection mode)
            
        Returns:
            True if successful, False otherwise
        """
        return self.delete_contexts([id], context_type)
    
    def delete_contexts(self, ids: List[str], context_type: str) -> bool:
        """
        Delete multiple contexts.
        
        Args:
            ids: List of context IDs to delete
            context_type: Type of context (not used in single collection mode)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False
        
        try:
            result = self._client.data_request(
                path="/api/vikingdb/data/delete",
                data={
                    "collection_name": self._collection_name,
                    "ids": ids,
                }
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
        Vector similarity search with context_type filtering.
        
        Args:
            query: Query vectorize object
            top_k: Maximum number of results
            context_types: List of context types to search (filter by context_type field)
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
        
        # Get query vector
        query_vector = None
        if query.vector and len(query.vector) > 0:
            query_vector = list(query.vector)
        else:
            if query.text:
                do_vectorize(query)
                query_vector = list(query.vector) if query.vector else None
        
        if not query_vector:
            all_contexts = self.get_all_processed_contexts(
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
        
        # Build filter with context_types
        filter_dict = self._build_filter_dict(
            filters=filters,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            context_types=context_types,
            data_type=DATA_TYPE_CONTEXT,
        )
        
        all_results = []
        
        try:
            # Build search request
            data = {
                "collection_name": self._collection_name,
                "index_name": self._index_name,
                "dense_vector": query_vector,
                "limit": top_k,
            }
            if filter_dict:
                data["filter"] = filter_dict
            
            result = self._client.data_request(
                path="/api/vikingdb/data/search/vector",
                data=data
            )
            
            if result.get("code") == "Success":
                output = result.get("result", {}).get("data", [])
                # VikingDB returns results in flat list
                for item in output:
                    # Reconstruct doc from id and fields
                    doc = {"id": item.get("id")}
                    doc.update(item.get("fields", {}))
                    context = self._doc_to_context(doc, need_vector)
                    if context:
                        score = item.get("score", 0.0)
                        all_results.append((context, score))
                            
        except Exception as e:
            logger.exception(f"Vector search failed: {e}")
        
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
            fields.pop(FIELD_DATA_TYPE, None)
            
            # Time fields that should be datetime
            TIME_FIELDS = {
                'create_time', 'event_time', 'update_time', 'last_call_time'
            }
            
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
                
                # Handle time fields - skip invalid values like 'default'
                if key in TIME_FIELDS:
                    if isinstance(val, str):
                        # Skip invalid time values
                        if val in ('default', '', 'null', 'None'):
                            continue
                        # Try to parse ISO format datetime string
                        try:
                            val = datetime.datetime.fromisoformat(val.replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            # If parsing fails, skip this field
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
        context_type: Optional[str] = None,
        context_types: Optional[List[str]] = None,
        data_type: Optional[str] = None,
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
            context_type: Single context type filter
            context_types: List of context types filter (OR condition)
            data_type: Data type filter (context or todo)
            
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
        
        # Fields that support range operator (must be in ScalarIndex and be int64/float32 type)
        # Based on VikingDB documentation: range operator only supports int64 and float32 fields
        # that are included in ScalarIndex
        # All timestamp and numeric fields are now included in ScalarIndex in _create_index method
        RANGE_SUPPORTED_FIELDS = {
            # Time timestamp fields (float32)
            FIELD_CREATED_AT_TS,
            FIELD_CREATE_TIME_TS,
            FIELD_EVENT_TIME_TS,
            FIELD_UPDATE_TIME_TS,
            FIELD_LAST_CALL_TIME_TS,
            # Numeric fields (float32)
            FIELD_CONFIDENCE,
            FIELD_IMPORTANCE,
            FIELD_CALL_COUNT,
            FIELD_MERGE_COUNT,
            FIELD_DURATION_COUNT,
            FIELD_HIERARCHY_LEVEL,
        }
        
        # Add data type filter using "must" operator
        if data_type:
            conditions.append({
                "op": "must",
                "field": FIELD_DATA_TYPE,
                "conds": [data_type]
            })
        
        # Add context type filter (single) using "must" operator
        if context_type:
            conditions.append({
                "op": "must",
                "field": FIELD_CONTEXT_TYPE,
                "conds": [context_type]
            })
        
        # Add context types filter (multiple - OR condition) using "must" operator
        if context_types and len(context_types) > 0:
            conditions.append({
                "op": "must",
                "field": FIELD_CONTEXT_TYPE,
                "conds": context_types
            })
        
        # Add user identity filters using "must" operator
        if user_id:
            conditions.append({
                "op": "must",
                "field": FIELD_USER_ID,
                "conds": [user_id]
            })
        if device_id:
            conditions.append({
                "op": "must",
                "field": FIELD_DEVICE_ID,
                "conds": [device_id]
            })
        if agent_id:
            conditions.append({
                "op": "must",
                "field": FIELD_AGENT_ID,
                "conds": [agent_id]
            })
        
        # Add custom filters
        if filters:
            for key, value in filters.items():
                if key in ('context_type', 'context_types', 'entities', 'data_type'):
                    continue
                if value is None:
                    continue
                
                is_time_field = key in TIME_FIELD_MAPPING or key.endswith('_ts')
                filter_key = TIME_FIELD_MAPPING.get(key, key)
                
                if isinstance(value, dict):
                    # Check if this field supports range operator
                    supports_range = filter_key in RANGE_SUPPORTED_FIELDS
                    
                    if supports_range:
                        # Handle comparison operators using "range" operator
                        range_filter = {
                            "op": "range",
                            "field": filter_key,
                        }
                        op_mapping = {
                            '$gte': 'gte',
                            '$lte': 'lte',
                            '$gt': 'gt',
                            '$lt': 'lt',
                        }
                        has_range = False
                        for op, range_key in op_mapping.items():
                            if op in value:
                                op_value = value[op]
                                if is_time_field:
                                    ts = self._parse_time_to_timestamp(op_value)
                                    if ts is not None:
                                        range_filter[range_key] = ts
                                        has_range = True
                                else:
                                    range_filter[range_key] = op_value
                                    has_range = True
                        if has_range:
                            conditions.append(range_filter)
                    else:
                        # For fields that don't support range, skip range operators
                        # and log a warning
                        range_ops = {'$gte', '$lte', '$gt', '$lt'}
                        if any(op in value for op in range_ops):
                            logger.warning(
                                f"Field '{filter_key}' does not support range operator. "
                                f"Only fields in ScalarIndex with int64/float32 type support range. "
                                f"Skipping range filter for this field."
                            )
                    
                    # Handle equality operators
                    if '$eq' in value:
                        conditions.append({
                            "op": "must",
                            "field": filter_key,
                            "conds": [value['$eq']]
                        })
                    if '$ne' in value:
                        conditions.append({
                            "op": "must_not",
                            "field": filter_key,
                            "conds": [value['$ne']]
                        })
                elif isinstance(value, list):
                    # Handle IN operator using "must" with multiple conds
                    conditions.append({
                        "op": "must",
                        "field": filter_key,
                        "conds": value
                    })
                elif isinstance(value, str):
                    if is_time_field and filter_key in RANGE_SUPPORTED_FIELDS:
                        ts = self._parse_time_to_timestamp(value)
                        if ts is not None:
                            # For exact time match, use range (only for supported fields)
                            conditions.append({
                                "op": "range",
                                "field": filter_key,
                                "gte": ts - 0.5,
                                "lte": ts + 0.5
                            })
                        else:
                            conditions.append({
                                "op": "must",
                                "field": key,
                                "conds": [value]
                            })
                    else:
                        conditions.append({
                            "op": "must",
                            "field": filter_key,
                            "conds": [value]
                        })
                elif isinstance(value, bool):
                    conditions.append({
                        "op": "must",
                        "field": filter_key,
                        "conds": [value]
                    })
                else:
                    conditions.append({
                        "op": "must",
                        "field": filter_key,
                        "conds": [value]
                    })
        
        if not conditions:
            return None
        
        # Wrap all conditions in an "and" operator
        if len(conditions) == 1:
            filter_dict = conditions[0]
        else:
            filter_dict = {
                "op": "and",
                "conds": conditions
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
        
        try:
            # Use scalar search with count
            filter_dict = self._build_filter_dict(
                context_type=context_type,
                data_type=DATA_TYPE_CONTEXT,
            )
            
            data = {
                "collection_name": self._collection_name,
                "index_name": self._index_name,
                "limit": 1,  # We just need the count
                "field": FIELD_CREATED_AT_TS,
                "order": "desc",
            }
            if filter_dict:
                data["filter"] = filter_dict
            
            result = self._client.data_request(
                path="/api/vikingdb/data/search/scalar",
                data=data
            )
            
            if result.get("code") == "Success":
                # Note: VikingDB returns filter_matched_count if filter is provided
                result_data = result.get("result", {})
                if "filter_matched_count" in result_data:
                    return result_data["filter_matched_count"]
                # Fallback to total_return_count
                return result_data.get("total_return_count", 0)
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
        for ct in ContextType:
            counts[ct.value] = self.get_processed_context_count(ct.value)
        return counts
    
    def delete_by_source_file(self, source_file_key: str, user_id: Optional[str] = None) -> bool:
        """
        Delete all chunks belonging to a source file (for document overwrite).

        This method searches for all records matching the source_file_key
        (and optionally user_id), then deletes them.

        Args:
            source_file_key: Source file key (format: "user_id:file_path")
            user_id: Optional user identifier for additional filtering

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False

        try:
            # Build filter to find all chunks with this source_file_key
            conditions = [
                {
                    "op": "must",
                    "field": FIELD_SOURCE_FILE_KEY,
                    "conds": [source_file_key],
                },
                {
                    "op": "must",
                    "field": FIELD_DATA_TYPE,
                    "conds": [DATA_TYPE_CONTEXT],
                },
            ]

            if user_id:
                conditions.append({
                    "op": "must",
                    "field": FIELD_USER_ID,
                    "conds": [user_id],
                })

            if len(conditions) == 1:
                filter_dict = conditions[0]
            else:
                filter_dict = {
                    "op": "and",
                    "conds": conditions,
                }

            # Search for all matching records to get their IDs
            # Use scalar search to find records by source_file_key
            data = {
                "collection_name": self._collection_name,
                "index_name": self._index_name,
                "limit": 10000,  # Large limit to get all chunks of the document
                "field": FIELD_CREATED_AT_TS,
                "order": "desc",
                "filter": filter_dict,
            }

            result = self._client.data_request(
                path="/api/vikingdb/data/search/scalar",
                data=data,
            )

            if result.get("code") != "Success":
                logger.error(
                    f"Failed to search for source_file_key '{source_file_key}': "
                    f"{result.get('message')}"
                )
                return False

            output = result.get("result", {}).get("data", [])
            if not output:
                logger.debug(
                    f"No records found for source_file_key '{source_file_key}'"
                )
                return True  # Nothing to delete

            # Collect all IDs to delete
            ids_to_delete = [item.get("id") for item in output if item.get("id")]

            if not ids_to_delete:
                return True

            # Delete all matching records
            delete_result = self._client.data_request(
                path="/api/vikingdb/data/delete",
                data={
                    "collection_name": self._collection_name,
                    "ids": ids_to_delete,
                },
            )

            if delete_result.get("code") == "Success":
                logger.info(
                    f"Deleted {len(ids_to_delete)} records for "
                    f"source_file_key '{source_file_key}'"
                )
                return True
            else:
                logger.error(
                    f"Failed to delete records for source_file_key "
                    f"'{source_file_key}': {delete_result.get('message')}"
                )
                return False

        except Exception as e:
            logger.exception(
                f"Failed to delete by source_file_key '{source_file_key}': {e}"
            )
            return False

    def search_by_hierarchy(
        self,
        context_type: str,
        hierarchy_level: int,
        time_bucket_start: Optional[str] = None,
        time_bucket_end: Optional[str] = None,
        user_id: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Tuple[ProcessedContext, float]]:
        """
        Search contexts by hierarchy level and time bucket range.

        Used for event hierarchical retrieval. Searches for records matching
        the given context_type and hierarchy_level, optionally filtering by
        a time bucket range.

        Args:
            context_type: Context type to search
            hierarchy_level: Hierarchy level (0=original, 1=daily, 2=weekly, 3=monthly)
            time_bucket_start: Start of time bucket range (inclusive), e.g. "2026-02-01"
            time_bucket_end: End of time bucket range (inclusive), e.g. "2026-02-28"
            user_id: User identifier for multi-user filtering
            top_k: Maximum number of results

        Returns:
            List of (context, score) tuples, sorted by time bucket
        """
        if not self._initialized:
            return []

        try:
            # Build filter conditions
            conditions = [
                {
                    "op": "must",
                    "field": FIELD_DATA_TYPE,
                    "conds": [DATA_TYPE_CONTEXT],
                },
                {
                    "op": "must",
                    "field": FIELD_CONTEXT_TYPE,
                    "conds": [context_type],
                },
                {
                    "op": "range",
                    "field": FIELD_HIERARCHY_LEVEL,
                    "gte": float(hierarchy_level),
                    "lte": float(hierarchy_level),
                },
            ]

            if user_id:
                conditions.append({
                    "op": "must",
                    "field": FIELD_USER_ID,
                    "conds": [user_id],
                })

            # For time bucket filtering, VikingDB string fields support enumeration
            # but not range. We use "must" with the time_bucket field if only one
            # boundary is specified. For a range, we need to fetch and filter in code
            # since string range filtering is not supported.
            # However, if both start and end are specified, we fetch a large set and
            # filter in Python.
            time_bucket_filter_start = time_bucket_start
            time_bucket_filter_end = time_bucket_end

            if time_bucket_start and not time_bucket_end:
                # Only start specified - we'll filter in Python after fetch
                pass
            elif time_bucket_end and not time_bucket_start:
                # Only end specified - we'll filter in Python after fetch
                pass
            elif time_bucket_start and time_bucket_end and time_bucket_start == time_bucket_end:
                # Exact match
                conditions.append({
                    "op": "must",
                    "field": FIELD_TIME_BUCKET,
                    "conds": [time_bucket_start],
                })
                # Clear range filters since exact match is handled
                time_bucket_filter_start = None
                time_bucket_filter_end = None

            if len(conditions) == 1:
                filter_dict = conditions[0]
            else:
                filter_dict = {
                    "op": "and",
                    "conds": conditions,
                }

            # Use scalar search to find matching records
            # Fetch more than top_k if we need to do in-code time bucket filtering
            fetch_limit = top_k
            if time_bucket_filter_start or time_bucket_filter_end:
                fetch_limit = max(top_k * 5, 100)  # Over-fetch for in-code filtering

            data = {
                "collection_name": self._collection_name,
                "index_name": self._index_name,
                "limit": fetch_limit,
                "field": FIELD_CREATED_AT_TS,
                "order": "desc",
                "filter": filter_dict,
            }

            result = self._client.data_request(
                path="/api/vikingdb/data/search/scalar",
                data=data,
            )

            if result.get("code") != "Success":
                logger.error(
                    f"Failed to search by hierarchy: {result.get('message')}"
                )
                return []

            output = result.get("result", {}).get("data", [])

            results = []
            for item in output:
                doc = {"id": item.get("id")}
                doc.update(item.get("fields", {}))

                # Apply in-code time bucket range filtering if needed
                if time_bucket_filter_start or time_bucket_filter_end:
                    item_time_bucket = doc.get(FIELD_TIME_BUCKET, "")
                    if not item_time_bucket:
                        continue
                    if time_bucket_filter_start and item_time_bucket < time_bucket_filter_start:
                        continue
                    if time_bucket_filter_end and item_time_bucket > time_bucket_filter_end:
                        continue

                context = self._doc_to_context(doc, need_vector=False)
                if context:
                    # Use 1.0 as default score for scalar search results
                    results.append((context, 1.0))

            # Limit to top_k
            return results[:top_k]

        except Exception as e:
            logger.exception(f"Failed to search by hierarchy: {e}")
            return []

    def get_by_ids(
        self, ids: List[str], context_type: Optional[str] = None
    ) -> List[ProcessedContext]:
        """
        Get contexts by their IDs.

        Used for drill-down from hierarchy summaries to retrieve specific
        child records.

        Args:
            ids: List of context IDs to retrieve
            context_type: Optional context type for additional validation

        Returns:
            List of ProcessedContext objects
        """
        if not self._initialized:
            return []

        if not ids:
            return []

        try:
            # Fetch data by IDs via data plane API
            result = self._client.data_request(
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

                # Optionally validate context_type
                if context_type and doc.get(FIELD_CONTEXT_TYPE) != context_type:
                    continue

                context = self._doc_to_context(doc, need_vector=False)
                if context:
                    contexts.append(context)

            return contexts

        except Exception as e:
            logger.exception(f"Failed to get contexts by IDs: {e}")
            return []

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
        Store a todo embedding in the unified collection.
        
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
                "id": f"todo_{todo_id}",  # Prefix to avoid ID collision
                "vector": list(embedding),
                FIELD_DATA_TYPE: DATA_TYPE_TODO,
                FIELD_TODO_ID: str(todo_id),
                FIELD_CONTENT: content,
                FIELD_CREATED_AT: datetime.datetime.now().isoformat(),
                FIELD_CREATED_AT_TS: datetime.datetime.now().timestamp(),
                FIELD_USER_ID: user_id or "",
                FIELD_DEVICE_ID: device_id or "",
                FIELD_AGENT_ID: agent_id or "",
            }
            
            result = self._client.data_request(
                path="/api/vikingdb/data/upsert",
                data={
                    "collection_name": self._collection_name,
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
        Search for similar todos in the unified collection.
        
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
            filter_dict = self._build_filter_dict(
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
                data_type=DATA_TYPE_TODO,
            )
            
            data = {
                "collection_name": self._collection_name,
                "index_name": self._index_name,
                "dense_vector": list(embedding),
                "limit": top_k,
            }
            if filter_dict:
                data["filter"] = filter_dict
            
            result = self._client.data_request(
                path="/api/vikingdb/data/search/vector",
                data=data
            )
            
            if result.get("code") == "Success":
                output = result.get("result", {}).get("data", [])
                results = []
                for item in output:
                    fields = item.get("fields", {})
                    todo_id_val = fields.get(FIELD_TODO_ID) or item.get("id", "").replace("todo_", "")
                    content = fields.get(FIELD_CONTENT, "")
                    score = item.get("score", 0.0)
                    results.append((todo_id_val, content, score))
                return results
                
        except Exception as e:
            logger.exception(f"Failed to search similar todos: {e}")
        
        return []
    
    def delete_todo_embedding(self, todo_id: str) -> bool:
        """
        Delete a todo embedding from the unified collection.
        
        Args:
            todo_id: Todo ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False
        
        try:
            result = self._client.data_request(
                path="/api/vikingdb/data/delete",
                data={
                    "collection_name": self._collection_name,
                    "ids": [f"todo_{todo_id}"],  # Use prefixed ID
                }
            )
            return result.get("code") == "Success"
            
        except Exception as e:
            logger.exception(f"Failed to delete todo embedding: {e}")
            return False
    
    def close(self):
        """Close the backend and release resources."""
        if self._client:
            self._client.close()
        self._initialized = False
        logger.info("VikingDB backend closed")


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    from opencontext.config.global_config import get_config, GlobalConfig
    
    # Load environment variables from .env file
    load_dotenv()

    # Explicitly initialize GlobalConfig to ensure config is loaded
    # Calculate project root relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    config_path = os.path.join(project_root, "config", "config.yaml")
    
    if os.path.exists(config_path):
        GlobalConfig.get_instance().initialize(config_path)
        print(f"Initialized GlobalConfig with: {config_path}")
    else:
        print(f"Warning: Config file not found at {config_path}")
    
    # Configure logging to output to stdout
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Get configuration for VikingDB
    config = get_config("vikingdb")
    if not config:
        print("Config 'vikingdb' not found directly.")
        # If no specific vikingdb config exists, try to get general storage config
        config = get_config("storage")
        if config:
            print("Found 'storage' config.")
            # Look for vikingdb configuration within storage config
            backends = config.get("backends", [])
            vikingdb_config = None
            for backend in backends:
                if backend.get("backend", "").lower() == "vikingdb":
                    vikingdb_config = backend
                    break
            if vikingdb_config:
                print("Found 'vikingdb' backend config in storage.")
                config = vikingdb_config
            else:
                print("Did not find 'vikingdb' backend in storage backends.")
        else:
            print("Config 'storage' not found.")
    
    if config:
        print(f"Initializing VikingDB with config keys: {config.keys()}")
        if 'config' in config:
             print(f"VikingDB internal config keys: {config['config'].keys()}")
    else:
        print("Final config object is None.")

    # Initialize VikingDB client
    client = VikingDBBackend()
    try:
        if client.initialize(config):
            print(f"VikingDB backend initialized successfully: {client.get_name()}")
            
            # Test basic functionality
            print("Testing basic functionality...")
            
            # Get collection names
            collections = client.get_collection_names()
            print(f"Collections: {collections}")
            
            # Example usage would go here
            # For example, you could test upsert, search, etc. methods

            # Test get_all_processed_contexts with offset
            print("\nTesting get_all_processed_contexts with offset...")
            
            # Fetch page 1
            limit = 1
            offset_page1 = 0
            print(f"Fetching Page 1 (limit={limit}, offset={offset_page1})...")
            results_page1 = client.get_all_processed_contexts(limit=limit, offset=offset_page1)
            print(f"Page 1 Results Structure: {type(results_page1)}")
            for ctx_type, contexts in results_page1.items():
                print(f"  Type: {ctx_type}, Count: {len(contexts)}")
                for ctx in contexts:
                    summary = ctx.extracted_data.summary if ctx.extracted_data and ctx.extracted_data.summary else 'None'
                    print(f"    - ID: {ctx.id}, Summary: {summary[:30]}")
            
            # Fetch page 2
            offset_page2 = 1
            print(f"\nFetching Page 2 (limit={limit}, offset={offset_page2})...")
            results_page2 = client.get_all_processed_contexts(limit=limit, offset=offset_page2)
            print(f"Page 2 Results Structure: {type(results_page2)}")
            for ctx_type, contexts in results_page2.items():
                print(f"  Type: {ctx_type}, Count: {len(contexts)}")
                for ctx in contexts:
                    summary = ctx.extracted_data.summary if ctx.extracted_data and ctx.extracted_data.summary else 'None'
                    print(f"    - ID: {ctx.id}, Summary: {summary[:30]}")
            
        else:
            print("Failed to initialize VikingDB backend (returned False)")
    except Exception as e:
        print(f"Exception during initialization: {e}")
        import traceback
        traceback.print_exc()