# -*- coding: utf-8 -*-

"""S3-compatible object storage backend."""

import asyncio
import hashlib
from typing import Optional

import aiohttp

from opencontext.storage.object_storage.base import IObjectStorage
from opencontext.storage.object_storage.s3_auth import S3Auth
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class S3CompatibleBackend(IObjectStorage):
    """
    S3-compatible object storage. Works with any S3-compatible service:
    - Volcengine TOS (tos-cn-beijing.volces.com)
    - Alibaba Cloud OSS (oss-cn-hangzhou.aliyuncs.com)
    - Tencent Cloud COS (cos.ap-guangzhou.myqcloud.com)
    - AWS S3 (s3.us-east-1.amazonaws.com)
    - MinIO (minio.internal:9000)
    """

    def __init__(
        self,
        endpoint: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        region: str = "cn-beijing",
        use_https: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.endpoint = endpoint
        self.bucket = bucket
        self.region = region
        self.scheme = "https" if use_https else "http"
        self.host = f"{bucket}.{endpoint}"
        self._auth = S3Auth(access_key_id, secret_access_key, region)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=50, limit_per_host=20, enable_cleanup_closed=True
                    )
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=60),
                    )
        return self._session

    async def upload(self, data: bytes, key: str, content_type: str) -> str:
        """Upload data via S3 PUT Object. Returns URL of the uploaded object."""
        path = f"/{key}"
        payload_hash = hashlib.sha256(data).hexdigest()

        headers = {"Content-Type": content_type}
        signed_headers = self._auth.sign_request(
            method="PUT",
            host=self.host,
            path=path,
            headers=headers,
            payload_hash=payload_hash,
        )

        url = f"{self.scheme}://{self.host}{path}"

        session = await self._get_session()
        for attempt in range(self._max_retries):
            try:
                async with session.put(url, data=data, headers=signed_headers) as resp:
                    if resp.status in (200, 201):
                        logger.info(f"Uploaded {key} ({len(data)} bytes)")
                        return self.get_url(key)
                    body = await resp.text()
                    logger.warning(
                        f"S3 upload {key} attempt {attempt + 1} failed: {resp.status} {body}"
                    )
            except Exception as e:
                logger.warning(f"S3 upload {key} attempt {attempt + 1} error: {e}")

            if attempt < self._max_retries - 1:
                await asyncio.sleep(self._retry_delay * (2**attempt))
                # Re-sign on retry (timestamp changes)
                headers = {"Content-Type": content_type}
                signed_headers = self._auth.sign_request(
                    method="PUT",
                    host=self.host,
                    path=path,
                    headers=headers,
                    payload_hash=payload_hash,
                )

        raise RuntimeError(f"Failed to upload {key} after {self._max_retries} attempts")

    def get_url(self, key: str) -> str:
        """Return URL for the object."""
        return f"{self.scheme}://{self.host}/{key}"

    async def delete(self, key: str) -> bool:
        """Delete object via S3 DELETE Object."""
        path = f"/{key}"
        payload_hash = hashlib.sha256(b"").hexdigest()

        headers = {}
        signed_headers = self._auth.sign_request(
            method="DELETE",
            host=self.host,
            path=path,
            headers=headers,
            payload_hash=payload_hash,
        )

        url = f"{self.scheme}://{self.host}{path}"
        session = await self._get_session()

        try:
            async with session.delete(url, headers=signed_headers) as resp:
                if resp.status in (200, 204):
                    logger.info(f"Deleted {key}")
                    return True
                body = await resp.text()
                logger.warning(f"S3 delete {key} failed: {resp.status} {body}")
                return False
        except Exception as e:
            logger.error(f"S3 delete {key} error: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
