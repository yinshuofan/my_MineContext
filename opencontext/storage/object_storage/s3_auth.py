# -*- coding: utf-8 -*-

"""AWS Signature Version 4 for S3-compatible object storage."""

import hashlib
import hmac
from datetime import datetime, timezone
from urllib.parse import quote


class S3Auth:
    """
    AWS Signature V4 signer for S3 requests.

    Compatible with all S3-compatible services: AWS S3, Volcengine TOS,
    Alibaba Cloud OSS, Tencent Cloud COS, MinIO.
    """

    def __init__(self, access_key_id: str, secret_access_key: str, region: str = "cn-beijing"):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self.service = "s3"

    def sign_request(
        self,
        method: str,
        host: str,
        path: str,
        headers: dict,
        payload_hash: str,
        params: dict = None,
    ) -> dict:
        """
        Sign an HTTP request with AWS Signature V4.

        Args:
            method: HTTP method (PUT, DELETE, GET)
            host: Host header value (e.g. "bucket.tos-cn-beijing.volces.com")
            path: URL path (e.g. "/media/user1/image/abc.jpg")
            headers: Existing headers dict (will be mutated with auth headers)
            payload_hash: SHA256 hex digest of the request body
            params: Optional query parameters

        Returns:
            Updated headers dict with Authorization, X-Amz-Date, X-Amz-Content-Sha256
        """
        now = datetime.now(tz=timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        # Add required S3 headers
        headers["Host"] = host
        headers["X-Amz-Date"] = amz_date
        headers["X-Amz-Content-Sha256"] = payload_hash

        # Step 1: Canonical Request
        canonical_uri = quote(path, safe="/")
        canonical_querystring = self._canonical_query_string(params or {})

        # Build canonical headers: lowercase keys, sorted, values trimmed
        header_map = {}
        for key, value in headers.items():
            header_map[key.lower()] = value.strip()

        signed_header_keys = sorted(header_map.keys())
        signed_headers = ";".join(signed_header_keys)
        canonical_headers = "".join(f"{k}:{header_map[k]}\n" for k in signed_header_keys)

        canonical_request = "\n".join(
            [
                method.upper(),
                canonical_uri,
                canonical_querystring,
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )

        # Step 2: String to Sign
        credential_scope = f"{date_stamp}/{self.region}/{self.service}/aws4_request"
        hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                credential_scope,
                hashed_canonical_request,
            ]
        )

        # Step 3: Signing Key
        signing_key = self._get_signing_key(date_stamp)

        # Step 4: Signature
        signature = hmac.new(
            signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # Step 5: Authorization header
        headers["Authorization"] = (
            f"AWS4-HMAC-SHA256 "
            f"Credential={self.access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )

        return headers

    def _get_signing_key(self, date_stamp: str) -> bytes:
        """Derive the signing key for AWS Signature V4."""
        k_date = hmac.new(
            f"AWS4{self.secret_access_key}".encode("utf-8"),
            date_stamp.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        k_region = hmac.new(k_date, self.region.encode("utf-8"), hashlib.sha256).digest()
        k_service = hmac.new(k_region, self.service.encode("utf-8"), hashlib.sha256).digest()
        k_signing = hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()
        return k_signing

    def _canonical_query_string(self, params: dict) -> str:
        """Build canonical query string: sorted, URI-encoded keys and values."""
        if not params:
            return ""
        return "&".join(
            f"{quote(str(k), safe='')}={quote(str(v), safe='')}" for k, v in sorted(params.items())
        )
