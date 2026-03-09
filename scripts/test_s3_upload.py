"""
Standalone validation script for S3-compatible object storage.

Tests upload (PUT), get_url, and delete operations against any S3-compatible
service using the S3Auth signer and S3CompatibleBackend.

Environment variables:
    S3_ENDPOINT          - Required. e.g. "tos-cn-beijing.volces.com"
    S3_ACCESS_KEY_ID     - Required. Access key ID.
    S3_SECRET_ACCESS_KEY - Required. Secret access key.
    S3_BUCKET            - Required. Bucket name. (default: opencontext-media)
    S3_REGION            - Optional. Region (default: cn-beijing).

Usage:
    python scripts/test_s3_upload.py
"""

import asyncio
import os
import sys
import uuid

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add project root to path so we can import opencontext modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opencontext.storage.object_storage.s3_auth import S3Auth
from opencontext.storage.object_storage.s3_backend import S3CompatibleBackend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENDPOINT = os.environ.get("S3_ENDPOINT", "tos-cn-beijing.volces.com")
ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID", "")
SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY", "")
BUCKET = os.environ.get("S3_BUCKET", "opencontext-media")
REGION = os.environ.get("S3_REGION", "cn-beijing")

# Fallback to VikingDB credentials (common in Volcengine TOS scenarios)
if not ACCESS_KEY_ID:
    ACCESS_KEY_ID = os.environ.get("VIKINGDB_ACCESS_KEY_ID", "")
if not SECRET_ACCESS_KEY:
    SECRET_ACCESS_KEY = os.environ.get("VIKINGDB_SECRET_ACCESS_KEY", "")

# Minimal 1x1 red PNG (67 bytes)
PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n"  # PNG signature
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx"
    b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_result(label: str, value: str) -> None:
    print(f"  {label}: {value}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def main() -> None:
    if not ACCESS_KEY_ID or not SECRET_ACCESS_KEY:
        print("ERROR: S3 credentials not set.")
        print("Set S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY (or VIKINGDB_* equivalents).")
        print("  export S3_ACCESS_KEY_ID='your-access-key'")
        print("  export S3_SECRET_ACCESS_KEY='your-secret-key'")
        sys.exit(1)

    print("S3 Object Storage Validation")
    print_result("Endpoint", ENDPOINT)
    print_result("Bucket", BUCKET)
    print_result("Region", REGION)
    print_result("Host", f"{BUCKET}.{ENDPOINT}")

    backend = S3CompatibleBackend(
        endpoint=ENDPOINT,
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        bucket=BUCKET,
        region=REGION,
        use_https=True,
    )

    test_key = f"media/test/{uuid.uuid4().hex}.png"
    passed = 0
    total = 3

    try:
        # Test 1: Upload
        print_header("Test 1: Upload (PUT Object)")
        try:
            url = await backend.upload(data=PNG_1X1, key=test_key, content_type="image/png")
            print_result("Key", test_key)
            print_result("Size", f"{len(PNG_1X1)} bytes")
            print_result("URL", url)
            print("  [PASS]")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            print("\n  Cannot proceed without upload. Aborting.")
            return

        # Test 2: get_url
        print_header("Test 2: get_url (URL format check)")
        expected_url = f"https://{BUCKET}.{ENDPOINT}/{test_key}"
        actual_url = backend.get_url(test_key)
        print_result("Expected", expected_url)
        print_result("Actual", actual_url)
        if actual_url == expected_url:
            print("  [PASS]")
            passed += 1
        else:
            print("  [FAIL] URL mismatch")

        # Test 3: Delete
        print_header("Test 3: Delete (DELETE Object)")
        try:
            ok = await backend.delete(test_key)
            print_result("Key", test_key)
            print_result("Deleted", str(ok))
            if ok:
                print("  [PASS]")
                passed += 1
            else:
                print("  [FAIL] Delete returned False")
        except Exception as e:
            print(f"  [FAIL] {e}")

    finally:
        await backend.close()

    # Summary
    print_header("Summary")
    print(f"  {passed}/{total} tests passed")
    if passed == total:
        print("\n  All tests PASSED.")
    else:
        print(f"\n  {total - passed} test(s) FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
