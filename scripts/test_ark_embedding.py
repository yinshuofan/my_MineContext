"""
Standalone validation script for the Ark multimodal embedding API.

Tests the doubao-embedding-vision-251215 model via direct HTTP calls (aiohttp).
This script validates API request formats and response parsing BEFORE any
business code is modified.

Environment variables:
    ARK_API_KEY          - Required. Bearer token for Ark API authentication.
    ARK_EMBEDDING_MODEL  - Optional. Model name (default: doubao-embedding-vision-251215).
    ARK_BASE_URL         - Optional. API endpoint (default: https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal).

Usage:
    python scripts/test_ark_embedding.py
"""

import asyncio
import base64
import os
import struct
import sys
import time
from typing import Any, Dict, List, Optional

import aiohttp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
ARK_EMBEDDING_MODEL = os.environ.get("ARK_EMBEDDING_MODEL", "doubao-embedding-vision-251215")
ARK_BASE_URL = os.environ.get(
    "ARK_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal",
)

# Default embedding dimensions (doubao-embedding-vision-251215 supports 2048 and 1024)
DIMENSIONS = 2048


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_tiny_png_base64() -> str:
    """
    Generate a minimal valid 1x1 red PNG image encoded as a data URI.
    This avoids needing an external image URL for testing.
    """
    # Minimal 1x1 red pixel PNG
    import io
    import zlib

    def _create_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk: width=1, height=1, bit_depth=8, color_type=2 (RGB)
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = _create_chunk(b"IHDR", ihdr_data)

    # IDAT chunk: raw image data (filter byte 0 + RGB red pixel)
    raw_data = b"\x00\xff\x00\x00"  # filter=0, R=255, G=0, B=0
    compressed = zlib.compress(raw_data)
    idat = _create_chunk(b"IDAT", compressed)

    # IEND chunk
    iend = _create_chunk(b"IEND", b"")

    png_bytes = signature + ihdr + idat + iend
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_result(label: str, value: Any) -> None:
    print(f"  {label}: {value}")


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def call_ark_embedding(
    session: aiohttp.ClientSession,
    input_data: List[Dict],
    instruction: str,
    dimensions: int = DIMENSIONS,
) -> Dict[str, Any]:
    """
    Call the Ark multimodal embedding API.

    Args:
        session: aiohttp session.
        input_data: List of input items (text, image_url, video_url).
        instruction: The instruction string for the model.
        dimensions: Output vector dimension.

    Returns:
        The full JSON response dict.

    Raises:
        Exception on HTTP or API errors.
    """
    payload = {
        "model": ARK_EMBEDDING_MODEL,
        "encoding_format": "float",
        "dimensions": dimensions,
        "instructions": instruction,
        "input": input_data,
    }

    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json",
    }

    async with session.post(ARK_BASE_URL, json=payload, headers=headers) as resp:
        body = await resp.json()
        if resp.status != 200:
            error_msg = body.get("error", {}).get("message", resp.reason)
            raise Exception(
                f"Ark API returned HTTP {resp.status}: {error_msg}\n"
                f"Full response: {body}"
            )
        return body


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

async def test_text_corpus_embedding(session: aiohttp.ClientSession) -> bool:
    """
    Test 1: Text-only embedding with corpus-side instruction.
    Validates that a plain text input returns a vector of the expected dimension.
    """
    print_header("Test 1: Text-only embedding (Corpus instruction)")

    instruction = "Instruction:Compress the text into one word.\nQuery:"
    input_data = [{"type": "text", "text": "The user visited West Lake yesterday and took photos."}]

    try:
        result = await call_ark_embedding(session, input_data, instruction)

        embedding = result.get("data", {}).get("embedding", [])
        usage = result.get("usage", {})
        total_tokens = usage.get("total_tokens", "N/A")

        print_result("Vector dimension", len(embedding))
        print_result("Total tokens", total_tokens)
        print_result("First 5 values", embedding[:5])
        print_result("Model", result.get("model", "N/A"))

        assert isinstance(embedding, list), "embedding should be a list"
        assert len(embedding) == DIMENSIONS, f"Expected {DIMENSIONS} dimensions, got {len(embedding)}"
        assert all(isinstance(v, (int, float)) for v in embedding[:10]), "Values should be numeric"
        assert total_tokens != "N/A", "total_tokens should be present"

        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


async def test_text_query_embedding(session: aiohttp.ClientSession) -> bool:
    """
    Test 2: Text-only embedding with query-side instruction.
    Validates query-side instruction format with Target_modality.
    """
    print_header("Test 2: Text-only embedding (Query instruction)")

    instruction = (
        "Target_modality: text/image/video.\n"
        "Instruction:Based on this query, find the most relevant memory content\n"
        "Query:"
    )
    input_data = [{"type": "text", "text": "photos of West Lake"}]

    try:
        result = await call_ark_embedding(session, input_data, instruction)

        embedding = result.get("data", {}).get("embedding", [])
        usage = result.get("usage", {})

        print_result("Vector dimension", len(embedding))
        print_result("Total tokens", usage.get("total_tokens", "N/A"))
        print_result("First 5 values", embedding[:5])

        assert len(embedding) == DIMENSIONS, f"Expected {DIMENSIONS} dimensions, got {len(embedding)}"

        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


async def test_image_embedding(session: aiohttp.ClientSession) -> bool:
    """
    Test 3: Image-only embedding using a tiny base64-encoded PNG.
    Validates that the API accepts base64 image input.
    """
    print_header("Test 3: Image-only embedding (base64 PNG)")

    instruction = "Instruction:Compress the image into one word.\nQuery:"
    image_data_uri = generate_tiny_png_base64()
    input_data = [{"type": "image_url", "image_url": {"url": image_data_uri}}]

    try:
        result = await call_ark_embedding(session, input_data, instruction)

        embedding = result.get("data", {}).get("embedding", [])
        usage = result.get("usage", {})
        image_tokens = usage.get("prompt_tokens_details", {}).get("image_tokens", "N/A")

        print_result("Vector dimension", len(embedding))
        print_result("Total tokens", usage.get("total_tokens", "N/A"))
        print_result("Image tokens", image_tokens)
        print_result("First 5 values", embedding[:5])

        assert len(embedding) == DIMENSIONS, f"Expected {DIMENSIONS} dimensions, got {len(embedding)}"

        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


async def test_text_image_combined_embedding(session: aiohttp.ClientSession) -> bool:
    """
    Test 4: Combined text + image embedding.
    Validates that mixed-modality input produces a single unified vector.
    """
    print_header("Test 4: Text + Image combined embedding")

    instruction = "Instruction:Compress the text and image into one word.\nQuery:"
    image_data_uri = generate_tiny_png_base64()
    input_data = [
        {"type": "text", "text": "A beautiful sunset over the ocean"},
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ]

    try:
        result = await call_ark_embedding(session, input_data, instruction)

        embedding = result.get("data", {}).get("embedding", [])
        usage = result.get("usage", {})
        text_tokens = usage.get("prompt_tokens_details", {}).get("text_tokens", "N/A")
        image_tokens = usage.get("prompt_tokens_details", {}).get("image_tokens", "N/A")

        print_result("Vector dimension", len(embedding))
        print_result("Total tokens", usage.get("total_tokens", "N/A"))
        print_result("Text tokens", text_tokens)
        print_result("Image tokens", image_tokens)
        print_result("First 5 values", embedding[:5])

        assert len(embedding) == DIMENSIONS, f"Expected {DIMENSIONS} dimensions, got {len(embedding)}"

        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


async def test_response_format(session: aiohttp.ClientSession) -> bool:
    """
    Test 5: Verify detailed response format structure.
    Checks that all expected fields exist in the API response.
    """
    print_header("Test 5: Response format validation")

    instruction = "Instruction:Compress the text into one word.\nQuery:"
    input_data = [{"type": "text", "text": "test response format"}]

    try:
        result = await call_ark_embedding(session, input_data, instruction)

        # Check top-level fields
        assert "data" in result, "Response missing 'data' field"
        assert "usage" in result, "Response missing 'usage' field"
        assert "model" in result, "Response missing 'model' field"
        assert "id" in result, "Response missing 'id' field"
        assert "object" in result, "Response missing 'object' field"

        # Check data structure
        data = result["data"]
        assert "embedding" in data, "data missing 'embedding' field"
        assert "object" in data, "data missing 'object' field"
        assert data["object"] == "embedding", f"data.object should be 'embedding', got '{data['object']}'"

        # Check embedding is list of floats
        embedding = data["embedding"]
        assert isinstance(embedding, list), "embedding should be a list"
        assert len(embedding) > 0, "embedding should not be empty"
        assert isinstance(embedding[0], (int, float)), "embedding values should be numeric"

        # Check usage structure
        usage = result["usage"]
        assert "total_tokens" in usage, "usage missing 'total_tokens'"
        assert "prompt_tokens" in usage, "usage missing 'prompt_tokens'"

        print_result("Response fields", list(result.keys()))
        print_result("data fields", list(data.keys()))
        print_result("usage fields", list(usage.keys()))
        print_result("data.object", data["object"])
        print_result("Embedding length", len(embedding))
        print_result("usage.total_tokens", usage["total_tokens"])

        # Check prompt_tokens_details if present
        if "prompt_tokens_details" in usage:
            details = usage["prompt_tokens_details"]
            print_result("prompt_tokens_details", details)

        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run all test cases sequentially."""
    if not ARK_API_KEY:
        print("ERROR: ARK_API_KEY environment variable is not set.")
        print("Please set it before running this script:")
        print("  export ARK_API_KEY='your-api-key-here'")
        sys.exit(1)

    print(f"Ark Multimodal Embedding API Validation")
    print(f"  Model:      {ARK_EMBEDDING_MODEL}")
    print(f"  Endpoint:   {ARK_BASE_URL}")
    print(f"  Dimensions: {DIMENSIONS}")

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = []

        results.append(await test_text_corpus_embedding(session))
        results.append(await test_text_query_embedding(session))
        results.append(await test_image_embedding(session))
        results.append(await test_text_image_combined_embedding(session))
        results.append(await test_response_format(session))

    # Summary
    print_header("Summary")
    passed = sum(results)
    total = len(results)
    print(f"  {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests PASSED. API formats are validated.")
    else:
        print(f"\n  {total - passed} test(s) FAILED. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
