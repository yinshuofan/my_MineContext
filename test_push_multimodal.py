"""Test multimodal push API with local image and video files."""
import base64
import json
import sys
import requests

BASE_URL = "http://localhost:8088"
HEADERS = {"X-API-Key": "default-key"}

def file_to_data_uri(path: str) -> str:
    import mimetypes
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        if path.endswith(".jpg") or path.endswith(".jpeg"):
            mime = "image/jpeg"
        elif path.endswith(".mp4"):
            mime = "video/mp4"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def test_text_only():
    """Test 1: Basic text-only push."""
    print("=" * 60)
    print("Test 1: Text-only push (direct mode)")
    print("=" * 60)
    resp = requests.post(f"{BASE_URL}/api/push/chat", headers=HEADERS, json={
        "messages": [
            {"role": "user", "content": "今天和小明一起去了故宫，天气很好，拍了很多照片。"},
            {"role": "assistant", "content": "听起来很棒！故宫的建筑确实很壮观。"},
        ],
        "user_id": "test_user",
        "process_mode": "direct",
    })
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")
    print()


def test_image_only():
    """Test 2: Multimodal push with image."""
    print("=" * 60)
    print("Test 2: Image + text push (direct mode)")
    print("=" * 60)
    image_uri = file_to_data_uri("test_image.jpg")
    print(f"Image data URI length: {len(image_uri)} chars")

    resp = requests.post(f"{BASE_URL}/api/push/chat", headers=HEADERS, json={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是我今天在故宫拍的照片，你看看好不好看"},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            },
            {
                "role": "assistant",
                "content": "这张照片拍得真好！光线和构图都很棒。",
            },
        ],
        "user_id": "test_user",
        "process_mode": "direct",
    })
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")
    print()


def test_video_only():
    """Test 3: Multimodal push with video."""
    print("=" * 60)
    print("Test 3: Video + text push (direct mode)")
    print("=" * 60)
    video_uri = file_to_data_uri("test_video.mp4")
    print(f"Video data URI length: {len(video_uri)} chars")

    resp = requests.post(f"{BASE_URL}/api/push/chat", headers=HEADERS, json={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是我拍的一段小视频"},
                    {"type": "video_url", "video_url": {"url": video_uri}},
                ],
            },
        ],
        "user_id": "test_user",
        "process_mode": "direct",
    })
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")
    print()


def test_mixed_multimodal():
    """Test 4: Mixed multimodal with image + video + text in conversation."""
    print("=" * 60)
    print("Test 4: Mixed image + video + text conversation (direct mode)")
    print("=" * 60)
    image_uri = file_to_data_uri("test_image.jpg")
    video_uri = file_to_data_uri("test_video.mp4")

    resp = requests.post(f"{BASE_URL}/api/push/chat", headers=HEADERS, json={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "今天去故宫玩了一天，给你看看我拍的照片和视频"},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "video_url", "video_url": {"url": video_uri}},
                ],
            },
            {
                "role": "assistant",
                "content": "哇，故宫看起来真壮观！照片和视频都拍得很好，尤其是那个建筑的细节。",
            },
            {
                "role": "user",
                "content": "谢谢！下次还想去颐和园看看。",
            },
        ],
        "user_id": "test_user",
        "process_mode": "direct",
    })
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")
    print()


if __name__ == "__main__":
    # Run specific test or all
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "text": test_text_only,
        "image": test_image_only,
        "video": test_video_only,
        "mixed": test_mixed_multimodal,
    }

    if test_name == "all":
        for name, fn in tests.items():
            try:
                fn()
            except Exception as e:
                print(f"FAILED: {e}")
                print()
    elif test_name in tests:
        tests[test_name]()
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available: {', '.join(tests.keys())}, all")
