# -*- coding: utf-8 -*-

"""
Media Upload API — upload images/videos to object storage for search queries.
"""

import os
import uuid

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import JSONResponse

from opencontext.server.middleware.auth import auth_dependency
from opencontext.storage.object_storage import get_object_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/media", tags=["media"])

_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "bmp"}
_VIDEO_EXTS = {"mp4", "avi", "mov", "webm"}
_ALL_EXTS = _IMAGE_EXTS | _VIDEO_EXTS
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_VIDEO_BYTES = 50 * 1024 * 1024  # 50 MB

_MIME_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "mp4": "video/mp4",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "webm": "video/webm",
}

_MEDIA_UPLOAD_DIR = "./uploads/media"


@router.post("/upload")
async def upload_media(
    file: UploadFile = File(...),
    user_id: str = Query(default="default"),
    _auth: str = auth_dependency,
):
    """
    Upload an image or video file for use in search queries.
    Returns the accessible URL of the uploaded file.
    """
    # Extract and validate extension
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALL_EXTS:
        return JSONResponse(
            status_code=400,
            content={"error": f"不支持的文件格式: .{ext}，支持: {', '.join(sorted(_ALL_EXTS))}"},
        )

    # Read file data
    data = await file.read()

    # Validate size
    media_type = "image" if ext in _IMAGE_EXTS else "video"
    max_size = _MAX_IMAGE_BYTES if media_type == "image" else _MAX_VIDEO_BYTES
    if len(data) > max_size:
        max_mb = max_size // (1024 * 1024)
        return JSONResponse(
            status_code=400,
            content={
                "error": f"文件过大（{len(data) / 1024 / 1024:.1f} MB），{media_type} 最大 {max_mb} MB"
            },
        )

    # Generate storage key
    key = f"search-media/{user_id}/{uuid.uuid4().hex}.{ext}"
    content_type = _MIME_MAP.get(ext, file.content_type or "application/octet-stream")

    # Upload to object storage or fallback to local
    obj_storage = get_object_storage()
    if obj_storage:
        try:
            url = await obj_storage.upload(data, key, content_type)
            logger.info(f"Media uploaded to object storage: {key} ({len(data)} bytes)")
        except Exception as e:
            logger.error(f"Object storage upload failed: {e}, falling back to local")
            url = _save_local(data, key)
    else:
        url = _save_local(data, key)

    return {"url": url, "type": media_type, "filename": filename}


def _save_local(data: bytes, key: str) -> str:
    """Fallback: save file locally and return the absolute path."""
    filename = key.rsplit("/", 1)[-1]
    os.makedirs(_MEDIA_UPLOAD_DIR, exist_ok=True)
    local_path = os.path.join(_MEDIA_UPLOAD_DIR, filename)
    with open(local_path, "wb") as f:
        f.write(data)
    return os.path.abspath(local_path)
