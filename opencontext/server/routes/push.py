#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Push API routes for external services to push data to the long-term memory backend.
This module provides HTTP endpoints for receiving context data from external services,
replacing the automatic capture/pull mechanisms with a push-based architecture.
"""

import asyncio
import base64
import datetime
import json
import mimetypes
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from opencontext.models.context import RawContextProperties
from opencontext.models.enums import ContentFormat, ContextSource
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.opencontext import OpenContext
from opencontext.server.utils import convert_resp, get_context_lab
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.time_utils import now as tz_now

logger = get_logger(__name__)
router = APIRouter(prefix="/api/push", tags=["push"])

# ============================================================================
# Task Scheduler Integration
# ============================================================================


async def _schedule_user_task(
    task_type: str,
    user_id: Optional[str],
    device_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> None:
    """Schedule a user task (compression, hierarchy summary, etc.) without failing the request."""
    if not user_id:
        return

    try:
        from opencontext.scheduler import get_scheduler

        scheduler = get_scheduler()
        if scheduler:
            await scheduler.schedule_user_task(
                task_type=task_type,
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            )
    except Exception as e:
        logger.warning(f"Failed to schedule {task_type} task: {e}")


# ============================================================================
# Request Models
# ============================================================================


class PushChatRequest(BaseModel):
    """Unified chat push request"""

    messages: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Chat messages (OpenAI format: [{role, content}])",
    )
    user_id: Optional[str] = Field(
        None, min_length=1, max_length=255, description="User identifier"
    )
    device_id: Optional[str] = Field(
        None, min_length=1, max_length=100, description="Device identifier"
    )
    agent_id: Optional[str] = Field(
        None, min_length=1, max_length=100, description="Agent identifier"
    )
    processors: List[str] = Field(
        default=["user_memory"],
        description="Processors to run: 'user_memory', 'agent_memory', etc.",
    )


class PushDocumentRequest(BaseModel):
    """Push a document"""

    # Either provide file_path (local) or base64_data (remote upload)
    file_path: Optional[str] = Field(None, description="Local file path")
    base64_data: Optional[str] = Field(None, description="Base64 encoded document data")
    filename: Optional[str] = Field(None, description="Filename for base64 uploads")
    content_type: Optional[str] = Field(None, description="MIME type of the document")
    user_id: Optional[str] = Field(
        None, min_length=1, max_length=255, description="User identifier"
    )
    device_id: Optional[str] = Field(
        None, min_length=1, max_length=100, description="Device identifier"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# ============================================================================
# Helper Functions
# ============================================================================


def _save_base64_to_temp_file(base64_data: str, filename: str, storage_dir: str = None) -> str:
    """
    Save base64 encoded data to a temporary file.

    Args:
        base64_data: Base64 encoded data
        filename: Original filename
        storage_dir: Directory to save the file (uses temp dir if None)

    Returns:
        Path to the saved file
    """
    try:
        # Decode base64 data
        file_data = base64.b64decode(base64_data)

        # Determine storage directory
        if storage_dir is None:
            storage_dir = tempfile.gettempdir()

        # Ensure directory exists
        os.makedirs(storage_dir, exist_ok=True)

        # Generate unique filename
        ext = os.path.splitext(filename)[1] if filename else ""
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(storage_dir, unique_filename)

        # Write file
        with open(file_path, "wb") as f:
            f.write(file_data)

        return file_path
    except Exception as e:
        logger.exception(f"Failed to save base64 data to file: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")


# ============================================================================
# Multimodal Media Processing
# ============================================================================

# Supported media formats and size limits
_IMAGE_FORMATS = {"jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"}
_VIDEO_FORMATS = {"mp4", "avi", "mov"}
_MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_VIDEO_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
_MEDIA_UPLOAD_DIR = "./uploads/media"


def _detect_media_ext_from_data_uri(data_uri: str) -> Optional[str]:
    """Extract file extension from a data URI (e.g., 'data:image/jpeg;base64,...' -> 'jpeg')."""
    if not data_uri.startswith("data:"):
        return None
    try:
        header = data_uri.split(",", 1)[0]  # "data:image/jpeg;base64"
        mime = header.split(";")[0].replace("data:", "")  # "image/jpeg"
        ext = mimetypes.guess_extension(mime, strict=False)
        if ext:
            return ext.lstrip(".")  # ".jpeg" -> "jpeg"
        # Fallback: use the subtype directly
        parts = mime.split("/")
        if len(parts) == 2:
            return parts[1]
    except Exception:
        pass
    return None


def _save_media_base64(data_uri: str, media_type: str) -> str:
    """
    Save a base64 data URI to a file in the media upload directory.

    Args:
        data_uri: Full data URI string (e.g., 'data:image/jpeg;base64,...')
        media_type: 'image' or 'video'

    Returns:
        Local file path of the saved file.

    Raises:
        HTTPException: If media is invalid, too large, or unsupported format.
    """
    # Extract the base64 data
    if "," not in data_uri:
        raise HTTPException(status_code=400, detail=f"Invalid {media_type} data URI format")

    base64_data = data_uri.split(",", 1)[1]

    # Detect extension
    ext = _detect_media_ext_from_data_uri(data_uri)
    if not ext:
        ext = "jpg" if media_type == "image" else "mp4"

    # Validate format
    allowed_formats = _IMAGE_FORMATS if media_type == "image" else _VIDEO_FORMATS
    if ext.lower() not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {media_type} format: .{ext}. Allowed: {', '.join(sorted(allowed_formats))}",
        )

    # Decode and validate size
    try:
        file_data = base64.b64decode(base64_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data for {media_type}: {e}")

    max_size = _MAX_IMAGE_SIZE_BYTES if media_type == "image" else _MAX_VIDEO_SIZE_BYTES
    if len(file_data) > max_size:
        max_mb = max_size // (1024 * 1024)
        actual_mb = len(file_data) / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"{media_type.capitalize()} too large: {actual_mb:.1f} MB (max {max_mb} MB)",
        )

    # Save to file
    os.makedirs(_MEDIA_UPLOAD_DIR, exist_ok=True)
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(_MEDIA_UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as f:
        f.write(file_data)

    return file_path


async def _process_multimodal_messages(
    messages: List[Dict[str, Any]], user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process multimodal messages: upload base64 media to object storage (or save locally).

    For each message, if `content` is a list (multimodal format), iterate over content parts:
    - text parts: pass through unchanged
    - image_url parts with base64 data URIs: upload to object storage, replace with URL
    - video_url parts with base64 data URIs: upload to object storage, replace with URL
    - HTTP URLs: pass through unchanged

    When object storage is configured (S3), base64 is uploaded and replaced with HTTPS URLs.
    When not configured, falls back to saving files locally.
    """
    from opencontext.storage.object_storage import get_object_storage

    obj_storage = get_object_storage()
    uid = user_id or "default"

    processed = []
    for msg in messages:
        content = msg.get("content")

        # Text-only message: pass through unchanged
        if not isinstance(content, list):
            processed.append(msg)
            continue

        # Multimodal message: process each content part
        new_content_parts = []
        for part in content:
            part_type = part.get("type", "")

            if part_type == "text":
                new_content_parts.append(part)

            elif part_type == "image_url":
                image_url_obj = part.get("image_url", {})
                url = image_url_obj.get("url", "")

                if url.startswith("data:"):
                    url = await _upload_or_save_media(obj_storage, url, "image", uid)
                    new_content_parts.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    new_content_parts.append(part)

            elif part_type == "video_url":
                video_url_obj = part.get("video_url", {})
                url = video_url_obj.get("url", "")

                if url.startswith("data:"):
                    url = await _upload_or_save_media(obj_storage, url, "video", uid)
                    new_part = {"type": "video_url", "video_url": {"url": url}}
                    if "fps" in video_url_obj:
                        new_part["video_url"]["fps"] = video_url_obj["fps"]
                    new_content_parts.append(new_part)
                else:
                    new_content_parts.append(part)

            else:
                new_content_parts.append(part)

        processed.append({**msg, "content": new_content_parts})

    return processed


async def _upload_or_save_media(obj_storage, data_uri: str, media_type: str, user_id: str) -> str:
    """Upload base64 data URI to object storage, or save locally as fallback.

    Returns:
        HTTPS URL (if object storage is available) or local file path.
    """
    # Validate and decode
    if "," not in data_uri:
        raise HTTPException(status_code=400, detail=f"Invalid {media_type} data URI format")

    base64_data = data_uri.split(",", 1)[1]
    ext = _detect_media_ext_from_data_uri(data_uri)
    if not ext:
        ext = "jpg" if media_type == "image" else "mp4"

    allowed_formats = _IMAGE_FORMATS if media_type == "image" else _VIDEO_FORMATS
    if ext.lower() not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {media_type} format: .{ext}. Allowed: {', '.join(sorted(allowed_formats))}",
        )

    try:
        file_data = base64.b64decode(base64_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data for {media_type}: {e}")

    max_size = _MAX_IMAGE_SIZE_BYTES if media_type == "image" else _MAX_VIDEO_SIZE_BYTES
    if len(file_data) > max_size:
        max_mb = max_size // (1024 * 1024)
        actual_mb = len(file_data) / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"{media_type.capitalize()} too large: {actual_mb:.1f} MB (max {max_mb} MB)",
        )

    # Determine content type
    _MIME_MAP = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "mp4": "video/mp4",
        "avi": "video/x-msvideo",
        "mov": "video/quicktime",
    }
    content_type = _MIME_MAP.get(ext.lower(), f"{media_type}/{ext}")

    key = f"media/{user_id}/{media_type}/{uuid.uuid4().hex}.{ext}"

    if obj_storage:
        # Upload to object storage (S3/TOS/OSS/etc.)
        return await obj_storage.upload(file_data, key, content_type)
    else:
        # Fallback: save to local file
        os.makedirs(_MEDIA_UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(_MEDIA_UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
        with open(file_path, "wb") as f:
            f.write(file_data)
        return file_path


# ============================================================================
# Chat Push Endpoint
# ============================================================================


@router.post("/chat", response_class=JSONResponse)
async def push_chat(
    request: PushChatRequest,
    background_tasks: BackgroundTasks,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push chat messages for processing.

    Messages are persisted to chat_batches, then dispatched to the requested
    processors (default: user_memory) in background.  Hierarchy summary and
    compression tasks are scheduled after processing.
    """
    try:
        # Validate reserved user_id
        if request.user_id == "__base__":
            raise HTTPException(
                status_code=400, detail="user_id '__base__' is reserved for system use"
            )

        # Validate agent exists when agent_memory processor is requested
        if (
            "agent_memory" in request.processors
            and request.agent_id is not None
            and request.agent_id != "default"
        ):
            storage = get_storage()
            agent = await storage.get_agent(request.agent_id)
            if not agent:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent '{request.agent_id}' is not registered. "
                    f"Please register the agent via POST /api/agents before using agent_memory processor.",
                )

        # Process multimodal messages: upload to object storage or save locally
        messages = await _process_multimodal_messages(request.messages, request.user_id)

        # Persist to chat_batches
        batch_id = str(uuid.uuid4())
        storage = get_storage()
        await storage.create_chat_batch(
            batch_id=batch_id,
            messages=messages,
            user_id=request.user_id,
            device_id=request.device_id or "default",
            agent_id=request.agent_id or "default",
        )

        # Detect content format from messages
        has_images = False
        has_videos = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    part_type = part.get("type", "")
                    if part_type == "image_url":
                        has_images = True
                    elif part_type == "video_url":
                        has_videos = True

        is_multimodal = has_images or has_videos
        content_format = ContentFormat.MULTIMODAL if is_multimodal else ContentFormat.TEXT

        # Build modalities list for additional_info
        modalities = ["text"]
        if has_images:
            modalities.append("image")
        if has_videos:
            modalities.append("video")

        # Build RawContextProperties
        raw_context = RawContextProperties(
            source=ContextSource.CHAT_LOG,
            content_format=content_format,
            create_time=tz_now(),
            content_text=json.dumps(messages, ensure_ascii=False),
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
            additional_info={
                "batch_id": batch_id,
                "message_count": len(messages),
                "roles": list(set(m.get("role", "user") for m in messages)),
                "modalities": modalities,
            },
        )

        # Dispatch to processors in background
        async def _process():
            await opencontext.processor_manager.process_batch(raw_context, request.processors)

        background_tasks.add_task(_process)

        # Schedule hierarchy summary and compression tasks
        background_tasks.add_task(
            _schedule_user_task,
            task_type="memory_compression",
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
        )
        background_tasks.add_task(
            _schedule_user_task,
            task_type="hierarchy_summary",
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
        )

        # Schedule agent profile update (only for registered agents)
        if (
            "agent_memory" in request.processors
            and request.agent_id is not None
            and request.agent_id != "default"
        ):
            background_tasks.add_task(
                _schedule_user_task,
                task_type="agent_profile_update",
                user_id=request.user_id,
                device_id=request.device_id,
                agent_id=request.agent_id,
            )

        return convert_resp(
            message="Chat messages submitted for processing",
            data={"batch_id": batch_id, "message_count": len(messages)},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error pushing chat: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


# ============================================================================
# Document Push Endpoints
# ============================================================================


@router.post("/document", response_class=JSONResponse)
async def push_document(
    request: PushDocumentRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push a document to the context capture system.
    Supports both local file path and base64 encoded data.
    """
    try:
        file_path = request.file_path
        if not file_path and request.base64_data:
            # Save base64 data to temp file
            file_path = _save_base64_to_temp_file(
                request.base64_data, request.filename or "document", "./uploads/documents"
            )

        if not file_path:
            return convert_resp(
                code=400, status=400, message="Either 'file_path' or 'base64_data' must be provided"
            )

        err_msg = await asyncio.wait_for(
            opencontext.add_document(file_path=file_path),
            timeout=60.0,
        )

        if err_msg:
            return convert_resp(code=400, status=400, message=err_msg)

        return convert_resp(message="Document pushed successfully", data={"path": file_path})
    except asyncio.TimeoutError:
        logger.warning("Push document timed out after 60s")
        return convert_resp(code=504, status=504, message="Request timed out")
    except Exception as e:
        logger.exception(f"Error pushing document: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


@router.post("/document/upload", response_class=JSONResponse)
async def upload_document_file(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    device_id: Optional[str] = Form(None),
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Upload a document file directly via multipart form.
    """
    try:
        # Save uploaded file
        upload_dir = "./uploads/documents"
        os.makedirs(upload_dir, exist_ok=True)

        ext = os.path.splitext(file.filename)[1] if file.filename else ""
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(upload_dir, unique_filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        err_msg = await opencontext.add_document(file_path=file_path)

        if err_msg:
            return convert_resp(code=400, status=400, message=err_msg)

        return convert_resp(
            message="Document uploaded and queued successfully",
            data={"path": file_path, "original_filename": file.filename},
        )
    except Exception as e:
        logger.exception(f"Error uploading document: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")
