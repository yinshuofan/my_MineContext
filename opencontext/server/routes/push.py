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
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from opencontext.models.context import RawContextProperties
from opencontext.models.enums import ContentFormat, ContextSource
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.opencontext import OpenContext
from opencontext.server.utils import convert_resp, get_context_lab
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/push", tags=["push"])


# ============================================================================
# Task Scheduler Integration
# ============================================================================


async def _schedule_user_compression(
    user_id: Optional[str], device_id: Optional[str] = None, agent_id: Optional[str] = None
) -> None:
    """
    Schedule a memory compression task for the user (async).
    This is called after data capture to trigger delayed compression.
    """
    if not user_id:
        return

    try:
        from opencontext.scheduler import get_scheduler

        scheduler = get_scheduler()
        if scheduler:
            await scheduler.schedule_user_task(
                task_type="memory_compression",
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            )
    except Exception as e:
        # Don't fail the request if scheduling fails
        logger.warning(f"Failed to schedule compression task: {e}")


async def _schedule_user_hierarchy_summary(
    user_id: Optional[str], device_id: Optional[str] = None, agent_id: Optional[str] = None
) -> None:
    """
    Schedule a hierarchy summary task for the user (async).
    Generates daily/weekly/monthly event summaries after a 24h delay.
    """
    if not user_id:
        return

    try:
        from opencontext.scheduler import get_scheduler

        scheduler = get_scheduler()
        if scheduler:
            await scheduler.schedule_user_task(
                task_type="hierarchy_summary",
                user_id=user_id,
                device_id=device_id,
                agent_id=agent_id,
            )
    except Exception as e:
        logger.warning(f"Failed to schedule hierarchy summary task: {e}")


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
    process_mode: Literal["buffer", "direct"] = Field(
        "buffer",
        description="buffer=Redis buffered (default), direct=bypass buffer and process immediately",
    )
    flush_immediately: bool = Field(
        False,
        description="In buffer mode, whether to flush the buffer immediately after pushing",
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


class PushContextRequest(BaseModel):
    """Push a generic context item"""

    source: str = Field(..., min_length=1, max_length=50, description="Context source type")
    content_format: str = Field("text", min_length=1, max_length=20, description="Content format")
    content_text: Optional[str] = Field(None, max_length=100000, description="Text content")
    content_images: Optional[List[str]] = Field(
        None, max_length=20, description="Image paths or base64 data"
    )
    create_time: Optional[str] = Field(None, description="Creation time (ISO format)")
    user_id: Optional[str] = Field(
        None, min_length=1, max_length=255, description="User identifier"
    )
    device_id: Optional[str] = Field(
        None, min_length=1, max_length=100, description="Device identifier"
    )
    agent_id: Optional[str] = Field(
        None, min_length=1, max_length=100, description="Agent identifier"
    )
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Additional context info")
    enable_merge: bool = Field(True, description="Whether to enable context merging")

    @model_validator(mode="after")
    def check_content_provided(self):
        if not self.content_text and not self.content_images:
            raise ValueError("At least one of content_text or content_images must be provided")
        return self


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


def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse datetime string to datetime object"""
    if not dt_str:
        return None
    try:
        # Try ISO format first
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        try:
            # Try common formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"]:
                try:
                    return datetime.strptime(dt_str, fmt)
                except ValueError:
                    continue
        except Exception:
            pass
    return None


def _get_source_enum(source_str: str) -> ContextSource:
    """Convert source string to ContextSource enum"""
    source_map = {
        "document": ContextSource.LOCAL_FILE,
        "chat": ContextSource.CHAT_LOG,
        "vault": ContextSource.VAULT,
        "input": ContextSource.INPUT,
        "local_file": ContextSource.LOCAL_FILE,
        "web_link": ContextSource.WEB_LINK,
    }
    return source_map.get(source_str.lower(), ContextSource.INPUT)


def _get_format_enum(format_str: str) -> ContentFormat:
    """Convert format string to ContentFormat enum"""
    format_map = {
        "text": ContentFormat.TEXT,
        "image": ContentFormat.IMAGE,
        "markdown": ContentFormat.MARKDOWN,
        "pdf": ContentFormat.PDF,
        "html": ContentFormat.HTML,
    }
    return format_map.get(format_str.lower(), ContentFormat.TEXT)


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
    Unified chat push endpoint.

    - process_mode="buffer" (default): Push messages to Redis buffer. Optionally flush immediately.
    - process_mode="direct": Bypass buffer and send messages directly to the processing pipeline.

    Both modes schedule compression and hierarchy summary tasks.
    """
    try:
        text_chat = opencontext.capture_manager.get_component("text_chat")
        if text_chat is None:
            return convert_resp(
                code=503, status=503, message="TextChatCapture component not available"
            )

        if request.process_mode == "buffer":
            async def _push_all_messages():
                for msg in request.messages:
                    await text_chat.push_message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        user_id=request.user_id,
                        device_id=request.device_id,
                        agent_id=request.agent_id,
                    )
                if request.flush_immediately:
                    await text_chat.flush_user_buffer(
                        user_id=request.user_id,
                        device_id=request.device_id,
                        agent_id=request.agent_id,
                    )

            await asyncio.wait_for(_push_all_messages(), timeout=60.0)

            await _schedule_user_compression(
                user_id=request.user_id,
                device_id=request.device_id,
                agent_id=request.agent_id,
            )
            await _schedule_user_hierarchy_summary(
                user_id=request.user_id,
                device_id=request.device_id,
                agent_id=request.agent_id,
            )

            return convert_resp(
                message="Chat messages pushed successfully",
                data={"count": len(request.messages)},
            )

        else:  # direct
            background_tasks.add_task(
                text_chat.process_messages_directly,
                messages=request.messages,
                user_id=request.user_id,
                device_id=request.device_id,
                agent_id=request.agent_id,
            )
            background_tasks.add_task(
                _schedule_user_compression,
                user_id=request.user_id,
                device_id=request.device_id,
                agent_id=request.agent_id,
            )
            background_tasks.add_task(
                _schedule_user_hierarchy_summary,
                user_id=request.user_id,
                device_id=request.device_id,
                agent_id=request.agent_id,
            )

            return convert_resp(
                message="Chat messages submitted for processing",
                data={"message_count": len(request.messages)},
            )

    except asyncio.TimeoutError:
        logger.warning("Push chat timed out after 60s")
        return convert_resp(code=504, status=504, message="Request timed out")
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
            asyncio.to_thread(opencontext.add_document, file_path=file_path),
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

        err_msg = opencontext.add_document(file_path=file_path)

        if err_msg:
            return convert_resp(code=400, status=400, message=err_msg)

        return convert_resp(
            message="Document uploaded and queued successfully",
            data={"path": file_path, "original_filename": file.filename},
        )
    except Exception as e:
        logger.exception(f"Error uploading document: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


# ============================================================================
# Generic Context Push Endpoints
# ============================================================================


@router.post("/context", response_class=JSONResponse)
async def push_context(
    request: PushContextRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push a generic context item to the processing pipeline.
    This is a flexible endpoint that can handle various context types.
    """
    try:
        create_time = _parse_datetime(request.create_time) or datetime.now()

        # Build RawContextProperties
        raw_context = RawContextProperties(
            source=_get_source_enum(request.source),
            content_format=_get_format_enum(request.content_format),
            content_text=request.content_text,
            content_images=request.content_images,
            create_time=create_time,
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
            additional_info=request.additional_info or {},
            enable_merge=request.enable_merge,
        )

        # Process context with timeout protection
        success = await asyncio.wait_for(
            asyncio.to_thread(opencontext.add_context, raw_context),
            timeout=60.0,
        )

        if success:
            # Schedule hierarchy summary task for the user
            await _schedule_user_hierarchy_summary(
                user_id=request.user_id,
                device_id=request.device_id,
                agent_id=request.agent_id,
            )
            return convert_resp(message="Context pushed successfully")
        else:
            return convert_resp(code=500, status=500, message="Failed to process context")
    except asyncio.TimeoutError:
        logger.warning("Push context timed out after 60s")
        return convert_resp(code=504, status=504, message="Request timed out")
    except Exception as e:
        logger.exception(f"Error pushing context: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


