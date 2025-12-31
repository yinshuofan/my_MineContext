#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Push API routes for external services to push data to the long-term memory backend.
This module provides HTTP endpoints for receiving context data from external services,
replacing the automatic capture/pull mechanisms with a push-based architecture.
"""

import base64
import json
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
    user_id: Optional[str],
    device_id: Optional[str] = None,
    agent_id: Optional[str] = None
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


# ============================================================================
# Request Models
# ============================================================================

class PushChatMessageRequest(BaseModel):
    """Push a single chat message"""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: List[Dict[str, Any]] = Field(
        ..., 
        description="Message content (OpenAI format: List of content blocks with type and text)"
    )
    user_id: Optional[str] = Field(None, description="User identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    agent_id: Optional[str] = Field(None, description="Agent identifier")
    timestamp: Optional[str] = Field(None, description="Message timestamp (ISO format)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PushChatMessagesRequest(BaseModel):
    """Push multiple chat messages in batch"""
    messages: List[Dict[str, Any]] = Field(
        ..., 
        description="List of chat messages (OpenAI format: role, content)"
    )
    user_id: Optional[str] = Field(None, description="Default user identifier")
    device_id: Optional[str] = Field(None, description="Default device identifier")
    agent_id: Optional[str] = Field(None, description="Default agent identifier")
    flush_immediately: bool = Field(False, description="Whether to flush buffer immediately")


class ProcessChatMessagesRequest(BaseModel):
    """Process chat messages directly (bypass buffer)"""
    messages: List[Dict[str, Any]] = Field(
        ..., 
        description="List of chat messages (OpenAI format: role, content[type, text] or other format)"
    )
    user_id: Optional[str] = Field(None, description="User identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    agent_id: Optional[str] = Field(None, description="Agent identifier")


class PushScreenshotRequest(BaseModel):
    """Push a screenshot"""
    # Either provide path (local file) or base64_data (remote upload)
    path: Optional[str] = Field(None, description="Local file path to screenshot")
    base64_data: Optional[str] = Field(None, description="Base64 encoded image data")
    filename: Optional[str] = Field(None, description="Filename for base64 uploads")
    window_title: str = Field("", description="Window title when screenshot was taken")
    app_name: str = Field("unknown", description="Application name")
    create_time: Optional[str] = Field(None, description="Screenshot creation time (ISO format)")
    user_id: Optional[str] = Field(None, description="User identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PushScreenshotsRequest(BaseModel):
    """Push multiple screenshots in batch"""
    screenshots: List[PushScreenshotRequest]


class PushDocumentRequest(BaseModel):
    """Push a document"""
    # Either provide file_path (local) or base64_data (remote upload)
    file_path: Optional[str] = Field(None, description="Local file path")
    base64_data: Optional[str] = Field(None, description="Base64 encoded document data")
    filename: Optional[str] = Field(None, description="Filename for base64 uploads")
    content_type: Optional[str] = Field(None, description="MIME type of the document")
    user_id: Optional[str] = Field(None, description="User identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PushActivityRequest(BaseModel):
    """Push an activity record"""
    title: str = Field(..., description="Activity title")
    content: str = Field(..., description="Activity content/description")
    start_time: Optional[str] = Field(None, description="Activity start time (ISO format)")
    end_time: Optional[str] = Field(None, description="Activity end time (ISO format)")
    resources: Optional[List[str]] = Field(None, description="Related resource paths/URLs")
    user_id: Optional[str] = Field(None, description="User identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PushContextRequest(BaseModel):
    """Push a generic context item"""
    source: str = Field(..., description="Context source type: screenshot, document, chat, activity, etc.")
    content_format: str = Field("text", description="Content format: text, image, markdown, etc.")
    content_text: Optional[str] = Field(None, description="Text content")
    content_images: Optional[List[str]] = Field(None, description="Image paths or base64 data")
    create_time: Optional[str] = Field(None, description="Creation time (ISO format)")
    user_id: Optional[str] = Field(None, description="User identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    agent_id: Optional[str] = Field(None, description="Agent identifier")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Additional context info")
    enable_merge: bool = Field(True, description="Whether to enable context merging")


class PushTodoRequest(BaseModel):
    """Push a todo item"""
    content: str = Field(..., description="Todo content")
    start_time: Optional[str] = Field(None, description="Start time (ISO format)")
    end_time: Optional[str] = Field(None, description="End time (ISO format)")
    status: int = Field(0, description="Status: 0=pending, 1=completed")
    urgency: int = Field(0, description="Urgency level: 0-3")
    assignee: Optional[str] = Field(None, description="Assignee")
    reason: Optional[str] = Field(None, description="Reason/notes")
    user_id: Optional[str] = Field(None, description="User identifier")


class PushTipRequest(BaseModel):
    """Push a tip"""
    content: str = Field(..., description="Tip content")
    user_id: Optional[str] = Field(None, description="User identifier")


class FlushBufferRequest(BaseModel):
    """Request to flush chat buffer"""
    user_id: Optional[str] = Field(None, description="User identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    agent_id: Optional[str] = Field(None, description="Agent identifier")


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
        "screenshot": ContextSource.SCREENSHOT,
        "document": ContextSource.LOCAL_FILE,
        "chat": ContextSource.CHAT_LOG,
        "activity": ContextSource.INPUT,
        "vault": ContextSource.VAULT,
        "text": ContextSource.TEXT,
        "input": ContextSource.INPUT,
        "local_file": ContextSource.LOCAL_FILE,
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
# Chat Message Push Endpoints
# ============================================================================

@router.post("/chat/message", response_class=JSONResponse)
async def push_chat_message(
    request: PushChatMessageRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push a single chat message to the context capture system.
    The message will be buffered and processed when the buffer is full.
    Uses async Redis operations to avoid blocking the event loop.
    """
    try:
        # Get TextChatCapture component
        text_chat = opencontext.capture_manager.get_component("text_chat")
        if text_chat is None:
            return convert_resp(code=503, status=503, message="TextChatCapture component not available")

        # Push message (async - non-blocking)
        await text_chat.push_message(
            role=request.role,
            content=request.content,
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
        )

        # Schedule compression task for the user
        await _schedule_user_compression(
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
        )

        return convert_resp(message="Chat message pushed successfully")
    except Exception as e:
        logger.exception(f"Error pushing chat message: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


@router.post("/chat/messages", response_class=JSONResponse)
async def push_chat_messages(
    request: PushChatMessagesRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push multiple chat messages in batch.
    Uses async Redis operations to avoid blocking the event loop.
    """
    try:
        text_chat = opencontext.capture_manager.get_component("text_chat")
        if text_chat is None:
            return convert_resp(code=503, status=503, message="TextChatCapture component not available")

        # Push each message (async - non-blocking)
        for msg in request.messages:
            await text_chat.push_message(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                user_id=msg.get("user_id") or request.user_id,
                device_id=msg.get("device_id") or request.device_id,
                agent_id=msg.get("agent_id") or request.agent_id,
            )

        # Flush buffer if requested (async - non-blocking)
        if request.flush_immediately:
            await text_chat.flush_user_buffer(
                user_id=request.user_id,
                device_id=request.device_id,
                agent_id=request.agent_id,
            )

        return convert_resp(
            message=f"Pushed {len(request.messages)} chat messages successfully",
            data={"count": len(request.messages)}
        )
    except Exception as e:
        logger.exception(f"Error pushing chat messages: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


@router.post("/chat/process", response_class=JSONResponse)
async def process_chat_messages(
    request: ProcessChatMessagesRequest,
    background_tasks: BackgroundTasks,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Process chat messages directly and send to pipeline, bypassing Redis buffer.
    Messages are immediately sent to the context processing pipeline without buffering.
    This endpoint returns immediately after task submission.
    """
    try:
        message_capturer = opencontext.capture_manager.get_component("text_chat")
        if message_capturer is None:
            return convert_resp(code=503, status=503, message="TextChatCapture component not available")

        background_tasks.add_task(
            message_capturer.process_messages_directly,
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

        return convert_resp(
            message="Chat messages submitted for processing",
            data={
                "message_count": len(request.messages),
                "user_id": request.user_id,
                "device_id": request.device_id,
                "agent_id": request.agent_id,
            }
        )
    except Exception as e:
        logger.exception(f"Error submitting chat messages: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


@router.post("/chat/flush", response_class=JSONResponse)
async def flush_chat_buffer(
    request: FlushBufferRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Manually flush the chat message buffer for a specific user.
    Uses async Redis operations to avoid blocking the event loop.
    """
    try:
        text_chat = opencontext.capture_manager.get_component("text_chat")
        if text_chat is None:
            return convert_resp(code=503, status=503, message="TextChatCapture component not available")

        # Flush buffer (async - non-blocking)
        await text_chat.flush_user_buffer(
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
        )

        return convert_resp(message="Chat buffer flushed successfully")
    except Exception as e:
        logger.exception(f"Error flushing chat buffer: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


# ============================================================================
# Screenshot Push Endpoints
# ============================================================================

@router.post("/screenshot", response_class=JSONResponse)
async def push_screenshot(
    request: PushScreenshotRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push a screenshot to the context capture system.
    Supports both local file path and base64 encoded data.
    """
    try:
        # Determine file path
        file_path = request.path
        if not file_path and request.base64_data:
            # Save base64 data to temp file
            from opencontext.config.global_config import GlobalConfig
            config = GlobalConfig.get_instance().get_config()
            storage_path = config.get("capture", {}).get("screenshot", {}).get("storage_path", "./screenshots")
            file_path = _save_base64_to_temp_file(
                request.base64_data,
                request.filename or "screenshot.png",
                storage_path
            )
        
        if not file_path:
            return convert_resp(code=400, status=400, message="Either 'path' or 'base64_data' must be provided")
        
        # Use create_time or current time
        create_time = request.create_time or datetime.now().isoformat()
        
        # Add screenshot via OpenContext
        err_msg = opencontext.add_screenshot(
            path=file_path,
            window=request.window_title,
            create_time=create_time,
            app=request.app_name,
        )
        
        if err_msg:
            return convert_resp(code=400, status=400, message=err_msg)

        # Schedule compression task for the user
        await _schedule_user_compression(
            user_id=request.user_id,
            device_id=request.device_id,
        )

        return convert_resp(message="Screenshot pushed successfully", data={"path": file_path})
    except Exception as e:
        logger.exception(f"Error pushing screenshot: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


@router.post("/screenshots", response_class=JSONResponse)
async def push_screenshots(
    request: PushScreenshotsRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push multiple screenshots in batch.
    """
    try:
        results = []
        for screenshot in request.screenshots:
            file_path = screenshot.path
            if not file_path and screenshot.base64_data:
                from opencontext.config.global_config import GlobalConfig
                config = GlobalConfig.get_instance().get_config()
                storage_path = config.get("capture", {}).get("screenshot", {}).get("storage_path", "./screenshots")
                file_path = _save_base64_to_temp_file(
                    screenshot.base64_data,
                    screenshot.filename or "screenshot.png",
                    storage_path
                )
            
            if not file_path:
                results.append({"success": False, "error": "No path or base64_data provided"})
                continue
            
            create_time = screenshot.create_time or datetime.now().isoformat()
            err_msg = opencontext.add_screenshot(
                path=file_path,
                window=screenshot.window_title,
                create_time=create_time,
                app=screenshot.app_name,
            )
            
            if err_msg:
                results.append({"success": False, "path": file_path, "error": err_msg})
            else:
                results.append({"success": True, "path": file_path})
        
        success_count = sum(1 for r in results if r.get("success"))
        return convert_resp(
            message=f"Pushed {success_count}/{len(results)} screenshots successfully",
            data={"results": results, "success_count": success_count, "total": len(results)}
        )
    except Exception as e:
        logger.exception(f"Error pushing screenshots: {e}")
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
                request.base64_data,
                request.filename or "document",
                "./uploads/documents"
            )
        
        if not file_path:
            return convert_resp(code=400, status=400, message="Either 'file_path' or 'base64_data' must be provided")
        
        err_msg = opencontext.add_document(file_path=file_path)
        
        if err_msg:
            return convert_resp(code=400, status=400, message=err_msg)
        
        return convert_resp(message="Document pushed successfully", data={"path": file_path})
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
            data={"path": file_path, "original_filename": file.filename}
        )
    except Exception as e:
        logger.exception(f"Error uploading document: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


# ============================================================================
# Activity Push Endpoints
# ============================================================================

@router.post("/activity", response_class=JSONResponse)
async def push_activity(
    request: PushActivityRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push an activity record to the storage.
    """
    try:
        storage = opencontext.storage
        if storage is None:
            return convert_resp(code=503, status=503, message="Storage not available")
        
        start_time = _parse_datetime(request.start_time) or datetime.now()
        end_time = _parse_datetime(request.end_time) or datetime.now()
        
        resources_json = json.dumps(request.resources, ensure_ascii=False) if request.resources else None
        metadata_json = json.dumps(request.metadata, ensure_ascii=False) if request.metadata else None
        
        activity_id = storage.insert_activity(
            title=request.title,
            content=request.content,
            resources=resources_json,
            metadata=metadata_json,
            start_time=start_time,
            end_time=end_time,
        )
        
        return convert_resp(
            message="Activity pushed successfully",
            data={"activity_id": activity_id}
        )
    except Exception as e:
        logger.exception(f"Error pushing activity: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


# ============================================================================
# Todo Push Endpoints
# ============================================================================

@router.post("/todo", response_class=JSONResponse)
async def push_todo(
    request: PushTodoRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push a todo item to the storage.
    """
    try:
        storage = opencontext.storage
        if storage is None:
            return convert_resp(code=503, status=503, message="Storage not available")
        
        start_time = _parse_datetime(request.start_time)
        end_time = _parse_datetime(request.end_time)
        
        todo_id = storage.insert_todo(
            content=request.content,
            start_time=start_time,
            end_time=end_time,
            status=request.status,
            urgency=request.urgency,
            assignee=request.assignee,
            reason=request.reason,
        )
        
        return convert_resp(
            message="Todo pushed successfully",
            data={"todo_id": todo_id}
        )
    except Exception as e:
        logger.exception(f"Error pushing todo: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


# ============================================================================
# Tip Push Endpoints
# ============================================================================

@router.post("/tip", response_class=JSONResponse)
async def push_tip(
    request: PushTipRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push a tip to the storage.
    """
    try:
        storage = opencontext.storage
        if storage is None:
            return convert_resp(code=503, status=503, message="Storage not available")
        
        tip_id = storage.insert_tip(content=request.content)
        
        return convert_resp(
            message="Tip pushed successfully",
            data={"tip_id": tip_id}
        )
    except Exception as e:
        logger.exception(f"Error pushing tip: {e}")
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
        
        # Process context
        success = opencontext.add_context(raw_context)
        
        if success:
            return convert_resp(message="Context pushed successfully")
        else:
            return convert_resp(code=500, status=500, message="Failed to process context")
    except Exception as e:
        logger.exception(f"Error pushing context: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")


# ============================================================================
# Batch Push Endpoint
# ============================================================================

class BatchPushItem(BaseModel):
    """A single item in a batch push request"""
    type: str = Field(..., description="Item type: chat, screenshot, document, activity, todo, tip, context")
    data: Dict[str, Any] = Field(..., description="Item data matching the corresponding push request schema")


class BatchPushRequest(BaseModel):
    """Batch push request containing multiple items of different types"""
    items: List[BatchPushItem]
    user_id: Optional[str] = Field(None, description="Default user identifier")
    device_id: Optional[str] = Field(None, description="Default device identifier")


@router.post("/batch", response_class=JSONResponse)
async def push_batch(
    request: BatchPushRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Push multiple items of different types in a single request.
    This is useful for syncing data from external services efficiently.
    Uses async Redis operations for chat items to avoid blocking the event loop.
    """
    try:
        results = []
        storage = opencontext.storage
        text_chat = opencontext.capture_manager.get_component("text_chat")

        for item in request.items:
            item_result = {"type": item.type, "success": False}
            try:
                if item.type == "chat":
                    if text_chat:
                        # Use async version to avoid blocking the event loop
                        await text_chat.push_message(
                            role=item.data.get("role", "user"),
                            content=item.data.get("content", ""),
                            user_id=item.data.get("user_id") or request.user_id,
                            device_id=item.data.get("device_id") or request.device_id,
                            agent_id=item.data.get("agent_id"),
                        )
                        item_result["success"] = True
                
                elif item.type == "screenshot":
                    path = item.data.get("path")
                    if path:
                        err = opencontext.add_screenshot(
                            path=path,
                            window=item.data.get("window_title", ""),
                            create_time=item.data.get("create_time", datetime.now().isoformat()),
                            app=item.data.get("app_name", "unknown"),
                        )
                        item_result["success"] = err is None
                        if err:
                            item_result["error"] = err
                
                elif item.type == "document":
                    file_path = item.data.get("file_path")
                    if file_path:
                        err = opencontext.add_document(file_path=file_path)
                        item_result["success"] = err is None
                        if err:
                            item_result["error"] = err
                
                elif item.type == "activity":
                    if storage:
                        activity_id = storage.insert_activity(
                            title=item.data.get("title", ""),
                            content=item.data.get("content", ""),
                            resources=json.dumps(item.data.get("resources")) if item.data.get("resources") else None,
                            metadata=json.dumps(item.data.get("metadata")) if item.data.get("metadata") else None,
                            start_time=_parse_datetime(item.data.get("start_time")),
                            end_time=_parse_datetime(item.data.get("end_time")),
                        )
                        item_result["success"] = True
                        item_result["id"] = activity_id
                
                elif item.type == "todo":
                    if storage:
                        todo_id = storage.insert_todo(
                            content=item.data.get("content", ""),
                            start_time=_parse_datetime(item.data.get("start_time")),
                            end_time=_parse_datetime(item.data.get("end_time")),
                            status=item.data.get("status", 0),
                            urgency=item.data.get("urgency", 0),
                            assignee=item.data.get("assignee"),
                            reason=item.data.get("reason"),
                        )
                        item_result["success"] = True
                        item_result["id"] = todo_id
                
                elif item.type == "tip":
                    if storage:
                        tip_id = storage.insert_tip(content=item.data.get("content", ""))
                        item_result["success"] = True
                        item_result["id"] = tip_id
                
                else:
                    item_result["error"] = f"Unknown item type: {item.type}"
            
            except Exception as e:
                item_result["error"] = str(e)
            
            results.append(item_result)
        
        success_count = sum(1 for r in results if r.get("success"))
        return convert_resp(
            message=f"Batch push completed: {success_count}/{len(results)} items succeeded",
            data={"results": results, "success_count": success_count, "total": len(results)}
        )
    except Exception as e:
        logger.exception(f"Error in batch push: {e}")
        return convert_resp(code=500, status=500, message=f"Internal server error: {e}")
