
# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Context management routes
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from opencontext.models.context import ProcessedContextModel
from opencontext.models.enums import ContentFormat, ContextSource
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.opencontext import OpenContext
from opencontext.server.utils import convert_resp, get_context_lab
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["context"])

project_root = Path(__file__).parent.parent.parent.parent.resolve()
templates_path = Path(__file__).parent.parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=templates_path)


class ContextIn(BaseModel):
    source: ContextSource
    content_format: ContentFormat
    data: Any
    metadata: dict | None = {}


class UpdateContextIn(BaseModel):
    title: str | None = None
    summary: str | None = None
    keywords: list[str] | None = None


class QueryIn(BaseModel):
    query: str


class ConsumeIn(BaseModel):
    query: str
    context_ids: list[str]


class ContextDetailRequest(BaseModel):
    id: str
    context_type: str


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    context_types: list[str] | None = None
    filters: dict[str, Any] | None = None
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1). Results below this score are filtered out.",
    )
    # Multi-user support fields
    user_id: str | None = None
    device_id: str | None = None
    agent_id: str | None = None


@router.post("/contexts/delete")
async def delete_context(
    detail_request: ContextDetailRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """Delete a processed context by its ID and context_type."""
    success = await opencontext.delete_context(detail_request.id, detail_request.context_type)
    if not success:
        raise HTTPException(status_code=404, detail="Context not found or failed to delete")
    return {"message": "Context deleted successfully"}


@router.post("/contexts/detail", response_class=HTMLResponse)
async def read_context_detail(
    detail_request: ContextDetailRequest,
    request: Request,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    context = await opencontext.get_context(detail_request.id, detail_request.context_type)
    if context is None:
        return templates.TemplateResponse(
            "error.html", {"request": request, "message": "Context not found"}, status_code=404
        )

    return templates.TemplateResponse(
        "context_detail.html",
        {
            "request": request,
            "context": ProcessedContextModel.from_processed_context(context, project_root),
        },
    )


@router.get("/api/contexts/{context_id}")
async def get_context_api(
    context_id: str,
    context_type: str = Query(..., description="Context type (event, knowledge, etc.)"),
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """Get a single context by ID as JSON (used for tree lazy loading)."""
    context = await opencontext.get_context(context_id, context_type)
    if context is None:
        raise HTTPException(status_code=404, detail="Context not found")
    model = ProcessedContextModel.from_processed_context(context, project_root)
    return model.model_dump()


@router.get("/api/context_types")
async def get_context_types(
    opencontext: OpenContext = Depends(get_context_lab), _auth: str = auth_dependency
):
    """Get all available context types."""
    try:
        context_types = opencontext.get_context_types()
        return context_types
    except Exception as e:
        logger.exception(f"Error getting context types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get context types: {str(e)}")


@router.post("/api/vector_search")
async def vector_search(
    request: VectorSearchRequest,
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """Directly search vector database without using LLM.

    Supports multi-user filtering through user_id, device_id, and agent_id parameters.
    """
    try:
        results = await opencontext.search(
            query=request.query,
            top_k=request.top_k,
            context_types=request.context_types,
            filters=request.filters,
            user_id=request.user_id,
            device_id=request.device_id,
            agent_id=request.agent_id,
            score_threshold=request.score_threshold,
        )

        return convert_resp(
            data={
                "results": results,
                "total": len(results),
                "query": request.query,
                "top_k": request.top_k,
                "context_types": request.context_types,
                "filters": request.filters,
                "user_id": request.user_id,
                "device_id": request.device_id,
                "agent_id": request.agent_id,
            }
        )

    except Exception as e:
        logger.exception(f"Error in vector search: {e}")
        return convert_resp(code=500, status=500, message=f"Vector search failed: {str(e)}")
