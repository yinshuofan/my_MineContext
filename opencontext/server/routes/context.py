# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Context management routes
"""

import datetime
import math
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
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.time_utils import get_timezone

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
    opencontext: OpenContext = Depends(get_context_lab),  # noqa: B008
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
    opencontext: OpenContext = Depends(get_context_lab),  # noqa: B008
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


@router.get("/api/contexts")
async def list_contexts_api(
    page: int = Query(1, ge=1, description="Page number"),  # noqa: B008
    limit: int = Query(15, ge=1, description="Items per page (max 100)"),  # noqa: B008
    type: str | None = Query(None, description="Filter by context type"),  # noqa: B008
    user_id: str | None = Query(None, description="Filter by user ID"),  # noqa: B008
    device_id: str | None = Query(None, description="Filter by device ID"),  # noqa: B008
    agent_id: str | None = Query(None, description="Filter by agent ID"),  # noqa: B008
    hierarchy_level: int | None = Query(None, description="Filter by hierarchy level"),  # noqa: B008
    start_date: str | None = Query(None, description="Start date filter"),  # noqa: B008
    end_date: str | None = Query(None, description="End date filter"),  # noqa: B008
    _auth: str = auth_dependency,
):
    """List processed contexts with filtering and pagination (JSON API)."""
    storage = get_storage()
    if storage is None:
        return convert_resp(
            code=503,
            status=503,
            message=(
                "Context storage is unavailable. Check vector database initialization, "
                "credentials, and network connectivity."
            ),
        )

    try:
        limit = min(max(limit, 1), 100)
        page = max(page, 1)
        types = []
        if type:
            types.append(type)

        # Build filter dict from query params
        storage_filter: dict = {}
        if hierarchy_level is not None:
            storage_filter["hierarchy_level"] = hierarchy_level
        if start_date:
            for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    start_dt = datetime.datetime.strptime(start_date, fmt).replace(
                        tzinfo=get_timezone()
                    )
                    storage_filter.setdefault("create_time_ts", {})["$gte"] = start_dt.timestamp()
                    break
                except ValueError:
                    continue
        if end_date:
            for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    end_dt = datetime.datetime.strptime(end_date, fmt).replace(
                        tzinfo=get_timezone()
                    )
                    if fmt == "%Y-%m-%d":
                        end_dt += datetime.timedelta(days=1)
                    else:
                        end_dt += datetime.timedelta(minutes=1)
                    storage_filter.setdefault("create_time_ts", {})["$lt"] = end_dt.timestamp()
                    break
                except ValueError:
                    continue

        context_types = [
            ct
            for ct in storage.get_available_context_types()
            if ct not in ("profile", "agent_profile", "agent_base_profile")
        ]
        types_for_query = list(types) if types else context_types

        # Get total count for pagination
        total_count = await storage.get_filtered_context_count(
            context_types=types_for_query,
            filter=storage_filter if storage_filter else None,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
        )
        total_pages = max(1, math.ceil(total_count / limit))
        page = min(page, total_pages)
        offset = (page - 1) * limit

        # Fetch data with skip_slice=True for correct cross-type global pagination.
        # Note: fetches (offset+limit) records per type, so cost is O(types * (offset+limit)).
        contexts_dict = await storage.get_all_processed_contexts(
            context_types=types_for_query,
            limit=limit + offset,
            offset=0,
            need_vector=False,
            filter=storage_filter if storage_filter else None,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            skip_slice=True,
        )
        contexts = []
        for backend_contexts in contexts_dict.values():
            contexts.extend(backend_contexts)

        # Sort with timezone-aware datetime handling, then global slice
        def get_sort_key(context):
            dt = context.properties.create_time
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=get_timezone())
            return dt

        contexts.sort(key=get_sort_key, reverse=True)
        contexts_to_display = contexts[offset : offset + limit]

        model_contexts = []
        for context in contexts_to_display:
            try:
                model = ProcessedContextModel.from_processed_context(context, project_root)
                model_contexts.append(model.model_dump(exclude={"embedding", "raw_contexts"}))
            except Exception:
                logger.exception("Failed to serialize context %s for /api/contexts", context.id)

        return convert_resp(
            data={
                "contexts": model_contexts,
                "page": page,
                "limit": limit,
                "total": total_count,
                "total_pages": total_pages,
                "context_types": context_types,
            }
        )

    except Exception as e:
        logger.exception("Failed to handle GET /api/contexts: %s", e)
        return convert_resp(code=500, status=500, message=f"Failed to load contexts: {str(e)}")


@router.get("/api/contexts/{context_id}")
async def get_context_api(
    context_id: str,
    context_type: str = Query(..., description="Context type (event, knowledge, etc.)"),  # noqa: B008
    opencontext: OpenContext = Depends(get_context_lab),  # noqa: B008
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
    opencontext: OpenContext = Depends(get_context_lab),  # noqa: B008
    _auth: str = auth_dependency,
):
    """Get all available context types."""
    try:
        context_types = opencontext.get_context_types()
        return context_types
    except Exception as e:
        logger.exception(f"Error getting context types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get context types: {str(e)}") from e


@router.post("/api/vector_search")
async def vector_search(
    request: VectorSearchRequest,
    opencontext: OpenContext = Depends(get_context_lab),  # noqa: B008
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
