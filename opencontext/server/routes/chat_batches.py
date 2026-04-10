#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Chat batches debug/tracing endpoints."""

import math
from pathlib import Path

from fastapi import APIRouter, HTTPException

from opencontext.models.context import ProcessedContextModel
from opencontext.models.enums import ContextType
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.utils import convert_resp
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/chat-batches", tags=["chat-batches"])


@router.get("")
async def list_chat_batches(
    user_id: str | None = None,
    device_id: str | None = None,
    agent_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    limit: int = 20,
    _auth: str = auth_dependency,
):
    """List chat batches with optional filters and pagination."""
    limit = min(max(limit, 1), 100)
    page = max(page, 1)
    storage = get_storage()
    offset = (page - 1) * limit

    batches = await storage.list_chat_batches(
        user_id=user_id,
        device_id=device_id,
        agent_id=agent_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )
    total = await storage.count_chat_batches(
        user_id=user_id,
        device_id=device_id,
        agent_id=agent_id,
        start_date=start_date,
        end_date=end_date,
    )

    return convert_resp(
        data={
            "batches": batches,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": math.ceil(total / limit) if total > 0 else 1,
        }
    )


@router.get("/{batch_id}")
async def get_chat_batch(batch_id: str, _auth: str = auth_dependency):
    """Get a single chat batch with its messages."""
    storage = get_storage()
    batch = await storage.get_chat_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Chat batch not found")
    return convert_resp(data={"batch": batch})


@router.get("/{batch_id}/contexts")
async def get_batch_contexts(batch_id: str, _auth: str = auth_dependency):
    """Get vector DB contexts produced by a specific chat batch."""
    storage = get_storage()

    # Exclude profile types (stored in relational DB, not vector DB)
    all_context_types = [
        ct.value
        for ct in ContextType
        if ct
        not in (ContextType.PROFILE, ContextType.AGENT_PROFILE, ContextType.AGENT_BASE_PROFILE)
    ]

    filter_dict = {"raw_type": {"$eq": "chat_batch"}, "raw_id": {"$eq": batch_id}}
    results = await storage.get_all_processed_contexts(
        context_types=all_context_types,
        filter=filter_dict,
        limit=100,
        need_vector=False,
    )

    contexts = []
    for type_contexts in results.values():
        for ctx in type_contexts:
            try:
                contexts.append(
                    ProcessedContextModel.from_processed_context(ctx, Path(".")).model_dump()
                )
            except Exception:
                logger.warning(f"Failed to serialize context {ctx.id}")

    return convert_resp(data={"contexts": contexts})
