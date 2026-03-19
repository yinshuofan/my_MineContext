#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Agent registry CRUD routes and agent base memory endpoints."""

import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    ProcessedContextModel,
    Vectorize,
)
from opencontext.models.enums import ContentFormat, ContextType
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.utils import convert_resp
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])


# ============================================================================
# Request Models
# ============================================================================


class CreateAgentRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""


class UpdateAgentRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None


class BaseProfileRequest(BaseModel):
    factual_profile: str
    behavioral_profile: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    importance: int = 0


class BaseEventItem(BaseModel):
    title: str
    summary: str
    event_time: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    importance: int = 5


class BaseEventsRequest(BaseModel):
    events: List[BaseEventItem] = Field(..., min_length=1)


# ============================================================================
# Agent CRUD Endpoints
# ============================================================================


@router.post("")
async def create_agent(request: CreateAgentRequest, _auth: str = auth_dependency):
    """Register a new agent."""
    if request.agent_id == "__base__":
        raise HTTPException(status_code=400, detail="agent_id '__base__' is reserved for system use")

    storage = get_storage()
    success = await storage.create_agent(request.agent_id, request.name, request.description)
    if not success:
        raise HTTPException(status_code=400, detail="Agent creation failed (ID may already exist)")
    return convert_resp(data={"agent_id": request.agent_id}, message="Agent created")


@router.get("")
async def list_agents(_auth: str = auth_dependency):
    """List all active agents."""
    storage = get_storage()
    agents = await storage.list_agents()
    return convert_resp(data={"agents": agents})


@router.get("/{agent_id}")
async def get_agent(agent_id: str, _auth: str = auth_dependency):
    """Get a single agent by ID."""
    storage = get_storage()
    agent = await storage.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return convert_resp(data={"agent": agent})


@router.put("/{agent_id}")
async def update_agent(
    agent_id: str, request: UpdateAgentRequest, _auth: str = auth_dependency
):
    """Update agent name and/or description."""
    storage = get_storage()
    success = await storage.update_agent(agent_id, name=request.name, description=request.description)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found or update failed")
    return convert_resp(message="Agent updated")


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, _auth: str = auth_dependency):
    """Soft-delete an agent."""
    storage = get_storage()
    success = await storage.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return convert_resp(message="Agent deleted")


# ============================================================================
# Agent Base Memory — Profile
# ============================================================================


@router.post("/{agent_id}/base/profile")
async def set_base_profile(
    agent_id: str, request: BaseProfileRequest, _auth: str = auth_dependency
):
    """Set or overwrite the agent's base profile.

    The base profile uses ``user_id="__base__"`` as a sentinel to distinguish
    it from per-user profiles that are generated during conversations.
    """
    storage = get_storage()
    agent = await storage.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    success = await storage.upsert_profile(
        user_id="__base__",
        device_id="default",
        agent_id=agent_id,
        context_type="agent_profile",
        factual_profile=request.factual_profile,
        behavioral_profile=request.behavioral_profile,
        entities=request.entities,
        importance=request.importance,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save base profile")
    return convert_resp(message="Base profile saved")


@router.get("/{agent_id}/base/profile")
async def get_base_profile(agent_id: str, _auth: str = auth_dependency):
    """Retrieve the agent's base profile."""
    storage = get_storage()
    profile = await storage.get_profile(
        user_id="__base__",
        device_id="default",
        agent_id=agent_id,
        context_type="agent_profile",
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Base profile not found")
    return convert_resp(data={"profile": profile})


# ============================================================================
# Agent Base Memory — Events
# ============================================================================


@router.post("/{agent_id}/base/events")
async def push_base_events(
    agent_id: str, request: BaseEventsRequest, _auth: str = auth_dependency
):
    """Push structured base events for an agent (no LLM extraction).

    Generates embeddings from ``title + summary + keywords`` and stores directly
    into the vector DB as ``AGENT_EVENT`` contexts.
    """
    storage = get_storage()
    agent = await storage.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    contexts: List[ProcessedContext] = []
    for event in request.events:
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        if event.event_time:
            event_time = datetime.datetime.fromisoformat(event.event_time)
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=datetime.timezone.utc)
        else:
            event_time = now

        text_for_embedding = f"{event.title}\n{event.summary}\n{', '.join(event.keywords)}"
        vectorize = Vectorize(
            input=[{"type": "text", "text": text_for_embedding}],
            content_format=ContentFormat.TEXT,
        )
        await do_vectorize(vectorize)

        ctx = ProcessedContext(
            properties=ContextProperties(
                raw_properties=[],
                create_time=now,
                update_time=now,
                event_time=event_time,
                time_bucket=event_time.strftime("%Y-%m-%dT%H:%M:%S"),
                is_processed=True,
                agent_id=agent_id,
            ),
            extracted_data=ExtractedData(
                title=event.title,
                summary=event.summary,
                keywords=event.keywords,
                entities=event.entities,
                context_type=ContextType.AGENT_EVENT,
                importance=event.importance,
                confidence=10,
            ),
            vectorize=vectorize,
        )
        contexts.append(ctx)

    result = await storage.batch_upsert_processed_context(contexts)
    success = result is not None
    return convert_resp(
        data={"count": len(contexts) if success else 0},
        message="Base events saved" if success else "Failed to save base events",
    )


@router.get("/{agent_id}/base/events")
async def list_base_events(
    agent_id: str,
    limit: int = 50,
    offset: int = 0,
    _auth: str = auth_dependency,
):
    """List base events for an agent.

    Base events are identified by having no ``user_id`` (they are pre-interaction
    background events pushed via this endpoint).
    """
    storage = get_storage()
    result = await storage.get_all_processed_contexts(
        context_types=[ContextType.AGENT_EVENT.value],
        agent_id=agent_id,
        limit=limit + offset,
    )
    contexts = result.get(ContextType.AGENT_EVENT.value, [])
    # Filter to base events (no user_id — base events have user_id=None)
    base_events = [c for c in contexts if not getattr(c.properties, "user_id", None)]
    page = base_events[offset : offset + limit]
    return convert_resp(
        data={
            "events": [
                ProcessedContextModel.from_processed_context(c, Path(".")).model_dump()
                for c in page
            ]
        }
    )


@router.delete("/{agent_id}/base/events/{event_id}")
async def delete_base_event(
    agent_id: str, event_id: str, _auth: str = auth_dependency
):
    """Delete a single base event by ID."""
    storage = get_storage()
    success = await storage.delete_processed_context(event_id, ContextType.AGENT_EVENT.value)
    if not success:
        raise HTTPException(status_code=404, detail="Event not found")
    return convert_resp(message="Event deleted")
