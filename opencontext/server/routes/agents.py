#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Agent registry CRUD routes and agent base memory endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
    name: str | None = Field(None, max_length=255)
    description: str | None = None


class BaseProfileRequest(BaseModel):
    factual_profile: str
    behavioral_profile: str | None = None
    entities: list[str] = Field(default_factory=list)
    importance: int = 0


# ============================================================================
# Agent CRUD Endpoints
# ============================================================================


@router.post("")
async def create_agent(request: CreateAgentRequest, _auth: str = auth_dependency):
    """Register a new agent."""
    if request.agent_id == "__base__":
        raise HTTPException(
            status_code=400, detail="agent_id '__base__' is reserved for system use"
        )

    storage = get_storage()
    success = await storage.create_agent(request.agent_id, request.name, request.description)  # type: ignore[union-attr]
    if not success:
        raise HTTPException(status_code=400, detail="Agent creation failed (ID may already exist)")
    return convert_resp(data={"agent_id": request.agent_id}, message="Agent created")


@router.get("")
async def list_agents(_auth: str = auth_dependency):
    """List all active agents."""
    storage = get_storage()
    agents = await storage.list_agents()  # type: ignore[union-attr]
    return convert_resp(data={"agents": agents})


@router.get("/{agent_id}")
async def get_agent(agent_id: str, _auth: str = auth_dependency):
    """Get a single agent by ID."""
    storage = get_storage()
    agent = await storage.get_agent(agent_id)  # type: ignore[union-attr]
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return convert_resp(data={"agent": agent})


@router.put("/{agent_id}")
async def update_agent(agent_id: str, request: UpdateAgentRequest, _auth: str = auth_dependency):
    """Update agent name and/or description."""
    storage = get_storage()
    success = await storage.update_agent(  # type: ignore[union-attr]
        agent_id, name=request.name, description=request.description
    )
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found or update failed")
    return convert_resp(message="Agent updated")


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, _auth: str = auth_dependency):
    """Soft-delete an agent."""
    storage = get_storage()
    success = await storage.delete_agent(agent_id)  # type: ignore[union-attr]
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

    The base profile is stored with ``context_type="agent_base_profile"``
    to distinguish it from per-user profiles.
    """
    storage = get_storage()
    agent = await storage.get_agent(agent_id)  # type: ignore[union-attr]
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    success = await storage.upsert_profile(  # type: ignore[union-attr]
        user_id="__base__",
        device_id="default",
        agent_id=agent_id,
        context_type="agent_base_profile",
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
    profile = await storage.get_profile(  # type: ignore[union-attr]
        user_id="__base__",
        device_id="default",
        agent_id=agent_id,
        context_type="agent_base_profile",
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Base profile not found")
    return convert_resp(data={"profile": profile})
