#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Agent registry CRUD routes."""

from typing import Optional

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
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None


# ============================================================================
# Endpoints
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
