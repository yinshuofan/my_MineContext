#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Users listing endpoint — distinct (user_id, device_id, agent_id) tuples.

NOTE: This is a demo/fallback implementation that queries the profiles table.
It only returns users who have generated a profile. In production, the embedding
platform should provide the user list from its own user system.
"""

from fastapi import APIRouter

from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.utils import convert_resp
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["users"])


@router.get("/api/users")
async def list_users(_auth: str = auth_dependency):
    """Return distinct (user_id, device_id, agent_id) tuples.

    Demo implementation: queries profiles table (only users with generated profiles).
    In production, replace with the embedding platform's user system.
    """
    storage = get_storage()
    if not storage:
        return convert_resp(code=503, status=503, message="Storage not available")

    users = await storage.list_distinct_users()
    return convert_resp(data={"users": users, "total": len(users)})
