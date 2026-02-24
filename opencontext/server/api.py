# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Server component: api - Main router configuration
"""

from pathlib import Path

from fastapi import APIRouter

from opencontext.utils.logging_utils import get_logger

# Import route modules
from .routes import (
    agent_chat,
    context,
    conversation,
    documents,
    health,
    memory_cache,
    messages,
    monitoring,
    push,
    search,
    settings,
    vaults,
    web,
)

logger = get_logger(__name__)

router = APIRouter()

project_root = Path(__file__).parent.parent.parent.resolve()


# Include all route modules
router.include_router(health.router)
router.include_router(web.router)
router.include_router(context.router)
router.include_router(monitoring.router)
router.include_router(vaults.router)
router.include_router(agent_chat.router)
router.include_router(settings.router)
router.include_router(conversation.router)
router.include_router(messages.router)
router.include_router(documents.router)
router.include_router(push.router)
router.include_router(search.router)
router.include_router(memory_cache.router)
