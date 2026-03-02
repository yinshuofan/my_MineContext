# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Health check routes
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from opencontext.server.middleware.auth import is_auth_enabled
from opencontext.server.opencontext import OpenContext
from opencontext.server.utils import convert_resp, get_context_lab

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return convert_resp(data={"status": "healthy", "service": "opencontext"})


@router.get("/api/health")
async def api_health_check(opencontext: OpenContext = Depends(get_context_lab)):
    """Detailed health check with service status"""
    try:
        health_data = {
            "status": "healthy",
            "service": "opencontext",
            "components": await opencontext.check_components_health(),
        }
        return convert_resp(data=health_data)
    except Exception as e:
        from opencontext.utils.logging_utils import get_logger

        logger = get_logger(__name__)
        logger.exception(f"Health check failed: {e}")
        return convert_resp(code=503, status=503, message="Service unhealthy")


@router.get("/api/auth/status")
async def auth_status():
    """Check if API authentication is enabled"""
    return convert_resp(data={"auth_enabled": is_auth_enabled()})


@router.get("/api/ready")
async def readiness_check(opencontext: OpenContext = Depends(get_context_lab)):
    """Readiness probe - checks all dependencies are connectable."""
    try:
        health_data = await opencontext.check_components_health()
        core_keys = ["config", "storage", "llm", "document_db", "redis"]
        all_healthy = all(health_data.get(k, False) for k in core_keys if k in health_data)
        status_code = 200 if all_healthy else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if all_healthy else "not_ready",
                "components": health_data,
            },
        )
    except Exception as e:
        from opencontext.utils.logging_utils import get_logger

        logger = get_logger(__name__)
        logger.exception(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e)},
        )
