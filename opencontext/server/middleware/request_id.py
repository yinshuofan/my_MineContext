# -*- coding: utf-8 -*-

"""
Request ID middleware for request tracing across components.
"""

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Context variable accessible from any async/sync code in the same request
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique request ID to each incoming request."""

    async def dispatch(self, request: Request, call_next):
        # Use client-provided ID or generate a short UUID
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
        request_id_var.set(rid)
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
