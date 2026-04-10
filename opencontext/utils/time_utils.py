#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Global timezone utilities.

Provides a single configured timezone for the entire application.
All code should use ``now()`` from this module instead of
``datetime.now()`` or ``datetime.now(tz=timezone.utc)``.

Initialize once at startup via ``init_timezone(tz_name)``.

**Important:** Do NOT call ``now()`` at module-import time or in
class-level expressions. The timezone is initialized at startup,
after module imports. Use only inside function/method bodies or
``default_factory`` lambdas.
"""

import datetime
from zoneinfo import ZoneInfo

from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

_configured_tz: datetime.tzinfo = datetime.UTC


def init_timezone(tz_name: str | None = None) -> None:
    """Initialize the global timezone from config.

    Args:
        tz_name: IANA timezone name (e.g. ``"Asia/Shanghai"``, ``"UTC"``).
                 Defaults to ``"UTC"`` if *None* or empty.
    """
    global _configured_tz
    name = tz_name or "UTC"
    try:
        _configured_tz = ZoneInfo(name)
        logger.info(f"Global timezone set to: {name}")
    except KeyError:
        logger.error(f"Unknown timezone '{name}', falling back to UTC")
        _configured_tz = datetime.UTC


def get_timezone() -> datetime.tzinfo:
    """Return the configured timezone object."""
    return _configured_tz


def now() -> datetime.datetime:
    """Return the current time in the configured timezone.

    This is the **primary** function for getting "now" throughout
    the codebase.
    """
    return datetime.datetime.now(tz=_configured_tz)


def utc_now() -> datetime.datetime:
    """Return the current time in UTC.

    Use this **only** when a protocol mandates UTC
    (e.g. AWS Signature V4, HTTP Date headers).
    For all other cases, use ``now()``.
    """
    return datetime.datetime.now(tz=datetime.UTC)


def today_start() -> datetime.datetime:
    """Return midnight of today in the configured timezone.

    Useful for "today events" boundary calculations.
    """
    n = now()
    return n.replace(hour=0, minute=0, second=0, microsecond=0)
