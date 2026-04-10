#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Storage backend package initialization file
"""

from .sqlite_backend import SQLiteBackend

__all__ = ["SQLiteBackend"]

# MySQL backend is optional, import only if pymysql is available
try:
    from .mysql_backend import MySQLBackend  # noqa: F401

    __all__.append("MySQLBackend")
except ImportError:
    pass

# Qdrant backend is optional
try:
    from .qdrant_backend import QdrantBackend  # noqa: F401

    __all__.append("QdrantBackend")
except ImportError:
    pass

# VikingDB backend is optional
try:
    from .vikingdb_backend import VikingDBBackend  # noqa: F401

    __all__.append("VikingDBBackend")
except ImportError:
    pass
