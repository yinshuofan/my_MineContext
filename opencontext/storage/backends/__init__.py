#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Storage backend package initialization file
"""

from .chromadb_backend import ChromaDBBackend
from .sqlite_backend import SQLiteBackend

__all__ = ["SQLiteBackend", "ChromaDBBackend"]

# MySQL backend is optional, import only if pymysql is available
try:
    from .mysql_backend import MySQLBackend
    __all__.append("MySQLBackend")
except ImportError:
    pass

# Qdrant backend is optional
try:
    from .qdrant_backend import QdrantBackend
    __all__.append("QdrantBackend")
except ImportError:
    pass

# DashVector backend is optional, import only if dashvector is available
try:
    from .dashvector_backend import DashVectorBackend
    __all__.append("DashVectorBackend")
except ImportError:
    pass

# VikingDB backend is optional
try:
    from .vikingdb_backend import VikingDBBackend
    __all__.append("VikingDBBackend")
except ImportError:
    pass
