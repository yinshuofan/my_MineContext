#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Storage backend package initialization file
"""

from .chromadb_backend import ChromaDBBackend
from .sqlite_backend import SQLiteBackend

# MySQL backend is optional, import only if pymysql is available
try:
    from .mysql_backend import MySQLBackend
    __all__ = ["SQLiteBackend", "ChromaDBBackend", "MySQLBackend"]
except ImportError:
    __all__ = ["SQLiteBackend", "ChromaDBBackend"]
