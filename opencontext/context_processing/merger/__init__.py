#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Merger module for context processing.
Contains merge-related functionality — only knowledge merge strategy remains.
"""

from .context_merger import ContextMerger
from .merge_strategies import (
    ContextTypeAwareStrategy,
    KnowledgeMergeStrategy,
    StrategyFactory,
)

__all__ = [
    "ContextMerger",
    "ContextTypeAwareStrategy",
    "KnowledgeMergeStrategy",
    "StrategyFactory",
]
