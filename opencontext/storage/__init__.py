# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Storage module - responsible for data storage and retrieval
"""

from opencontext.storage.redis_cache import (
    RedisCache,
    RedisCacheConfig,
    InMemoryCache,
    get_redis_cache,
    init_redis_cache,
    close_redis_cache,
    get_cache,
)

__all__ = [
    "RedisCache",
    "RedisCacheConfig",
    "InMemoryCache",
    "get_redis_cache",
    "init_redis_cache",
    "close_redis_cache",
    "get_cache",
]
