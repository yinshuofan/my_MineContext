#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
MySQL document note storage backend implementation (async via asyncmy)
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import asyncmy
import asyncmy.cursors

from opencontext.storage.base_storage import (
    DataType,
    DocumentData,
    IDocumentStorageBackend,
    QueryResult,
    StorageType,
)
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MySQLBackend(IDocumentStorageBackend):
    """
    MySQL document note storage backend (async)
    Specialized for storing activity generated markdown content and notes
    """

    def __init__(self):
        self.db_config: Optional[Dict[str, Any]] = None
        self._initialized = False
        self._pool = None

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize MySQL database"""
        try:
            # Get MySQL configuration
            db_config = config.get("config", {})
            self.db_config = {
                "host": db_config.get("host", "localhost"),
                "port": db_config.get("port", 3306),
                "user": db_config.get("user", "root"),
                "password": db_config.get("password", ""),
                "db": db_config.get("database", "opencontext"),
                "charset": db_config.get("charset", "utf8mb4"),
                "autocommit": False,
            }

            # Create database if not exists
            temp_config = {
                "host": self.db_config["host"],
                "port": self.db_config["port"],
                "user": self.db_config["user"],
                "password": self.db_config["password"],
                "charset": self.db_config["charset"],
            }
            temp_conn = await asyncmy.connect(**temp_config)
            try:
                async with temp_conn.cursor() as cur:
                    await cur.execute(
                        f"CREATE DATABASE IF NOT EXISTS `{self.db_config['db']}` "
                        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                await temp_conn.commit()
            finally:
                await temp_conn.ensure_closed()

            # Create connection pool
            self._pool = await asyncmy.create_pool(
                host=self.db_config["host"],
                port=self.db_config["port"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                db=self.db_config["db"],
                charset=self.db_config["charset"],
                autocommit=False,
                minsize=5,
                maxsize=20,
            )

            # Create table structure
            await self._create_tables()
            await self._migrate_schema_v2()

            self._initialized = True
            logger.info(
                f"MySQL backend initialized successfully, database: {self.db_config['db']}"
            )
            return True

        except Exception as e:
            logger.exception(f"MySQL backend initialization failed: {e}")
            return False

    @asynccontextmanager
    async def _get_connection(self):
        """Get a connection from the pool. Auto-returns on exit."""
        async with self._pool.acquire() as conn:
            try:
                yield conn
            except Exception:
                try:
                    await conn.rollback()
                except Exception as e:
                    logger.debug(f"Nested rollback failed: {e}")
                raise

    async def _create_tables(self):
        """Create database table structure"""
        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                # vaults table
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vaults (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        title TEXT,
                        summary TEXT,
                        content LONGTEXT,
                        tags TEXT,
                        parent_id BIGINT,
                        is_folder BOOLEAN DEFAULT FALSE,
                        is_deleted BOOLEAN DEFAULT FALSE,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        document_type VARCHAR(50) DEFAULT 'vaults',
                        sort_order INT DEFAULT 0,
                        INDEX idx_vaults_created (created_at),
                        INDEX idx_vaults_type (document_type),
                        INDEX idx_vaults_folder (is_folder),
                        INDEX idx_vaults_deleted (is_deleted),
                        FOREIGN KEY (parent_id) REFERENCES vaults (id) ON DELETE SET NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                # Todo table
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS todo (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        content TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                        end_time DATETIME,
                        status INT DEFAULT 0,
                        urgency INT DEFAULT 0,
                        assignee VARCHAR(255),
                        reason TEXT,
                        INDEX idx_todo_status (status),
                        INDEX idx_todo_urgency (urgency),
                        INDEX idx_todo_created (created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                # Tips table
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tips (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        content TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_tips_time (created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                # Profiles table
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS profiles (
                        user_id VARCHAR(255) NOT NULL,
                        device_id VARCHAR(100) NOT NULL DEFAULT 'default',
                        agent_id VARCHAR(100) NOT NULL DEFAULT 'default',
                        content LONGTEXT NOT NULL,
                        summary TEXT,
                        keywords JSON,
                        entities JSON,
                        importance INT DEFAULT 0,
                        metadata JSON,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, device_id, agent_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                # Entities table
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entities (
                        id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        device_id VARCHAR(100) NOT NULL DEFAULT 'default',
                        agent_id VARCHAR(100) NOT NULL DEFAULT 'default',
                        entity_name VARCHAR(255) NOT NULL,
                        entity_type VARCHAR(50),
                        content LONGTEXT NOT NULL,
                        summary TEXT,
                        keywords JSON,
                        aliases JSON,
                        metadata JSON,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        UNIQUE KEY uk_user_entity (user_id, device_id, agent_id, entity_name),
                        INDEX idx_entity_user (user_id),
                        INDEX idx_entity_type (entity_type)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                # Monitoring tables
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monitoring_token_usage (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        time_bucket VARCHAR(20) NOT NULL,
                        model VARCHAR(100) NOT NULL,
                        prompt_tokens INT DEFAULT 0,
                        completion_tokens INT DEFAULT 0,
                        total_tokens INT DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uk_time_model (time_bucket, model),
                        INDEX idx_monitoring_token_created (created_at),
                        INDEX idx_monitoring_token_model (model)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monitoring_stage_timing (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        time_bucket VARCHAR(20) NOT NULL,
                        stage_name VARCHAR(100) NOT NULL,
                        count INT DEFAULT 1,
                        total_duration_ms BIGINT NOT NULL,
                        min_duration_ms BIGINT NOT NULL,
                        max_duration_ms BIGINT NOT NULL,
                        avg_duration_ms BIGINT NOT NULL,
                        success_count INT DEFAULT 0,
                        error_count INT DEFAULT 0,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uk_time_stage (time_bucket, stage_name),
                        INDEX idx_monitoring_stage_created (created_at),
                        INDEX idx_monitoring_stage_name (stage_name)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monitoring_data_stats (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        time_bucket VARCHAR(20) NOT NULL,
                        data_type VARCHAR(50) NOT NULL,
                        count INT DEFAULT 1,
                        context_type VARCHAR(50),
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uk_time_type_context (time_bucket, data_type, context_type),
                        INDEX idx_monitoring_data_created (created_at),
                        INDEX idx_monitoring_data_type (data_type)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                # Conversation tables
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        title VARCHAR(500),
                        user_id VARCHAR(100),
                        page_name VARCHAR(20) DEFAULT 'home',
                        status VARCHAR(20) DEFAULT 'active',
                        metadata JSON DEFAULT (JSON_OBJECT()),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_conversations_updated_at (updated_at DESC),
                        INDEX idx_conversations_user_page (user_id, page_name)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        conversation_id BIGINT NOT NULL,
                        parent_message_id VARCHAR(100),
                        role VARCHAR(50) NOT NULL,
                        content LONGTEXT,
                        status VARCHAR(20) NOT NULL DEFAULT 'pending',
                        token_count INT DEFAULT 0,
                        metadata JSON DEFAULT (JSON_OBJECT()),
                        latency_ms INT DEFAULT 0,
                        error_message TEXT,
                        completed_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_messages_status (status),
                        INDEX idx_messages_conversation_id (conversation_id),
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS message_thinking (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        message_id BIGINT NOT NULL,
                        content TEXT NOT NULL,
                        stage VARCHAR(100),
                        progress FLOAT DEFAULT 0.0,
                        sequence INT DEFAULT 0,
                        metadata JSON DEFAULT (JSON_OBJECT()),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_message_thinking_message_id (message_id),
                        INDEX idx_message_thinking_stage (message_id, stage),
                        INDEX idx_message_thinking_sequence (message_id, sequence),
                        FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

                await conn.commit()

    async def _migrate_schema_v2(self):
        """Add device_id to profiles and device_id+agent_id to entities tables (idempotent)."""
        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute(
                        "ALTER TABLE profiles ADD COLUMN device_id VARCHAR(100) NOT NULL DEFAULT 'default' AFTER user_id"
                    )
                    await conn.commit()
                    logger.info("Migration: added device_id column to profiles table")
                except Exception:
                    await conn.rollback()

                try:
                    await cursor.execute(
                        "ALTER TABLE profiles DROP PRIMARY KEY, ADD PRIMARY KEY (user_id, device_id, agent_id)"
                    )
                    await conn.commit()
                    logger.info(
                        "Migration: updated profiles primary key to (user_id, device_id, agent_id)"
                    )
                except Exception:
                    await conn.rollback()

                try:
                    await cursor.execute(
                        "ALTER TABLE entities ADD COLUMN device_id VARCHAR(100) NOT NULL DEFAULT 'default' AFTER user_id"
                    )
                    await conn.commit()
                    logger.info("Migration: added device_id column to entities table")
                except Exception:
                    await conn.rollback()

                try:
                    await cursor.execute(
                        "ALTER TABLE entities ADD COLUMN agent_id VARCHAR(100) NOT NULL DEFAULT 'default' AFTER device_id"
                    )
                    await conn.commit()
                    logger.info("Migration: added agent_id column to entities table")
                except Exception:
                    await conn.rollback()

                try:
                    await cursor.execute(
                        "ALTER TABLE entities MODIFY COLUMN entity_name VARCHAR(255) NOT NULL"
                    )
                    await conn.commit()
                    logger.info("Migration: reduced entity_name to VARCHAR(255)")
                except Exception:
                    await conn.rollback()

                try:
                    await cursor.execute(
                        "ALTER TABLE entities DROP INDEX uk_user_entity, "
                        "ADD UNIQUE KEY uk_user_entity (user_id, device_id, agent_id, entity_name)"
                    )
                    await conn.commit()
                    logger.info("Migration: updated entities unique key")
                except Exception:
                    await conn.rollback()

    # Report table operations
    async def insert_vaults(
        self,
        title: str,
        summary: str,
        content: str,
        document_type: str,
        tags: str = None,
        parent_id: int = None,
        is_folder: bool = False,
    ) -> int:
        if not self._initialized:
            raise RuntimeError("MySQL backend not initialized")

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute(
                        """
                        INSERT INTO vaults (title, summary, content, tags, parent_id, is_folder, document_type, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
                            title, summary, content, tags, parent_id, is_folder,
                            document_type, datetime.now(), datetime.now(),
                        ),
                    )
                    vault_id = cursor.lastrowid
                    await conn.commit()
                    logger.info(f"Report inserted, ID: {vault_id}")
                    return vault_id
                except Exception as e:
                    logger.exception(f"Failed to insert report: {e}")
                    raise

    async def get_reports(
        self, limit: int = 100, offset: int = 0, is_deleted: bool = False
    ) -> List[Dict]:
        return await self.get_vaults(
            limit=limit, offset=offset, is_deleted=is_deleted, document_type="report"
        )

    async def get_vaults(
        self,
        limit: int = 100,
        offset: int = 0,
        is_deleted: bool = False,
        document_type: str = None,
        created_after: datetime = None,
        created_before: datetime = None,
        updated_after: datetime = None,
        updated_before: datetime = None,
    ) -> List[Dict]:
        if not self._initialized:
            return []

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    where_clauses = ["is_deleted = %s"]
                    params = [is_deleted]

                    if document_type:
                        where_clauses.append("document_type = %s")
                        params.append(document_type)
                    if created_after:
                        where_clauses.append("created_at >= %s")
                        params.append(created_after)
                    if created_before:
                        where_clauses.append("created_at <= %s")
                        params.append(created_before)
                    if updated_after:
                        where_clauses.append("updated_at >= %s")
                        params.append(updated_after)
                    if updated_before:
                        where_clauses.append("updated_at <= %s")
                        params.append(updated_before)

                    params.extend([limit, offset])
                    where_clause = " AND ".join(where_clauses)
                    sql = f"""
                        SELECT id, title, summary, content, tags, parent_id, is_folder, is_deleted,
                               created_at, updated_at, document_type
                        FROM vaults
                        WHERE {where_clause}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """
                    await cursor.execute(sql, params)
                    rows = await cursor.fetchall()
                    return list(rows)
                except Exception as e:
                    logger.exception(f"Failed to get vaults list: {e}")
                    return []

    async def get_vault(self, vault_id: int) -> Optional[Dict]:
        if not self._initialized:
            return None

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    await cursor.execute(
                        """
                        SELECT id, title, summary, content, tags, parent_id, is_folder, is_deleted,
                               created_at, updated_at, document_type
                        FROM vaults WHERE id = %s
                    """,
                        (vault_id,),
                    )
                    row = await cursor.fetchone()
                    return row
                except Exception as e:
                    logger.exception(f"Failed to get vaults: {e}")
                    return None

    async def update_vault(self, vault_id: int, **kwargs) -> bool:
        if not self._initialized:
            return False

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    set_clauses = []
                    params = []
                    for key, value in kwargs.items():
                        if key in ["title", "summary", "content", "tags", "parent_id", "is_folder", "is_deleted"]:
                            set_clauses.append(f"{key} = %s")
                            params.append(value)
                    if not set_clauses:
                        return False
                    params.append(vault_id)
                    sql = f"UPDATE vaults SET {', '.join(set_clauses)} WHERE id = %s"
                    await cursor.execute(sql, params)
                    success = cursor.rowcount > 0
                    await conn.commit()
                    return success
                except Exception as e:
                    logger.exception(f"Failed to update report: {e}")
                    return False

    # Todo table operations
    async def insert_todo(
        self,
        content: str,
        start_time: datetime = None,
        end_time: datetime = None,
        status: int = 0,
        urgency: int = 0,
        assignee: str = None,
        reason: str = None,
    ) -> int:
        if not self._initialized:
            raise RuntimeError("MySQL backend not initialized")

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute(
                        """
                        INSERT INTO todo (content, start_time, end_time, status, urgency, assignee, reason, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                        (content, start_time or datetime.now(), end_time, status, urgency, assignee, reason, datetime.now()),
                    )
                    todo_id = cursor.lastrowid
                    await conn.commit()
                    logger.info(f"Todo item inserted, ID: {todo_id}")
                    return todo_id
                except Exception as e:
                    logger.exception(f"Failed to insert todo item: {e}")
                    raise

    async def get_todos(
        self,
        status: int = None,
        limit: int = 100,
        offset: int = 0,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> List[Dict]:
        if not self._initialized:
            return []

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    where_conditions = []
                    params = []
                    if start_time:
                        where_conditions.append("start_time >= %s")
                        params.append(start_time)
                    if end_time:
                        where_conditions.append("end_time <= %s")
                        params.append(end_time)
                    if status is not None:
                        where_conditions.append("status = %s")
                        params.append(status)
                    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                    params.extend([limit, offset])
                    await cursor.execute(
                        f"""
                        SELECT id, content, created_at, start_time, end_time, status, urgency, assignee, reason
                        FROM todo WHERE {where_clause}
                        ORDER BY urgency DESC, created_at DESC
                        LIMIT %s OFFSET %s
                    """,
                        params,
                    )
                    rows = await cursor.fetchall()
                    return list(rows)
                except Exception as e:
                    logger.exception(f"Failed to get todo item list: {e}")
                    return []

    async def update_todo_status(self, todo_id: int, status: int, end_time: datetime = None) -> bool:
        if not self._initialized:
            return False

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    if status == 1 and end_time is None:
                        end_time = datetime.now()
                    await cursor.execute(
                        "UPDATE todo SET status = %s, end_time = %s WHERE id = %s",
                        (status, end_time, todo_id),
                    )
                    success = cursor.rowcount > 0
                    await conn.commit()
                    return success
                except Exception as e:
                    logger.exception(f"Failed to update todo item status: {e}")
                    return False

    # Tips table operations
    async def insert_tip(self, content: str) -> int:
        if not self._initialized:
            raise RuntimeError("MySQL backend not initialized")

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute(
                        "INSERT INTO tips (content, created_at) VALUES (%s, %s)",
                        (content, datetime.now()),
                    )
                    tip_id = cursor.lastrowid
                    await conn.commit()
                    logger.info(f"Tip inserted, ID: {tip_id}")
                    return tip_id
                except Exception as e:
                    logger.exception(f"Failed to insert tip: {e}")
                    raise

    async def get_tips(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        if not self._initialized:
            return []

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    where_conditions = []
                    params = []
                    if start_time:
                        where_conditions.append("created_at >= %s")
                        params.append(start_time)
                    if end_time:
                        where_conditions.append("created_at <= %s")
                        params.append(end_time)
                    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                    params.extend([limit, offset])
                    await cursor.execute(
                        f"""
                        SELECT id, content, created_at FROM tips
                        WHERE {where_clause}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """,
                        params,
                    )
                    rows = await cursor.fetchall()
                    return list(rows)
                except Exception as e:
                    logger.exception(f"Failed to get tip list: {e}")
                    return []

    # ── Profile CRUD ──

    async def upsert_profile(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        content: str = "",
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        importance: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self._initialized:
            return False

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    now = datetime.now()
                    keywords_json = json.dumps(keywords or [], ensure_ascii=False)
                    entities_json = json.dumps(entities or [], ensure_ascii=False)
                    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

                    await cursor.execute(
                        """
                        INSERT INTO profiles (user_id, device_id, agent_id, content, summary, keywords, entities,
                                              importance, metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            content = VALUES(content),
                            summary = VALUES(summary),
                            keywords = VALUES(keywords),
                            entities = VALUES(entities),
                            importance = VALUES(importance),
                            metadata = VALUES(metadata),
                            updated_at = VALUES(updated_at)
                        """,
                        (user_id, device_id, agent_id, content, summary, keywords_json,
                         entities_json, importance, metadata_json, now, now),
                    )
                    await conn.commit()
                    logger.info(f"Profile upserted for user_id={user_id}, device_id={device_id}, agent_id={agent_id}")
                    return True
                except Exception as e:
                    logger.exception(f"Failed to upsert profile: {e}")
                    return False

    async def get_profile(
        self, user_id: str, device_id: str = "default", agent_id: str = "default"
    ) -> Optional[Dict]:
        if not self._initialized:
            return None

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    await cursor.execute(
                        """
                        SELECT user_id, device_id, agent_id, content, summary, keywords, entities,
                               importance, metadata, created_at, updated_at
                        FROM profiles
                        WHERE user_id = %s AND device_id = %s AND agent_id = %s
                        """,
                        (user_id, device_id, agent_id),
                    )
                    row = await cursor.fetchone()
                    if row:
                        result = dict(row)
                        if isinstance(result.get("keywords"), str):
                            result["keywords"] = json.loads(result["keywords"])
                        if isinstance(result.get("entities"), str):
                            result["entities"] = json.loads(result["entities"])
                        if isinstance(result.get("metadata"), str):
                            result["metadata"] = json.loads(result["metadata"])
                        return result
                    return None
                except Exception as e:
                    logger.exception(f"Failed to get profile: {e}")
                    return None

    async def delete_profile(
        self, user_id: str, device_id: str = "default", agent_id: str = "default"
    ) -> bool:
        if not self._initialized:
            return False

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute(
                        "DELETE FROM profiles WHERE user_id = %s AND device_id = %s AND agent_id = %s",
                        (user_id, device_id, agent_id),
                    )
                    await conn.commit()
                    return cursor.rowcount > 0
                except Exception as e:
                    logger.exception(f"Failed to delete profile: {e}")
                    return False

    # ── Entity CRUD ──

    async def upsert_entity(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        entity_name: str = "",
        content: str = "",
        entity_type: Optional[str] = None,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self._initialized:
            raise RuntimeError("MySQL backend not initialized")

        import uuid

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    now = datetime.now()
                    entity_id = str(uuid.uuid4())
                    keywords_json = json.dumps(keywords or [], ensure_ascii=False)
                    aliases_json = json.dumps(aliases or [], ensure_ascii=False)
                    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

                    await cursor.execute(
                        """
                        INSERT INTO entities (id, user_id, device_id, agent_id, entity_name, entity_type, content, summary,
                                              keywords, aliases, metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            entity_type = VALUES(entity_type),
                            content = VALUES(content),
                            summary = VALUES(summary),
                            keywords = VALUES(keywords),
                            aliases = VALUES(aliases),
                            metadata = VALUES(metadata),
                            updated_at = VALUES(updated_at)
                        """,
                        (entity_id, user_id, device_id, agent_id, entity_name, entity_type,
                         content, summary, keywords_json, aliases_json, metadata_json, now, now),
                    )
                    await conn.commit()

                    # Retrieve the actual persisted ID (may differ from generated one on update)
                    await cursor.execute(
                        "SELECT id FROM entities WHERE user_id = %s AND device_id = %s AND agent_id = %s AND entity_name = %s",
                        (user_id, device_id, agent_id, entity_name),
                    )
                    row = await cursor.fetchone()
                    if row:
                        entity_id = row["id"]

                    logger.info(f"Entity upserted: {entity_name} for user_id={user_id}")
                    return entity_id
                except Exception as e:
                    logger.exception(f"Failed to upsert entity: {e}")
                    raise

    async def get_entity(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        entity_name: str = "",
    ) -> Optional[Dict]:
        if not self._initialized:
            return None

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    await cursor.execute(
                        """
                        SELECT id, user_id, device_id, agent_id, entity_name, entity_type, content, summary,
                               keywords, aliases, metadata, created_at, updated_at
                        FROM entities
                        WHERE user_id = %s AND device_id = %s AND agent_id = %s AND entity_name = %s
                        """,
                        (user_id, device_id, agent_id, entity_name),
                    )
                    row = await cursor.fetchone()
                    if row:
                        result = dict(row)
                        if isinstance(result.get("keywords"), str):
                            result["keywords"] = json.loads(result["keywords"])
                        if isinstance(result.get("aliases"), str):
                            result["aliases"] = json.loads(result["aliases"])
                        if isinstance(result.get("metadata"), str):
                            result["metadata"] = json.loads(result["metadata"])
                        return result
                    return None
                except Exception as e:
                    logger.exception(f"Failed to get entity: {e}")
                    return None

    async def list_entities(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        entity_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        if not self._initialized:
            return []

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    where_clauses = ["user_id = %s", "device_id = %s", "agent_id = %s"]
                    params: list = [user_id, device_id, agent_id]
                    if entity_type:
                        where_clauses.append("entity_type = %s")
                        params.append(entity_type)
                    params.extend([limit, offset])
                    where_sql = " AND ".join(where_clauses)

                    await cursor.execute(
                        f"""
                        SELECT id, user_id, device_id, agent_id, entity_name, entity_type, content, summary,
                               keywords, aliases, metadata, created_at, updated_at
                        FROM entities WHERE {where_sql}
                        ORDER BY updated_at DESC LIMIT %s OFFSET %s
                        """,
                        params,
                    )
                    rows = await cursor.fetchall()
                    results = []
                    for row in rows:
                        result = dict(row)
                        if isinstance(result.get("keywords"), str):
                            result["keywords"] = json.loads(result["keywords"])
                        if isinstance(result.get("aliases"), str):
                            result["aliases"] = json.loads(result["aliases"])
                        if isinstance(result.get("metadata"), str):
                            result["metadata"] = json.loads(result["metadata"])
                        results.append(result)
                    return results
                except Exception as e:
                    logger.exception(f"Failed to list entities: {e}")
                    return []

    async def search_entities(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        query_text: str = "",
        limit: int = 20,
    ) -> List[Dict]:
        if not self._initialized:
            return []

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    like_pattern = f"%{query_text}%"
                    await cursor.execute(
                        """
                        SELECT id, user_id, device_id, agent_id, entity_name, entity_type, content, summary,
                               keywords, aliases, metadata, created_at, updated_at
                        FROM entities
                        WHERE user_id = %s AND device_id = %s AND agent_id = %s
                          AND (entity_name LIKE %s OR content LIKE %s OR aliases LIKE %s)
                        ORDER BY updated_at DESC LIMIT %s
                        """,
                        (user_id, device_id, agent_id, like_pattern, like_pattern, like_pattern, limit),
                    )
                    rows = await cursor.fetchall()
                    results = []
                    for row in rows:
                        result = dict(row)
                        if isinstance(result.get("keywords"), str):
                            result["keywords"] = json.loads(result["keywords"])
                        if isinstance(result.get("aliases"), str):
                            result["aliases"] = json.loads(result["aliases"])
                        if isinstance(result.get("metadata"), str):
                            result["metadata"] = json.loads(result["metadata"])
                        results.append(result)
                    return results
                except Exception as e:
                    logger.exception(f"Failed to search entities: {e}")
                    return []

    async def delete_entity(
        self,
        user_id: str,
        device_id: str = "default",
        agent_id: str = "default",
        entity_name: str = "",
    ) -> bool:
        if not self._initialized:
            return False

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute(
                        "DELETE FROM entities WHERE user_id = %s AND device_id = %s AND agent_id = %s AND entity_name = %s",
                        (user_id, device_id, agent_id, entity_name),
                    )
                    await conn.commit()
                    return cursor.rowcount > 0
                except Exception as e:
                    logger.exception(f"Failed to delete entity: {e}")
                    return False

    def get_name(self) -> str:
        return "mysql"

    def get_storage_type(self) -> StorageType:
        return StorageType.DOCUMENT_DB

    # Monitoring data operations
    async def save_monitoring_token_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ) -> bool:
        if not self._initialized:
            return False

        try:
            async with self._get_connection() as conn:
                async with conn.cursor() as cursor:
                    now = datetime.now()
                    time_bucket = now.strftime("%Y-%m-%d %H:00:00")
                    await cursor.execute(
                        """
                        INSERT INTO monitoring_token_usage (time_bucket, model, prompt_tokens, completion_tokens, total_tokens, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            prompt_tokens = prompt_tokens + VALUES(prompt_tokens),
                            completion_tokens = completion_tokens + VALUES(completion_tokens),
                            total_tokens = total_tokens + VALUES(total_tokens)
                        """,
                        (time_bucket, model, prompt_tokens, completion_tokens, total_tokens, now),
                    )
                    await conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Failed to save token usage: {e}")
            return False

    async def save_monitoring_stage_timing(
        self, stage_name: str, duration_ms: int, status: str = "success", metadata: Optional[str] = None,
    ) -> bool:
        if not self._initialized:
            return False

        try:
            async with self._get_connection() as conn:
                async with conn.cursor() as cursor:
                    now = datetime.now()
                    time_bucket = now.strftime("%Y-%m-%d %H:00:00")
                    success_inc = 1 if status == "success" else 0
                    error_inc = 0 if status == "success" else 1

                    await cursor.execute(
                        """
                        INSERT INTO monitoring_stage_timing
                        (time_bucket, stage_name, count, total_duration_ms, min_duration_ms,
                         max_duration_ms, avg_duration_ms, success_count, error_count, metadata, created_at)
                        VALUES (%s, %s, 1, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            count = count + 1,
                            total_duration_ms = total_duration_ms + VALUES(total_duration_ms),
                            min_duration_ms = LEAST(min_duration_ms, VALUES(min_duration_ms)),
                            max_duration_ms = GREATEST(max_duration_ms, VALUES(max_duration_ms)),
                            success_count = success_count + VALUES(success_count),
                            error_count = error_count + VALUES(error_count),
                            avg_duration_ms = total_duration_ms DIV count
                        """,
                        (time_bucket, stage_name, duration_ms, duration_ms, duration_ms, duration_ms,
                         success_inc, error_inc, metadata, now),
                    )
                    await conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Failed to save stage timing: {e}")
            return False

    async def save_monitoring_data_stats(
        self, data_type: str, count: int = 1, context_type: Optional[str] = None, metadata: Optional[str] = None,
    ) -> bool:
        if not self._initialized:
            return False

        try:
            async with self._get_connection() as conn:
                async with conn.cursor() as cursor:
                    now = datetime.now()
                    time_bucket = now.strftime("%Y-%m-%d %H:00:00")
                    await cursor.execute(
                        """
                        INSERT INTO monitoring_data_stats (time_bucket, data_type, count, context_type, metadata, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE count = count + VALUES(count)
                        """,
                        (time_bucket, data_type, count, context_type, metadata, now),
                    )
                    await conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Failed to save data stats: {e}")
            return False

    async def query_monitoring_token_usage(self, hours: int = 24) -> List[Dict[str, Any]]:
        if not self._initialized:
            return []

        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_bucket = cutoff_time.strftime("%Y-%m-%d %H:00:00")
            async with self._get_connection() as conn:
                async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                    await cursor.execute(
                        """
                        SELECT model, prompt_tokens, completion_tokens, total_tokens, time_bucket
                        FROM monitoring_token_usage WHERE time_bucket >= %s ORDER BY time_bucket DESC
                        """,
                        (cutoff_bucket,),
                    )
                    rows = await cursor.fetchall()
                    return list(rows)
        except Exception as e:
            logger.error(f"Failed to query token usage: {e}")
            return []

    async def query_monitoring_stage_timing(self, hours: int = 24) -> List[Dict[str, Any]]:
        if not self._initialized:
            return []

        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_bucket = cutoff_time.strftime("%Y-%m-%d %H:00:00")
            async with self._get_connection() as conn:
                async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                    await cursor.execute(
                        """
                        SELECT stage_name, count, total_duration_ms, min_duration_ms, max_duration_ms, avg_duration_ms, success_count, error_count, time_bucket
                        FROM monitoring_stage_timing WHERE time_bucket >= %s ORDER BY time_bucket DESC
                        """,
                        (cutoff_bucket,),
                    )
                    rows = await cursor.fetchall()
                    result = []
                    for row in rows:
                        result.append({
                            "stage_name": row["stage_name"], "count": row["count"],
                            "total_duration": row["total_duration_ms"], "min_duration": row["min_duration_ms"],
                            "max_duration": row["max_duration_ms"], "duration_ms": row["avg_duration_ms"],
                            "success_count": row["success_count"], "error_count": row["error_count"],
                            "status": "success" if row["success_count"] > 0 else "error",
                            "time_bucket": row["time_bucket"],
                        })
                    return result
        except Exception as e:
            logger.error(f"Failed to query stage timing: {e}")
            return []

    async def query_monitoring_data_stats(self, hours: int = 24) -> List[Dict[str, Any]]:
        if not self._initialized:
            return []

        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_bucket = cutoff_time.strftime("%Y-%m-%d %H:00:00")
            async with self._get_connection() as conn:
                async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                    await cursor.execute(
                        """
                        SELECT data_type, SUM(count) as total_count, context_type
                        FROM monitoring_data_stats WHERE time_bucket >= %s GROUP BY data_type, context_type
                        """,
                        (cutoff_bucket,),
                    )
                    rows = await cursor.fetchall()
                    return [{"data_type": r["data_type"], "count": r["total_count"], "context_type": r["context_type"]} for r in rows]
        except Exception as e:
            logger.error(f"Failed to query data stats: {e}")
            return []

    async def query_monitoring_data_stats_by_range(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        if not self._initialized:
            return []

        try:
            start_bucket = start_time.strftime("%Y-%m-%d %H:00:00")
            end_bucket = end_time.strftime("%Y-%m-%d %H:00:00")
            async with self._get_connection() as conn:
                async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                    await cursor.execute(
                        """
                        SELECT data_type, SUM(count) as total_count, context_type
                        FROM monitoring_data_stats WHERE time_bucket >= %s AND time_bucket <= %s GROUP BY data_type, context_type
                        """,
                        (start_bucket, end_bucket),
                    )
                    rows = await cursor.fetchall()
                    return [{"data_type": r["data_type"], "count": r["total_count"], "context_type": r["context_type"]} for r in rows]
        except Exception as e:
            logger.error(f"Failed to query data stats by range: {e}")
            return []

    async def query_monitoring_data_stats_trend(
        self, hours: int = 24, interval_hours: int = 1
    ) -> List[Dict[str, Any]]:
        if not self._initialized:
            return []

        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_bucket = cutoff_time.strftime("%Y-%m-%d %H:00:00")
            async with self._get_connection() as conn:
                async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                    await cursor.execute(
                        """
                        SELECT time_bucket, data_type, SUM(count) as total_count, context_type
                        FROM monitoring_data_stats WHERE time_bucket >= %s
                        GROUP BY time_bucket, data_type, context_type ORDER BY time_bucket ASC
                        """,
                        (cutoff_bucket,),
                    )
                    rows = await cursor.fetchall()
                    return [{"timestamp": r["time_bucket"], "data_type": r["data_type"], "count": r["total_count"], "context_type": r["context_type"]} for r in rows]
        except Exception as e:
            logger.error(f"Failed to query data stats trend: {e}")
            return []

    async def cleanup_old_monitoring_data(self, days: int = 7) -> bool:
        if not self._initialized:
            return False

        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            cutoff_bucket = cutoff_time.strftime("%Y-%m-%d %H:00:00")
            async with self._get_connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("DELETE FROM monitoring_token_usage WHERE time_bucket < %s", (cutoff_bucket,))
                    await cursor.execute("DELETE FROM monitoring_stage_timing WHERE time_bucket < %s", (cutoff_bucket,))
                    await cursor.execute("DELETE FROM monitoring_data_stats WHERE time_bucket < %s", (cutoff_bucket,))
                    await conn.commit()
                    logger.info(f"Cleaned up monitoring data older than {days} days")
                    return True
        except Exception as e:
            logger.error(f"Failed to cleanup old monitoring data: {e}")
            return False

    # Conversation/Message operations
    async def create_conversation(
        self, page_name: str, user_id: Optional[str] = None, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("MySQL backend not initialized")

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    now = datetime.now()
                    meta_str = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"
                    await cursor.execute(
                        """
                        INSERT INTO conversations (page_name, user_id, title, metadata, status, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (page_name, user_id, title, meta_str, "active", now, now),
                    )
                    conversation_id = cursor.lastrowid
                    await conn.commit()
                    logger.info(f"Conversation created, ID: {conversation_id}")
                    return await self.get_conversation(conversation_id)
                except Exception as e:
                    logger.exception(f"Failed to create conversation: {e}")
                    return None

    async def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            return None

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    await cursor.execute(
                        "SELECT id, title, user_id, page_name, status, metadata, created_at, updated_at FROM conversations WHERE id = %s",
                        (conversation_id,),
                    )
                    row = await cursor.fetchone()
                    return row
                except Exception as e:
                    logger.exception(f"Failed to get conversation: {e}")
                    return None

    async def get_conversation_list(
        self, limit: int = 20, offset: int = 0, page_name: Optional[str] = None,
        user_id: Optional[str] = None, status: str = "active",
    ) -> Dict[str, Any]:
        if not self._initialized:
            return {"items": [], "total": 0}

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    where_clauses = []
                    params = []
                    if status:
                        where_clauses.append("status = %s")
                        params.append(status)
                    if page_name:
                        where_clauses.append("page_name = %s")
                        params.append(page_name)
                    if user_id:
                        where_clauses.append("user_id = %s")
                        params.append(user_id)
                    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

                    count_params = params[:]
                    await cursor.execute(f"SELECT COUNT(*) as cnt FROM conversations WHERE {where_sql}", count_params)
                    result = await cursor.fetchone()
                    total = result["cnt"] if result else 0

                    list_params = params + [limit, offset]
                    await cursor.execute(
                        f"""
                        SELECT id, title, user_id, page_name, status, metadata, created_at, updated_at
                        FROM conversations WHERE {where_sql} ORDER BY updated_at DESC LIMIT %s OFFSET %s
                        """,
                        list_params,
                    )
                    rows = await cursor.fetchall()
                    return {"items": list(rows), "total": total}
                except Exception as e:
                    logger.exception(f"Failed to get conversation list: {e}")
                    return {"items": [], "total": 0}

    async def update_conversation(
        self, conversation_id: int, title: Optional[str] = None, status: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            return None

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    set_clauses = []
                    params = []
                    if title is not None:
                        set_clauses.append("title = %s")
                        params.append(title)
                    if status is not None:
                        set_clauses.append("status = %s")
                        params.append("deleted" if status == "delected" else status)
                    if not set_clauses:
                        return await self.get_conversation(conversation_id)
                    params.append(conversation_id)
                    sql = f"UPDATE conversations SET {', '.join(set_clauses)} WHERE id = %s"
                    await cursor.execute(sql, params)
                    await conn.commit()
                    if cursor.rowcount > 0:
                        logger.info(f"Conversation {conversation_id} updated.")
                        return await self.get_conversation(conversation_id)
                    else:
                        logger.warning(f"Failed to update conversation {conversation_id}, row not found or no change.")
                        return None
                except Exception as e:
                    logger.exception(f"Failed to update conversation: {e}")
                    return None

    async def delete_conversation(self, conversation_id: int) -> Dict[str, Any]:
        updated_convo = await self.update_conversation(conversation_id=conversation_id, status="deleted")
        success = updated_convo is not None
        return {"success": success, "id": conversation_id}

    async def get_message(self, message_id: int, include_thinking: bool = True) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            return None

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    await cursor.execute("SELECT * FROM messages WHERE id = %s", (message_id,))
                    row = await cursor.fetchone()
                    if row:
                        message = dict(row)
                        if include_thinking:
                            message["thinking"] = await self.get_message_thinking(message_id)
                        return message
                    return None
                except Exception as e:
                    logger.exception(f"Failed to get message: {e}")
                    return None

    async def create_message(
        self, conversation_id: int, role: str, content: str, is_complete: bool = True,
        token_count: int = 0, parent_message_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("MySQL backend not initialized")

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    now = datetime.now()
                    status = "completed" if is_complete else "streaming"
                    completed_at = now if is_complete else None
                    meta_str = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"
                    await cursor.execute(
                        """
                        INSERT INTO messages (conversation_id, role, content, status, token_count,
                                              parent_message_id, metadata, completed_at, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (conversation_id, role, content, status, token_count, parent_message_id, meta_str, completed_at, now, now),
                    )
                    message_id = cursor.lastrowid
                    await cursor.execute("UPDATE conversations SET updated_at = %s WHERE id = %s", (now, conversation_id))
                    await conn.commit()
                    logger.info(f"Message created, ID: {message_id}")
                    return await self.get_message(message_id)
                except Exception as e:
                    logger.exception(f"Failed to create message: {e}")
                    return None

    async def create_streaming_message(
        self, conversation_id: int, role: str, parent_message_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self.create_message(
            conversation_id=conversation_id, role=role, content="", is_complete=False,
            token_count=0, parent_message_id=parent_message_id, metadata=metadata,
        )

    async def update_message(
        self, message_id: int, new_content: str, is_complete: Optional[bool] = None, token_count: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            return None

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    now = datetime.now()
                    set_clauses = ["content = %s", "updated_at = %s"]
                    params = [new_content, now]
                    if token_count is not None:
                        set_clauses.append("token_count = %s")
                        params.append(token_count)
                    if is_complete is True:
                        set_clauses.append("status = %s")
                        params.append("completed")
                        set_clauses.append("completed_at = %s")
                        params.append(now)
                    elif is_complete is False:
                        set_clauses.append("status = %s")
                        params.append("streaming")
                        set_clauses.append("completed_at = NULL")
                    params.append(message_id)
                    sql = f"UPDATE messages SET {', '.join(set_clauses)} WHERE id = %s"
                    await cursor.execute(sql, params)
                    await cursor.execute(
                        "UPDATE conversations SET updated_at = %s WHERE id = (SELECT conversation_id FROM messages WHERE id = %s)",
                        (now, message_id),
                    )
                    await conn.commit()
                    if cursor.rowcount > 0:
                        return await self.get_message(message_id)
                    else:
                        logger.warning(f"Failed to update message {message_id}, not found.")
                        return None
                except Exception as e:
                    logger.exception(f"Failed to update message: {e}")
                    return None

    async def append_message_content(self, message_id: int, content_chunk: str, token_count: int = 0) -> bool:
        if not self._initialized:
            return False

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    now = datetime.now()
                    await cursor.execute(
                        """
                        UPDATE messages SET content = CONCAT(content, %s), token_count = token_count + %s,
                            status = CASE WHEN status = 'pending' THEN 'streaming' ELSE status END, updated_at = %s
                        WHERE id = %s
                        """,
                        (content_chunk, token_count, now, message_id),
                    )
                    if cursor.rowcount == 0:
                        logger.warning(f"Failed to append message {message_id}, not found.")
                        return False
                    await cursor.execute(
                        "UPDATE conversations SET updated_at = %s WHERE id = (SELECT conversation_id FROM messages WHERE id = %s)",
                        (now, message_id),
                    )
                    await conn.commit()
                    return True
                except Exception as e:
                    logger.exception(f"Failed to append message content: {e}")
                    return False

    async def update_message_metadata(self, message_id: int, metadata: Dict[str, Any]) -> bool:
        if not self._initialized:
            return False

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    now = datetime.now()
                    meta_str = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"
                    await cursor.execute("UPDATE messages SET metadata = %s, updated_at = %s WHERE id = %s", (meta_str, now, message_id))
                    success = cursor.rowcount > 0
                    await conn.commit()
                    return success
                except Exception as e:
                    logger.exception(f"Failed to update message metadata: {e}")
                    return False

    async def mark_message_finished(
        self, message_id: int, status: str = "completed", error_message: Optional[str] = None
    ) -> bool:
        if not self._initialized:
            return False

        if status not in ["completed", "failed", "cancelled"]:
            status = "completed"

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    now = datetime.now()
                    set_clauses = ["status = %s", "completed_at = %s", "updated_at = %s"]
                    params = [status, now, now]
                    if error_message:
                        set_clauses.append("error_message = %s")
                        params.append(error_message)
                    params.append(message_id)
                    params.append(status)
                    sql = f"UPDATE messages SET {', '.join(set_clauses)} WHERE id = %s AND status != %s"
                    await cursor.execute(sql, params)
                    success = cursor.rowcount > 0
                    if not success:
                        await cursor.execute("SELECT status FROM messages WHERE id = %s", (message_id,))
                        row = await cursor.fetchone()
                        if row and row["status"] == status:
                            success = True
                        else:
                            logger.warning(f"Failed to mark message {message_id} as {status}, not found or no change.")
                    await cursor.execute(
                        "UPDATE conversations SET updated_at = %s WHERE id = (SELECT conversation_id FROM messages WHERE id = %s)",
                        (now, message_id),
                    )
                    await conn.commit()
                    return success
                except Exception as e:
                    logger.exception(f"Failed to mark message {status}: {e}")
                    return False

    async def interrupt_message(self, message_id: int) -> bool:
        return await self.mark_message_finished(
            message_id=message_id, status="cancelled", error_message="Message interrupted by user."
        )

    async def get_conversation_messages(self, conversation_id: int) -> List[Dict[str, Any]]:
        if not self._initialized:
            return []

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    await cursor.execute(
                        "SELECT * FROM messages WHERE conversation_id = %s ORDER BY created_at ASC",
                        (conversation_id,),
                    )
                    rows = await cursor.fetchall()
                    messages = []
                    for row in rows:
                        message = dict(row)
                        message["thinking"] = await self.get_message_thinking(message["id"])
                        messages.append(message)
                    return messages
                except Exception as e:
                    logger.exception(f"Failed to get conversation messages: {e}")
                    return []

    async def delete_message(self, message_id: int) -> bool:
        if not self._initialized:
            logger.warning("Storage not initialized")
            return False

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute("DELETE FROM messages WHERE id = %s", (message_id,))
                    await conn.commit()
                    return cursor.rowcount > 0
                except Exception as e:
                    logger.exception(f"Failed to delete message {message_id}: {e}")
                    return False

    # Message Thinking Management Methods
    async def add_message_thinking(
        self, message_id: int, content: str, stage: Optional[str] = None,
        progress: float = 0.0, sequence: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        if not self._initialized:
            logger.warning("Storage not initialized")
            return None

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    if sequence is None:
                        await cursor.execute(
                            "SELECT COALESCE(MAX(sequence), -1) + 1 as next_seq FROM message_thinking WHERE message_id = %s",
                            (message_id,),
                        )
                        result = await cursor.fetchone()
                        sequence = result["next_seq"] if result else 0
                    meta_str = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"
                    await cursor.execute(
                        """
                        INSERT INTO message_thinking (message_id, content, stage, progress, sequence, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (message_id, content, stage, progress, sequence, meta_str),
                    )
                    thinking_id = cursor.lastrowid
                    await conn.commit()
                    logger.debug(f"Added thinking record {thinking_id} to message {message_id}")
                    return thinking_id
                except Exception as e:
                    logger.exception(f"Failed to add thinking to message {message_id}: {e}")
                    return None

    async def get_message_thinking(self, message_id: int) -> List[Dict[str, Any]]:
        if not self._initialized:
            return []

        async with self._get_connection() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
                try:
                    await cursor.execute(
                        """
                        SELECT id, message_id, content, stage, progress, sequence, metadata, created_at
                        FROM message_thinking WHERE message_id = %s ORDER BY sequence ASC, created_at ASC
                        """,
                        (message_id,),
                    )
                    rows = await cursor.fetchall()
                    return list(rows)
                except Exception as e:
                    logger.exception(f"Failed to get thinking for message {message_id}: {e}")
                    return []

    async def clear_message_thinking(self, message_id: int) -> bool:
        if not self._initialized:
            return False

        async with self._get_connection() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute("DELETE FROM message_thinking WHERE message_id = %s", (message_id,))
                    await conn.commit()
                    return True
                except Exception as e:
                    logger.exception(f"Failed to clear thinking for message {message_id}: {e}")
                    return False

    async def query(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> QueryResult:
        if not self._initialized:
            return QueryResult(documents=[], total_count=0)
        logger.warning("MySQL query method is not fully implemented")
        return QueryResult(documents=[], total_count=0)

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
        self._initialized = False
        logger.info("MySQL connection pool closed")
