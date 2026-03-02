#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import uuid
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from qdrant_client import AsyncQdrantClient, models

from opencontext.llm.global_embedding_client import do_vectorize, do_vectorize_batch
from opencontext.models.context import ContextProperties, ExtractedData, ProcessedContext, Vectorize
from opencontext.models.enums import ContentFormat, ContextType
from opencontext.storage.base_storage import IVectorStorageBackend, StorageType
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

TODO_COLLECTION = "todo"

FIELD_DOCUMENT = "document"
FIELD_ORIGINAL_ID = "original_id"
FIELD_TODO_ID = "todo_id"
FIELD_CONTENT = "content"
FIELD_CREATED_AT = "created_at"


class QdrantBackend(IVectorStorageBackend):
    """
    Qdrant vector storage backend - https://qdrant.tech/
    """

    def __init__(self):
        self._client: Optional[AsyncQdrantClient] = None
        self._collections: Dict[str, str] = {}
        self._initialized = False
        self._config = None
        self._vector_size = None

    async def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            self._config = config
            qdrant_config = config.get("config", {})

            self._vector_size = qdrant_config.get("vector_size", None)
            client_config = {k: v for k, v in qdrant_config.items() if k != "vector_size"}
            self._client = AsyncQdrantClient(**client_config)

            context_types = [ct.value for ct in ContextType]

            for context_type in context_types:
                collection_name = f"{context_type}"
                await self._ensure_collection(collection_name, context_type)
                self._collections[context_type] = collection_name

            # Create todo collection only if consumption is enabled
            from opencontext.config.global_config import GlobalConfig

            consumption_enabled = (
                GlobalConfig.get_instance().get_config().get("consumption", {}).get("enabled", True)
            )
            if consumption_enabled:
                await self._ensure_collection(TODO_COLLECTION, TODO_COLLECTION)
                self._collections[TODO_COLLECTION] = TODO_COLLECTION
                logger.info("Todo collection initialized")
            else:
                logger.info("Todo collection skipped (consumption disabled)")

            self._initialized = True
            logger.info(
                f"Qdrant vector backend initialized successfully, created {len(self._collections)} collections"
            )
            return True

        except Exception as e:
            logger.exception(f"Qdrant vector backend initialization failed: {e}")
            return False

    async def _ensure_collection(self, collection_name: str, context_type: str) -> None:
        if not await self._client.collection_exists(collection_name):
            vector_size = self._vector_size or 1536

            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            logger.debug(f"Qdrant collection already exists: {collection_name}")

    async def _check_connection(self) -> bool:
        if not self._client:
            return False

        try:
            await self._client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    def get_name(self) -> str:
        return "qdrant"

    async def get_collection_names(self) -> Optional[List[str]]:
        return list(self._collections.keys())

    def get_storage_type(self) -> StorageType:
        return StorageType.VECTOR_DB

    def _string_to_uuid(self, string_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

    async def _ensure_vectorized(self, context: ProcessedContext) -> List[float]:
        if not context.vectorize:
            raise ValueError("Vectorize not set")
        if context.vectorize.vector:
            if not self._vector_size:
                self._vector_size = len(context.vectorize.vector)
            return context.vectorize.vector

        await do_vectorize(context.vectorize)
        if not self._vector_size and context.vectorize.vector:
            self._vector_size = len(context.vectorize.vector)
        return context.vectorize.vector

    def _context_to_qdrant_format(self, context: ProcessedContext) -> Dict[str, Any]:
        payload = context.model_dump(
            exclude_none=True,
            exclude={"properties", "extracted_data", "vectorize", "metadata"},
        )

        if context.extracted_data:
            extracted_data_dict = context.extracted_data.model_dump(exclude_none=True)
            payload.update(extracted_data_dict)

        if context.metadata:
            payload.update(context.metadata)

        if context.vectorize:
            if context.vectorize.content_format == ContentFormat.TEXT:
                payload[FIELD_DOCUMENT] = context.vectorize.text

        if context.properties:
            properties_dict = context.properties.model_dump(exclude_none=True)
            payload.update(properties_dict)

        def default_json_serializer(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value

        for key, value in list(payload.items()):
            if key == "id":
                continue
            if value is None:
                del payload[key]
                continue
            if isinstance(value, datetime.datetime):
                payload[f"{key}_ts"] = int(value.timestamp())
                payload[key] = value.isoformat()
            elif isinstance(value, Enum):
                payload[key] = value.value
            elif isinstance(value, (dict, list)):
                try:
                    payload[key] = json.dumps(
                        value, ensure_ascii=False, default=default_json_serializer
                    )
                except (TypeError, ValueError):
                    payload[key] = str(value)
        return payload

    async def upsert_processed_context(self, context: ProcessedContext) -> str:
        return (await self.batch_upsert_processed_context([context]))[0]

    async def batch_upsert_processed_context(self, contexts: List[ProcessedContext]) -> List[str]:
        if not self._initialized:
            raise RuntimeError("Qdrant backend not initialized")

        if not await self._check_connection():
            raise RuntimeError("Qdrant connection not available")

        contexts_by_type = {}
        for context in contexts:
            context_type = context.extracted_data.context_type.value
            if context_type not in contexts_by_type:
                contexts_by_type[context_type] = []
            contexts_by_type[context_type].append(context)

        stored_ids = []

        for context_type, type_contexts in contexts_by_type.items():
            collection_name = self._collections.get(context_type)
            if not collection_name:
                logger.warning(
                    f"No collection found for context_type '{context_type}', skipping storage"
                )
                continue

            # Batch pre-vectorize all contexts (fewer API calls)
            vectorizes = [
                c.vectorize for c in type_contexts if c.vectorize and not c.vectorize.vector
            ]
            if vectorizes:
                await do_vectorize_batch(vectorizes)

            points = []
            point_to_context_id = {}

            for context in type_contexts:
                try:
                    vector = await self._ensure_vectorized(context)
                    payload = self._context_to_qdrant_format(context)
                    payload[FIELD_ORIGINAL_ID] = context.id

                    uuid_id = self._string_to_uuid(context.id)
                    point = models.PointStruct(
                        id=uuid_id,
                        vector=vector,
                        payload=payload,
                    )
                    points.append(point)
                    point_to_context_id[uuid_id] = context.id

                except Exception as e:
                    logger.exception(f"Failed to process context {context.id}: {e}")
                    continue

            if not points:
                continue

            try:
                await self._client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                stored_ids.extend(point_to_context_id.values())

            except Exception as e:
                logger.error(f"Batch storing context to {context_type} collection failed: {e}")
                continue

        return stored_ids

    async def get_processed_context(
        self, id: str, context_type: str, need_vector: bool = False
    ) -> Optional[ProcessedContext]:
        if not self._initialized:
            return None

        if context_type not in self._collections:
            return None

        collection_name = self._collections[context_type]
        try:
            result = await self._client.retrieve(
                collection_name=collection_name,
                ids=[self._string_to_uuid(id)],
                with_payload=True,
                with_vectors=need_vector,
            )

            if result and len(result) > 0:
                point = result[0]
                return self._qdrant_result_to_context(point, need_vector)

        except Exception as e:
            logger.debug(f"Failed to retrieve context {id} from {context_type} collection: {e}")
            return None

    async def get_all_processed_contexts(
        self,
        context_types: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        filter: Optional[Dict[str, Any]] = None,
        need_vector: bool = False,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, List[ProcessedContext]]:
        if not self._initialized:
            return {}

        result = {}
        if not context_types:
            context_types = [k for k in self._collections.keys() if k != TODO_COLLECTION]

        # Merge multi-user fields into filter dict
        merged_filter = dict(filter) if filter else {}
        if user_id:
            merged_filter["user_id"] = user_id
        if device_id:
            merged_filter["device_id"] = device_id
        if agent_id:
            merged_filter["agent_id"] = agent_id

        for context_type in context_types:
            if context_type not in self._collections:
                continue
            collection_name = self._collections[context_type]
            try:
                filter_condition = self._build_filter_condition(merged_filter)

                fetch_limit = limit + offset

                records, _ = await self._client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_condition,
                    limit=fetch_limit,
                    with_payload=True,
                    with_vectors=need_vector,
                )

                if offset > 0 and len(records) > offset:
                    records = records[offset:]
                elif offset > 0:
                    records = []

                if len(records) > limit:
                    records = records[:limit]

                contexts = []
                for point in records:
                    context = self._qdrant_result_to_context(point, need_vector)
                    if context:
                        contexts.append(context)

                if contexts:
                    result[context_type] = contexts

            except Exception as e:
                logger.exception(f"Failed to get contexts from {context_type} collection: {e}")
                continue

        return result

    async def scroll_processed_contexts(
        self,
        context_types: Optional[List[str]] = None,
        batch_size: int = 100,
        filter: Optional[Dict[str, Any]] = None,
        need_vector: bool = False,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> AsyncGenerator[ProcessedContext, None]:
        """Cursor-based scrolling using Qdrant's native next_page_offset.

        O(n) total — each batch fetches exactly batch_size records using the
        cursor from the previous call. Qdrant's scroll cursor is stateless
        (a point ID), so there is no server-side session to expire.
        """
        if not self._initialized:
            return

        if not context_types:
            context_types = [k for k in self._collections.keys() if k != TODO_COLLECTION]

        merged_filter = dict(filter) if filter else {}
        if user_id:
            merged_filter["user_id"] = user_id
        if device_id:
            merged_filter["device_id"] = device_id
        if agent_id:
            merged_filter["agent_id"] = agent_id

        for context_type in context_types:
            if context_type not in self._collections:
                continue
            collection_name = self._collections[context_type]

            try:
                filter_condition = self._build_filter_condition(merged_filter)
                next_offset = None

                while True:
                    records, next_offset = await self._client.scroll(
                        collection_name=collection_name,
                        scroll_filter=filter_condition,
                        limit=batch_size,
                        offset=next_offset,
                        with_payload=True,
                        with_vectors=need_vector,
                    )

                    if not records:
                        break

                    for point in records:
                        context = self._qdrant_result_to_context(point, need_vector)
                        if context:
                            yield context

                    if next_offset is None:
                        break

            except Exception as e:
                logger.exception(f"Failed to scroll contexts from {context_type} collection: {e}")
                continue

    async def delete_processed_context(self, id: str, context_type: str) -> bool:
        return await self.delete_contexts([id], context_type)

    async def search(
        self,
        query: Vectorize,
        top_k: int = 10,
        context_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        need_vector: bool = False,
    ) -> List[Tuple[ProcessedContext, float]]:
        if not self._initialized:
            return []

        target_collections = {}
        if context_types:
            for context_type in context_types:
                if context_type in self._collections:
                    target_collections[context_type] = self._collections[context_type]
                else:
                    logger.warning(f"Collection not found: {context_type}")
        else:
            target_collections = {
                k: v for k, v in self._collections.items() if k != TODO_COLLECTION
            }

        query_vector = None
        if query.vector and len(query.vector) > 0:
            query_vector = query.vector
        else:
            await do_vectorize(query)
            query_vector = query.vector

        if not query_vector:
            logger.warning("Unable to get query vector, search failed")
            return []

        # Merge multi-user fields into filters
        merged_filters = dict(filters) if filters else {}
        if user_id:
            merged_filters["user_id"] = user_id
        if device_id:
            merged_filters["device_id"] = device_id
        if agent_id:
            merged_filters["agent_id"] = agent_id

        all_results = []

        for context_type, collection_name in target_collections.items():
            try:
                collection_info = await self._client.get_collection(collection_name)
                if collection_info.points_count == 0:
                    continue

                filter_condition = self._build_filter_condition(merged_filters)

                results = (
                    await self._client.query_points(
                        collection_name=collection_name,
                        query=query_vector,
                        query_filter=filter_condition,
                        limit=top_k,
                        with_payload=True,
                        with_vectors=need_vector,
                    )
                ).points

                for scored_point in results:
                    context = self._qdrant_result_to_context(scored_point, need_vector)
                    if context:
                        score = scored_point.score
                        all_results.append((context, score))

            except Exception as e:
                logger.exception(f"Vector search failed in {context_type} collection: {e}")
                continue

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def _qdrant_result_to_context(
        self, point: models.Record, need_vector: bool = True
    ) -> Optional[ProcessedContext]:
        try:
            if not point.id:
                logger.warning("Qdrant result missing id field")
                return None

            extracted_data_field_names = set(ExtractedData.model_fields.keys())
            properties_field_names = set(ContextProperties.model_fields.keys())
            vectorize_field_names = set(Vectorize.model_fields.keys())

            extracted_data_dict = {}
            properties_dict = {}
            context_dict = {}
            vectorize_dict = {}
            metadata_dict = {}

            payload = dict(point.payload) if point.payload else {}
            document = payload.pop(FIELD_DOCUMENT, None)
            vector = point.vector if need_vector else None

            if document:
                vectorize_dict["text"] = document
            if vector:
                vectorize_dict["vector"] = vector

            metadata_field_names = set()
            context_type_value = payload.get("context_type")

            original_id = payload.pop(FIELD_ORIGINAL_ID, str(point.id))

            for key, value in payload.items():
                if key.endswith("_ts"):
                    continue

                val = value
                if isinstance(value, str) and value.startswith(("{", "[")):
                    try:
                        val = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if key in extracted_data_field_names:
                    extracted_data_dict[key] = val
                elif key in properties_field_names:
                    properties_dict[key] = val
                elif key in vectorize_field_names:
                    vectorize_dict[key] = val
                elif metadata_field_names and key in metadata_field_names:
                    metadata_dict[key] = val
                else:
                    metadata_dict[key] = val

            context_dict["id"] = original_id
            context_dict["extracted_data"] = ExtractedData.model_validate(extracted_data_dict)
            context_dict["properties"] = ContextProperties.model_validate(properties_dict)
            context_dict["vectorize"] = Vectorize.model_validate(vectorize_dict)

            if metadata_dict:
                context_dict["metadata"] = metadata_dict

            context = ProcessedContext.model_validate(context_dict)
            if not need_vector:
                context.vectorize.vector = None
            return context

        except Exception as e:
            logger.exception(f"Failed to convert Qdrant result to ProcessedContext: {e}")
            return None

    def _build_filter_condition(self, filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        if not filters:
            return None

        must_conditions = []

        for key, value in filters.items():
            if key == "context_type":
                continue
            elif key == "entities":
                continue
            elif not value:
                continue
            elif key.endswith("_ts") and isinstance(value, dict):
                if "$gte" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(gte=value["$gte"]),
                        )
                    )
                if "$lte" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(lte=value["$lte"]),
                        )
                    )
            else:
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )

        if not must_conditions:
            return None
        elif len(must_conditions) == 1:
            return models.Filter(must=must_conditions)
        else:
            return models.Filter(must=must_conditions)

    async def delete_contexts(self, ids: List[str], context_type: str) -> bool:
        if not self._initialized:
            return False

        if context_type not in self._collections:
            return False

        collection_name = self._collections[context_type]
        try:
            uuid_ids = [self._string_to_uuid(id) for id in ids]
            await self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=uuid_ids),
            )
            return True
        except Exception as e:
            logger.exception(f"Failed to delete Qdrant contexts: {e}")
            return False

    async def get_processed_context_count(self, context_type: str) -> int:
        if not self._initialized:
            return 0

        if context_type not in self._collections:
            return 0

        collection_name = self._collections[context_type]
        return (await self._client.count(collection_name)).count

    async def get_all_processed_context_counts(self) -> Dict[str, int]:
        if not self._initialized:
            return {}

        result = {}
        for context_type in self._collections.keys():
            if context_type != TODO_COLLECTION:
                result[context_type] = await self.get_processed_context_count(context_type)

        return result

    async def upsert_todo_embedding(
        self,
        todo_id: int,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
    ) -> bool:
        if not self._initialized:
            logger.warning("Qdrant not initialized, cannot store todo embedding")
            return False

        try:
            collection_name = self._collections.get(TODO_COLLECTION)
            if not collection_name:
                logger.error("Todo collection not found")
                return False

            payload = {
                FIELD_TODO_ID: todo_id,
                FIELD_CONTENT: content,
                FIELD_CREATED_AT: datetime.datetime.now().isoformat(),
            }
            if metadata:
                payload.update(metadata)

            point = models.PointStruct(
                id=todo_id,
                vector=embedding,
                payload=payload,
            )

            await self._client.upsert(
                collection_name=collection_name,
                points=[point],
            )

            return True

        except Exception as e:
            logger.error(f"Failed to store todo embedding (id={todo_id}): {e}")
            return False

    async def search_similar_todos(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[int, str, float]]:
        if not self._initialized:
            logger.warning("Qdrant not initialized, cannot search todos")
            return []

        try:
            collection_name = self._collections.get(TODO_COLLECTION)
            if not collection_name:
                logger.error("Todo collection not found")
                return []

            if (await self._client.count(collection_name)).count == 0:
                return []

            results = (
                await self._client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    limit=top_k,
                    with_payload=True,
                )
            ).points

            similar_todos = []
            for scored_point in results:
                similarity = scored_point.score

                if similarity >= similarity_threshold:
                    payload = scored_point.payload
                    similar_todos.append(
                        (
                            payload[FIELD_TODO_ID],
                            payload[FIELD_CONTENT],
                            similarity,
                        )
                    )

            return similar_todos

        except Exception as e:
            logger.error(f"Failed to search similar todos: {e}")
            return []

    async def delete_todo_embedding(self, todo_id: int) -> bool:
        if not self._initialized:
            logger.warning("Qdrant not initialized, cannot delete todo embedding")
            return False

        try:
            collection_name = self._collections.get(TODO_COLLECTION)
            if not collection_name:
                logger.error("Todo collection not found")
                return False

            await self._client.delete(
                collection_name=collection_name,
                points_selector=[todo_id],
            )
            logger.debug(f"Deleted todo embedding: id={todo_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete todo embedding (id={todo_id}): {e}")
            return False

    async def delete_by_source_file(
        self, source_file_key: str, user_id: Optional[str] = None
    ) -> bool:
        if not self._initialized:
            return False

        must_conditions = [
            models.FieldCondition(
                key="source_file_key",
                match=models.MatchValue(value=source_file_key),
            )
        ]

        if user_id:
            must_conditions.append(
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                )
            )

        filter_condition = models.Filter(must=must_conditions)

        success = True
        for context_type, collection_name in self._collections.items():
            if context_type == TODO_COLLECTION:
                continue
            try:
                await self._client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(filter=filter_condition),
                )
                logger.debug(
                    f"Deleted points with source_file_key='{source_file_key}' "
                    f"from collection '{collection_name}'"
                )
            except Exception as e:
                logger.error(
                    f"Failed to delete by source_file_key='{source_file_key}' "
                    f"from collection '{collection_name}': {e}"
                )
                success = False

        return success

    async def search_by_hierarchy(
        self,
        context_type: str,
        hierarchy_level: int,
        time_bucket_start: Optional[str] = None,
        time_bucket_end: Optional[str] = None,
        user_id: Optional[str] = None,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Tuple[ProcessedContext, float]]:
        if not self._initialized:
            return []

        if context_type not in self._collections:
            logger.warning(f"Collection not found for context_type: {context_type}")
            return []

        collection_name = self._collections[context_type]

        must_conditions = [
            models.FieldCondition(
                key="hierarchy_level",
                match=models.MatchValue(value=hierarchy_level),
            )
        ]

        if user_id:
            must_conditions.append(
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                )
            )
        if device_id:
            must_conditions.append(
                models.FieldCondition(
                    key="device_id",
                    match=models.MatchValue(value=device_id),
                )
            )
        if agent_id:
            must_conditions.append(
                models.FieldCondition(
                    key="agent_id",
                    match=models.MatchValue(value=agent_id),
                )
            )

        filter_condition = models.Filter(must=must_conditions)

        # Fetch more results to allow for in-code time_bucket filtering
        fetch_limit = top_k * 3 if (time_bucket_start or time_bucket_end) else top_k

        try:
            records, _ = await self._client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=fetch_limit,
                with_payload=True,
                with_vectors=False,
            )

            results = []
            for point in records:
                # In-code string comparison filtering for time_bucket
                # (Qdrant Range filter does not support string fields)
                if time_bucket_start or time_bucket_end:
                    payload = point.payload or {}
                    tb = payload.get("time_bucket", "")
                    if time_bucket_start and tb < time_bucket_start:
                        continue
                    if time_bucket_end and tb > time_bucket_end:
                        continue

                context = self._qdrant_result_to_context(point, need_vector=False)
                if context:
                    results.append((context, 1.0))
                    if len(results) >= top_k:
                        break

            return results

        except Exception as e:
            logger.exception(
                f"search_by_hierarchy failed for context_type={context_type}, "
                f"hierarchy_level={hierarchy_level}: {e}"
            )
            return []

    async def get_by_ids(
        self,
        ids: List[str],
        context_type: Optional[str] = None,
        need_vector: bool = False,
    ) -> List[ProcessedContext]:
        if not self._initialized:
            return []

        if not ids:
            return []

        uuid_ids = [self._string_to_uuid(id) for id in ids]

        target_collections = {}
        if context_type and context_type in self._collections:
            target_collections[context_type] = self._collections[context_type]
        else:
            target_collections = {
                k: v for k, v in self._collections.items() if k != TODO_COLLECTION
            }

        results = []

        for ct, collection_name in target_collections.items():
            try:
                points = await self._client.retrieve(
                    collection_name=collection_name,
                    ids=uuid_ids,
                    with_payload=True,
                    with_vectors=need_vector,
                )

                for point in points:
                    context = self._qdrant_result_to_context(point, need_vector=need_vector)
                    if context:
                        results.append(context)

            except Exception as e:
                logger.debug(f"Failed to retrieve IDs from collection '{collection_name}': {e}")
                continue

        return results

    async def batch_set_parent_id(
        self,
        children_ids: List[str],
        parent_id: str,
        context_type: str,
    ) -> int:
        if not self._initialized or not children_ids:
            return 0
        collection_name = self._collections.get(context_type)
        if not collection_name:
            return 0
        uuid_ids = [self._string_to_uuid(cid) for cid in children_ids]
        try:
            await self._client.set_payload(
                collection_name=collection_name,
                payload={"parent_id": parent_id},
                points=uuid_ids,
            )
            return len(children_ids)
        except Exception as e:
            logger.warning(f"batch_set_parent_id failed: {e}")
            return 0
