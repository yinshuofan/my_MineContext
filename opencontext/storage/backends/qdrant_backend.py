#!/usr/bin/env python

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import datetime
import json
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from opencontext.llm.global_embedding_client import do_vectorize, do_vectorize_batch
from opencontext.models.context import ContextProperties, ExtractedData, ProcessedContext, Vectorize
from opencontext.models.enums import ContextType
from opencontext.storage.base_storage import IVectorStorageBackend, StorageType
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

FIELD_DOCUMENT = "document"


class QdrantBackend(IVectorStorageBackend):
    """
    Qdrant vector storage backend - https://qdrant.tech/
    """

    def __init__(self):
        self._client: AsyncQdrantClient | None = None
        self._collections: dict[str, str] = {}
        self._initialized = False
        self._config = None
        self._vector_size = None

    async def initialize(self, config: dict[str, Any]) -> bool:
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

            self._initialized = True
            logger.info(
                f"Qdrant vector backend initialized successfully, "
                f"created {len(self._collections)} collections"
            )
            return True

        except Exception as e:
            logger.exception(f"Qdrant vector backend initialization failed: {e}")
            return False

    async def _ensure_collection(self, collection_name: str, context_type: str) -> None:
        if not await self._client.collection_exists(collection_name):  # type: ignore[union-attr]
            vector_size = self._vector_size or 1536

            await self._client.create_collection(  # type: ignore[union-attr]
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

    async def get_collection_names(self) -> list[str] | None:
        return list(self._collections.keys())

    def get_storage_type(self) -> StorageType:
        return StorageType.VECTOR_DB

    async def _ensure_vectorized(self, context: ProcessedContext) -> list[float]:
        if not context.vectorize:
            raise ValueError("Vectorize not set")
        if context.vectorize.vector:
            if not self._vector_size:
                self._vector_size = len(context.vectorize.vector)
            return context.vectorize.vector

        await do_vectorize(context.vectorize)
        if not self._vector_size and context.vectorize.vector:
            self._vector_size = len(context.vectorize.vector)
        return context.vectorize.vector  # type: ignore[return-value]

    def _context_to_qdrant_format(self, context: ProcessedContext) -> dict[str, Any]:
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
            text = context.vectorize.get_text()
            if text:
                payload[FIELD_DOCUMENT] = text

        if context.properties:
            properties_dict = context.properties.model_dump(exclude_none=True)
            payload.update(properties_dict)

        # Explicit refs serialization — always present for backward compatibility
        payload["refs"] = (
            json.dumps(context.properties.refs)
            if context.properties and context.properties.refs
            else "{}"
        )

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

    async def batch_upsert_processed_context(self, contexts: list[ProcessedContext]) -> list[str]:
        if not self._initialized:
            raise RuntimeError("Qdrant backend not initialized")

        if not await self._check_connection():
            raise RuntimeError("Qdrant connection not available")

        contexts_by_type = {}  # type: ignore[var-annotated]
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
            stored_context_ids = []

            for context in type_contexts:
                try:
                    vector = await self._ensure_vectorized(context)
                    payload = self._context_to_qdrant_format(context)

                    point = models.PointStruct(
                        id=context.id,
                        vector=vector,
                        payload=payload,
                    )
                    points.append(point)
                    stored_context_ids.append(context.id)

                except Exception as e:
                    logger.exception(f"Failed to process context {context.id}: {e}")
                    continue

            if not points:
                continue

            try:
                await self._client.upsert(  # type: ignore[union-attr]
                    collection_name=collection_name,
                    points=points,
                )
                stored_ids.extend(stored_context_ids)

            except Exception as e:
                logger.error(f"Batch storing context to {context_type} collection failed: {e}")
                continue

        return stored_ids

    async def get_processed_context(  # type: ignore[override, return]
        self, id: str, context_type: str, need_vector: bool = False
    ) -> ProcessedContext | None:
        if not self._initialized:
            return None

        if context_type not in self._collections:
            return None

        collection_name = self._collections[context_type]
        try:
            result = await self._client.retrieve(  # type: ignore[union-attr]
                collection_name=collection_name,
                ids=[id],
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
        context_types: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
        filter: dict[str, Any] | None = None,
        need_vector: bool = False,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
        skip_slice: bool = False,
    ) -> dict[str, list[ProcessedContext]]:
        if not self._initialized:
            return {}

        result = {}
        if not context_types:
            context_types = list(self._collections.keys())

        # Base filter (user_id and device_id applied per-collection below)
        base_filter = dict(filter) if filter else {}
        if agent_id:
            base_filter["agent_id"] = agent_id

        for context_type in context_types:
            if context_type not in self._collections:
                continue
            collection_name = self._collections[context_type]
            try:
                # Skip user_id and device_id for agent_base_* types
                merged_filter = dict(base_filter)
                is_base = context_type.startswith("agent_base")
                if user_id and not is_base:
                    merged_filter["user_id"] = user_id
                if device_id and not is_base:
                    merged_filter["device_id"] = device_id

                filter_condition = self._build_filter_condition(merged_filter)

                fetch_limit = limit + offset

                records, _ = await self._client.scroll(  # type: ignore[union-attr]
                    collection_name=collection_name,
                    scroll_filter=filter_condition,
                    limit=fetch_limit,
                    with_payload=True,
                    with_vectors=need_vector,
                )

                if not skip_slice:
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
        context_types: list[str] | None = None,
        batch_size: int = 100,
        filter: dict[str, Any] | None = None,
        need_vector: bool = False,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
    ) -> AsyncGenerator[ProcessedContext, None]:
        """Cursor-based scrolling using Qdrant's native next_page_offset.

        O(n) total — each batch fetches exactly batch_size records using the
        cursor from the previous call. Qdrant's scroll cursor is stateless
        (a point ID), so there is no server-side session to expire.
        """
        if not self._initialized:
            return

        if not context_types:
            context_types = list(self._collections.keys())

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
                    records, next_offset = await self._client.scroll(  # type: ignore[union-attr]
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

    async def search(  # type: ignore[override]
        self,
        query: Vectorize,
        top_k: int = 10,
        context_types: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
        need_vector: bool = False,
        score_threshold: float | None = None,
    ) -> list[tuple[ProcessedContext, float]]:
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
            target_collections = {k: v for k, v in self._collections.items()}

        query_vector = None
        if query.vector and len(query.vector) > 0:
            query_vector = query.vector
        else:
            await do_vectorize(query, role="query")
            query_vector = query.vector

        if not query_vector:
            logger.warning("Unable to get query vector, search failed")
            return []

        # Base filters (user_id and device_id applied per-collection below)
        base_filters = dict(filters) if filters else {}
        if agent_id:
            base_filters["agent_id"] = agent_id

        all_results = []

        for context_type, collection_name in target_collections.items():
            try:
                collection_info = await self._client.get_collection(collection_name)  # type: ignore[union-attr]
                if collection_info.points_count == 0:
                    continue

                # Skip user_id and device_id filters for agent_base_* types
                # (base memories are agent-level, not scoped to a specific user or device)
                merged_filters = dict(base_filters)
                is_base = context_type.startswith("agent_base")
                if user_id and not is_base:
                    merged_filters["user_id"] = user_id
                if device_id and not is_base:
                    merged_filters["device_id"] = device_id

                filter_condition = self._build_filter_condition(merged_filters)

                results = (
                    await self._client.query_points(  # type: ignore[union-attr]
                        collection_name=collection_name,
                        query=query_vector,
                        query_filter=filter_condition,
                        limit=top_k,
                        score_threshold=score_threshold,
                        with_payload=True,
                        with_vectors=need_vector,
                    )
                ).points

                for scored_point in results:
                    context = self._qdrant_result_to_context(scored_point, need_vector)  # type: ignore[arg-type]
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
    ) -> ProcessedContext | None:
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
                vectorize_dict["input"] = [{"type": "text", "text": document}]
            if vector:
                vectorize_dict["vector"] = vector  # type: ignore[assignment]

            metadata_field_names = set()  # type: ignore[var-annotated]
            payload.get("context_type")

            for key, value in payload.items():
                if key.endswith("_ts"):
                    continue

                val = value
                if isinstance(value, str) and value.startswith(("{", "[")):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        val = json.loads(value)

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

            context_dict["id"] = str(point.id)
            context_dict["extracted_data"] = ExtractedData.model_validate(extracted_data_dict)  # type: ignore[assignment]
            context_dict["properties"] = ContextProperties.model_validate(properties_dict)  # type: ignore[assignment]
            context_dict["vectorize"] = Vectorize.model_validate(vectorize_dict)  # type: ignore[assignment]

            if metadata_dict:
                context_dict["metadata"] = metadata_dict  # type: ignore[assignment]

            context = ProcessedContext.model_validate(context_dict)
            if not need_vector:
                context.vectorize.vector = None
            return context

        except Exception as e:
            logger.exception(f"Failed to convert Qdrant result to ProcessedContext: {e}")
            return None

    def _build_filter_condition(self, filters: dict[str, Any] | None) -> models.Filter | None:
        if not filters:
            return None

        must_conditions = []

        for key, value in filters.items():
            if key == "context_type":
                continue
            elif key == "entities":
                # TODO: entities filter is skipped because entities are stored as a
                # JSON-serialized string (e.g. '["Alice","Bob"]') in the payload.
                # Qdrant's MatchAny/MatchValue cannot match individual elements
                # inside a JSON string. To enable entity filtering, store entities
                # as a native list (not json.dumps) and use MatchAny on the field.
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
                if "$lt" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(lt=value["$lt"]),
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

    async def delete_contexts(self, ids: list[str], context_type: str) -> bool:
        if not self._initialized:
            return False

        if context_type not in self._collections:
            return False

        collection_name = self._collections[context_type]
        try:
            await self._client.delete(  # type: ignore[union-attr]
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=ids),
            )
            return True
        except Exception as e:
            logger.exception(f"Failed to delete Qdrant contexts: {e}")
            return False

    async def delete_contexts_bulk(self, ids_by_type: dict[str, list[str]]) -> dict[str, bool]:
        if not self._initialized:
            return {ct: False for ct in ids_by_type}
        if not ids_by_type:
            return {}

        types = list(ids_by_type.keys())
        results = await asyncio.gather(
            *(self.delete_contexts(ids_by_type[ct], ct) for ct in types),
            return_exceptions=True,
        )
        return {
            ct: (not isinstance(r, BaseException) and bool(r))
            for ct, r in zip(types, results, strict=True)
        }

    async def get_processed_context_count(
        self,
        context_type: str,
        filter: dict[str, Any] | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
    ) -> int:
        if not self._initialized:
            return 0

        if context_type not in self._collections:
            return 0

        collection_name = self._collections[context_type]

        merged_filter = dict(filter) if filter else {}
        if user_id:
            merged_filter["user_id"] = user_id
        if device_id:
            merged_filter["device_id"] = device_id
        if agent_id:
            merged_filter["agent_id"] = agent_id

        count_filter = self._build_filter_condition(merged_filter) if merged_filter else None
        return (await self._client.count(collection_name, count_filter=count_filter)).count  # type: ignore[union-attr]

    async def get_all_processed_context_counts(self) -> dict[str, int]:
        if not self._initialized:
            return {}

        result = {}
        for context_type in self._collections:
            result[context_type] = await self.get_processed_context_count(context_type)

        return result

    async def search_by_hierarchy(
        self,
        context_type: str,
        hierarchy_level: int,
        time_start: float | None = None,
        time_end: float | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        agent_id: str | None = None,
        top_k: int = 20,
    ) -> list[tuple[ProcessedContext, float]]:
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

        # Skip user_id and device_id filters for agent_base_* types
        is_base = context_type.startswith("agent_base")
        if user_id and not is_base:
            must_conditions.append(
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                )
            )
        if device_id and not is_base:
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

        # Numeric range overlap
        if time_end is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="event_time_start_ts",
                    range=models.Range(lte=time_end),
                )
            )
        if time_start is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="event_time_end_ts",
                    range=models.Range(gte=time_start),
                )
            )

        filter_condition = models.Filter(must=must_conditions)

        try:
            records, _ = await self._client.scroll(  # type: ignore[union-attr]
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            results = []
            for point in records:
                context = self._qdrant_result_to_context(point, need_vector=False)
                if context:
                    results.append((context, 1.0))

            return results

        except Exception as e:
            logger.exception(
                f"search_by_hierarchy failed for context_type={context_type}, "
                f"hierarchy_level={hierarchy_level}: {e}"
            )
            return []

    async def get_by_ids(
        self,
        ids: list[str],
        context_type: str | None = None,
        need_vector: bool = False,
    ) -> list[ProcessedContext]:
        if not self._initialized:
            return []

        if not ids:
            return []

        target_collections = {}
        if context_type and context_type in self._collections:
            target_collections[context_type] = self._collections[context_type]
        else:
            target_collections = {k: v for k, v in self._collections.items()}

        results = []

        for _ct, collection_name in target_collections.items():
            try:
                points = await self._client.retrieve(  # type: ignore[union-attr]
                    collection_name=collection_name,
                    ids=ids,
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

    async def batch_update_refs(
        self,
        context_ids: list[str],
        ref_key: str,
        ref_value: str,
        context_type: str,
    ) -> int:
        """Add a ref entry to multiple contexts."""
        if not self._initialized or not context_ids:
            return 0
        collection_name = self._collections.get(context_type)
        if not collection_name:
            return 0

        updated = 0
        for ctx_id in context_ids:
            try:
                points = await self._client.retrieve(  # type: ignore[union-attr]
                    collection_name=collection_name,
                    ids=[ctx_id],
                    with_payload=True,
                )
                if not points:
                    continue

                existing_refs = json.loads(points[0].payload.get("refs", "{}"))  # type: ignore[union-attr]
                if ref_key not in existing_refs:
                    existing_refs[ref_key] = []
                if ref_value not in existing_refs[ref_key]:
                    existing_refs[ref_key].append(ref_value)

                await self._client.set_payload(  # type: ignore[union-attr]
                    collection_name=collection_name,
                    payload={"refs": json.dumps(existing_refs)},
                    points=[ctx_id],
                )
                updated += 1
            except Exception as e:
                logger.warning(f"batch_update_refs failed for {ctx_id}: {e}")
        return updated
