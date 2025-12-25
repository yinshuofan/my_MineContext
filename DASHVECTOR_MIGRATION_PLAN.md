# DashVector 替代 Qdrant/ChromaDB 方案

## 一、概述

本方案旨在使用阿里云 DashVector 向量检索服务替代现有的 Qdrant 和 ChromaDB 向量数据库后端，以获得更好的云服务支持和可扩展性。

## 二、API 对比分析

### 2.1 核心概念映射

| 概念 | Qdrant/ChromaDB | DashVector | 说明 |
|------|-----------------|------------|------|
| 连接 | QdrantClient / chromadb.Client | dashvector.Client | 客户端初始化 |
| 集合 | Collection | Collection | 向量集合 |
| 向量记录 | Point / Document | Doc | 单条向量数据 |
| 向量维度 | vector_size / dimension | dimension | 创建时指定 |
| 距离度量 | Distance.COSINE | metric='cosine' | 支持 cosine/euclidean/dotproduct |
| 元数据 | payload / metadata | fields | 附加字段 |
| 分区 | - | Partition | DashVector 支持数据分区 |

### 2.2 接口方法映射

| 功能 | Qdrant | ChromaDB | DashVector |
|------|--------|----------|------------|
| 初始化客户端 | `QdrantClient(url, api_key)` | `chromadb.Client()` | `dashvector.Client(api_key, endpoint)` |
| 创建集合 | `create_collection()` | `get_or_create_collection()` | `client.create(name, dimension)` |
| 获取集合 | `get_collection()` | `get_collection()` | `client.get(name)` |
| 检查集合存在 | `collection_exists()` | 异常处理 | `client.list()` 检查 |
| 删除集合 | `delete_collection()` | `delete_collection()` | `client.delete(name)` |
| 插入向量 | `upsert(points=[])` | `add(ids, embeddings, metadatas)` | `collection.insert(docs)` |
| 更新向量 | `upsert()` | `update()` | `collection.upsert(docs)` |
| 删除向量 | `delete(ids=[])` | `delete(ids=[])` | `collection.delete(ids)` |
| 向量搜索 | `query_points(query, limit)` | `query(query_embeddings, n_results)` | `collection.query(vector, topk)` |
| 按ID获取 | `retrieve(ids=[])` | `get(ids=[])` | `collection.fetch(ids)` |
| 统计数量 | `count()` | `count()` | `collection.stats()` |

## 三、实现方案

### 3.1 新建 DashVectorBackend 类

创建文件 `opencontext/storage/backends/dashvector_backend.py`，实现 `IVectorStorageBackend` 接口。

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DashVector vector storage backend - Aliyun DashVector Service
"""

import datetime
import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import dashvector
from dashvector import Doc

from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.models.context import (
    ContextProperties,
    ExtractedData,
    ProcessedContext,
    Vectorize,
)
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


class DashVectorBackend(IVectorStorageBackend):
    """
    DashVector vector storage backend - https://help.aliyun.com/product/2510217.html
    """

    def __init__(self):
        self._client: Optional[dashvector.Client] = None
        self._collections: Dict[str, Any] = {}  # context_type -> collection
        self._initialized = False
        self._config = None
        self._vector_size = None

    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化 DashVector 后端"""
        try:
            self._config = config
            dashvector_config = config.get("config", {})

            # 获取配置
            api_key = dashvector_config.get("api_key")
            endpoint = dashvector_config.get("endpoint")
            self._vector_size = dashvector_config.get("vector_size", 1536)
            timeout = dashvector_config.get("timeout", 30.0)

            if not api_key or not endpoint:
                raise ValueError("DashVector requires api_key and endpoint")

            # 创建客户端
            self._client = dashvector.Client(
                api_key=api_key,
                endpoint=endpoint,
                timeout=timeout
            )

            # 检查连接
            if not self._client:
                raise RuntimeError("Failed to create DashVector client")

            # 为每个 context_type 创建 collection
            context_types = [ct.value for ct in ContextType]
            for context_type in context_types:
                collection_name = f"{context_type}"
                self._ensure_collection(collection_name, context_type)
                self._collections[context_type] = self._client.get(collection_name)

            # 创建 todo collection
            self._ensure_collection(TODO_COLLECTION, TODO_COLLECTION)
            self._collections[TODO_COLLECTION] = self._client.get(TODO_COLLECTION)

            self._initialized = True
            logger.info(
                f"DashVector backend initialized successfully, created {len(self._collections)} collections"
            )
            return True

        except Exception as e:
            logger.exception(f"DashVector backend initialization failed: {e}")
            return False

    def _ensure_collection(self, collection_name: str, context_type: str) -> None:
        """确保 collection 存在，不存在则创建"""
        try:
            # 检查 collection 是否存在
            existing = self._client.list()
            if collection_name not in existing:
                # 创建新 collection
                ret = self._client.create(
                    name=collection_name,
                    dimension=self._vector_size,
                    metric='cosine',
                    dtype=float,
                    fields_schema={
                        'original_id': str,
                        'context_type': str,
                        'document': str,
                        'created_at': str,
                        'user_id': str,
                        'device_id': str,
                        'agent_id': str,
                    },
                    timeout=-1  # 异步创建
                )
                if ret:
                    logger.info(f"Created DashVector collection: {collection_name}")
                else:
                    logger.warning(f"Failed to create collection {collection_name}: {ret.message}")
            else:
                logger.debug(f"DashVector collection already exists: {collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name}: {e}")
            raise

    # ... 其他方法实现见下文
```

### 3.2 关键方法实现

#### 3.2.1 upsert_processed_context

```python
def upsert_processed_context(self, context: ProcessedContext) -> str:
    """存储单个 ProcessedContext"""
    return self.batch_upsert_processed_context([context])[0]

def batch_upsert_processed_context(self, contexts: List[ProcessedContext]) -> List[str]:
    """批量存储 ProcessedContext"""
    if not self._initialized:
        raise RuntimeError("DashVector backend not initialized")

    # 按 context_type 分组
    contexts_by_type = {}
    for context in contexts:
        context_type = context.extracted_data.context_type.value
        if context_type not in contexts_by_type:
            contexts_by_type[context_type] = []
        contexts_by_type[context_type].append(context)

    stored_ids = []

    for context_type, type_contexts in contexts_by_type.items():
        collection = self._collections.get(context_type)
        if not collection:
            logger.warning(f"No collection found for context_type '{context_type}'")
            continue

        docs = []
        for context in type_contexts:
            try:
                vector = self._ensure_vectorized(context)
                fields = self._context_to_dashvector_format(context)
                fields['original_id'] = context.id

                doc = Doc(
                    id=context.id,
                    vector=vector,
                    fields=fields
                )
                docs.append(doc)
            except Exception as e:
                logger.exception(f"Failed to process context {context.id}: {e}")
                continue

        if not docs:
            continue

        try:
            # 使用 upsert 确保更新或插入
            ret = collection.upsert(docs)
            if ret:
                stored_ids.extend([doc.id for doc in docs])
            else:
                logger.error(f"Batch upsert failed: {ret.message}")
        except Exception as e:
            logger.error(f"Batch storing context to {context_type} failed: {e}")
            continue

    return stored_ids
```

#### 3.2.2 search (向量相似性搜索)

```python
def search(
    self,
    query: Vectorize,
    top_k: int = 10,
    context_types: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    device_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> List[Tuple[ProcessedContext, float]]:
    """向量相似性搜索"""
    if not self._initialized:
        return []

    # 确定目标 collections
    target_collections = {}
    if context_types:
        for context_type in context_types:
            if context_type in self._collections:
                target_collections[context_type] = self._collections[context_type]
    else:
        target_collections = {
            k: v for k, v in self._collections.items() if k != TODO_COLLECTION
        }

    # 获取查询向量
    query_vector = None
    if query.vector and len(query.vector) > 0:
        query_vector = query.vector
    else:
        do_vectorize(query)
        query_vector = query.vector

    if not query_vector:
        logger.warning("Unable to get query vector, search failed")
        return []

    # 构建过滤条件
    filter_str = self._build_filter_string(filters, user_id, device_id, agent_id)

    all_results = []

    for context_type, collection in target_collections.items():
        try:
            # 执行查询
            ret = collection.query(
                vector=query_vector,
                topk=top_k,
                filter=filter_str,
                include_vector=False,
                output_fields=None  # 返回所有字段
            )

            if ret:
                for doc in ret:
                    context = self._dashvector_result_to_context(doc)
                    if context:
                        all_results.append((context, doc.score))

        except Exception as e:
            logger.exception(f"Vector search failed in {context_type}: {e}")
            continue

    # 按分数排序
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:top_k]
```

#### 3.2.3 delete_contexts

```python
def delete_contexts(self, ids: List[str], context_type: str) -> bool:
    """删除指定的 contexts"""
    if not self._initialized:
        return False

    if context_type not in self._collections:
        return False

    collection = self._collections[context_type]
    try:
        ret = collection.delete(ids)
        return bool(ret)
    except Exception as e:
        logger.exception(f"Failed to delete contexts: {e}")
        return False
```

#### 3.2.4 过滤条件构建

```python
def _build_filter_string(
    self,
    filters: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    device_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Optional[str]:
    """构建 DashVector 过滤条件字符串
    
    DashVector 使用 SQL WHERE 子句风格的过滤语法
    """
    conditions = []

    # 添加用户过滤条件
    if user_id:
        conditions.append(f"user_id = '{user_id}'")
    if device_id:
        conditions.append(f"device_id = '{device_id}'")
    if agent_id:
        conditions.append(f"agent_id = '{agent_id}'")

    # 添加其他过滤条件
    if filters:
        for key, value in filters.items():
            if key in ('context_type', 'entities'):
                continue
            if value is None:
                continue
            
            if isinstance(value, dict):
                # 范围查询
                if '$gte' in value:
                    conditions.append(f"{key} >= {value['$gte']}")
                if '$lte' in value:
                    conditions.append(f"{key} <= {value['$lte']}")
            elif isinstance(value, list):
                # IN 查询
                values_str = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                conditions.append(f"{key} IN ({values_str})")
            elif isinstance(value, str):
                conditions.append(f"{key} = '{value}'")
            else:
                conditions.append(f"{key} = {value}")

    if not conditions:
        return None

    return ' AND '.join(conditions)
```

### 3.3 Todo 相关方法

```python
def upsert_todo_embedding(
    self,
    todo_id: int,
    content: str,
    embedding: List[float],
    metadata: Optional[Dict] = None,
) -> bool:
    """存储 todo embedding"""
    if not self._initialized:
        return False

    try:
        collection = self._collections.get(TODO_COLLECTION)
        if not collection:
            return False

        fields = {
            'todo_id': todo_id,
            'content': content,
            'created_at': datetime.datetime.now().isoformat(),
        }
        if metadata:
            fields.update(metadata)

        doc = Doc(
            id=str(todo_id),
            vector=embedding,
            fields=fields
        )

        ret = collection.upsert([doc])
        return bool(ret)

    except Exception as e:
        logger.error(f"Failed to store todo embedding: {e}")
        return False

def search_similar_todos(
    self,
    query_embedding: List[float],
    top_k: int = 10,
    similarity_threshold: float = 0.85,
) -> List[Tuple[int, str, float]]:
    """搜索相似的 todos"""
    if not self._initialized:
        return []

    try:
        collection = self._collections.get(TODO_COLLECTION)
        if not collection:
            return []

        ret = collection.query(
            vector=query_embedding,
            topk=top_k,
            include_vector=False
        )

        similar_todos = []
        if ret:
            for doc in ret:
                if doc.score >= similarity_threshold:
                    similar_todos.append((
                        doc.fields.get('todo_id'),
                        doc.fields.get('content'),
                        doc.score
                    ))

        return similar_todos

    except Exception as e:
        logger.error(f"Failed to search similar todos: {e}")
        return []

def delete_todo_embedding(self, todo_id: int) -> bool:
    """删除 todo embedding"""
    if not self._initialized:
        return False

    try:
        collection = self._collections.get(TODO_COLLECTION)
        if not collection:
            return False

        ret = collection.delete([str(todo_id)])
        return bool(ret)

    except Exception as e:
        logger.error(f"Failed to delete todo embedding: {e}")
        return False
```

## 四、配置说明

### 4.1 配置文件示例

在 `config/config_dashvector.yaml` 中添加：

```yaml
storage:
  enabled: true
  backends:
    # DashVector 向量数据库配置
    - name: "default_vector"
      storage_type: "vector_db"
      backend: "dashvector"
      config:
        api_key: "${DASHVECTOR_API_KEY}"
        endpoint: "${DASHVECTOR_ENDPOINT}"
        vector_size: 1536
        timeout: 30.0
```

### 4.2 环境变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `DASHVECTOR_API_KEY` | DashVector API Key | `sk-xxxxxxxxxxxx` |
| `DASHVECTOR_ENDPOINT` | DashVector Cluster Endpoint | `vrs-cn-xxxxxxxxxx.dashvector.cn-hangzhou.aliyuncs.com` |

## 五、代码修改清单

### 5.1 新增文件

| 文件路径 | 说明 |
|----------|------|
| `opencontext/storage/backends/dashvector_backend.py` | DashVector 后端实现 |
| `config/config_dashvector.yaml` | DashVector 配置示例 |

### 5.2 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `opencontext/storage/backends/__init__.py` | 导出 DashVectorBackend |
| `opencontext/storage/unified_storage.py` | 添加 DashVector 工厂方法 |
| `pyproject.toml` | 添加 dashvector 依赖 |

### 5.3 unified_storage.py 修改

```python
# 在 _create_vector_backend 方法中添加：
elif backend_type == "dashvector":
    from opencontext.storage.backends.dashvector_backend import DashVectorBackend
    backend = DashVectorBackend()
```

### 5.4 依赖添加

在 `pyproject.toml` 中添加：

```toml
dependencies = [
    ...
    "dashvector",
]
```

## 六、注意事项

### 6.1 DashVector 特性差异

1. **Collection 创建**：DashVector 需要预先定义 `fields_schema`，建议在创建时定义常用字段
2. **ID 类型**：DashVector 的 Doc ID 为字符串类型，与现有实现兼容
3. **过滤语法**：DashVector 使用 SQL WHERE 子句风格，需要转换现有的过滤条件格式
4. **Partition 支持**：DashVector 支持数据分区，可用于多租户场景优化

### 6.2 性能考虑

1. **批量操作**：DashVector 支持批量插入，建议使用 `batch_upsert_processed_context`
2. **异步创建**：Collection 创建时使用 `timeout=-1` 开启异步模式
3. **连接池**：DashVector SDK 内部管理连接，无需额外配置

### 6.3 错误处理

1. **返回值检查**：DashVector 操作返回 `DashVectorResponse` 对象，需检查 `ret.code`
2. **重试机制**：建议对网络错误添加重试逻辑
3. **日志记录**：记录所有失败操作的详细信息

## 七、迁移步骤

1. **安装依赖**：`pip install dashvector`
2. **创建 DashVector Cluster**：在阿里云控制台创建
3. **获取 API Key 和 Endpoint**：从控制台获取
4. **配置环境变量**：设置 `DASHVECTOR_API_KEY` 和 `DASHVECTOR_ENDPOINT`
5. **修改配置文件**：使用 `config_dashvector.yaml`
6. **数据迁移**：如需迁移现有数据，编写迁移脚本
7. **测试验证**：运行测试确保功能正常

## 八、测试计划

### 8.1 单元测试

- [ ] Client 初始化测试
- [ ] Collection 创建/获取/删除测试
- [ ] Doc 插入/更新/删除测试
- [ ] 向量搜索测试
- [ ] 过滤条件测试
- [ ] Todo 相关方法测试

### 8.2 集成测试

- [ ] 完整的 ProcessedContext 存储流程
- [ ] 多 context_type 并发操作
- [ ] 大批量数据插入性能
- [ ] 搜索结果准确性验证

## 九、总结

DashVector 作为阿里云的托管向量检索服务，具有以下优势：

1. **云原生**：无需自建和维护向量数据库
2. **高可用**：阿里云提供 SLA 保障
3. **弹性扩展**：支持按需扩容
4. **多租户**：通过 Partition 支持数据隔离

本方案通过实现 `IVectorStorageBackend` 接口，可以无缝替换现有的 Qdrant 和 ChromaDB 后端，同时保持 API 兼容性。
