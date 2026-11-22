"""Milvus 向量存储。

设计要点：

1. 单 collection + ``partition_key=kb_id``：库数量无上限，查询带 ``kb_id`` 等值过滤时
   只扫对应分区。**分区键不是隔离边界**——不带过滤仍会扫全部分区，所以端口实现强制加
   filter。
2. 只写入 ``level=child`` 的子块；父块只用于回答回填。
3. ``consistency_level="Strong"``：摄取后立即检索、删除后断言 0 命中的测试不会因最终
   一致而偶发失败；代价是每次查询略慢。
4. 维度不一致直接抛 ``ConfigurationError`` 并提示重建，避免"能启动但检索恒空"。
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from uuid import UUID

import anyio
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

from rag_kb.domain.errors import (
    ConfigurationError,
    DependencyUnavailableError,
    ValidationError,
)
from rag_kb.domain.models import ChunkLevel, VectorHit, VectorRecord


def _to_thread[**P, R](func: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await anyio.to_thread.run_sync(lambda: func(*args, **kwargs))

    return wrapper


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """把 pymilvus 的异常翻译成领域异常。

    必须写在 ``_to_thread`` 里层（同步方法层）：异常是子线程里抛出的，只有在那里转换，
    上层才能拿到稳定的错误码。不转换的后果是 Milvus 掉线表现为 500 而不是 503，
    而且异常文本里可能带上连接串；领域异常只保留"依赖不可用: milvus"这一层信息。
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except MilvusException as exc:
            raise DependencyUnavailableError("milvus") from exc

    return wrapper


def _delete_count(result: Any) -> int:
    if isinstance(result, dict):
        return int(result.get("delete_count", result.get("deleteCount", 0)))
    return int(getattr(result, "delete_count", 0))


class MilvusVectorStore:
    def __init__(
        self,
        uri: str,
        *,
        collection_name: str = "chunk_vec",
        token: str = "",
    ) -> None:
        kwargs: dict[str, Any] = {"uri": uri}
        if token:
            kwargs["token"] = token
        self._client = MilvusClient(**kwargs)
        self._collection = collection_name
        self._dimension: int | None = None

    async def close(self) -> None:
        await anyio.to_thread.run_sync(self._client.close)

    @_to_thread
    @_guard
    def ensure_collection(self, dimension: int) -> None:
        if dimension < 1:
            raise ValidationError(f"dimension 必须 >= 1，当前 {dimension}")

        if self._client.has_collection(self._collection):
            actual = self._collection_dimension()
            if actual != dimension:
                raise ConfigurationError(
                    f"Milvus collection {self._collection} 维度为 {actual}，"
                    f"当前 EMBEDDING_DIM={dimension}；请重建 collection（make reset）"
                )
            self._dimension = dimension
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("kb_id", DataType.VARCHAR, max_length=64, is_partition_key=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("level", DataType.VARCHAR, max_length=16)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        self._dimension = dimension

    def _collection_dimension(self) -> int:
        info = self._client.describe_collection(self._collection)
        for field in info.get("fields", []):
            if field.get("name") == "vector":
                params = field.get("params") or {}
                return int(params.get("dim", 0))
        raise ConfigurationError(f"collection {self._collection} 缺少 vector 字段")

    @_to_thread
    @_guard
    def upsert(self, kb_id: UUID, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        if self._dimension is None:
            raise ConfigurationError("调用 upsert 前必须先 ensure_collection")
        rows = []
        for record in records:
            if record.level is not ChunkLevel.CHILD:
                raise ValidationError("只有 level=child 的块可以写入向量库")
            if len(record.vector) != self._dimension:
                raise ValidationError(
                    f"向量维度不一致: expected={self._dimension} actual={len(record.vector)}"
                )
            rows.append(
                {
                    "id": record.chunk_id,
                    "kb_id": str(kb_id),
                    "doc_id": str(record.doc_id),
                    "level": record.level.value,
                    "vector": list(record.vector),
                }
            )
        self._client.upsert(collection_name=self._collection, data=rows)

    @_to_thread
    @_guard
    def search(self, kb_id: UUID, vector: Sequence[float], top_k: int) -> list[VectorHit]:
        if self._dimension is None:
            raise ConfigurationError("调用 search 前必须先 ensure_collection")
        if len(vector) != self._dimension:
            raise ValidationError(
                f"查询向量维度不一致: expected={self._dimension} actual={len(vector)}"
            )
        results = self._client.search(
            collection_name=self._collection,
            data=[list(vector)],
            anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            filter=f'kb_id == "{kb_id}"',
            output_fields=["doc_id", "level"],
            consistency_level="Strong",
        )
        hits = results[0] if results else []
        return [
            VectorHit(chunk_id=str(hit["id"]), score=float(hit.get("distance", 0.0)), rank=rank)
            for rank, hit in enumerate(hits)
        ]

    @_to_thread
    @_guard
    def delete_by_doc(self, kb_id: UUID, doc_id: UUID) -> int:
        result = self._client.delete(
            collection_name=self._collection,
            filter=f'kb_id == "{kb_id}" and doc_id == "{doc_id}"',
        )
        return _delete_count(result)

    @_to_thread
    @_guard
    def count(self, kb_id: UUID, doc_id: UUID | None = None) -> int:
        expr = f'kb_id == "{kb_id}"'
        if doc_id is not None:
            expr += f' and doc_id == "{doc_id}"'
        rows = self._client.query(
            collection_name=self._collection,
            filter=expr,
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        if not rows:
            return 0
        return int(next(iter(rows[0].values())))

    async def flush(self) -> None:
        """测试/运维用：显式刷盘，确保 Strong 读能看到最新写入。"""
        await anyio.to_thread.run_sync(self._client.flush, self._collection)

    async def drop_collection(self) -> None:
        await anyio.to_thread.run_sync(self._client.drop_collection, self._collection)
