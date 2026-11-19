"""Milvus 向量存储集成测试（真容器）。"""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_kb.config import get_settings
from rag_kb.domain.errors import ConfigurationError, ValidationError
from rag_kb.domain.models import ChunkLevel, VectorRecord
from rag_kb.infra.milvus.store import MilvusVectorStore

pytestmark = pytest.mark.integration


async def test_collection_search_isolation_and_delete() -> None:
    settings = get_settings()
    store = MilvusVectorStore(
        settings.milvus_uri,
        collection_name=f"chunk_vec_it_{uuid4().hex[:8]}",
        token=settings.milvus_token,
    )
    try:
        await store.ensure_collection(4)
        await store.ensure_collection(4)  # 幂等

        with pytest.raises(ConfigurationError, match="维度"):
            await store.ensure_collection(8)

        kb1, kb2 = uuid4(), uuid4()
        doc1, doc2 = uuid4(), uuid4()
        await store.upsert(
            kb1,
            [
                VectorRecord("a" * 32, doc1, ChunkLevel.CHILD, (1.0, 0.0, 0.0, 0.0)),
                VectorRecord("b" * 32, doc2, ChunkLevel.CHILD, (0.0, 1.0, 0.0, 0.0)),
            ],
        )
        await store.flush()

        hits = await store.search(kb1, [1.0, 0.0, 0.0, 0.0], top_k=10)
        assert {hit.chunk_id for hit in hits} == {"a" * 32, "b" * 32}
        assert hits[0].chunk_id == "a" * 32

        # 知识库隔离：kb2 不应看到 kb1 的数据
        assert await store.search(kb2, [1.0, 0.0, 0.0, 0.0], top_k=10) == []
        assert await store.count(kb1) == 2
        assert await store.count(kb2) == 0

        deleted = await store.delete_by_doc(kb1, doc1)
        assert deleted == 1
        assert await store.count(kb1) == 1
        assert await store.count(kb1, doc1) == 0

        # 父块不允许写入向量库
        with pytest.raises(ValidationError, match="child"):
            await store.upsert(
                kb1,
                [VectorRecord("c" * 32, doc1, ChunkLevel.PARENT, (1.0, 0.0, 0.0, 0.0))],
            )
    finally:
        await store.drop_collection()
        await store.close()
