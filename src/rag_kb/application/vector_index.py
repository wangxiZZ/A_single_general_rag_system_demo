"""向量索引用例。

- 只对 ``level=child`` 的块做 embedding；父块不参与检索。
- 先完成全部 embedding，再一次性 upsert：embedding 失败不会留下半批向量。
- 删除按 ``(kb_id, doc_id)`` 过滤，返回实际删除计数。
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from rag_kb.application.embedding import embed_texts
from rag_kb.domain.models import Chunk, ChunkLevel, VectorRecord
from rag_kb.domain.ports import Embedder, VectorStore


async def index_chunks(
    chunks: Sequence[Chunk],
    *,
    kb_id: UUID,
    embedder: Embedder | None,
    vector_store: VectorStore,
) -> int:
    children = [chunk for chunk in chunks if chunk.level is ChunkLevel.CHILD]
    if not children:
        return 0

    vectors = await embed_texts(embedder, [chunk.content for chunk in children])
    records = [
        VectorRecord(
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            level=ChunkLevel.CHILD,
            vector=tuple(vector),
        )
        for chunk, vector in zip(children, vectors, strict=True)
    ]
    await vector_store.upsert(kb_id, records)
    return len(records)


async def delete_document_vectors(vector_store: VectorStore, kb_id: UUID, doc_id: UUID) -> int:
    return await vector_store.delete_by_doc(kb_id, doc_id)
