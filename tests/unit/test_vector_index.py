"""向量索引用例单测：fake embedder + fake vector store。"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from rag_kb.application.vector_index import delete_document_vectors, index_chunks
from rag_kb.domain.errors import ConfigurationError
from rag_kb.domain.models import Chunk, ChunkKind, ChunkLevel, VectorRecord

KB_ID = uuid4()
DOC_ID = uuid4()


class FakeEmbedder:
    dimension = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), 0.0, 0.0, 1.0] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0, 1.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[UUID, list[VectorRecord]]] = []
        self.deleted: list[tuple[UUID, UUID]] = []

    async def upsert(self, kb_id: UUID, records: list[VectorRecord]) -> None:
        self.upserts.append((kb_id, list(records)))

    async def delete_by_doc(self, kb_id: UUID, doc_id: UUID) -> int:
        self.deleted.append((kb_id, doc_id))
        return 2


def _chunk(chunk_id: str, level: ChunkLevel, content: str, seq: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        level=level,
        kind=ChunkKind.TEXT,
        content=content,
        seq=seq,
        char_start=0,
        char_end=len(content),
        token_count=len(content),
        chunker_version="v1",
        parent_id="p" * 32 if level is ChunkLevel.CHILD else None,
    )


async def test_indexes_only_children_in_order() -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    chunks = [
        _chunk("p" * 32, ChunkLevel.PARENT, "父块内容", 0),
        _chunk("a" * 32, ChunkLevel.CHILD, "子块A", 1),
        _chunk("b" * 32, ChunkLevel.CHILD, "子块BB", 2),
    ]

    count = await index_chunks(chunks, kb_id=KB_ID, embedder=embedder, vector_store=store)

    assert count == 2
    assert embedder.calls == [["子块A", "子块BB"]]
    kb_id, records = store.upserts[0]
    assert kb_id == KB_ID
    assert [record.chunk_id for record in records] == ["a" * 32, "b" * 32]
    assert [record.vector[0] for record in records] == [3.0, 4.0]
    assert all(record.level is ChunkLevel.CHILD for record in records)


async def test_no_children_does_not_embed() -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    chunks = [_chunk("p" * 32, ChunkLevel.PARENT, "父块", 0)]

    assert await index_chunks(chunks, kb_id=KB_ID, embedder=embedder, vector_store=store) == 0
    assert embedder.calls == []
    assert store.upserts == []


async def test_missing_embedder_fails_before_writing() -> None:
    store = FakeVectorStore()
    chunks = [_chunk("a" * 32, ChunkLevel.CHILD, "子块", 0)]

    with pytest.raises(ConfigurationError, match="EMBED_API_KEY"):
        await index_chunks(chunks, kb_id=KB_ID, embedder=None, vector_store=store)
    assert store.upserts == []


async def test_delete_returns_store_count() -> None:
    store = FakeVectorStore()
    assert await delete_document_vectors(store, KB_ID, DOC_ID) == 2
    assert store.deleted == [(KB_ID, DOC_ID)]
