"""检索编排单测：fake 三路端口与仓储。"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from rag_kb.application.search_service import SearchService
from rag_kb.domain.errors import ConfigurationError
from rag_kb.domain.models import (
    Chunk,
    ChunkKind,
    ChunkLevel,
    Document,
    Entity,
    GraphEvidence,
    LexicalHit,
    SearchMode,
    VectorHit,
)

KB_ID = uuid4()
DOC_ID = uuid4()


class FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    async def search(self, kb_id: UUID, vector, top_k: int) -> list[VectorHit]:
        return [VectorHit("a" * 32, 0.9, 1), VectorHit("b" * 32, 0.8, 2)]


class FakeLexicalIndex:
    async def search(self, kb_id: UUID, query: str, top_k: int) -> list[LexicalHit]:
        return [LexicalHit("b" * 32, 5.0, 1), LexicalHit("c" * 32, 4.0, 2)]


class FakeGraphStore:
    async def candidate_entities(self, kb_id: UUID, terms, limit: int, *, contains=False):
        return [Entity(id="e1", kb_id=KB_ID, name="年假", normalized="年假", type="条款")]

    async def expand(self, kb_id: UUID, seed_ids, max_hops: int, neighbor_limit: int):
        return [GraphEvidence("b" * 32, 0.5, 1, ("e1",))]


class FakeRepository:
    def __init__(self) -> None:
        self.chunks = {
            chunk.id: chunk
            for chunk in [
                _chunk("a" * 32, "A 内容", 0),
                _chunk("b" * 32, "B 内容", 1),
                _chunk("c" * 32, "C 内容", 2),
            ]
        }

    async def get_chunks(self, kb_id: UUID, chunk_ids) -> list[Chunk]:
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    async def get_document(self, kb_id: UUID, doc_id: UUID) -> Document | None:
        return Document(
            id=doc_id,
            kb_id=kb_id,
            filename="a.pdf",
            mime="application/pdf",
            sha256="a" * 64,
            size=1,
        )


class FakeReranker:
    async def rerank(self, query, candidates, top_k: int):
        return list(reversed(list(candidates)))[:top_k]


def _chunk(chunk_id: str, content: str, seq: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        level=ChunkLevel.CHILD,
        kind=ChunkKind.TEXT,
        content=content,
        seq=seq,
        char_start=0,
        char_end=len(content),
        token_count=len(content),
        chunker_version="v1",
        page_no=1,
    )


def _service(**kwargs) -> SearchService:
    defaults = {
        "embedder": FakeEmbedder(),
        "vector_store": FakeVectorStore(),
        "lexical_index": FakeLexicalIndex(),
        "graph_store": FakeGraphStore(),
        "top_k": 20,
    }
    defaults.update(kwargs)
    return SearchService(FakeRepository(), **defaults)


async def test_vector_mode_hydrates_source_chunks() -> None:
    hits = await _service().search(KB_ID, "查询", mode=SearchMode.VECTOR)
    assert [hit.chunk_id for hit in hits] == ["a" * 32, "b" * 32]
    assert hits[0].channels == ("vector",)
    assert hits[0].doc_name == "a.pdf"
    assert hits[0].snippet == "A 内容"


async def test_hybrid_rrf_merges_channels() -> None:
    hits = await _service().search(KB_ID, "年假", mode=SearchMode.HYBRID)
    assert [hit.chunk_id for hit in hits] == ["b" * 32, "a" * 32, "c" * 32]
    assert hits[0].channels == ("vector", "keyword", "graph")
    assert hits[1].channels == ("vector",)


async def test_weighted_mode_runs() -> None:
    hits = await _service().search(KB_ID, "年假", mode=SearchMode.HYBRID_WEIGHTED)
    assert {hit.chunk_id for hit in hits} == {"a" * 32, "b" * 32, "c" * 32}


async def test_rerank_applied_when_enabled() -> None:
    hits = await _service(reranker=FakeReranker()).search(
        KB_ID, "查询", mode=SearchMode.VECTOR, rerank=True
    )
    assert [hit.chunk_id for hit in hits] == ["b" * 32, "a" * 32]


async def test_rerank_without_reranker_raises() -> None:
    with pytest.raises(ConfigurationError, match="RERANK"):
        await _service().search(KB_ID, "查询", mode=SearchMode.VECTOR, rerank=True)


async def test_vector_mode_without_embedder_raises() -> None:
    with pytest.raises(ConfigurationError, match="EMBED_API_KEY"):
        await _service(embedder=None, vector_store=None).search(
            KB_ID, "查询", mode=SearchMode.VECTOR
        )


async def test_hybrid_degrades_when_vector_channel_missing() -> None:
    hits = await _service(embedder=None, vector_store=None).search(
        KB_ID, "年假", mode=SearchMode.HYBRID
    )
    assert {hit.chunk_id for hit in hits} == {"b" * 32, "c" * 32}
    assert hits[0].channels == ("keyword", "graph")
