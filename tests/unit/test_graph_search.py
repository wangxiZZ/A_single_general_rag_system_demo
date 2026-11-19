"""图谱召回用例单测：fake graph store + fake repository。"""

from __future__ import annotations

from uuid import UUID, uuid4

from rag_kb.application.graph_search import graph_search
from rag_kb.domain.models import (
    Chunk,
    ChunkKind,
    ChunkLevel,
    Document,
    Entity,
    GraphEvidence,
)

KB_ID = uuid4()
DOC_ID = uuid4()


class FakeGraphStore:
    def __init__(self, seeds: list[Entity], evidence: list[GraphEvidence]) -> None:
        self.seeds = seeds
        self.evidence = evidence
        self.expand_calls: list[tuple[list[str], int, int]] = []

    async def candidate_entities(
        self, kb_id: UUID, terms, limit: int, *, contains: bool = False
    ) -> list[Entity]:
        return self.seeds

    async def expand(
        self, kb_id: UUID, seed_ids, max_hops: int, neighbor_limit: int
    ) -> list[GraphEvidence]:
        self.expand_calls.append((list(seed_ids), max_hops, neighbor_limit))
        return self.evidence


class FakeRepository:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}

    async def get_chunks(self, kb_id: UUID, chunk_ids) -> list[Chunk]:
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    async def get_document(self, kb_id: UUID, doc_id: UUID) -> Document | None:
        return Document(
            id=doc_id,
            kb_id=kb_id,
            filename="manual.pdf",
            mime="application/pdf",
            sha256="a" * 64,
            size=1,
        )


def _chunk(chunk_id: str, content: str, seq: int = 0) -> Chunk:
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
        heading_path=("Manual",),
        page_no=1,
    )


def _seed() -> Entity:
    return Entity(id="e1", kb_id=KB_ID, name="年假政策", normalized="年假政策", type="条款")


async def test_graph_search_returns_original_chunks() -> None:
    chunk1 = _chunk("a" * 32, "年假政策规定员工每年享有带薪年假。")
    chunk2 = _chunk("b" * 32, "休假申请需要审批。", seq=1)
    store = FakeGraphStore(
        [_seed()],
        [
            GraphEvidence(chunk_id=chunk1.id, score=0.5, hops=1, seed_ids=("e1",)),
            GraphEvidence(chunk_id=chunk2.id, score=0.3, hops=2, seed_ids=("e1",)),
        ],
    )

    hits = await graph_search(
        KB_ID,
        "年假政策",
        graph_store=store,  # type: ignore[arg-type]
        repository=FakeRepository([chunk1, chunk2]),  # type: ignore[arg-type]
    )

    assert [hit.chunk_id for hit in hits] == [chunk1.id, chunk2.id]
    assert hits[0].channels == ("graph",)
    assert hits[0].doc_name == "manual.pdf"
    assert hits[0].snippet.startswith("年假政策")
    assert hits[1].debug["hops"] == 2
    assert store.expand_calls == [(["e1"], 2, 50)]


async def test_no_seeds_returns_empty() -> None:
    store = FakeGraphStore([], [])
    hits = await graph_search(
        KB_ID,
        "不存在",
        graph_store=store,
        repository=FakeRepository([]),  # type: ignore[arg-type]
    )
    assert hits == []
    assert store.expand_calls == []


async def test_missing_chunk_is_dropped() -> None:
    store = FakeGraphStore(
        [_seed()],
        [GraphEvidence(chunk_id="z" * 32, score=0.5, hops=1, seed_ids=("e1",))],
    )
    hits = await graph_search(
        KB_ID,
        "年假政策",
        graph_store=store,  # type: ignore[arg-type]
        repository=FakeRepository([]),  # type: ignore[arg-type]
    )
    assert hits == []
