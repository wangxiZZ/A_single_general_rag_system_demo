"""证据拼装与上下文预算单测。"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from rag_kb.application.qa_service import QAService
from rag_kb.domain.errors import ConfigurationError
from rag_kb.domain.models import (
    Chunk,
    ChunkKind,
    ChunkLevel,
    SearchHit,
)

KB_ID = uuid4()
DOC_ID = uuid4()


class FakeRepository:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}

    async def get_chunks(self, kb_id: UUID, chunk_ids) -> list[Chunk]:
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]


def _chunk(
    chunk_id: str,
    content: str,
    *,
    level: ChunkLevel,
    parent_id: str | None = None,
    seq: int = 0,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        parent_id=parent_id,
        level=level,
        kind=ChunkKind.TEXT,
        content=content,
        seq=seq,
        char_start=0,
        char_end=len(content),
        token_count=len(content),
        chunker_version="v1",
        heading_path=("Manual",),
    )


def _hit(chunk_id: str, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=DOC_ID,
        doc_name="a.pdf",
        level=ChunkLevel.CHILD,
        score=score,
        channels=("hybrid",),
        heading_path=("Manual",),
    )


async def test_child_hit_backfills_parent() -> None:
    parent = _chunk("p" * 32, "父块完整上下文。", level=ChunkLevel.PARENT)
    child = _chunk("c" * 32, "子块", level=ChunkLevel.CHILD, parent_id=parent.id, seq=1)
    service = QAService(
        FakeRepository([parent, child]),  # type: ignore[arg-type]
        max_context_tokens=100,
        parent_max_tokens=10,
        chars_per_token=1.0,
    )

    evidence = await service.assemble_evidence(KB_ID, [_hit(child.id, 0.9)])

    assert len(evidence) == 1
    assert evidence[0].hit.chunk_id == child.id
    assert evidence[0].body.id == parent.id
    assert evidence[0].text == parent.content
    assert evidence[0].index == 1


async def test_same_parent_is_deduplicated_by_highest_score() -> None:
    parent = _chunk("p" * 32, "父块。", level=ChunkLevel.PARENT)
    child1 = _chunk("a" * 32, "子块A", level=ChunkLevel.CHILD, parent_id=parent.id, seq=1)
    child2 = _chunk("b" * 32, "子块B", level=ChunkLevel.CHILD, parent_id=parent.id, seq=2)
    service = QAService(
        FakeRepository([parent, child1, child2]),  # type: ignore[arg-type]
        max_context_tokens=100,
        parent_max_tokens=10,
        chars_per_token=1.0,
    )

    evidence = await service.assemble_evidence(KB_ID, [_hit(child1.id, 0.5), _hit(child2.id, 0.9)])

    assert len(evidence) == 1
    assert evidence[0].hit.chunk_id == child2.id


async def test_budget_drops_lowest_whole_items() -> None:
    chunks = [
        _chunk("a" * 32, "甲" * 8, level=ChunkLevel.CHILD),
        _chunk("b" * 32, "乙" * 8, level=ChunkLevel.CHILD),
        _chunk("c" * 32, "丙" * 8, level=ChunkLevel.CHILD),
    ]
    service = QAService(
        FakeRepository(chunks),  # type: ignore[arg-type]
        max_context_tokens=20,
        parent_max_tokens=10,
        chars_per_token=1.0,
    )

    evidence = await service.assemble_evidence(
        KB_ID,
        [_hit("a" * 32, 0.9), _hit("b" * 32, 0.5), _hit("c" * 32, 0.3)],
    )

    assert [item.hit.chunk_id for item in evidence] == ["a" * 32, "b" * 32]
    assert [item.index for item in evidence] == [1, 2]


async def test_parent_missing_falls_back_to_child() -> None:
    child = _chunk("c" * 32, "子块内容", level=ChunkLevel.CHILD, parent_id="missing")
    service = QAService(
        FakeRepository([child]),  # type: ignore[arg-type]
        max_context_tokens=100,
        parent_max_tokens=10,
        chars_per_token=1.0,
    )
    evidence = await service.assemble_evidence(KB_ID, [_hit(child.id, 0.9)])
    assert evidence[0].body.id == child.id


def test_budget_must_cover_parent_max() -> None:
    with pytest.raises(ConfigurationError, match="QA_MAX_CONTEXT_TOKENS"):
        QAService(
            FakeRepository([]),  # type: ignore[arg-type]
            max_context_tokens=10,
            parent_max_tokens=20,
            chars_per_token=1.0,
        )
