"""图谱证据 → 子块的字符区间映射测试。"""

from __future__ import annotations

from uuid import uuid4

from rag_kb.application.graph_extract import map_evidence_to_child
from rag_kb.domain.models import Chunk, ChunkKind, ChunkLevel


def _chunk(chunk_id: str, start: int, end: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        kb_id=uuid4(),
        doc_id=uuid4(),
        level=ChunkLevel.CHILD,
        kind=ChunkKind.TEXT,
        content="x" * (end - start),
        seq=start,
        char_start=start,
        char_end=end,
        token_count=end - start,
        chunker_version="v1",
    )


def test_maps_to_covering_child() -> None:
    children = [_chunk("a" * 32, 0, 10), _chunk("b" * 32, 10, 20)]
    assert map_evidence_to_child((2, 8), children) == "a" * 32
    assert map_evidence_to_child((10, 12), children) == "b" * 32
    assert map_evidence_to_child((8, 15), children) == "b" * 32


def test_tie_keeps_first_child() -> None:
    children = [_chunk("a" * 32, 0, 10), _chunk("b" * 32, 10, 20)]
    assert map_evidence_to_child((5, 15), children) == "a" * 32


def test_no_overlap_returns_none() -> None:
    children = [_chunk("a" * 32, 0, 10)]
    assert map_evidence_to_child((20, 30), children) is None
    assert map_evidence_to_child((5, 5), children) is None
