"""领域模型的纯单测：不发网络请求、不连数据库。"""

from __future__ import annotations

from itertools import pairwise
from uuid import uuid4

import pytest

from rag_kb.domain.errors import (
    ConflictError,
    ErrorCode,
    FileTooLargeError,
    NoExtractableTextError,
    NotFoundError,
)
from rag_kb.domain.models import (
    BBox,
    Chunk,
    ChunkKind,
    ChunkLevel,
    Citation,
    DocumentStatus,
    can_transition,
)

KB_ID = uuid4()
DOC_ID = uuid4()


def test_document_status_happy_path() -> None:
    path = [
        DocumentStatus.PENDING,
        DocumentStatus.PARSING,
        DocumentStatus.CHUNKED,
        DocumentStatus.INDEXING,
        DocumentStatus.READY,
    ]
    assert all(can_transition(a, b) for a, b in pairwise(path))


def test_ready_is_not_reachable_directly_from_chunked() -> None:
    # 核心约束：M2 只到 chunked，M3 建完三路索引才到 ready。
    assert not can_transition(DocumentStatus.CHUNKED, DocumentStatus.READY)


def test_failed_only_returns_to_pending() -> None:
    assert can_transition(DocumentStatus.FAILED, DocumentStatus.PENDING)
    assert not can_transition(DocumentStatus.FAILED, DocumentStatus.INDEXING)


def test_ready_can_reenter_indexing_for_reindex() -> None:
    assert can_transition(DocumentStatus.READY, DocumentStatus.INDEXING)


def test_chunk_indexable_only_for_child() -> None:
    child = Chunk(
        id="a" * 32,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        level=ChunkLevel.CHILD,
        kind=ChunkKind.TEXT,
        content="子块",
        seq=1,
        char_start=0,
        char_end=2,
        token_count=2,
        chunker_version="v1",
        parent_id="p" * 32,
    )
    parent = Chunk(
        id="b" * 32,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        level=ChunkLevel.PARENT,
        kind=ChunkKind.TEXT,
        content="父块",
        seq=0,
        char_start=0,
        char_end=2,
        token_count=2,
        chunker_version="v1",
    )
    assert child.is_indexable is True
    assert parent.is_indexable is False


def test_docx_chunk_allows_null_page_and_bbox() -> None:
    chunk = Chunk(
        id="c" * 32,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        level=ChunkLevel.CHILD,
        kind=ChunkKind.TEXT,
        content="docx 段落",
        seq=3,
        char_start=10,
        char_end=16,
        token_count=4,
        chunker_version="v1",
        page_no=None,
        bbox=None,
    )
    assert chunk.page_no is None
    assert chunk.bbox is None


def test_chunk_rejects_inverted_char_range() -> None:
    with pytest.raises(ValueError):
        Chunk(
            id="d" * 32,
            kb_id=KB_ID,
            doc_id=DOC_ID,
            level=ChunkLevel.CHILD,
            kind=ChunkKind.TEXT,
            content="bad",
            seq=0,
            char_start=10,
            char_end=5,
            token_count=1,
            chunker_version="v1",
        )


def test_bbox_rejects_inverted_corners() -> None:
    with pytest.raises(ValueError):
        BBox(x0=10, y0=10, x1=5, y1=20)


def test_citation_index_is_one_based() -> None:
    with pytest.raises(ValueError):
        Citation(index=0, doc_id=DOC_ID, doc_name="a.pdf", chunk_id="e" * 32)
    citation = Citation(index=1, doc_id=DOC_ID, doc_name="a.pdf", chunk_id="e" * 32)
    assert citation.index == 1


def test_error_codes_are_stable() -> None:
    assert NotFoundError("document", DOC_ID).code is ErrorCode.NOT_FOUND
    assert ConflictError("duplicate").code is ErrorCode.CONFLICT
    assert FileTooLargeError("a.pdf", 100, 50).code is ErrorCode.FILE_TOO_LARGE
    assert NoExtractableTextError("scan.pdf").code is ErrorCode.NO_EXTRACTABLE_TEXT
