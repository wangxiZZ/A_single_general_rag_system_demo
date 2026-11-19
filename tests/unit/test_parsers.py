"""基于固定夹具的 parser 回归：块序、seq、bbox 与分块长度分布。"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from rag_kb.application.chunking import chunk_document
from rag_kb.domain.errors import NoExtractableTextError
from rag_kb.domain.models import BlockType, ChunkLevel
from rag_kb.infra.parsers.docx import DocxParser
from rag_kb.infra.parsers.pdf import PdfParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "documents"
PDF = FIXTURES / "sample.pdf"
DOCX = FIXTURES / "sample.docx"
SCANNED = FIXTURES / "scanned.pdf"


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    return PDF.read_bytes()


@pytest.fixture(scope="module")
def docx_bytes() -> bytes:
    return DOCX.read_bytes()


def test_pdf_fixture_block_order_seq_and_bbox(pdf_bytes: bytes) -> None:
    parsed = PdfParser().parse(pdf_bytes, filename="sample.pdf", mime="application/pdf")

    assert parsed.page_count == 2
    assert [block.seq for block in parsed.blocks] == list(range(len(parsed.blocks)))
    assert all(block.page_no in (1, 2) for block in parsed.blocks)
    assert all(block.bbox is not None for block in parsed.blocks)

    texts = [block.text for block in parsed.blocks if block.text]
    assert texts[0] == "Fixture Manual"
    assert "3.2 Dependencies" in texts
    assert "Appendix" in texts

    images = [block for block in parsed.blocks if block.type is BlockType.IMAGE]
    assert len(images) == 1
    assert images[0].page_no == 1
    assert images[0].bbox is not None
    assert images[0].bbox.x1 - images[0].bbox.x0 == 200

    for block in parsed.blocks:
        if block.text:
            assert block.char_end - block.char_start == len(block.text)


def test_docx_fixture_block_order_and_missing_geometry(docx_bytes: bytes) -> None:
    parsed = DocxParser().parse(docx_bytes, filename="sample.docx", mime="x")

    assert parsed.page_count is None
    assert [block.seq for block in parsed.blocks] == list(range(len(parsed.blocks)))
    assert all(block.page_no is None and block.bbox is None for block in parsed.blocks)
    assert "HEADER SHOULD NOT APPEAR" not in [block.text for block in parsed.blocks]

    heading = next(block for block in parsed.blocks if block.text == "3.2 Dependencies")
    assert heading.type is BlockType.HEADING
    assert heading.heading_level == 2

    assert any(block.type is BlockType.IMAGE for block in parsed.blocks)
    table = next(block for block in parsed.blocks if block.type is BlockType.TABLE)
    assert "Name | Value" in table.text


def test_scanned_fixture_has_no_text() -> None:
    with pytest.raises(NoExtractableTextError):
        PdfParser().parse(SCANNED.read_bytes(), filename="scanned.pdf", mime="application/pdf")


def test_chunk_length_distribution_on_pdf_fixture(pdf_bytes: bytes) -> None:
    parsed = PdfParser().parse(pdf_bytes, filename="sample.pdf", mime="application/pdf")
    chunks = chunk_document(
        list(parsed.blocks),
        kb_id=uuid4(),
        doc_id=uuid4(),
        doc_sha256="d" * 64,
        chunker_version="v1",
        target_tokens=40,
        overlap_ratio=0.15,
        parent_max_tokens=120,
        chars_per_token=1.5,
    )

    parents = [chunk for chunk in chunks if chunk.level is ChunkLevel.PARENT]
    children = [chunk for chunk in chunks if chunk.level is ChunkLevel.CHILD]
    assert parents and children
    assert all(parent.token_count <= 120 for parent in parents)
    # 子块正文受 target 控制；内容含标题前缀，允许前缀带来的少量超出。
    assert all(child.token_count <= 80 for child in children)
    assert all(child.parent_id in {parent.id for parent in parents} for child in children)
    assert all(child.heading_path for child in children)
    assert [chunk.seq for chunk in chunks] == list(range(len(chunks)))
