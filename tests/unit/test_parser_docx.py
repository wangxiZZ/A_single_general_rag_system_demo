"""docx 解析器单测：用 python-docx 现场生成 fixture。"""

from __future__ import annotations

import io
import struct
import zlib
from itertools import pairwise

import pytest
from docx import Document

from rag_kb.domain.errors import ParseError
from rag_kb.domain.models import BlockType
from rag_kb.infra.parsers.docx import DocxParser


def _png(size: int) -> bytes:
    raw = b"".join(b"\x00" + bytes([30, 120, 200]) * size for _ in range(size))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _sample_docx() -> bytes:
    document = Document()
    document.add_heading("Product Manual", level=1)
    document.add_paragraph("This is body text.")
    document.add_heading("3.2 Dependencies", level=2)
    document.add_paragraph("Install dependencies before running.")
    document.add_picture(io.BytesIO(_png(100)))

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "A"
    table.cell(1, 1).text = "B"

    # 页眉不在 body 内，解析结果不应包含它
    document.sections[0].header.paragraphs[0].text = "HEADER SHOULD NOT APPEAR"

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_docx_parses_order_headings_image_and_table() -> None:
    parsed = DocxParser().parse(_sample_docx(), filename="manual.docx", mime="x")

    assert parsed.page_count is None
    assert [block.seq for block in parsed.blocks] == list(range(len(parsed.blocks)))
    assert all(block.page_no is None and block.bbox is None for block in parsed.blocks)

    texts = [block.text for block in parsed.blocks]
    assert "HEADER SHOULD NOT APPEAR" not in texts

    heading1 = next(block for block in parsed.blocks if block.text == "Product Manual")
    heading2 = next(block for block in parsed.blocks if block.text == "3.2 Dependencies")
    assert heading1.type is BlockType.HEADING
    assert heading1.heading_level == 1
    assert heading2.heading_level == 2

    images = [block for block in parsed.blocks if block.type is BlockType.IMAGE]
    assert len(images) == 1
    assert images[0].image_ref is not None
    assert images[0].image_ref.mime == "image/png"
    assert images[0].image_ref.data.startswith(b"\x89PNG")

    tables = [block for block in parsed.blocks if block.type is BlockType.TABLE]
    assert len(tables) == 1
    assert "Name | Value" in tables[0].text
    assert "A | B" in tables[0].text

    # 图片在第二段正文之后、表格之前
    image_index = parsed.blocks.index(images[0])
    table_index = parsed.blocks.index(tables[0])
    assert image_index < table_index
    assert any(
        block.text.startswith("Install dependencies") for block in parsed.blocks[:image_index]
    )

    for block in parsed.blocks:
        if block.text:
            assert block.char_end - block.char_start == len(block.text)
    text_blocks = [block for block in parsed.blocks if block.text]
    assert all(a.char_start <= b.char_start for a, b in pairwise(text_blocks))


def test_empty_docx_raises_parse_error() -> None:
    document = Document()
    buffer = io.BytesIO()
    document.save(buffer)
    with pytest.raises(ParseError, match="未解析出任何内容"):
        DocxParser().parse(buffer.getvalue(), filename="empty.docx")


def test_corrupt_docx_is_wrapped_as_parse_error() -> None:
    with pytest.raises(ParseError):
        DocxParser().parse(b"not a zip", filename="broken.docx")
