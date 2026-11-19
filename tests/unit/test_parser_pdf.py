"""PDF 解析器单测：用 PyMuPDF 现场生成 fixture，不依赖二进制样例。"""

from __future__ import annotations

import random
import struct
import zlib
from itertools import pairwise

import pymupdf
import pytest

from rag_kb.domain.errors import NoExtractableTextError, ParseError
from rag_kb.domain.models import BlockType
from rag_kb.infra.parsers.pdf import PdfParser


def _png(size: int, *, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    raw = b"".join(b"\x00" + rng.randbytes(size * 3) for _ in range(size))

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
        + chunk(b"IDAT", zlib.compress(raw, level=0))
        + chunk(b"IEND", b"")
    )


def _sample_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Product Manual", fontsize=24)
    page.insert_text((72, 130), "This is body text for the manual.", fontsize=10)
    page.insert_text((72, 170), "3.2 Dependencies", fontsize=16)
    page.insert_text((72, 200), "Install the dependencies before running.", fontsize=10)
    page.insert_image(pymupdf.Rect(72, 240, 272, 440), stream=_png(200, seed=1))
    page.insert_image(pymupdf.Rect(72, 460, 82, 470), stream=_png(10, seed=2))

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 80), "Appendix", fontsize=24)
    page2.insert_text((72, 130), "Additional notes.", fontsize=10)

    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_parses_text_images_and_headings() -> None:
    parsed = PdfParser().parse(_sample_pdf(), filename="manual.pdf", mime="application/pdf")

    assert parsed.page_count == 2
    assert [block.seq for block in parsed.blocks] == list(range(len(parsed.blocks)))

    by_text = {block.text: block for block in parsed.blocks if block.text}
    assert by_text["Product Manual"].type is BlockType.HEADING
    assert by_text["Product Manual"].heading_level == 1
    assert by_text["3.2 Dependencies"].heading_level == 2
    assert by_text["Appendix"].page_no == 2

    images = [block for block in parsed.blocks if block.type is BlockType.IMAGE]
    # 10×10 装饰图被过滤，只保留 200×200。
    assert len(images) == 1
    image = images[0]
    assert image.image_ref is not None
    assert image.page_no == 1
    assert image.bbox is not None
    assert image.bbox.x1 - image.bbox.x0 >= 120
    assert len(image.image_ref.data) >= 8 * 1024

    for block in parsed.blocks:
        if block.text:
            assert block.char_end - block.char_start == len(block.text)

    text_blocks = [block for block in parsed.blocks if block.text]
    assert all(a.char_start <= b.char_start for a, b in pairwise(text_blocks))


def test_pdf_without_text_raises() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(72, 72, 200, 200), color=(1, 0, 0), fill=(1, 0, 0))
    data = doc.tobytes()
    doc.close()

    with pytest.raises(NoExtractableTextError):
        PdfParser().parse(data, filename="scan.pdf", mime="application/pdf")


def test_corrupt_pdf_is_wrapped_as_parse_error() -> None:
    with pytest.raises(ParseError):
        PdfParser().parse(b"not a pdf", filename="broken.pdf", mime="application/pdf")
