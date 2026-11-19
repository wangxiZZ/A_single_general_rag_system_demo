"""生成解析测试夹具。

夹具是确定性的（固定随机种子），生成后提交到 ``tests/fixtures/documents/``；
测试直接读取它们，避免每次运行依赖 Pillow/Word 等外部工具。

用法：

    uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

import io
import random
import struct
import zlib
from pathlib import Path

import pymupdf
from docx import Document

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "documents"


def _png(size: int, *, seed: int) -> bytes:
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


def make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Fixture Manual", fontsize=24)
    page.insert_text((72, 130), "This is body text for the fixture manual.", fontsize=10)
    page.insert_text((72, 170), "3.2 Dependencies", fontsize=16)
    page.insert_text((72, 200), "Install dependencies before running the service.", fontsize=10)
    page.insert_image(pymupdf.Rect(72, 240, 272, 440), stream=_png(200, seed=7))

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 80), "Appendix", fontsize=24)
    page2.insert_text((72, 130), "Additional fixture notes.", fontsize=10)

    doc.save(path)
    doc.close()


def make_docx(path: Path) -> None:
    document = Document()
    document.add_heading("Fixture Manual", level=1)
    document.add_paragraph("This is body text for the fixture manual.")
    document.add_heading("3.2 Dependencies", level=2)
    document.add_paragraph("Install dependencies before running the service.")
    document.add_picture(io.BytesIO(_png(80, seed=9)))

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "A"
    table.cell(1, 1).text = "B"

    document.sections[0].header.paragraphs[0].text = "HEADER SHOULD NOT APPEAR"
    document.save(path)


def make_scanned(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(72, 72, 240, 240), color=(1, 0, 0), fill=(1, 0, 0))
    doc.save(path)
    doc.close()


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    make_pdf(FIXTURE_DIR / "sample.pdf")
    make_docx(FIXTURE_DIR / "sample.docx")
    make_scanned(FIXTURE_DIR / "scanned.pdf")
    print(f"fixtures written to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
