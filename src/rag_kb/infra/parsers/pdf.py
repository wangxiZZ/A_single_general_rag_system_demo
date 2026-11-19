"""PDF 解析器（PyMuPDF）。

关键取舍：

1. **阅读顺序**按页面内 ``(y0, x0)`` 排序。真正的多栏排版需要版面分析，这里显式
   接受"单栏优先"的已知限制，而不是假装能处理所有版式。
2. **图片与文本共用同一个有序 Block 序列**：图片按 bbox 插入，后续分块才能知道它前后
   是哪些段落。
3. **装饰图过滤**：``<120×120 px`` 或 ``<8 KB`` 的图片直接丢弃，避免图标污染向量空间。
4. **无文本 PDF 直接失败**：题目不要求 OCR，产出空 chunk 比失败更难排查。
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any

import pymupdf

from rag_kb.domain.errors import NoExtractableTextError, ParseError
from rag_kb.domain.models import (
    BBox,
    Block,
    BlockType,
    ImageRef,
    ParsedDocument,
)
from rag_kb.infra.parsers.base import BaseParser

MIN_IMAGE_SIDE = 120
MIN_IMAGE_BYTES = 8 * 1024
_HEADING_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})[\s、.．]")

_MIME_BY_EXT = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


@dataclass(slots=True)
class _Item:
    bbox: BBox
    kind: str  # text / image
    payload: Any


def _to_bbox(rect: Any) -> BBox:
    return BBox(x0=float(rect.x0), y0=float(rect.y0), x1=float(rect.x1), y1=float(rect.y1))


def _heading_level(text: str, size: float, body_size: float) -> int | None:
    match = _HEADING_NUMBER_RE.match(text)
    if match and len(text) <= 160:
        return min(match.group(1).count(".") + 1, 6)
    if not body_size:
        return None
    if size >= body_size * 1.5:
        return 1
    if size >= body_size * 1.3:
        return 2
    if size >= body_size * 1.15:
        return 3
    return None


class PdfParser(BaseParser):
    extensions = frozenset({".pdf"})

    def _parse(self, data: bytes, *, filename: str, mime: str) -> ParsedDocument:
        try:
            document = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise ParseError(f"无法打开 PDF: {filename}: {exc}") from exc

        try:
            if document.needs_pass:
                raise ParseError(f"PDF 已加密，无法解析: {filename}")

            body_size = self._body_font_size(document)
            blocks: list[Block] = []
            seq = 0
            cursor = 0
            text_found = False

            for page_index, page in enumerate(document):
                page_no = page_index + 1
                items = self._collect_page_items(page, page_no)
                if not items:
                    continue

                for item in sorted(items, key=lambda x: (round(x.bbox.y0, 1), round(x.bbox.x0, 1))):
                    if item.kind == "text":
                        text, size = item.payload
                        if not text.strip():
                            continue
                        text_found = True
                        level = _heading_level(text, size, body_size)
                        blocks.append(
                            Block(
                                type=BlockType.HEADING if level else BlockType.TEXT,
                                text=text,
                                seq=seq,
                                char_start=cursor,
                                char_end=cursor + len(text),
                                page_no=page_no,
                                bbox=item.bbox,
                                heading_level=level,
                            )
                        )
                        cursor += len(text)
                        seq += 1
                    else:
                        image_ref = item.payload
                        blocks.append(
                            Block(
                                type=BlockType.IMAGE,
                                text="",
                                seq=seq,
                                char_start=cursor,
                                char_end=cursor,
                                page_no=page_no,
                                bbox=item.bbox,
                                image_ref=image_ref,
                            )
                        )
                        seq += 1

            if not text_found:
                raise NoExtractableTextError(filename)

            return ParsedDocument(blocks=tuple(blocks), page_count=document.page_count)
        finally:
            document.close()

    def _body_font_size(self, document: pymupdf.Document) -> float:
        sizes: list[float] = []
        for page in document:
            for block in page.get_text("dict", sort=True).get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.append(float(span.get("size", 0)))
        return statistics.median(sizes) if sizes else 0.0

    def _collect_page_items(self, page: pymupdf.Page, page_no: int) -> list[_Item]:
        items: list[_Item] = []

        for block in page.get_text("dict", sort=True).get("blocks", []):
            if block.get("type") != 0:
                continue
            lines: list[str] = []
            max_size = 0.0
            for line in block.get("lines", []):
                spans = [span.get("text", "") for span in line.get("spans", [])]
                line_text = "".join(spans)
                if line_text.strip():
                    lines.append(line_text)
                for span in line.get("spans", []):
                    max_size = max(max_size, float(span.get("size", 0)))
            if not lines:
                continue
            items.append(
                _Item(
                    bbox=_to_bbox(pymupdf.Rect(block["bbox"])),
                    kind="text",
                    payload=("\n".join(lines), max_size),
                )
            )

        for index, image in enumerate(page.get_images(full=True)):
            xref = image[0]
            try:
                info = page.parent.extract_image(xref)
            except Exception:
                continue
            image_bytes = info.get("image", b"")
            if len(image_bytes) < MIN_IMAGE_BYTES:
                continue

            ext = str(info.get("ext", "png")).lower()
            mime_type = _MIME_BY_EXT.get(ext, "application/octet-stream")
            for rect_index, rect in enumerate(page.get_image_rects(xref)):
                if rect.width < MIN_IMAGE_SIDE or rect.height < MIN_IMAGE_SIDE:
                    continue
                items.append(
                    _Item(
                        bbox=_to_bbox(rect),
                        kind="image",
                        payload=ImageRef(
                            id=f"p{page_no}-x{xref}-{index}-{rect_index}",
                            data=image_bytes,
                            mime=mime_type,
                        ),
                    )
                )
        return items
