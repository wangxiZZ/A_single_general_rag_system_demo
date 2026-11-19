"""docx 解析器（python-docx + lxml）。

docx 没有页码与坐标，因此：

- 所有 Block 的 ``page_no/bbox`` 都为 ``None``；定位靠 ``seq`` 与 ``char_start/char_end``；
- 必须按 **XML body 顺序** 遍历 ``w:p`` / ``w:tbl``，不能依赖 ``inline_shapes``（顺序不可靠）；
- 图片通过 ``a:blip`` 的 ``r:embed`` 找 ``related_parts``，作为独立 Block 插在文本中间；
- 页眉/页脚不在 ``document.element.body`` 内，天然不会被解析（已知限制，题目不要求）。

表格线性化为 ``列1 | 列2`` 的文本行；表格内嵌图片与合并单元格不做还原。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

from docx import Document as DocxDocument
from docx.text.paragraph import Paragraph

from rag_kb.domain.errors import ParseError
from rag_kb.domain.models import Block, BlockType, ImageRef, ParsedDocument
from rag_kb.infra.parsers.base import BaseParser

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_HEADING_STYLE_RE = re.compile(r"^(?:Heading|标题)\s*(\d+)$", re.IGNORECASE)


@dataclass(slots=True)
class _Cursor:
    seq: int = 0
    char: int = 0


class DocxParser(BaseParser):
    extensions = frozenset({".docx"})

    def _parse(self, data: bytes, *, filename: str, mime: str) -> ParsedDocument:
        try:
            document = DocxDocument(io.BytesIO(data))
        except Exception as exc:
            raise ParseError(f"无法打开 docx: {filename}: {exc}") from exc

        blocks: list[Block] = []
        cursor = _Cursor()
        paragraphs = {paragraph._p: paragraph for paragraph in document.paragraphs}

        for child in document.element.body.iterchildren():
            tag = child.tag
            if tag == f"{_W}p":
                paragraph = paragraphs.get(child) or Paragraph(child, document)
                self._handle_paragraph(paragraph, blocks, cursor)
            elif tag == f"{_W}tbl":
                text = self._table_text(child)
                if text:
                    self._append_text(blocks, cursor, text, BlockType.TABLE, heading_level=None)

        if not blocks:
            raise ParseError(f"docx 未解析出任何内容: {filename}")
        return ParsedDocument(blocks=tuple(blocks), page_count=None)

    def _handle_paragraph(self, paragraph: Paragraph, blocks: list[Block], cursor: _Cursor) -> None:
        level = self._heading_level(paragraph)
        buffer: list[str] = []

        def flush() -> None:
            text = "".join(buffer).strip()
            buffer.clear()
            if text:
                self._append_text(
                    blocks,
                    cursor,
                    text,
                    BlockType.HEADING if level else BlockType.TEXT,
                    heading_level=level,
                )

        for element in paragraph._p.iter():
            tag = element.tag
            if tag == f"{_W}t":
                buffer.append(element.text or "")
            elif tag == f"{_W}tab":
                buffer.append("\t")
            elif tag in (f"{_W}br", f"{_W}cr"):
                buffer.append("\n")
            elif tag == f"{_A}blip":
                rid = element.get(f"{_R}embed")
                if not rid:
                    continue
                try:
                    image_part = paragraph.part.related_parts[rid]
                except KeyError:
                    continue
                flush()
                blocks.append(
                    Block(
                        type=BlockType.IMAGE,
                        text="",
                        seq=cursor.seq,
                        char_start=cursor.char,
                        char_end=cursor.char,
                        page_no=None,
                        bbox=None,
                        image_ref=ImageRef(
                            id=f"docx-img-{cursor.seq}",
                            data=image_part.blob,
                            mime=image_part.content_type,
                        ),
                    )
                )
                cursor.seq += 1
        flush()

    def _append_text(
        self,
        blocks: list[Block],
        cursor: _Cursor,
        text: str,
        block_type: BlockType,
        *,
        heading_level: int | None,
    ) -> None:
        blocks.append(
            Block(
                type=block_type,
                text=text,
                seq=cursor.seq,
                char_start=cursor.char,
                char_end=cursor.char + len(text),
                page_no=None,
                bbox=None,
                heading_level=heading_level,
            )
        )
        cursor.char += len(text)
        cursor.seq += 1

    def _heading_level(self, paragraph: Paragraph) -> int | None:
        style = paragraph.style
        style_name = style.name if style is not None else ""
        if match := _HEADING_STYLE_RE.match(style_name):
            return min(int(match.group(1)), 6)
        if style_name.lower() == "title":
            return 1

        p_pr = paragraph._p.pPr
        if p_pr is not None and p_pr.outlineLvl is not None:
            value = p_pr.outlineLvl.val
            if value is not None:
                return min(int(value) + 1, 6)
        return None

    def _table_text(self, table_element: Any) -> str:
        rows: list[str] = []
        for row in table_element.findall(f"{_W}tr"):
            cells: list[str] = []
            for cell in row.findall(f"{_W}tc"):
                parts: list[str] = []
                for paragraph in cell.findall(f".//{_W}p"):
                    text = "".join(t.text or "" for t in paragraph.iter(f"{_W}t")).strip()
                    if text:
                        parts.append(text)
                cells.append(" ".join(parts))
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows).strip()
