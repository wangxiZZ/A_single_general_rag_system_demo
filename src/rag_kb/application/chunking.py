"""分块策略：标题树 + 父子双层。

目标与取舍（ADR-013/018）：

- **父块**按标题分节，整节超过 ``parent_max_tokens`` 时按段落/句子二次切分，保证单块
  不超上限；父块不进入向量索引，只用于回答回填。
- **子块**在父块内按句子打包，目标 ``chunk_target_tokens``、重叠 ``overlap_ratio``；
  不切断句子，也不跨父块——跨父块的重叠会让证据归属变得模糊。
- **char_start/char_end** 映射回原文串联文本；子块内容会加标题前缀，但偏移只指向正文。
- **chunk_id** 由 ``sha256(kb_id:doc_id:seq:chunker_version)`` 确定性生成，保证重跑幂等，
  同时避免同一文件上传到多个知识库时撞主键。

图片块不在这里处理：它需要 VLM 描述与图片落盘（M2-05），由后续步骤基于同一 Block
序列构造，避免这里引入模型调用。
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from uuid import UUID

from rag_kb.domain.models import (
    BBox,
    Block,
    BlockType,
    Chunk,
    ChunkKind,
    ChunkLevel,
)

_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?|\n+")


@dataclass(frozen=True, slots=True)
class _Para:
    text: str
    char_start: int
    char_end: int
    page_no: int | None
    bbox: BBox | None
    kind: ChunkKind


@dataclass(frozen=True, slots=True)
class _Sentence:
    text: str
    para_index: int
    offset: int


def estimate_tokens(text: str, chars_per_token: float) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))


def make_chunk_id(kb_id: UUID, doc_id: UUID, seq: int, chunker_version: str) -> str:
    """确定性 chunk id。

    必须包含 kb_id/doc_id：只拿 doc_sha256 会让"同一文件上传到两个知识库"产生相同
    chunk 主键，第二个文档直接撞唯一约束（实测踩过）。
    """
    raw = f"{kb_id}:{doc_id}:{seq}:{chunker_version}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """按中文/英文句末标点与换行切分，返回 ``(句子, 段内偏移)``；偏移保持原文映射。"""
    sentences: list[tuple[str, int]] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group(0)
        if not raw.strip():
            continue
        stripped = raw.lstrip()
        offset = match.start() + (len(raw) - len(stripped))
        if stripped:
            sentences.append((stripped, offset))
    return sentences


def _hard_pieces(text: str, offset: int, max_chars: int) -> list[tuple[str, int]]:
    """句子本身超过目标长度时的兜底：按字符硬切，至少保证进度。"""
    if max_chars < 1:
        max_chars = 1
    return [(text[i : i + max_chars], offset + i) for i in range(0, len(text), max_chars)]


def _union_bbox(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return BBox(
        x0=min(box.x0 for box in boxes),
        y0=min(box.y0 for box in boxes),
        x1=max(box.x1 for box in boxes),
        y1=max(box.y1 for box in boxes),
    )


def _span(paras: list[_Para]) -> tuple[int | None, int | None, BBox | None]:
    pages = [para.page_no for para in paras if para.page_no is not None]
    page_no = min(pages) if pages else None
    page_to = max(pages) if pages else None
    boxes = [para.bbox for para in paras if para.bbox is not None]
    bbox = _union_bbox(boxes) if boxes and len(boxes) == len(paras) and page_no == page_to else None
    return page_no, page_to, bbox


def _kind(paras: list[_Para]) -> ChunkKind:
    return (
        ChunkKind.TABLE if all(para.kind is ChunkKind.TABLE for para in paras) else ChunkKind.TEXT
    )


def _para_from_block(block: Block) -> _Para | None:
    text = block.text.strip()
    if not text:
        return None
    leading = len(block.text) - len(block.text.lstrip())
    char_start = block.char_start + leading
    return _Para(
        text=text,
        char_start=char_start,
        char_end=char_start + len(text),
        page_no=block.page_no,
        bbox=block.bbox,
        kind=ChunkKind.TABLE if block.type is BlockType.TABLE else ChunkKind.TEXT,
    )


def _build_sections(blocks: list[Block]) -> list[tuple[tuple[str, ...], list[_Para]]]:
    sections: list[tuple[tuple[str, ...], list[_Para]]] = []
    stack: list[tuple[int, str]] = []
    current: list[_Para] = []

    def flush() -> None:
        if current:
            sections.append((tuple(text for _, text in stack), list(current)))
            current.clear()

    for block in sorted(blocks, key=lambda item: item.seq):
        if block.type is BlockType.HEADING:
            flush()
            level = block.heading_level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            heading_text = block.text.strip()
            if heading_text:
                stack.append((level, heading_text))
            continue
        if block.type is BlockType.IMAGE:
            continue
        para = _para_from_block(block)
        if para is not None:
            current.append(para)

    flush()
    return sections


def _split_long_para(para: _Para, max_tokens: int, chars_per_token: float) -> list[_Para]:
    if estimate_tokens(para.text, chars_per_token) <= max_tokens:
        return [para]

    pieces: list[_Para] = []
    for sentence, offset in _split_sentences(para.text):
        if estimate_tokens(sentence, chars_per_token) > max_tokens:
            for text, piece_offset in _hard_pieces(
                sentence, offset, max(1, int(max_tokens * chars_per_token))
            ):
                pieces.append(
                    _Para(
                        text=text,
                        char_start=para.char_start + piece_offset,
                        char_end=para.char_start + piece_offset + len(text),
                        page_no=para.page_no,
                        bbox=para.bbox,
                        kind=para.kind,
                    )
                )
        else:
            pieces.append(
                _Para(
                    text=sentence,
                    char_start=para.char_start + offset,
                    char_end=para.char_start + offset + len(sentence),
                    page_no=para.page_no,
                    bbox=para.bbox,
                    kind=para.kind,
                )
            )
    return pieces


def _split_paragraphs(
    paras: list[_Para], max_tokens: int, chars_per_token: float
) -> list[list[_Para]]:
    groups: list[list[_Para]] = []
    current: list[_Para] = []
    current_tokens = 0

    for para in paras:
        for part in _split_long_para(para, max_tokens, chars_per_token):
            tokens = estimate_tokens(part.text, chars_per_token)
            if current and current_tokens + tokens > max_tokens:
                groups.append(current)
                current = []
                current_tokens = 0
            current.append(part)
            current_tokens += tokens
    if current:
        groups.append(current)
    return groups


def _child_sentences(
    paras: list[_Para], target_tokens: int, chars_per_token: float
) -> list[_Sentence]:
    sentences: list[_Sentence] = []
    for index, para in enumerate(paras):
        for sentence, offset in _split_sentences(para.text):
            if estimate_tokens(sentence, chars_per_token) > target_tokens:
                for text, piece_offset in _hard_pieces(
                    sentence, offset, max(1, int(target_tokens * chars_per_token))
                ):
                    sentences.append(_Sentence(text, index, piece_offset))
            else:
                sentences.append(_Sentence(sentence, index, offset))
    return sentences


def _next_start(
    sentences: list[_Sentence], start: int, end: int, overlap_tokens: int, cpt: float
) -> int:
    if overlap_tokens <= 0:
        return end
    accumulated = 0
    index = end - 1
    candidate = end
    while index >= start and accumulated < overlap_tokens:
        accumulated += estimate_tokens(sentences[index].text, cpt)
        candidate = index
        index -= 1
    return candidate if candidate > start else end


def chunk_document(
    blocks: list[Block],
    *,
    kb_id: UUID,
    doc_id: UUID,
    doc_sha256: str,
    chunker_version: str,
    target_tokens: int = 320,
    overlap_ratio: float = 0.15,
    parent_max_tokens: int = 1200,
    chars_per_token: float = 1.5,
) -> list[Chunk]:
    if target_tokens < 1 or parent_max_tokens < target_tokens:
        raise ValueError("要求 1 <= target_tokens <= parent_max_tokens")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio 必须在 [0, 1) 内")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token 必须大于 0")

    chunks: list[Chunk] = []
    seq = 0
    overlap_tokens = int(target_tokens * overlap_ratio)

    for heading_path, paras in _build_sections(blocks):
        for parent_group in _split_paragraphs(paras, parent_max_tokens, chars_per_token):
            page_no, page_to, bbox = _span(parent_group)
            kind = _kind(parent_group)
            parent_content = "\n".join(para.text for para in parent_group)
            parent_id = make_chunk_id(kb_id, doc_id, seq, chunker_version)
            parent = Chunk(
                id=parent_id,
                kb_id=kb_id,
                doc_id=doc_id,
                level=ChunkLevel.PARENT,
                kind=kind,
                content=parent_content,
                seq=seq,
                char_start=parent_group[0].char_start,
                char_end=parent_group[-1].char_end,
                token_count=estimate_tokens(parent_content, chars_per_token),
                chunker_version=chunker_version,
                heading_path=heading_path,
                page_no=page_no,
                page_to=page_to,
                bbox=bbox,
            )
            chunks.append(parent)
            seq += 1

            sentences = _child_sentences(parent_group, target_tokens, chars_per_token)
            prefix = f"{' > '.join(heading_path)}\n" if heading_path else ""
            start = 0
            while start < len(sentences):
                end = start
                tokens = 0
                while end < len(sentences):
                    sentence_tokens = estimate_tokens(sentences[end].text, chars_per_token)
                    if end > start and tokens + sentence_tokens > target_tokens:
                        break
                    tokens += sentence_tokens
                    end += 1
                if end == start:  # 理论上不会发生，防御性保证进度
                    end = start + 1

                selected = sentences[start:end]
                content = prefix + "".join(sentence.text for sentence in selected)
                first = selected[0]
                last = selected[-1]
                used_paras = parent_group[first.para_index : last.para_index + 1]
                child_page, child_page_to, child_bbox = _span(used_paras)
                chunks.append(
                    Chunk(
                        id=make_chunk_id(kb_id, doc_id, seq, chunker_version),
                        kb_id=kb_id,
                        doc_id=doc_id,
                        parent_id=parent_id,
                        level=ChunkLevel.CHILD,
                        kind=_kind(used_paras),
                        content=content,
                        seq=seq,
                        char_start=parent_group[first.para_index].char_start + first.offset,
                        char_end=parent_group[last.para_index].char_start
                        + last.offset
                        + len(last.text),
                        token_count=estimate_tokens(content, chars_per_token),
                        chunker_version=chunker_version,
                        heading_path=heading_path,
                        page_no=child_page,
                        page_to=child_page_to,
                        bbox=child_bbox,
                    )
                )
                seq += 1

                if end >= len(sentences):
                    break
                start = _next_start(sentences, start, end, overlap_tokens, chars_per_token)

    return chunks
