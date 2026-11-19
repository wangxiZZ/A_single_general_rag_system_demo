"""图片块构造：VLM 描述 + 前后文上下文 + 落盘记录。

为什么单独放在 application：

1. 需要调用 VLM 端口（异步、可能失败），不能塞进纯函数分块器；
2. 图片描述失败时按 ADR-017 降级为占位文本 + ``meta.low_weight``，不阻塞文档 ready；
3. 图片必须落盘并生成 ``DocImage`` 记录，`chunk.image_id` 才能回到原图。

图片 chunk 的 seq 追加在文本 chunk 之后（当前不重排文本 chunk 的 id），但 ``char_start``
仍指向图片在原文中的位置，图谱/引用都按字符位置回溯。
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

from rag_kb.application.chunking import estimate_tokens, make_chunk_id
from rag_kb.domain.errors import LlmError
from rag_kb.domain.models import (
    Block,
    BlockType,
    Chunk,
    ChunkKind,
    ChunkLevel,
    DocImage,
)
from rag_kb.domain.ports import VLM, Storage

_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


def _neighbor_text(blocks: Sequence[Block], seq: int, *, before: bool, limit: int) -> str:
    candidates = (
        [block for block in blocks if block.seq < seq]
        if before
        else [block for block in blocks if block.seq > seq]
    )
    if not candidates:
        return ""
    chosen = (
        max(candidates, key=lambda block: block.seq)
        if before
        else min(candidates, key=lambda block: block.seq)
    )
    text = chosen.text.strip()
    return text[-limit:] if before else text[:limit]


def _parent_for(parents: Sequence[Chunk], char_start: int) -> Chunk | None:
    if not parents:
        return None
    preceding = [parent for parent in parents if parent.char_start <= char_start]
    if preceding:
        return max(preceding, key=lambda parent: parent.char_start)
    return min(parents, key=lambda parent: parent.char_start)


async def _describe(vlm: VLM | None, image: bytes, mime: str) -> tuple[str, str, bool]:
    if vlm is None or not image:
        return "未生成描述（VLM 未启用）", "placeholder", True
    try:
        caption = (await vlm.describe_image(image, mime=mime)).strip()
    except LlmError:
        return "未生成描述（VLM 调用失败）", "placeholder", True
    if not caption:
        return "未生成描述（VLM 返回空）", "placeholder", True
    return caption, "ok", False


async def build_image_chunks(
    blocks: Sequence[Block],
    chunks: Sequence[Chunk],
    *,
    kb_id: UUID,
    doc_id: UUID,
    doc_sha256: str,
    chunker_version: str,
    storage: Storage,
    vlm: VLM | None,
    chars_per_token: float = 1.5,
    context_chars: int = 200,
) -> tuple[list[Chunk], list[DocImage]]:
    text_blocks = [
        block
        for block in blocks
        if block.type in (BlockType.TEXT, BlockType.HEADING, BlockType.TABLE) and block.text.strip()
    ]
    image_blocks = sorted(
        (
            block
            for block in blocks
            if block.type is BlockType.IMAGE and block.image_ref is not None
        ),
        key=lambda block: block.seq,
    )
    if not image_blocks:
        return [], []

    parents = [chunk for chunk in chunks if chunk.level is ChunkLevel.PARENT]
    next_seq = max((chunk.seq for chunk in chunks), default=-1) + 1
    image_chunks: list[Chunk] = []
    doc_images: list[DocImage] = []

    for index, block in enumerate(image_blocks):
        ref = block.image_ref
        assert ref is not None
        caption, caption_status, low_weight = await _describe(vlm, ref.data, ref.mime)

        image_id = uuid4()
        ext = _EXT_BY_MIME.get(ref.mime.lower(), ".png")
        file_path = await storage.save_image(kb_id, doc_id, str(image_id), ref.data, ext=ext)
        doc_image = DocImage(
            id=image_id,
            kb_id=kb_id,
            doc_id=doc_id,
            seq=index,
            page_no=block.page_no,
            bbox=block.bbox,
            file_path=file_path,
            caption=caption,
        )
        doc_images.append(doc_image)

        previous = _neighbor_text(text_blocks, block.seq, before=True, limit=context_chars)
        following = _neighbor_text(text_blocks, block.seq, before=False, limit=context_chars)
        context = "\n".join(part for part in (previous, following) if part)
        content = f"【图片】{caption}"
        if context:
            content += f"\n【上下文】{context}"

        parent = _parent_for(parents, block.char_start)
        seq = next_seq + index
        image_chunks.append(
            Chunk(
                id=make_chunk_id(doc_sha256, seq, chunker_version),
                kb_id=kb_id,
                doc_id=doc_id,
                parent_id=parent.id if parent else None,
                level=ChunkLevel.CHILD,
                kind=ChunkKind.IMAGE,
                content=content,
                seq=seq,
                char_start=block.char_start,
                char_end=block.char_end,
                token_count=estimate_tokens(content, chars_per_token),
                chunker_version=chunker_version,
                heading_path=parent.heading_path if parent else (),
                page_no=block.page_no,
                page_to=block.page_no,
                bbox=block.bbox,
                image_id=image_id,
                meta={"caption_status": caption_status, "low_weight": low_weight},
            )
        )

    return image_chunks, doc_images
