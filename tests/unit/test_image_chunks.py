"""图片块构造单测：fake VLM + fake storage。"""

from __future__ import annotations

from uuid import uuid4

from rag_kb.application.chunking import chunk_document
from rag_kb.application.image_chunks import build_image_chunks
from rag_kb.domain.errors import LlmError
from rag_kb.domain.models import (
    BBox,
    Block,
    BlockType,
    ChunkKind,
    ChunkLevel,
    ImageRef,
)
from tests.unit.fakes import FakeStorage

KB_ID = uuid4()
DOC_ID = uuid4()
SHA = "b" * 64


class RecordingVLM:
    def __init__(self, caption: str = "一张架构图", error: Exception | None = None) -> None:
        self.caption = caption
        self.error = error
        self.calls: list[tuple[int, str]] = []

    async def describe_image(
        self, image: bytes, *, mime: str = "image/png", prompt: str = ""
    ) -> str:
        self.calls.append((len(image), mime))
        if self.error is not None:
            raise self.error
        return self.caption


def _blocks() -> list[Block]:
    return [
        Block(
            type=BlockType.HEADING,
            text="Manual",
            seq=0,
            char_start=0,
            char_end=6,
            heading_level=1,
        ),
        Block(type=BlockType.TEXT, text="前置说明。", seq=1, char_start=6, char_end=11),
        Block(
            type=BlockType.IMAGE,
            text="",
            seq=2,
            char_start=11,
            char_end=11,
            page_no=1,
            bbox=BBox(10, 20, 210, 220),
            image_ref=ImageRef(id="img-0", data=b"\x89PNG-data", mime="image/png"),
        ),
        Block(type=BlockType.TEXT, text="后置说明。", seq=3, char_start=11, char_end=16),
    ]


def _text_chunks(blocks: list[Block]):
    return chunk_document(
        blocks,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        doc_sha256=SHA,
        chunker_version="v1",
        target_tokens=40,
        parent_max_tokens=200,
        chars_per_token=1.0,
    )


async def test_image_chunk_uses_caption_and_context() -> None:
    blocks = _blocks()
    text_chunks = _text_chunks(blocks)
    storage = FakeStorage()
    vlm = RecordingVLM()

    image_chunks, doc_images = await build_image_chunks(
        blocks,
        text_chunks,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        doc_sha256=SHA,
        chunker_version="v1",
        storage=storage,
        vlm=vlm,
        chars_per_token=1.0,
    )

    assert len(image_chunks) == 1
    chunk = image_chunks[0]
    image = doc_images[0]

    assert vlm.calls == [(len(b"\x89PNG-data"), "image/png")]
    assert image.caption == "一张架构图"
    assert image.page_no == 1
    assert image.bbox == BBox(10, 20, 210, 220)
    assert storage.files[image.file_path] == b"\x89PNG-data"

    parent = next(c for c in text_chunks if c.level is ChunkLevel.PARENT)
    assert chunk.kind is ChunkKind.IMAGE
    assert chunk.level is ChunkLevel.CHILD
    assert chunk.image_id == image.id
    assert chunk.parent_id == parent.id
    assert chunk.heading_path == ("Manual",)
    assert chunk.seq == max(c.seq for c in text_chunks) + 1
    assert chunk.meta == {"caption_status": "ok", "low_weight": False}
    assert "【图片】一张架构图" in chunk.content
    assert "前置说明" in chunk.content
    assert "后置说明" in chunk.content


async def test_vlm_failure_degrades_to_placeholder() -> None:
    blocks = _blocks()
    text_chunks = _text_chunks(blocks)
    storage = FakeStorage()

    image_chunks, doc_images = await build_image_chunks(
        blocks,
        text_chunks,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        doc_sha256=SHA,
        chunker_version="v1",
        storage=storage,
        vlm=RecordingVLM(error=LlmError("boom")),
        chars_per_token=1.0,
    )

    assert "未生成描述" in image_chunks[0].content
    assert doc_images[0].caption is not None
    assert image_chunks[0].meta["low_weight"] is True
    assert image_chunks[0].meta["caption_status"] == "placeholder"


async def test_vlm_disabled_degrades_to_placeholder() -> None:
    blocks = _blocks()
    text_chunks = _text_chunks(blocks)

    image_chunks, _ = await build_image_chunks(
        blocks,
        text_chunks,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        doc_sha256=SHA,
        chunker_version="v1",
        storage=FakeStorage(),
        vlm=None,
        chars_per_token=1.0,
    )
    assert "VLM 未启用" in image_chunks[0].content
    assert image_chunks[0].meta["low_weight"] is True


async def test_image_without_text_chunks_has_no_parent() -> None:
    blocks = [
        Block(
            type=BlockType.IMAGE,
            text="",
            seq=0,
            char_start=0,
            char_end=0,
            image_ref=ImageRef(id="img-0", data=b"x", mime="image/png"),
        )
    ]
    image_chunks, _ = await build_image_chunks(
        blocks,
        [],
        kb_id=KB_ID,
        doc_id=DOC_ID,
        doc_sha256=SHA,
        chunker_version="v1",
        storage=FakeStorage(),
        vlm=RecordingVLM(),
        chars_per_token=1.0,
    )
    assert image_chunks[0].parent_id is None
    assert image_chunks[0].heading_path == ()
