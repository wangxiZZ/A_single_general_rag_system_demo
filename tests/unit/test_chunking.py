"""分块算法单测：纯函数，不连数据库、不调模型。"""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_kb.application.chunking import chunk_document
from rag_kb.domain.models import BBox, Block, BlockType, ChunkKind, ChunkLevel

KB_ID = uuid4()
DOC_ID = uuid4()
SHA = "a" * 64


def _block(
    text: str,
    seq: int,
    start: int,
    *,
    type_: BlockType = BlockType.TEXT,
    level: int | None = None,
    page_no: int | None = 1,
    bbox: BBox | None = None,
) -> Block:
    return Block(
        type=type_,
        text=text,
        seq=seq,
        char_start=start,
        char_end=start + len(text),
        page_no=page_no,
        bbox=bbox or BBox(0, 0, 100, 10),
        heading_level=level,
    )


def _chunks(blocks: list[Block], **kwargs):
    params = {
        "target_tokens": 20,
        "overlap_ratio": 0.0,
        "parent_max_tokens": 100,
        "chars_per_token": 1.0,
    }
    params.update(kwargs)
    return chunk_document(
        blocks,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        doc_sha256=SHA,
        chunker_version="v1",
        **params,
    )


def test_heading_tree_and_child_prefix() -> None:
    blocks = [
        _block("Manual", 0, 0, type_=BlockType.HEADING, level=1),
        _block("Intro。", 1, 6),
        _block("3.2 Deps", 2, 12, type_=BlockType.HEADING, level=2),
        _block("Dependency text。", 3, 20),
    ]
    chunks = _chunks(blocks)
    children = [chunk for chunk in chunks if chunk.level is ChunkLevel.CHILD]

    intro = next(chunk for chunk in children if "Intro" in chunk.content)
    assert intro.heading_path == ("Manual",)
    assert intro.content.startswith("Manual\n")

    dependency = next(chunk for chunk in children if "Dependency" in chunk.content)
    assert dependency.heading_path == ("Manual", "3.2 Deps")
    assert dependency.content.startswith("Manual > 3.2 Deps\n")

    parent = next(chunk for chunk in chunks if chunk.id == dependency.parent_id)
    assert parent.level is ChunkLevel.PARENT
    assert parent.heading_path == dependency.heading_path


def test_parent_is_split_at_paragraph_boundary() -> None:
    blocks = [
        _block("A" * 10 + "。", 0, 0),
        _block("B" * 10 + "。", 1, 11),
    ]
    chunks = _chunks(blocks, target_tokens=10, parent_max_tokens=15)
    parents = [chunk for chunk in chunks if chunk.level is ChunkLevel.PARENT]
    children = [chunk for chunk in chunks if chunk.level is ChunkLevel.CHILD]

    assert len(parents) == 2
    assert all(parent.token_count <= 15 for parent in parents)
    assert {child.parent_id for child in children} == {parent.id for parent in parents}


def test_children_do_not_cut_sentences() -> None:
    blocks = [_block("第一句。第二句。第三句。", 0, 0)]
    chunks = _chunks(blocks, target_tokens=6)
    children = [chunk for chunk in chunks if chunk.level is ChunkLevel.CHILD]

    assert len(children) >= 3
    assert all(chunk.content.endswith("。") for chunk in children)


def test_overlap_reuses_trailing_sentences() -> None:
    blocks = [_block("AAAA。BBBB。CCCC。DDDD。", 0, 0)]
    chunks = _chunks(blocks, target_tokens=10, overlap_ratio=0.3)
    contents = [chunk.content for chunk in chunks if chunk.level is ChunkLevel.CHILD]

    assert len(contents) >= 2
    assert "BBBB。" in contents[0]
    assert "BBBB。" in contents[1]


def test_long_sentence_falls_back_to_hard_split() -> None:
    blocks = [_block("X" * 50, 0, 0)]
    chunks = _chunks(blocks, target_tokens=10)
    children = [chunk for chunk in chunks if chunk.level is ChunkLevel.CHILD]

    assert len(children) == 5
    assert all(len(chunk.content) <= 10 for chunk in children)


def test_chunk_ids_include_kb_and_doc_identity() -> None:
    blocks = [_block("同一份内容。", 0, 0)]
    first = chunk_document(
        blocks,
        kb_id=KB_ID,
        doc_id=uuid4(),
        doc_sha256=SHA,
        chunker_version="v1",
        target_tokens=20,
        overlap_ratio=0.0,
        parent_max_tokens=100,
        chars_per_token=1.0,
    )
    second = chunk_document(
        blocks,
        kb_id=KB_ID,
        doc_id=uuid4(),
        doc_sha256=SHA,
        chunker_version="v1",
        target_tokens=20,
        overlap_ratio=0.0,
        parent_max_tokens=100,
        chars_per_token=1.0,
    )
    assert {chunk.id for chunk in first}.isdisjoint({chunk.id for chunk in second})


def test_chunk_ids_are_deterministic() -> None:
    blocks = [_block("一段文本。", 0, 0)]
    first = _chunks(blocks)
    second = _chunks(blocks)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == len(first)
    assert [chunk.seq for chunk in first] == list(range(len(first)))


def test_no_heading_uses_root_section() -> None:
    blocks = [_block("Root text。", 0, 0)]
    chunks = _chunks(blocks)
    assert all(chunk.heading_path == () for chunk in chunks)


def test_table_blocks_keep_table_kind() -> None:
    blocks = [_block("A | B\nC | D", 0, 0, type_=BlockType.TABLE)]
    chunks = _chunks(blocks)
    assert all(chunk.kind is ChunkKind.TABLE for chunk in chunks)


def test_child_offsets_stay_within_source() -> None:
    blocks = [
        _block("Intro。", 0, 0),
        _block("Body text。", 1, 6),
    ]
    chunks = _chunks(blocks)
    children = [chunk for chunk in chunks if chunk.level is ChunkLevel.CHILD]
    total = sum(len(block.text) for block in blocks)
    for chunk in sorted(children, key=lambda item: item.char_start):
        assert 0 <= chunk.char_start <= chunk.char_end <= total


def test_invalid_parameters_rejected() -> None:
    blocks = [_block("x", 0, 0)]
    with pytest.raises(ValueError):
        _chunks(blocks, target_tokens=10, parent_max_tokens=5)
    with pytest.raises(ValueError):
        _chunks(blocks, overlap_ratio=1.0)
