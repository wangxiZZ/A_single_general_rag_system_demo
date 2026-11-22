"""摄取管线单测：fake parser / storage / VLM / repository。"""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_kb.application.ingest_pipeline import IngestPipeline
from rag_kb.domain.errors import IngestAbortedError, ParseError
from rag_kb.domain.models import (
    KB,
    BBox,
    Block,
    BlockType,
    Document,
    DocumentStatus,
    ImageRef,
    IngestJob,
    ParsedDocument,
)
from tests.unit.fakes import FakeMetadataRepository, FakeStorage

KB_ID = uuid4()
DOC_ID = uuid4()
SHA = "c" * 64


class FakeParser:
    def __init__(self, parsed: ParsedDocument | Exception) -> None:
        self.parsed = parsed
        self.calls = 0

    def parse(self, data: bytes, *, filename: str, mime: str) -> ParsedDocument:
        self.calls += 1
        if isinstance(self.parsed, Exception):
            raise self.parsed
        return self.parsed


class FakeVLM:
    async def describe_image(
        self, image: bytes, *, mime: str = "image/png", prompt: str = ""
    ) -> str:
        return "一张架构图"


def _parsed() -> ParsedDocument:
    return ParsedDocument(
        blocks=(
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
                image_ref=ImageRef(id="img-0", data=b"png-bytes", mime="image/png"),
            ),
            Block(type=BlockType.TEXT, text="后置说明。", seq=3, char_start=11, char_end=16),
        ),
        page_count=1,
    )


async def _seed(repo: FakeMetadataRepository, storage: FakeStorage) -> tuple[str, str, IngestJob]:
    kb = await repo.create_kb(KB(id=uuid4(), name="kb"))
    file_path = f"uploads/{kb.id}/{DOC_ID}/a.pdf"
    storage.files[file_path] = b"%PDF-1.7"
    doc = await repo.create_document(
        Document(
            id=DOC_ID,
            kb_id=kb.id,
            filename="a.pdf",
            mime="application/pdf",
            sha256=SHA,
            size=8,
            file_path=file_path,
        )
    )
    job = await repo.enqueue_ingest_job(kb.id, doc.id)
    return str(kb.id), str(doc.id), job


def _pipeline(
    repo: FakeMetadataRepository,
    storage: FakeStorage,
    parser: FakeParser,
    *,
    vlm=None,
    indexer=None,
) -> IngestPipeline:
    return IngestPipeline(
        repo,
        storage,
        parser,
        vlm,
        chunker_version="v1",
        target_tokens=40,
        overlap_ratio=0.0,
        parent_max_tokens=200,
        chars_per_token=1.0,
        embed_model="qwen-test",
        indexer=indexer,
    )


async def test_pipeline_parses_chunks_images_and_marks_chunked() -> None:
    repo = FakeMetadataRepository()
    storage = FakeStorage()
    _, _, job = await _seed(repo, storage)
    parser = FakeParser(_parsed())

    result = await _pipeline(repo, storage, parser, vlm=FakeVLM()).process(job)

    assert result.stage == "chunked"
    assert parser.calls == 1
    assert {"parse_ms", "chunk_ms", "image_ms", "persist_ms"} <= set(result.timings)

    document = repo.documents[job.doc_id]
    assert document.status is DocumentStatus.CHUNKED
    assert document.page_count == 1
    assert document.chunker_version == "v1"
    assert document.embed_model == "qwen-test"
    assert document.chunk_count == len(repo.chunks[job.doc_id])

    chunks = repo.chunks[job.doc_id]
    assert any(chunk.kind.value == "image" for chunk in chunks)
    assert await repo.count_images(job.kb_id, job.doc_id) == 1
    assert storage.files


async def test_pipeline_is_idempotent() -> None:
    repo = FakeMetadataRepository()
    storage = FakeStorage()
    _, _, job = await _seed(repo, storage)
    pipeline = _pipeline(repo, storage, FakeParser(_parsed()), vlm=FakeVLM())

    await pipeline.process(job)
    first_count = len(repo.chunks[job.doc_id])
    await pipeline.process(job)
    assert len(repo.chunks[job.doc_id]) == first_count
    assert await repo.count_images(job.kb_id, job.doc_id) == 1


async def test_pipeline_aborts_when_document_deleted() -> None:
    repo = FakeMetadataRepository()
    storage = FakeStorage()
    _, _, job = await _seed(repo, storage)
    await repo.delete_document(job.kb_id, job.doc_id)

    with pytest.raises(IngestAbortedError):
        await _pipeline(repo, storage, FakeParser(_parsed())).process(job)


class FakeIndexer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    async def index_document(self, kb_id, doc_id):
        self.calls.append((kb_id, doc_id))
        return None


async def test_pipeline_hands_off_to_indexer_and_reports_ready() -> None:
    repo = FakeMetadataRepository()
    storage = FakeStorage()
    _, _, job = await _seed(repo, storage)
    indexer = FakeIndexer()

    result = await _pipeline(
        repo, storage, FakeParser(_parsed()), vlm=FakeVLM(), indexer=indexer
    ).process(job)

    assert result.stage == "ready"
    assert indexer.calls == [(job.kb_id, job.doc_id)]


async def test_pipeline_propagates_parse_error_without_dirty_index() -> None:
    repo = FakeMetadataRepository()
    storage = FakeStorage()
    _, _, job = await _seed(repo, storage)

    with pytest.raises(ParseError):
        await _pipeline(repo, storage, FakeParser(ParseError("broken"))).process(job)

    assert repo.chunks.get(job.doc_id, []) == []
    assert await repo.count_images(job.kb_id, job.doc_id) == 0


class ExplodingIndexer:
    """在落库之后失败，模拟"索引阶段挂掉"这条最容易留下脏数据的路径。"""

    def __init__(self) -> None:
        self.calls = 0

    async def index_document(self, kb_id, doc_id):
        self.calls += 1
        raise RuntimeError("milvus down")


async def test_failure_after_persist_leaves_no_residue() -> None:
    """回归：persist 之后的失败也必须清 chunk/图片行/图片文件，并把文档置 failed。

    历史缺陷：只有 NotFound 分支清文件，其它失败由 worker 只改状态，于是 failed
    文档仍能从 /chunks 读到内容、chunk_count 也不为 0。
    """
    repo = FakeMetadataRepository()
    storage = FakeStorage()
    _, _, job = await _seed(repo, storage)
    indexer = ExplodingIndexer()

    with pytest.raises(RuntimeError, match="milvus down"):
        await _pipeline(
            repo, storage, FakeParser(_parsed()), vlm=FakeVLM(), indexer=indexer
        ).process(job)

    assert indexer.calls == 1
    assert repo.chunks.get(job.doc_id, []) == []
    assert await repo.count_images(job.kb_id, job.doc_id) == 0
    assert not any(path.startswith("images/") for path in storage.files)
    document = repo.documents[job.doc_id]
    assert document.status is DocumentStatus.FAILED
    assert document.chunk_count == 0
    assert "milvus down" in (document.error or "")
    # 上传件必须保留，否则重试没东西可解析。
    assert any(path.startswith("uploads/") for path in storage.files)
