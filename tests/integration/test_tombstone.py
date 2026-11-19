"""Tombstone 集成测试：真实 PG + 本地存储，验证删除后无残留。"""

from __future__ import annotations

import contextlib
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest

from rag_kb.application.ingest_pipeline import IngestPipeline
from rag_kb.config import get_settings
from rag_kb.domain.errors import IngestAbortedError
from rag_kb.domain.models import (
    KB,
    BBox,
    Block,
    BlockType,
    Document,
    ImageRef,
    IngestJob,
    ParsedDocument,
)
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.infra.postgres.repository import PostgresMetadataRepository
from rag_kb.infra.storage.local import LocalStorage

pytestmark = pytest.mark.integration


class FakeParser:
    def __init__(self, parsed: ParsedDocument) -> None:
        self.parsed = parsed

    def parse(self, data: bytes, *, filename: str, mime: str) -> ParsedDocument:
        return self.parsed


class DeletingVLM:
    def __init__(self, delete) -> None:
        self._delete = delete

    async def describe_image(
        self, image: bytes, *, mime: str = "image/png", prompt: str = ""
    ) -> str:
        await self._delete()
        return "一张图"


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
            Block(type=BlockType.TEXT, text="正文。", seq=1, char_start=6, char_end=9),
            Block(
                type=BlockType.IMAGE,
                text="",
                seq=2,
                char_start=9,
                char_end=9,
                page_no=1,
                bbox=BBox(10, 20, 210, 220),
                image_ref=ImageRef(id="img", data=b"png-bytes", mime="image/png"),
            ),
        ),
        page_count=1,
    )


async def _seed(
    repo: PostgresMetadataRepository, storage: LocalStorage
) -> tuple[KB, Document, IngestJob]:
    kb = await repo.create_kb(KB(id=uuid4(), name=f"tombstone-{uuid4().hex[:6]}"))
    doc_id = uuid4()
    file_path = await storage.save_upload(kb.id, doc_id, "a.pdf", b"%PDF-1.7")
    document = await repo.create_document(
        Document(
            id=doc_id,
            kb_id=kb.id,
            filename="a.pdf",
            mime="application/pdf",
            sha256=uuid4().hex * 2,
            size=8,
            file_path=file_path,
        )
    )
    job = await repo.enqueue_ingest_job(kb.id, document.id)
    return kb, document, job


def _pipeline(
    repo: PostgresMetadataRepository,
    storage: LocalStorage,
    vlm=None,
) -> IngestPipeline:
    return IngestPipeline(
        repo,
        storage,
        FakeParser(_parsed()),
        vlm,
        chunker_version="v1",
        target_tokens=40,
        overlap_ratio=0.0,
        parent_max_tokens=200,
        chars_per_token=1.0,
    )


def _db_counts(settings, kb_id: UUID, doc_id: UUID) -> tuple[int, int, int]:
    with psycopg.connect(settings.pg_dsn) as conn:
        chunks = conn.execute(
            "SELECT count(*) FROM chunk WHERE kb_id=%s AND doc_id=%s", (kb_id, doc_id)
        ).fetchone()[0]
        images = conn.execute(
            "SELECT count(*) FROM doc_image WHERE kb_id=%s AND doc_id=%s", (kb_id, doc_id)
        ).fetchone()[0]
        jobs = conn.execute(
            "SELECT count(*) FROM ingest_job WHERE doc_id=%s", (doc_id,)
        ).fetchone()[0]
    return chunks, images, jobs


async def test_document_deleted_before_processing_leaves_no_residue(tmp_path: Path) -> None:
    settings = get_settings()
    apply_migrations(settings.pg_dsn)
    repo = PostgresMetadataRepository(settings.pg_dsn, min_size=1, max_size=2)
    await repo.open()
    storage = LocalStorage(tmp_path)
    kb = None
    try:
        kb, document, job = await _seed(repo, storage)
        await repo.delete_document(kb.id, document.id)
        await storage.delete_document(kb.id, document.id)

        with pytest.raises(IngestAbortedError):
            await _pipeline(repo, storage).process(job)

        assert await repo.get_document(kb.id, document.id) is None
        assert _db_counts(settings, kb.id, document.id) == (0, 0, 0)
        assert not (tmp_path / "uploads" / str(kb.id) / str(document.id)).exists()
    finally:
        if kb is not None:
            with contextlib.suppress(Exception):
                await repo.delete_kb(kb.id)
        await repo.close()


async def test_delete_during_image_description_cleans_recreated_files(tmp_path: Path) -> None:
    settings = get_settings()
    apply_migrations(settings.pg_dsn)
    repo = PostgresMetadataRepository(settings.pg_dsn, min_size=1, max_size=2)
    await repo.open()
    storage = LocalStorage(tmp_path)
    kb = None
    try:
        kb, document, job = await _seed(repo, storage)

        async def delete() -> None:
            await repo.delete_document(kb.id, document.id)
            await storage.delete_document(kb.id, document.id)

        with pytest.raises(IngestAbortedError):
            await _pipeline(repo, storage, DeletingVLM(delete)).process(job)

        chunks, images, jobs = _db_counts(settings, kb.id, document.id)
        assert (chunks, images, jobs) == (0, 0, 0)
        assert await repo.get_document(kb.id, document.id) is None
        # 图片描述后可能又写回了文件；pipeline 的 abort 清理应把它删掉。
        assert not (tmp_path / "images" / str(kb.id) / str(document.id)).exists()
        assert not (tmp_path / "uploads" / str(kb.id) / str(document.id)).exists()
    finally:
        if kb is not None:
            with contextlib.suppress(Exception):
                await repo.delete_kb(kb.id)
        await repo.close()
