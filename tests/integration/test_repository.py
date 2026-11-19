"""PostgreSQL 仓储集成测试：需要 docker compose up 与已执行的迁移。"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from uuid import uuid4

import pytest

from rag_kb.config import get_settings
from rag_kb.domain.errors import NotFoundError
from rag_kb.domain.models import (
    KB,
    BBox,
    Chunk,
    ChunkKind,
    ChunkLevel,
    DocImage,
    Document,
    DocumentStatus,
    QueryLog,
    SearchMode,
)
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.infra.postgres.repository import PostgresMetadataRepository
from rag_kb.main import create_app

pytestmark = pytest.mark.integration


async def test_repository_roundtrip_and_isolation() -> None:
    settings = get_settings()
    apply_migrations(settings.pg_dsn)

    repo = PostgresMetadataRepository(settings.pg_dsn, min_size=1, max_size=2)
    await repo.open()
    kb_id = uuid4()
    doc_id = uuid4()

    try:
        await repo.create_kb(KB(id=kb_id, name="repo-test"))
        doc = await repo.create_document(
            Document(
                id=doc_id,
                kb_id=kb_id,
                filename="a.pdf",
                mime="application/pdf",
                sha256=uuid4().hex * 2,
                size=10,
            )
        )
        assert doc.status is DocumentStatus.PENDING
        assert await repo.find_document_by_sha(kb_id, doc.sha256) is not None

        parent = Chunk(
            id="p" * 32,
            kb_id=kb_id,
            doc_id=doc_id,
            level=ChunkLevel.PARENT,
            kind=ChunkKind.TEXT,
            content="父块",
            seq=0,
            char_start=0,
            char_end=2,
            token_count=2,
            chunker_version="v1",
            heading_path=("1", "1.1"),
        )
        child = Chunk(
            id="c" * 32,
            kb_id=kb_id,
            doc_id=doc_id,
            parent_id=parent.id,
            level=ChunkLevel.CHILD,
            kind=ChunkKind.TEXT,
            content="子块",
            seq=1,
            char_start=0,
            char_end=2,
            token_count=2,
            chunker_version="v1",
            heading_path=("1", "1.1"),
            meta={"low_weight": True},
        )
        # 故意乱序传入：仓储应先写父块，DEFERRABLE 外键保证批量插入可行。
        await repo.replace_chunks(kb_id, doc_id, [child, parent])
        assert await repo.count_chunks(kb_id, doc_id) == 2

        fetched = await repo.get_chunks(kb_id, [child.id])
        assert [c.id for c in fetched] == [child.id]
        assert fetched[0].parent_id == parent.id
        assert fetched[0].meta == {"low_weight": True}

        # 跨库隔离：另一个 kb_id 查同一个 chunk id 必须为空。
        assert await repo.get_chunks(uuid4(), [child.id]) == []

        image = DocImage(
            id=uuid4(),
            kb_id=kb_id,
            doc_id=doc_id,
            seq=0,
            page_no=1,
            bbox=BBox(10, 20, 210, 220),
            file_path=f"images/{kb_id}/{doc_id}/x.png",
            caption="一张图",
        )
        await repo.replace_images(kb_id, doc_id, [image])
        images = await repo.list_images(kb_id, doc_id)
        assert images == [image]
        assert await repo.count_images(kb_id, doc_id) == 1

        updated = await repo.update_document(
            kb_id, replace(doc, status=DocumentStatus.CHUNKED, chunk_count=2)
        )
        assert updated.status is DocumentStatus.CHUNKED
        assert updated.chunk_count == 2

        job = await repo.enqueue_ingest_job(doc_id)
        claimed = await repo.claim_ingest_job()
        assert claimed is not None
        assert claimed.id == job.id
        assert claimed.status.value == "running"

        await repo.add_query_log(
            QueryLog(
                kb_id=kb_id,
                mode=SearchMode.HYBRID,
                query="test",
                latency_ms=1.5,
                hit_ids=(child.id,),
                channels={"vector": 1, "keyword": 1},
            )
        )

        await repo.delete_document(kb_id, doc_id)
        assert await repo.count_chunks(kb_id, doc_id) == 0
        assert await repo.count_images(kb_id, doc_id) == 0
        assert await repo.list_documents(kb_id) == []
    finally:
        with contextlib.suppress(NotFoundError):
            await repo.delete_kb(kb_id)
        await repo.close()


async def test_app_lifespan_opens_and_closes_repository() -> None:
    app = create_app()
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.repository, PostgresMetadataRepository)
        assert await app.state.repository.get_kb(uuid4()) is None
