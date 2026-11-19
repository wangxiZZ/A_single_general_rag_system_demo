"""失败边界集成测试：真实三存储 + fake 模型。

覆盖 ADR-022 的三类失败语义：
1. 上传超限在 API 层就是 413，且**不创建 document 行**；
2. 损坏 PDF / 扫描件 / 缺少 EMBED_API_KEY → document `failed`、error 可读、
   `ingest_job` 记录失败阶段、chunk 与三路索引均无残留；
3. 修好配置后 retry 能把同一个文档推进到 `ready`。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient
from psycopg.rows import dict_row

from rag_kb.application.document_service import DocumentService
from rag_kb.application.indexing import IndexingService
from rag_kb.application.ingest_pipeline import IngestPipeline
from rag_kb.config import get_settings
from rag_kb.domain.models import KB, DocumentStatus
from rag_kb.infra.milvus.store import MilvusVectorStore
from rag_kb.infra.neo4j.store import Neo4jGraphStore
from rag_kb.infra.parsers import build_parser_registry
from rag_kb.infra.postgres.lexical import PgLexicalIndex
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.infra.postgres.repository import PostgresMetadataRepository
from rag_kb.infra.storage.local import LocalStorage
from rag_kb.main import create_app
from rag_kb.worker.loop import IngestWorker

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class FakeEmbedder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed_documents(self, texts):
        return [[1.0] * self.dimension for _ in texts]

    async def embed_query(self, text: str):
        return [1.0] * self.dimension


class FakeVLM:
    async def describe_image(
        self, image: bytes, *, mime: str = "image/png", prompt: str = ""
    ) -> str:
        return "一张图"


class FakeLLM:
    async def structured_json(self, messages, schema):
        text = messages[0].content
        if "年假" in text:
            return {
                "entities": [{"name": "年假政策", "type": "条款", "span": "年假"}],
                "relations": [],
            }
        return {"entities": [], "relations": []}

    async def chat(self, messages, *, temperature: float = 0.0, max_tokens=None) -> str:
        return ""


async def _make_stack(tmp_path: Path, *, with_embedder: bool) -> SimpleNamespace:
    settings = get_settings()
    apply_migrations(settings.pg_dsn)
    repo = PostgresMetadataRepository(settings.pg_dsn, min_size=1, max_size=2)
    lexical = PgLexicalIndex(settings.pg_dsn, min_size=1, max_size=2)
    vector = MilvusVectorStore(settings.milvus_uri, collection_name=settings.milvus_collection)
    graph = Neo4jGraphStore(
        settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    await repo.open()
    await lexical.open()
    await vector.ensure_collection(settings.embedding_dim)
    await graph.ensure_schema()

    storage = LocalStorage(tmp_path / "data")
    embedder = FakeEmbedder(settings.embedding_dim) if with_embedder else None
    indexer = IndexingService(
        repo,
        embedder=embedder,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
        llm=FakeLLM(),
        chunker_version=settings.chunker_version,
        embed_model="fake-embed",
    )
    pipeline = IngestPipeline(
        repo,
        storage,
        build_parser_registry(),
        FakeVLM(),
        chunker_version=settings.chunker_version,
        target_tokens=settings.chunk_target_tokens,
        overlap_ratio=settings.chunk_overlap_ratio,
        parent_max_tokens=settings.parent_max_tokens,
        chars_per_token=settings.chars_per_token,
        embed_model="fake-embed",
        indexer=indexer,
    )
    documents = DocumentService(
        repo,
        storage,
        max_upload_mb=settings.max_upload_mb,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
    )
    worker = IngestWorker(repo, pipeline.process, poll_interval_s=0.05)
    return SimpleNamespace(
        settings=settings,
        repo=repo,
        lexical=lexical,
        vector=vector,
        graph=graph,
        storage=storage,
        documents=documents,
        worker=worker,
        pipeline=pipeline,
        indexer=indexer,
    )


async def _close(stack: SimpleNamespace) -> None:
    await stack.vector.close()
    await stack.graph.close()
    await stack.lexical.close()
    await stack.repo.close()


async def _process_next(stack: SimpleNamespace, doc_id) -> None:
    job = await stack.repo.claim_ingest_job()
    assert job is not None and job.doc_id == doc_id, "任务队列里出现了其他文档的任务"
    await stack.worker._process(job)


def _job_row(settings, job_doc_id) -> dict:
    with psycopg.connect(settings.pg_dsn, row_factory=dict_row) as conn:
        row = conn.execute(
            "SELECT status, stage, error FROM ingest_job WHERE doc_id = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (job_doc_id,),
        ).fetchone()
    assert row is not None
    return row


async def _assert_no_residue(stack: SimpleNamespace, kb_id, doc_id) -> None:
    assert await stack.repo.count_chunks(kb_id, doc_id) == 0
    assert await stack.repo.count_images(kb_id, doc_id) == 0
    assert await stack.vector.count(kb_id, doc_id) == 0
    assert await stack.lexical.count(kb_id, doc_id) == 0
    assert await stack.graph.count(kb_id, doc_id) == 0


async def test_oversized_upload_returns_413_without_document_row(tmp_path: Path) -> None:
    stack = await _make_stack(tmp_path, with_embedder=False)
    settings = stack.settings.model_copy(update={"max_upload_mb": 1})
    app = create_app(settings=settings, repository=stack.repo, start_worker=False)

    try:
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        ):
            kb = (await client.post("/api/kbs", json={"name": "超限测试"})).json()
            response = await client.post(
                f"/api/kbs/{kb['id']}/documents",
                files={
                    "file": (
                        "a.pdf",
                        b"%PDF-1.7\n" + b"x" * (1024 * 1024),
                        "application/pdf",
                    )
                },
            )
            assert response.status_code == 413
            assert response.json()["code"] == "file_too_large"

            assert (await client.get(f"/api/kbs/{kb['id']}/documents")).json() == []
            assert await stack.repo.list_documents(kb["id"]) == []

            await client.delete(f"/api/kbs/{kb['id']}")
    finally:
        await _close(stack)


@pytest.mark.parametrize(
    ("filename", "data", "expected_error"),
    [
        ("broken.pdf", b"%PDF-1.7\nbroken", "无法打开 PDF"),
        (
            "scanned.pdf",
            (FIXTURES / "documents" / "scanned.pdf").read_bytes(),
            "OCR",
        ),
    ],
)
async def test_parse_failure_marks_failed_without_residue(
    tmp_path: Path, filename: str, data: bytes, expected_error: str
) -> None:
    stack = await _make_stack(tmp_path, with_embedder=True)
    kb = await stack.repo.create_kb(KB(id=uuid4(), name=f"fail-{uuid4().hex[:8]}"))
    try:
        document = await stack.documents.upload_document(
            kb.id, filename=filename, mime="application/pdf", data=data
        )
        await _process_next(stack, document.id)

        failed = await stack.documents.get_document(kb.id, document.id)
        assert failed.status is DocumentStatus.FAILED
        assert failed.error and expected_error in failed.error

        row = _job_row(stack.settings, document.id)
        assert row["status"] == "failed"
        assert row["stage"] == "failed"
        assert row["error"] and expected_error in row["error"]

        await _assert_no_residue(stack, kb.id, document.id)
    finally:
        await stack.repo.delete_kb(kb.id)
        await _close(stack)


async def test_missing_embed_key_fails_then_retry_recovers(tmp_path: Path) -> None:
    stack = await _make_stack(tmp_path, with_embedder=False)
    data = (FIXTURES / "demo" / "employee_handbook.pdf").read_bytes()

    kb = await stack.repo.create_kb(KB(id=uuid4(), name=f"retry-{uuid4().hex[:8]}"))
    try:
        document = await stack.documents.upload_document(
            kb.id, filename="employee_handbook.pdf", mime="application/pdf", data=data
        )
        await _process_next(stack, document.id)

        failed = await stack.documents.get_document(kb.id, document.id)
        assert failed.status is DocumentStatus.FAILED
        assert failed.error and "EMBED_API_KEY" in failed.error
        assert _job_row(stack.settings, document.id)["status"] == "failed"
        await _assert_no_residue(stack, kb.id, document.id)

        # 模拟配好 Key：换一个带 embedder 的 indexer，retry 后应重新解析并建好三路索引。
        # 同一个 tmp_path = 同一个存储根；retry 需要从原文件重新解析。
        good = await _make_stack(tmp_path, with_embedder=True)
        try:
            good.documents = DocumentService(
                stack.repo,
                stack.storage,
                max_upload_mb=stack.settings.max_upload_mb,
                vector_store=good.vector,
                lexical_index=good.lexical,
                graph_store=good.graph,
            )
            retried = await good.documents.retry_document(kb.id, document.id)
            assert retried.status is DocumentStatus.PENDING

            await _process_next(
                SimpleNamespace(
                    repo=stack.repo,
                    worker=IngestWorker(stack.repo, good.pipeline.process, poll_interval_s=0.05),
                ),
                document.id,
            )

            recovered = await good.documents.get_document(kb.id, document.id)
            assert recovered.status is DocumentStatus.READY
            assert recovered.chunk_count > 0
            assert await stack.repo.count_chunks(kb.id, document.id) > 0
            assert await stack.vector.count(kb.id, document.id) > 0
            assert await stack.lexical.count(kb.id, document.id) > 0
        finally:
            await _close(good)
    finally:
        await stack.vector.delete_by_doc(kb.id, document.id)
        await stack.lexical.delete_by_doc(kb.id, document.id)
        await stack.graph.delete_by_doc(kb.id, document.id)
        await stack.repo.delete_kb(kb.id)
        await _close(stack)
