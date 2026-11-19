"""删除边界集成测试：删文档 / 删整库 / 删除与摄取并发。

验收口径（ADR-022、AGENTS §3.7）不是接口返回 204，而是：
三通道 0 命中 + PG/Milvus/Neo4j 计数清零 + 磁盘文件消失。
Milvus 使用 Strong 一致性，删除后可以立即断言，无需 sleep 或重试。
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from rag_kb.application.document_service import DocumentService
from rag_kb.application.indexing import IndexingService
from rag_kb.application.ingest_pipeline import IngestPipeline
from rag_kb.application.kb_service import KbService
from rag_kb.application.search_service import SearchService
from rag_kb.config import get_settings
from rag_kb.domain.errors import IngestAbortedError, NotFoundError
from rag_kb.domain.models import KB, SearchMode
from rag_kb.infra.milvus.store import MilvusVectorStore
from rag_kb.infra.neo4j.store import Neo4jGraphStore
from rag_kb.infra.parsers import build_parser_registry
from rag_kb.infra.postgres.lexical import PgLexicalIndex
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.infra.postgres.repository import PostgresMetadataRepository
from rag_kb.infra.storage.local import LocalStorage

pytestmark = pytest.mark.integration

PDF = Path(__file__).resolve().parents[1] / "fixtures" / "demo" / "employee_handbook.pdf"
SEARCH_MODES = [SearchMode.VECTOR, SearchMode.KEYWORD, SearchMode.GRAPH, SearchMode.HYBRID]


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
        if "年假" in messages[0].content:
            return {
                "entities": [
                    {"name": "年假政策", "type": "条款", "span": "年假"},
                    {"name": "员工", "type": "人物", "span": "员工"},
                ],
                "relations": [
                    {
                        "head": "年假政策",
                        "tail": "员工",
                        "relation": "关联",
                        "evidence_span": "年假",
                    }
                ],
            }
        return {"entities": [], "relations": []}

    async def chat(self, messages, *, temperature: float = 0.0, max_tokens=None) -> str:
        return ""


class DeletingVLM(FakeVLM):
    def __init__(self, delete) -> None:
        self._delete = delete

    async def describe_image(self, image, *, mime="image/png", prompt="") -> str:
        await self._delete()
        return "一张图"


async def _make_stack(tmp_path: Path, *, vlm=None) -> SimpleNamespace:
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
    embedder = FakeEmbedder(settings.embedding_dim)
    llm = FakeLLM()
    indexer = IndexingService(
        repo,
        embedder=embedder,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
        llm=llm,
        chunker_version=settings.chunker_version,
        embed_model="fake-embed",
    )
    pipeline = IngestPipeline(
        repo,
        storage,
        build_parser_registry(),
        vlm or FakeVLM(),
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
    kbs = KbService(
        repo,
        storage,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
    )
    search = SearchService(
        repo,
        embedder=embedder,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
    )
    return SimpleNamespace(
        settings=settings,
        repo=repo,
        lexical=lexical,
        vector=vector,
        graph=graph,
        storage=storage,
        documents=documents,
        kbs=kbs,
        search=search,
        pipeline=pipeline,
        indexer=indexer,
        root=tmp_path / "data",
    )


async def _close(stack: SimpleNamespace) -> None:
    await stack.vector.close()
    await stack.graph.close()
    await stack.lexical.close()
    await stack.repo.close()


async def _ingest(stack: SimpleNamespace, kb_id, filename: str = PDF.name):
    document = await stack.documents.upload_document(
        kb_id, filename=filename, mime="application/pdf", data=PDF.read_bytes()
    )
    job = await stack.repo.claim_ingest_job()
    assert job is not None and job.doc_id == document.id
    await stack.pipeline.process(job)
    return await stack.documents.get_document(kb_id, document.id)


async def _assert_all_channels_empty(stack, kb_id, doc_id) -> None:
    assert await stack.repo.count_chunks(kb_id, doc_id) == 0
    assert await stack.repo.count_images(kb_id, doc_id) == 0
    assert await stack.vector.count(kb_id, doc_id) == 0
    assert await stack.lexical.count(kb_id, doc_id) == 0
    assert await stack.graph.count(kb_id, doc_id) == 0


def _doc_dirs(root: Path, kb_id, doc_id) -> tuple[Path, Path]:
    return root / "uploads" / str(kb_id) / str(doc_id), root / "images" / str(kb_id) / str(doc_id)


async def test_delete_document_clears_indexes_rows_and_files(tmp_path: Path) -> None:
    stack = await _make_stack(tmp_path)
    kb = await stack.repo.create_kb(KB(id=uuid4(), name=f"del-doc-{uuid4().hex[:6]}"))
    doc_id = None
    try:
        document = await _ingest(stack, kb.id)
        doc_id = document.id
        upload_dir, image_dir = _doc_dirs(stack.root, kb.id, doc_id)
        assert upload_dir.is_dir()
        assert image_dir.is_dir()

        # 删除前确认三通道确实能召回，否则「删后 0 命中」没有意义。
        for mode in SEARCH_MODES:
            assert await stack.search.search(kb.id, "年假", mode=mode), mode

        await stack.documents.delete_document(kb.id, doc_id)

        for mode in SEARCH_MODES:
            assert await stack.search.search(kb.id, "年假", mode=mode) == [], mode

        assert await stack.repo.get_document(kb.id, doc_id) is None
        assert await stack.repo.list_documents(kb.id) == []
        await _assert_all_channels_empty(stack, kb.id, doc_id)
        assert not upload_dir.exists()
        assert not image_dir.exists()
    finally:
        if doc_id is not None:
            with contextlib.suppress(Exception):
                await stack.vector.delete_by_doc(kb.id, doc_id)
            with contextlib.suppress(Exception):
                await stack.lexical.delete_by_doc(kb.id, doc_id)
            with contextlib.suppress(Exception):
                await stack.graph.delete_by_doc(kb.id, doc_id)
        with contextlib.suppress(Exception):
            await stack.repo.delete_kb(kb.id)
        await _close(stack)


async def test_delete_kb_clears_everything(tmp_path: Path) -> None:
    stack = await _make_stack(tmp_path)
    kb = await stack.repo.create_kb(KB(id=uuid4(), name=f"del-kb-{uuid4().hex[:6]}"))
    doc_id = None
    try:
        document = await _ingest(stack, kb.id)
        doc_id = document.id
        upload_dir, image_dir = _doc_dirs(stack.root, kb.id, doc_id)
        assert await stack.vector.count(kb.id) > 0
        assert await stack.lexical.count(kb.id) > 0
        assert await stack.graph.count(kb.id) > 0

        await stack.kbs.delete_kb(kb.id)

        assert await stack.repo.get_kb(kb.id) is None
        assert await stack.vector.count(kb.id) == 0
        assert await stack.lexical.count(kb.id) == 0
        assert await stack.graph.count(kb.id) == 0
        assert not upload_dir.exists()
        assert not image_dir.exists()
        with pytest.raises(NotFoundError):
            await stack.search.search(kb.id, "年假", mode=SearchMode.HYBRID)
    finally:
        if doc_id is not None:
            with contextlib.suppress(Exception):
                await stack.vector.delete_by_doc(kb.id, doc_id)
            with contextlib.suppress(Exception):
                await stack.lexical.delete_by_doc(kb.id, doc_id)
            with contextlib.suppress(Exception):
                await stack.graph.delete_by_doc(kb.id, doc_id)
        with contextlib.suppress(Exception):
            await stack.repo.delete_kb(kb.id)
        await _close(stack)


async def test_delete_during_ingest_aborts_without_residue(tmp_path: Path) -> None:
    stack = await _make_stack(tmp_path)
    kb = await stack.repo.create_kb(KB(id=uuid4(), name=f"del-race-{uuid4().hex[:6]}"))
    doc_id = None
    try:
        document = await stack.documents.upload_document(
            kb.id, filename=PDF.name, mime="application/pdf", data=PDF.read_bytes()
        )
        doc_id = document.id

        async def delete_now() -> None:
            await stack.documents.delete_document(kb.id, doc_id)

        # 用带删除副作用的 VLM 重建管线：图片描述阶段删除文档，模拟与 worker 并发。
        pipeline = IngestPipeline(
            stack.repo,
            stack.storage,
            build_parser_registry(),
            DeletingVLM(delete_now),
            chunker_version=stack.settings.chunker_version,
            target_tokens=stack.settings.chunk_target_tokens,
            overlap_ratio=stack.settings.chunk_overlap_ratio,
            parent_max_tokens=stack.settings.parent_max_tokens,
            chars_per_token=stack.settings.chars_per_token,
            embed_model="fake-embed",
            indexer=stack.indexer,
        )

        job = await stack.repo.claim_ingest_job()
        assert job is not None and job.doc_id == doc_id
        with pytest.raises(IngestAbortedError):
            await pipeline.process(job)

        assert await stack.repo.get_document(kb.id, doc_id) is None
        await _assert_all_channels_empty(stack, kb.id, doc_id)
        upload_dir, image_dir = _doc_dirs(stack.root, kb.id, doc_id)
        assert not upload_dir.exists()
        assert not image_dir.exists()
    finally:
        if doc_id is not None:
            with contextlib.suppress(Exception):
                await stack.vector.delete_by_doc(kb.id, doc_id)
            with contextlib.suppress(Exception):
                await stack.lexical.delete_by_doc(kb.id, doc_id)
            with contextlib.suppress(Exception):
                await stack.graph.delete_by_doc(kb.id, doc_id)
        with contextlib.suppress(Exception):
            await stack.repo.delete_kb(kb.id)
        await _close(stack)
