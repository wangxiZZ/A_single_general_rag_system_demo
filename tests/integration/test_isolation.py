"""知识库隔离与删除后不可召回的集成测试（真实三存储）。"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from rag_kb.application.document_service import DocumentService
from rag_kb.application.indexing import IndexingService
from rag_kb.application.ingest_pipeline import IngestPipeline
from rag_kb.application.kb_service import KbService
from rag_kb.application.search_service import SearchService
from rag_kb.config import get_settings
from rag_kb.domain.models import (
    KB,
    Chunk,
    ChunkKind,
    ChunkLevel,
    Document,
    DocumentStatus,
    SearchMode,
)
from rag_kb.infra.milvus.store import MilvusVectorStore
from rag_kb.infra.neo4j.store import Neo4jGraphStore
from rag_kb.infra.parsers import build_parser_registry
from rag_kb.infra.postgres.lexical import PgLexicalIndex
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.infra.postgres.repository import PostgresMetadataRepository
from rag_kb.infra.storage.local import LocalStorage

pytestmark = pytest.mark.integration


class FakeEmbedder:
    dimension = 4

    async def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, text: str):
        return [1.0, 0.0, 0.0, 0.0]


class FakeLLM:
    async def structured_json(self, messages, schema):
        text = messages[0].content
        if "外包" in text:
            return {
                "entities": [
                    {"name": "年假政策", "type": "条款", "span": "年假政策"},
                    {"name": "外包人员", "type": "人物", "span": "外包人员"},
                ],
                "relations": [
                    {
                        "head": "年假政策",
                        "tail": "外包人员",
                        "relation": "适用于",
                        "evidence_span": "年假政策适用于外包人员",
                    }
                ],
            }
        return {
            "entities": [
                {"name": "年假政策", "type": "条款", "span": "年假政策"},
                {"name": "员工", "type": "人物", "span": "员工"},
            ],
            "relations": [
                {
                    "head": "年假政策",
                    "tail": "员工",
                    "relation": "适用于",
                    "evidence_span": "年假",
                }
            ],
        }

    async def chat(self, messages, *, temperature: float = 0.0, max_tokens=None) -> str:
        return ""


async def _seed_doc(
    repo: PostgresMetadataRepository, kb: KB, chunk_id: str, content: str
) -> tuple[Document, Chunk]:
    doc = await repo.create_document(
        Document(
            id=uuid4(),
            kb_id=kb.id,
            filename=f"{kb.name}.pdf",
            mime="application/pdf",
            sha256=uuid4().hex * 2,
            size=1,
            status=DocumentStatus.CHUNKED,
            chunker_version="v1",
            chunk_count=1,
        )
    )
    chunk = Chunk(
        id=chunk_id,
        kb_id=kb.id,
        doc_id=doc.id,
        level=ChunkLevel.CHILD,
        kind=ChunkKind.TEXT,
        content=content,
        seq=0,
        char_start=0,
        char_end=len(content),
        token_count=len(content),
        chunker_version="v1",
        page_no=1,
    )
    await repo.replace_chunks(kb.id, doc.id, [chunk])
    return doc, chunk


async def test_kb_isolation_and_deletion(tmp_path: Path) -> None:
    settings = get_settings()
    apply_migrations(settings.pg_dsn)
    repo = PostgresMetadataRepository(settings.pg_dsn, min_size=1, max_size=2)
    lexical = PgLexicalIndex(settings.pg_dsn, min_size=1, max_size=2)
    vector = MilvusVectorStore(
        settings.milvus_uri,
        collection_name=f"chunk_vec_iso_{uuid4().hex[:8]}",
    )
    graph = Neo4jGraphStore(
        settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    await repo.open()
    await lexical.open()
    await vector.ensure_collection(4)
    await graph.ensure_schema()

    kb_a = await repo.create_kb(KB(id=uuid4(), name=f"iso-a-{uuid4().hex[:6]}"))
    kb_b = await repo.create_kb(KB(id=uuid4(), name=f"iso-b-{uuid4().hex[:6]}"))
    doc_a, _ = await _seed_doc(repo, kb_a, "a" * 32, "年假政策规定员工每年享有带薪年假。")
    doc_b, _ = await _seed_doc(repo, kb_b, "b" * 32, "年假政策适用于外包人员。")

    indexer = IndexingService(
        repo,
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        vector_store=vector,  # type: ignore[arg-type]
        lexical_index=lexical,  # type: ignore[arg-type]
        graph_store=graph,  # type: ignore[arg-type]
        llm=FakeLLM(),  # type: ignore[arg-type]
        chunker_version="v1",
        embed_model="fake",
    )
    await indexer.index_document(kb_a.id, doc_a.id)
    await indexer.index_document(kb_b.id, doc_b.id)

    search = SearchService(
        repo,
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        vector_store=vector,  # type: ignore[arg-type]
        lexical_index=lexical,  # type: ignore[arg-type]
        graph_store=graph,  # type: ignore[arg-type]
        top_k=5,
    )

    try:
        for mode in (
            SearchMode.VECTOR,
            SearchMode.KEYWORD,
            SearchMode.GRAPH,
            SearchMode.HYBRID,
        ):
            a_hits = await search.search(kb_a.id, "年假政策", mode=mode)
            b_hits = await search.search(kb_b.id, "年假政策", mode=mode)
            assert {hit.chunk_id for hit in a_hits} == {"a" * 32}, mode
            assert {hit.chunk_id for hit in b_hits} == {"b" * 32}, mode

        storage = LocalStorage(tmp_path)
        docs_service = DocumentService(
            repo,
            storage,
            vector_store=vector,  # type: ignore[arg-type]
            lexical_index=lexical,  # type: ignore[arg-type]
            graph_store=graph,  # type: ignore[arg-type]
        )
        await docs_service.delete_document(kb_a.id, doc_a.id)

        assert await vector.count(kb_a.id) == 0
        assert await lexical.count(kb_a.id) == 0
        assert await graph.count(kb_a.id) == 0
        assert await vector.count(kb_b.id) == 1
        assert await lexical.count(kb_b.id) == 1

        for mode in (
            SearchMode.VECTOR,
            SearchMode.KEYWORD,
            SearchMode.GRAPH,
            SearchMode.HYBRID,
        ):
            assert await search.search(kb_a.id, "年假政策", mode=mode) == [], mode
            assert {
                hit.chunk_id for hit in await search.search(kb_b.id, "年假政策", mode=mode)
            } == {"b" * 32}, mode

        kb_service = KbService(
            repo,
            storage,
            vector_store=vector,  # type: ignore[arg-type]
            lexical_index=lexical,  # type: ignore[arg-type]
            graph_store=graph,  # type: ignore[arg-type]
        )
        await kb_service.delete_kb(kb_b.id)
        assert await vector.count(kb_b.id) == 0
        assert await lexical.count(kb_b.id) == 0
        assert await graph.count(kb_b.id) == 0
    finally:
        with contextlib.suppress(Exception):
            await repo.delete_kb(kb_a.id)
        with contextlib.suppress(Exception):
            await repo.delete_kb(kb_b.id)
        with contextlib.suppress(Exception):
            await graph.delete_by_doc(kb_a.id, doc_a.id)
        with contextlib.suppress(Exception):
            await graph.delete_by_doc(kb_b.id, doc_b.id)
        await vector.drop_collection()
        await graph.close()
        await lexical.close()
        await repo.close()


class FakeVLM:
    async def describe_image(
        self, image: bytes, *, mime: str = "image/png", prompt: str = ""
    ) -> str:
        return "一张图"


async def _build_ingest_stack(tmp_path: Path, collection_name: str) -> SimpleNamespace:
    """完整摄取链路：同一份 PDF 要真的解析、分块、写三路索引，才能验证 id 级隔离。"""
    settings = get_settings()
    apply_migrations(settings.pg_dsn)
    repo = PostgresMetadataRepository(settings.pg_dsn, min_size=1, max_size=2)
    lexical = PgLexicalIndex(settings.pg_dsn, min_size=1, max_size=2)
    vector = MilvusVectorStore(settings.milvus_uri, collection_name=collection_name)
    graph = Neo4jGraphStore(
        settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    await repo.open()
    await lexical.open()
    await vector.ensure_collection(4)
    await graph.ensure_schema()

    storage = LocalStorage(tmp_path)
    indexer = IndexingService(
        repo,
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        vector_store=vector,  # type: ignore[arg-type]
        lexical_index=lexical,  # type: ignore[arg-type]
        graph_store=graph,  # type: ignore[arg-type]
        llm=FakeLLM(),  # type: ignore[arg-type]
        chunker_version="v1",
        embed_model="fake",
    )
    pipeline = IngestPipeline(
        repo,
        storage,
        build_parser_registry(),
        FakeVLM(),  # type: ignore[arg-type]
        chunker_version="v1",
        target_tokens=settings.chunk_target_tokens,
        overlap_ratio=settings.chunk_overlap_ratio,
        parent_max_tokens=settings.parent_max_tokens,
        chars_per_token=settings.chars_per_token,
        embed_model="fake",
        indexer=indexer,
    )
    documents = DocumentService(
        repo,
        storage,
        vector_store=vector,  # type: ignore[arg-type]
        lexical_index=lexical,  # type: ignore[arg-type]
        graph_store=graph,  # type: ignore[arg-type]
    )
    kbs = KbService(
        repo,
        storage,
        vector_store=vector,  # type: ignore[arg-type]
        lexical_index=lexical,  # type: ignore[arg-type]
        graph_store=graph,  # type: ignore[arg-type]
    )
    search = SearchService(
        repo,
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        vector_store=vector,  # type: ignore[arg-type]
        lexical_index=lexical,  # type: ignore[arg-type]
        graph_store=graph,  # type: ignore[arg-type]
        top_k=10,
    )
    return SimpleNamespace(
        repo=repo,
        lexical=lexical,
        vector=vector,
        graph=graph,
        documents=documents,
        kbs=kbs,
        search=search,
        pipeline=pipeline,
    )


async def test_same_file_and_entity_names_stay_isolated(tmp_path: Path) -> None:
    """两个库上传同一份 PDF、抽出同名实体，三通道仍不能互串。"""
    stack = await _build_ingest_stack(tmp_path, f"chunk_vec_iso_same_{uuid4().hex[:8]}")
    pdf = Path(__file__).resolve().parents[1] / "fixtures" / "demo" / "employee_handbook.pdf"
    kb_a = await stack.repo.create_kb(KB(id=uuid4(), name=f"same-a-{uuid4().hex[:6]}"))
    kb_b = await stack.repo.create_kb(KB(id=uuid4(), name=f"same-b-{uuid4().hex[:6]}"))
    docs: list[tuple[UUID, UUID]] = []
    try:
        for kb in (kb_a, kb_b):
            document = await stack.documents.upload_document(
                kb.id, filename=pdf.name, mime="application/pdf", data=pdf.read_bytes()
            )
            job = await stack.repo.claim_ingest_job()
            assert job is not None and job.doc_id == document.id
            await stack.pipeline.process(job)
            docs.append((kb.id, document.id))

        doc_a = await stack.documents.get_document(kb_a.id, docs[0][1])
        doc_b = await stack.documents.get_document(kb_b.id, docs[1][1])
        assert doc_a.filename == doc_b.filename == pdf.name
        assert doc_a.sha256 == doc_b.sha256  # 同一文件，只是分属不同库
        assert doc_a.status is doc_b.status is DocumentStatus.READY

        for mode in (
            SearchMode.VECTOR,
            SearchMode.KEYWORD,
            SearchMode.GRAPH,
            SearchMode.HYBRID,
        ):
            a_hits = await stack.search.search(kb_a.id, "年假政策", mode=mode)
            b_hits = await stack.search.search(kb_b.id, "年假政策", mode=mode)
            assert a_hits and b_hits, mode
            assert {hit.doc_id for hit in a_hits} == {doc_a.id}, mode
            assert {hit.doc_id for hit in b_hits} == {doc_b.id}, mode
            # 同一内容的确定性 id 含 kb_id，跨库不能复用主键。
            assert {hit.chunk_id for hit in a_hits}.isdisjoint({hit.chunk_id for hit in b_hits})

        entities_a = await stack.graph.candidate_entities(kb_a.id, ["年假政策"], 10, contains=True)
        entities_b = await stack.graph.candidate_entities(kb_b.id, ["年假政策"], 10, contains=True)
        assert entities_a and entities_b
        assert {entity.name for entity in entities_a} & {entity.name for entity in entities_b}, (
            "两库应抽到同名实体，否则隔离断言没有意义"
        )
        assert {entity.id for entity in entities_a}.isdisjoint({entity.id for entity in entities_b})

        await stack.documents.delete_document(kb_a.id, doc_a.id)
        for mode in (
            SearchMode.VECTOR,
            SearchMode.KEYWORD,
            SearchMode.GRAPH,
            SearchMode.HYBRID,
        ):
            assert await stack.search.search(kb_a.id, "年假政策", mode=mode) == [], mode
            b_hits = await stack.search.search(kb_b.id, "年假政策", mode=mode)
            assert b_hits, mode
            assert {hit.doc_id for hit in b_hits} == {doc_b.id}, mode
    finally:
        with contextlib.suppress(Exception):
            await stack.kbs.delete_kb(kb_a.id)
        with contextlib.suppress(Exception):
            await stack.kbs.delete_kb(kb_b.id)
        with contextlib.suppress(Exception):
            await stack.vector.drop_collection()
        await stack.graph.close()
        await stack.lexical.close()
        await stack.repo.close()
