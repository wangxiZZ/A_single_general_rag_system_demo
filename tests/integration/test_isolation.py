"""知识库隔离与删除后不可召回的集成测试（真实三存储）。"""

from __future__ import annotations

import contextlib
from pathlib import Path
from uuid import uuid4

import pytest

from rag_kb.application.document_service import DocumentService
from rag_kb.application.indexing import IndexingService
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
                    "evidence_span": "年假政策规定员工",
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
