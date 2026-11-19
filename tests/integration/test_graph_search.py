"""图谱多跳召回集成测试：Neo4j 路径 → PostgreSQL 原文。"""

from __future__ import annotations

import contextlib
from uuid import uuid4

import pytest

from rag_kb.application.graph_search import graph_search
from rag_kb.config import get_settings
from rag_kb.domain.models import (
    KB,
    Chunk,
    ChunkKind,
    ChunkLevel,
    Document,
    Entity,
    Mention,
    Relation,
)
from rag_kb.infra.neo4j.store import Neo4jGraphStore
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.infra.postgres.repository import PostgresMetadataRepository

pytestmark = pytest.mark.integration


async def test_multi_hop_graph_search_returns_source_chunks() -> None:
    settings = get_settings()
    apply_migrations(settings.pg_dsn)
    repo = PostgresMetadataRepository(settings.pg_dsn, min_size=1, max_size=2)
    graph = Neo4jGraphStore(
        settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    await repo.open()
    await graph.ensure_schema()
    kb_id = uuid4()
    doc_id = uuid4()
    chunk1 = "a" * 32
    chunk2 = "b" * 32

    try:
        await repo.create_kb(KB(id=kb_id, name=f"graph-search-{uuid4().hex[:6]}"))
        await repo.create_document(
            Document(
                id=doc_id,
                kb_id=kb_id,
                filename="manual.pdf",
                mime="application/pdf",
                sha256="a" * 64,
                size=1,
            )
        )
        await repo.replace_chunks(
            kb_id,
            doc_id,
            [
                Chunk(
                    id=chunk1,
                    kb_id=kb_id,
                    doc_id=doc_id,
                    level=ChunkLevel.CHILD,
                    kind=ChunkKind.TEXT,
                    content="年假政策规定员工每年享有带薪年假。",
                    seq=0,
                    char_start=0,
                    char_end=20,
                    token_count=20,
                    chunker_version="v1",
                    heading_path=("Manual",),
                    page_no=1,
                ),
                Chunk(
                    id=chunk2,
                    kb_id=kb_id,
                    doc_id=doc_id,
                    level=ChunkLevel.CHILD,
                    kind=ChunkKind.TEXT,
                    content="员工提交审批流程后由主管批准。",
                    seq=1,
                    char_start=20,
                    char_end=40,
                    token_count=20,
                    chunker_version="v1",
                    heading_path=("Manual",),
                    page_no=2,
                ),
            ],
        )

        e1 = Entity(id="e1", kb_id=kb_id, name="年假政策", normalized="年假政策", type="条款")
        e2 = Entity(id="e2", kb_id=kb_id, name="员工", normalized="员工", type="人物")
        e3 = Entity(id="e3", kb_id=kb_id, name="审批流程", normalized="审批流程", type="流程")
        await graph.upsert(
            kb_id,
            doc_id,
            [e1, e2, e3],
            [
                Relation(head_id="e1", tail_id="e2", type="适用于", chunk_id=chunk1, weight=1.0),
                Relation(head_id="e2", tail_id="e3", type="触发", chunk_id=chunk2, weight=1.0),
            ],
            [
                Mention(chunk_id=chunk1, entity_id="e1", span=(0, 4)),
                Mention(chunk_id=chunk1, entity_id="e2", span=(4, 6)),
                Mention(chunk_id=chunk2, entity_id="e3", span=(4, 8)),
            ],
        )

        one_hop = await graph.expand(kb_id, ["e1"], max_hops=1, neighbor_limit=10)
        assert [item.chunk_id for item in one_hop] == [chunk1]

        two_hops = await graph.expand(kb_id, ["e1"], max_hops=2, neighbor_limit=10)
        assert {item.chunk_id for item in two_hops} == {chunk1, chunk2}
        assert two_hops[0].score > two_hops[1].score

        hits = await graph_search(
            kb_id,
            "年假政策",
            graph_store=graph,
            repository=repo,
            max_hops=2,
            top_k=10,
        )
        assert {hit.chunk_id for hit in hits} == {chunk1, chunk2}
        assert all(hit.channels == ("graph",) for hit in hits)
        assert all(hit.doc_name == "manual.pdf" for hit in hits)
        assert any("审批流程" in hit.snippet for hit in hits)
    finally:
        with contextlib.suppress(Exception):
            await graph.delete_by_doc(kb_id, doc_id)
        with contextlib.suppress(Exception):
            await repo.delete_kb(kb_id)
        await graph.close()
        await repo.close()
