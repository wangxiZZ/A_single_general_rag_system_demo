"""Neo4j 图存储集成测试。"""

from __future__ import annotations

from uuid import uuid4

import pytest
from neo4j import GraphDatabase

from rag_kb.config import get_settings
from rag_kb.domain.models import Entity, Mention, Relation
from rag_kb.infra.neo4j.store import Neo4jGraphStore

pytestmark = pytest.mark.integration


async def test_upsert_isolation_delete_and_orphan_cleanup() -> None:
    settings = get_settings()
    store = Neo4jGraphStore(
        settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    await store.ensure_schema()
    kb1, kb2 = uuid4(), uuid4()
    doc1, doc2 = uuid4(), uuid4()
    chunk1, chunk2 = "a" * 32, "b" * 32
    e1 = Entity(id="e1", kb_id=kb1, name="年假政策", normalized="年假政策", type="条款")
    e2 = Entity(id="e2", kb_id=kb1, name="员工", normalized="员工", type="人物")
    other = Entity(id="e9", kb_id=kb2, name="年假政策", normalized="年假政策", type="条款")

    try:
        await store.upsert(
            kb1,
            doc1,
            [e1, e2],
            [
                Relation(
                    head_id="e1",
                    tail_id="e2",
                    type="适用于",
                    chunk_id=chunk1,
                    evidence_span=(0, 4),
                )
            ],
            [
                Mention(chunk_id=chunk1, entity_id="e1", span=(0, 4)),
                Mention(chunk_id=chunk1, entity_id="e2", span=(4, 6)),
                Mention(chunk_id=chunk2, entity_id="e1", span=(0, 4)),
            ],
        )
        await store.upsert(
            kb2,
            doc2,
            [other],
            [],
            [Mention(chunk_id="c" * 32, entity_id="e9", span=(0, 4))],
        )

        # Doc1 + 2 Chunk + 2 Entity
        assert await store.count(kb1) == 5
        # Doc1 + 2 Chunk（Entity 没有 doc_id）
        assert await store.count(kb1, doc1) == 3
        assert await store.count(kb2) == 3  # Doc2 + Chunk + Entity

        candidates = await store.candidate_entities(kb1, ["年假政策"], limit=10)
        assert [entity.id for entity in candidates] == ["e1"]

        # 重复 upsert 幂等
        await store.upsert(
            kb1,
            doc1,
            [e1, e2],
            [],
            [Mention(chunk_id=chunk1, entity_id="e1", span=(0, 4))],
        )
        assert await store.count(kb1) == 5

        deleted = await store.delete_by_doc(kb1, doc1)
        assert deleted == 2
        assert await store.count(kb1) == 0
        # 另一个库不受影响
        assert await store.count(kb2) == 3

        with (
            GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            ) as driver,
            driver.session(database=settings.neo4j_database) as session,
        ):
            rels = session.run(
                """
                    MATCH ()-[r:REL {kb_id: $kb, doc_id: $doc}]-()
                    RETURN count(r) AS n
                    """,
                kb=str(kb1),
                doc=str(doc1),
            ).single()
            assert rels["n"] == 0
    finally:
        with (
            GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            ) as driver,
            driver.session(database=settings.neo4j_database) as session,
        ):
            session.run(
                "MATCH (n) WHERE n.kb_id IN $kbs DETACH DELETE n",
                kbs=[str(kb1), str(kb2)],
            ).consume()
        await store.close()
