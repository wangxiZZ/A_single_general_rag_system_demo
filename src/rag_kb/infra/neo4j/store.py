"""Neo4j 图谱存储。

- ``upsert`` 用 MERGE 保证幂等：Doc/Chunk/Entity 复用，MENTIONS/REL 边按
  ``(chunk_id, type)`` 合并。
- 关系边同时带 ``chunk_id``、``evidence_span``、``kb_id``、``doc_id``：前者用于回溯原文，
  后两者让按文档删除不必先反查边。
- ``delete_by_doc`` 先删该文档的边，再删 Chunk/Doc，最后清理失去所有关系的 Entity；
  返回删除的 Chunk 数，便于集成测试断言。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager
from uuid import UUID

import anyio
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from rag_kb.domain.errors import DependencyUnavailableError, StorageError
from rag_kb.domain.models import Entity, Mention, Relation
from rag_kb.infra.neo4j.schema import apply_schema


def _to_thread[**P, R](func: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await anyio.to_thread.run_sync(lambda: func(*args, **kwargs))

    return wrapper


class Neo4jGraphStore:
    def __init__(
        self,
        uri: str,
        *,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    async def close(self) -> None:
        await anyio.to_thread.run_sync(self._driver.close)

    @contextmanager
    def _session(self):
        try:
            with self._driver.session(database=self._database) as session:
                yield session
        except ServiceUnavailable as exc:
            raise DependencyUnavailableError("neo4j") from exc
        except Neo4jError as exc:
            raise StorageError(f"Neo4j 操作失败: {exc}") from exc

    async def ensure_schema(self) -> None:
        try:
            await anyio.to_thread.run_sync(apply_schema, self._driver, self._database)
        except ServiceUnavailable as exc:
            raise DependencyUnavailableError("neo4j") from exc

    @_to_thread
    def upsert(
        self,
        kb_id: UUID,
        doc_id: UUID,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
        mentions: Sequence[Mention],
    ) -> None:
        chunk_ids = sorted(
            {mention.chunk_id for mention in mentions}
            | {relation.chunk_id for relation in relations}
        )
        entity_rows = [
            {
                "id": entity.id,
                "name": entity.name,
                "normalized": entity.normalized,
                "type": entity.type,
                "aliases": list(entity.aliases),
            }
            for entity in entities
        ]
        mention_rows = [
            {
                "chunk_id": mention.chunk_id,
                "entity_id": mention.entity_id,
                "span": list(mention.span),
            }
            for mention in mentions
        ]
        relation_rows = [
            {
                "head_id": relation.head_id,
                "tail_id": relation.tail_id,
                "type": relation.type,
                "chunk_id": relation.chunk_id,
                "weight": relation.weight,
                "evidence_span": list(relation.evidence_span)
                if relation.evidence_span is not None
                else None,
            }
            for relation in relations
        ]
        kb, doc = str(kb_id), str(doc_id)

        with self._session() as session:
            session.execute_write(
                self._upsert_tx,
                kb,
                doc,
                chunk_ids,
                entity_rows,
                mention_rows,
                relation_rows,
            )

    @staticmethod
    def _upsert_tx(
        tx,
        kb: str,
        doc: str,
        chunk_ids: list[str],
        entities: list[dict],
        mentions: list[dict],
        relations: list[dict],
    ) -> None:
        tx.run(
            "MERGE (d:Doc {kb_id: $kb, doc_id: $doc})",
            kb=kb,
            doc=doc,
        ).consume()
        if chunk_ids:
            tx.run(
                """
                MATCH (d:Doc {kb_id: $kb, doc_id: $doc})
                UNWIND $chunk_ids AS chunk_id
                MERGE (c:Chunk {kb_id: $kb, chunk_id: chunk_id})
                SET c.doc_id = $doc, c.level = 'child'
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                kb=kb,
                doc=doc,
                chunk_ids=chunk_ids,
            ).consume()
        if entities:
            tx.run(
                """
                UNWIND $entities AS row
                MERGE (e:Entity {
                    kb_id: $kb, normalized: row.normalized, type: row.type
                })
                SET e.id = row.id, e.name = row.name, e.aliases = row.aliases
                """,
                kb=kb,
                entities=entities,
            ).consume()
        if mentions:
            tx.run(
                """
                UNWIND $mentions AS row
                MATCH (c:Chunk {kb_id: $kb, chunk_id: row.chunk_id})
                MATCH (e:Entity {kb_id: $kb, id: row.entity_id})
                MERGE (c)-[m:MENTIONS]->(e)
                SET m.span = row.span
                """,
                kb=kb,
                mentions=mentions,
            ).consume()
        if relations:
            tx.run(
                """
                UNWIND $relations AS row
                MATCH (h:Entity {kb_id: $kb, id: row.head_id})
                MATCH (t:Entity {kb_id: $kb, id: row.tail_id})
                MERGE (h)-[r:REL {
                    kb_id: $kb, doc_id: $doc,
                    chunk_id: row.chunk_id, type: row.type
                }]->(t)
                SET r.weight = row.weight, r.evidence_span = row.evidence_span
                """,
                kb=kb,
                doc=doc,
                relations=relations,
            ).consume()

    @_to_thread
    def delete_by_doc(self, kb_id: UUID, doc_id: UUID) -> int:
        kb, doc = str(kb_id), str(doc_id)

        def delete_tx(tx) -> int:
            record = tx.run(
                """
                MATCH (c:Chunk {kb_id: $kb, doc_id: $doc})
                RETURN count(c) AS n
                """,
                kb=kb,
                doc=doc,
            ).single()
            chunk_count = int(record["n"]) if record else 0
            tx.run(
                """
                MATCH ()-[r:REL {kb_id: $kb, doc_id: $doc}]-()
                DELETE r
                """,
                kb=kb,
                doc=doc,
            ).consume()
            tx.run(
                """
                MATCH (c:Chunk {kb_id: $kb, doc_id: $doc})
                DETACH DELETE c
                """,
                kb=kb,
                doc=doc,
            ).consume()
            tx.run(
                """
                MATCH (d:Doc {kb_id: $kb, doc_id: $doc})
                DETACH DELETE d
                """,
                kb=kb,
                doc=doc,
            ).consume()
            tx.run(
                """
                MATCH (e:Entity {kb_id: $kb})
                WHERE NOT (e)<-[:MENTIONS]-() AND NOT (e)-[:REL]-()
                DETACH DELETE e
                """,
                kb=kb,
            ).consume()
            return chunk_count

        with self._session() as session:
            return session.execute_write(delete_tx)

    @_to_thread
    def count(self, kb_id: UUID, doc_id: UUID | None = None) -> int:
        kb = str(kb_id)
        if doc_id is None:
            query = "MATCH (n {kb_id: $kb}) RETURN count(n) AS n"
            params = {"kb": kb}
        else:
            query = "MATCH (n {kb_id: $kb, doc_id: $doc}) RETURN count(n) AS n"
            params = {"kb": kb, "doc": str(doc_id)}
        with self._session() as session:
            record = session.run(query, **params).single()
        return int(record["n"]) if record else 0

    @_to_thread
    def candidate_entities(
        self,
        kb_id: UUID,
        terms: Sequence[str],
        limit: int,
        *,
        contains: bool = False,
    ) -> list[Entity]:
        if not terms:
            return []
        with self._session() as session:
            rows = session.run(
                """
                MATCH (e:Entity {kb_id: $kb})
                WHERE (
                    $contains AND ANY(
                        term IN $terms
                        WHERE e.normalized CONTAINS term OR e.name CONTAINS term
                    )
                ) OR (
                    NOT $contains AND (
                        e.normalized IN $terms OR e.name IN $terms
                        OR ANY(alias IN coalesce(e.aliases, []) WHERE alias IN $terms)
                    )
                )
                RETURN e
                LIMIT $limit
                """,
                kb=str(kb_id),
                terms=list(terms),
                limit=limit,
                contains=contains,
            ).data()
        return [
            Entity(
                id=row["e"]["id"],
                kb_id=kb_id,
                name=row["e"]["name"],
                normalized=row["e"]["normalized"],
                type=row["e"]["type"],
                aliases=tuple(row["e"].get("aliases") or ()),
            )
            for row in rows
        ]
