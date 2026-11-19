"""Neo4j schema 集成测试：约束与索引可幂等创建。"""

from __future__ import annotations

import pytest
from neo4j import GraphDatabase

from rag_kb.config import get_settings
from rag_kb.infra.neo4j.schema import apply_schema, schema_statements

pytestmark = pytest.mark.integration


def test_schema_statements_are_complete() -> None:
    text = "\n".join(schema_statements())
    assert "doc_key" in text
    assert "chunk_key" in text
    assert "entity_key" in text
    assert "CREATE INDEX entity_lookup" in text


def test_apply_schema_is_idempotent() -> None:
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        apply_schema(driver, settings.neo4j_database)
        apply_schema(driver, settings.neo4j_database)

        with driver.session(database=settings.neo4j_database) as session:
            constraints = {
                record["name"]: record
                for record in session.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties")
            }
            indexes = {record["name"] for record in session.run("SHOW INDEXES YIELD name")}

        assert constraints["doc_key"]["properties"] == ["kb_id", "doc_id"]
        assert constraints["chunk_key"]["properties"] == ["kb_id", "chunk_id"]
        assert constraints["entity_key"]["properties"] == ["kb_id", "normalized", "type"]
        assert constraints["entity_key"]["labelsOrTypes"] == ["Entity"]
        assert {"doc_kb", "chunk_kb_doc", "entity_lookup"} <= indexes
    finally:
        driver.close()
