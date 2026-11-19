"""Neo4j schema 加载与执行。

schema 文件与代码同包分发；语句用 ``IF NOT EXISTS``，启动时重复执行是幂等的。
"""

from __future__ import annotations

from pathlib import Path

from neo4j import Driver

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.cypher"


def schema_statements() -> list[str]:
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    return [part.strip() for part in text.split(";") if part.strip()]


def apply_schema(driver: Driver, database: str = "neo4j") -> None:
    with driver.session(database=database) as session:
        for statement in schema_statements():
            session.run(statement).consume()
