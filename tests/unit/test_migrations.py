"""迁移文件发现的纯单测；真正的执行验证属于 integration。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_kb.infra.postgres.migrations import default_sql_dir, discover_migrations


def test_discover_migrations_sorts_by_filename(tmp_path: Path) -> None:
    (tmp_path / "002_add_lex.sql").write_text("-- noop", encoding="utf-8")
    (tmp_path / "001_init.sql").write_text("-- noop", encoding="utf-8")

    names = [path.name for path in discover_migrations(tmp_path)]
    assert names == ["001_init.sql", "002_add_lex.sql"]


def test_discover_migrations_rejects_bad_name(tmp_path: Path) -> None:
    (tmp_path / "init.sql").write_text("-- noop", encoding="utf-8")
    with pytest.raises(ValueError, match=r"001_name\.sql"):
        discover_migrations(tmp_path)


def test_discover_migrations_requires_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_migrations(tmp_path)


def test_initial_migration_is_present_and_has_core_tables() -> None:
    sql_dir = default_sql_dir()
    files = discover_migrations(sql_dir)
    assert files[0].name == "001_init.sql"

    sql = files[0].read_text(encoding="utf-8")
    for table in ("kb", "document", "doc_image", "chunk", "chunk_lex", "ingest_job", "query_log"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
