"""手写迁移执行器。

为什么不用 Alembic：本项目以检索查询为主，schema 变化不频繁，手写 SQL 更容易看清
索引与约束；迁移框架的 autogenerate 反而会遮蔽执行计划。

约定：

- 迁移文件放在 ``sql/`` 下，文件名 ``<三位序号>_<snake_case>.sql``，按文件名排序执行；
- 已执行版本记录在 ``schema_migration`` 表，重复运行是幂等的；
- 每个文件在独立事务里执行，失败则整个文件回滚，不会留下"执行了一半"的 schema；
- 执行器不隐藏错误：连接失败、SQL 语法错误直接向上抛，让 CLI/运维看到真实原因。
"""

from __future__ import annotations

import re
from pathlib import Path

import psycopg

MIGRATION_VERSION_RE = re.compile(r"^\d{3}_[a-z0-9_]+$")

MIGRATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migration (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def default_sql_dir() -> Path:
    """优先使用包内 SQL；非 editable 安装时回退到源码树，便于本机与容器共用。"""
    packaged = Path(__file__).resolve().parent / "sql"
    if packaged.is_dir():
        return packaged
    return Path.cwd() / "src" / "rag_kb" / "infra" / "postgres" / "sql"


def discover_migrations(sql_dir: Path | None = None) -> list[Path]:
    """返回按版本号排序的迁移文件；文件名不合法或为空目录时直接失败。"""
    directory = sql_dir or default_sql_dir()
    if not directory.is_dir():
        raise FileNotFoundError(f"迁移目录不存在: {directory}")

    files = sorted(
        (path for path in directory.glob("*.sql") if path.is_file()), key=lambda p: p.name
    )
    if not files:
        raise FileNotFoundError(f"迁移目录里没有 .sql 文件: {directory}")

    for path in files:
        if not MIGRATION_VERSION_RE.fullmatch(path.stem):
            raise ValueError(f"迁移文件名不合法（应为 001_name.sql）: {path.name}")

    versions = [path.stem for path in files]
    if len(versions) != len(set(versions)):
        raise ValueError(f"迁移版本重复: {versions}")
    return files


def apply_migrations(dsn: str, sql_dir: Path | None = None) -> list[str]:
    """执行所有未应用的迁移，返回本次应用的版本列表。"""
    files = discover_migrations(sql_dir)

    # autocommit=True 让建表语句立即提交；每个迁移文件再用 transaction() 包住，
    # 保证 DDL 与"写入 schema_migration"要么一起成功，要么一起回滚。
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(MIGRATION_TABLE_SQL)
        rows = conn.execute("SELECT version FROM schema_migration").fetchall()
        applied = {row[0] for row in rows}

        newly_applied: list[str] = []
        for path in files:
            version = path.stem
            if version in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_migration (version) VALUES (%s)",
                    (version,),
                )
            newly_applied.append(version)
            applied.add(version)

    return newly_applied
