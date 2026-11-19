"""执行数据库迁移。

用法（宿主或容器内）：

    uv run python scripts/migrate.py
    docker compose exec app rag-kb migrate   # M2-13 落地 CLI 后

连接串来自 Settings（``PG_DSN``），不在这里硬编码。
"""

from __future__ import annotations

from rag_kb.config import get_settings
from rag_kb.infra.postgres.migrations import apply_migrations


def main() -> None:
    settings = get_settings()
    applied = apply_migrations(settings.pg_dsn)
    if applied:
        print("applied migrations: " + ", ".join(applied))
    else:
        print("no pending migrations")


if __name__ == "__main__":
    main()
