"""``rag-kb`` 命令行入口。

子命令：

- ``serve``：启动 uvicorn（默认，保持旧行为）；
- ``migrate``：执行手写迁移；
- ``demo``：通过 REST 建演示知识库并上传固定夹具，轮询到 chunked。

``demo`` 走 HTTP 而不是直接操作仓储，原因有两个：验证的是真实用户路径；worker 在
服务进程里运行，直接写库会绕过任务队列。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from rag_kb.config import get_settings
from rag_kb.infra.postgres.migrations import apply_migrations


def _fixtures_dir() -> Path:
    candidates = (
        Path.cwd() / "tests" / "fixtures" / "documents",
        Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "documents",
    )
    for candidate in candidates:
        if (candidate / "sample.pdf").exists():
            return candidate
    raise FileNotFoundError("找不到 tests/fixtures/documents，请在仓库根目录运行 demo")


def _migrate() -> int:
    settings = get_settings()
    applied = apply_migrations(settings.pg_dsn)
    if applied:
        print("applied migrations: " + ", ".join(applied))
    else:
        print("no pending migrations")
    return 0


def _wait_until_chunked(
    client: httpx.Client, base: str, kb_id: str, doc_id: str, name: str
) -> bool:
    for _ in range(120):
        detail = client.get(f"{base}/kbs/{kb_id}/documents/{doc_id}").json()
        status = detail["status"]
        if status in {"chunked", "ready"}:
            print(f"{name}: {status}, chunks={detail['chunk_count']}")
            return True
        if status == "failed":
            print(f"{name}: failed — {detail.get('error')}", file=sys.stderr)
            return False
        time.sleep(0.5)
    print(f"{name}: 等待超时，请检查 worker 日志", file=sys.stderr)
    return False


def _demo(base_url: str | None) -> int:
    settings = get_settings()
    base = (base_url or f"http://localhost:{settings.app_port}").rstrip("/") + "/api"
    fixtures = _fixtures_dir()

    try:
        with httpx.Client(timeout=60) as client:
            kb = client.post(
                f"{base}/kbs",
                json={
                    "name": f"演示知识库-{time.strftime('%H%M%S')}",
                    "description": "rag-kb demo",
                },
            )
            kb.raise_for_status()
            kb_id = kb.json()["id"]
            print(f"知识库: {kb_id}")

            for filename in ("sample.pdf", "sample.docx"):
                path = fixtures / filename
                with path.open("rb") as handle:
                    response = client.post(
                        f"{base}/kbs/{kb_id}/documents",
                        files={"file": (filename, handle)},
                    )
                response.raise_for_status()
                document = response.json()
                _wait_until_chunked(client, base, kb_id, document["id"], filename)

            print(f"完成。打开 http://localhost:{settings.app_port}/web/kbs/{kb_id}")
    except httpx.HTTPError as exc:
        print(f"demo 失败: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-kb", description="通用 RAG 知识库服务")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="启动服务（默认）")
    sub.add_parser("migrate", help="执行数据库迁移")
    demo = sub.add_parser("demo", help="装载演示知识库与文档")
    demo.add_argument(
        "--base-url", default=None, help="API 根地址，默认 http://localhost:<APP_PORT>"
    )

    args = parser.parse_args(argv)
    if args.command in (None, "serve"):
        from rag_kb.main import run

        run()
        return 0
    if args.command == "migrate":
        return _migrate()
    if args.command == "demo":
        return _demo(args.base_url)
    parser.error(f"未知命令: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
