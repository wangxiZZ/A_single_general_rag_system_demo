"""``rag-kb`` 命令行入口。

子命令：

- ``serve``：启动 uvicorn（默认，保持旧行为）；
- ``migrate``：执行手写迁移；
- ``demo``：通过 REST 建演示知识库并上传固定夹具，轮询到 ready。

``demo`` 走 HTTP 而不是直接操作仓储，原因有两个：验证的是真实用户路径；worker 在
服务进程里运行，直接写库会绕过任务队列。
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from rag_kb.config import get_settings
from rag_kb.infra.postgres.migrations import apply_migrations


def _fixtures_dir(explicit: Path | None = None) -> Path:
    """定位演示夹具：优先 ``tests/fixtures/demo``，回退到解析测试用的 ``documents``。

    两个目录都在仓库里，从容器或任意工作目录调用时必须按文件定位而不是按 cwd，
    否则 ``docker compose exec app rag-kb demo`` 会找不到素材。
    """
    if explicit is not None:
        if not explicit.is_dir():
            raise FileNotFoundError(f"演示夹具目录不存在：{explicit}")
        return explicit

    roots = (Path.cwd(), Path(__file__).resolve().parents[2])
    for root in roots:
        demo = root / "tests" / "fixtures" / "demo"
        if demo.is_dir() and any(demo.glob("*.pdf")):
            return demo
    for root in roots:
        documents = root / "tests" / "fixtures" / "documents"
        if (documents / "sample.pdf").exists():
            return documents
    raise FileNotFoundError("找不到 tests/fixtures/demo，请在仓库根目录运行 demo")


def _migrate() -> int:
    settings = get_settings()
    applied = apply_migrations(settings.pg_dsn)
    if applied:
        print("applied migrations: " + ", ".join(applied))
    else:
        print("no pending migrations")
    return 0


def _wait_until_ready(
    client: httpx.Client, base: str, kb_id: str, doc_id: str, name: str, timeout_s: float
) -> bool:
    """轮询到 ``ready``。

    不能停在 ``chunked``：三路索引与图片描述都在之后的 indexing 阶段，提前返回会让演示
    立刻检索却 0 命中。失败时打印服务端 ``error``（可读原因），而不是只报超时。
    """
    deadline = time.monotonic() + timeout_s
    last_status = "unknown"
    while time.monotonic() < deadline:
        response = client.get(f"{base}/kbs/{kb_id}/documents/{doc_id}")
        response.raise_for_status()
        detail = response.json()
        status = detail["status"]
        if status != last_status:
            print(f"{name}: {status}")
            last_status = status
        if status == "ready":
            print(f"{name}: ready, chunks={detail['chunk_count']}")
            return True
        if status == "failed":
            print(
                f"{name}: failed — {detail.get('error') or '服务端未返回错误信息'}", file=sys.stderr
            )
            return False
        time.sleep(0.5)
    print(
        f"{name}: 等待 {timeout_s:.0f}s 仍未 ready（最后状态 {last_status}），请检查 worker 日志",
        file=sys.stderr,
    )
    return False


def _reindex(kb_id: str, base_url: str | None) -> int:
    settings = get_settings()
    base = (base_url or f"http://localhost:{settings.app_port}").rstrip("/") + "/api"
    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(f"{base}/kbs/{kb_id}/reindex")
            response.raise_for_status()
            task_id = response.json().get("task_id")
            print(f"reindex accepted: {task_id}")

            # 先等后台任务真正结束；否则旧状态的 ready 文档会让下面的轮询立即返回。
            for _ in range(240):
                status_response = client.get(f"{base}/kbs/{kb_id}/reindex/{task_id}")
                if status_response.status_code == 404:
                    break
                if status_response.json()["status"] == "done":
                    break
                time.sleep(0.5)

            for _ in range(240):
                documents = client.get(f"{base}/kbs/{kb_id}/documents").json()
                active = [
                    doc
                    for doc in documents
                    if doc["status"] in {"pending", "parsing", "chunked", "indexing"}
                ]
                if not active:
                    failed = [doc for doc in documents if doc["status"] == "failed"]
                    ready = [doc for doc in documents if doc["status"] == "ready"]
                    print(f"ready={len(ready)} failed={len(failed)}")
                    return 0 if not failed else 1
                time.sleep(0.5)
            print("reindex 等待超时，请检查 worker 日志", file=sys.stderr)
            return 1
    except httpx.HTTPError as exc:
        print(f"reindex 失败: {exc}", file=sys.stderr)
        return 1


def load_demo(
    base_url: str | None = None,
    fixtures_dir: Path | None = None,
    kb_name: str | None = None,
    timeout_s: float = 240.0,
) -> int:
    """建一个演示知识库并上传夹具，等到每个文档 ``ready``。

    走 HTTP 而不是直接写仓储：演示验证的是真实用户路径，且 worker 在服务进程内运行，
    直接写库会绕过任务队列。``scripts/load_demo.py`` 只是本函数的参数包装。
    """
    settings = get_settings()
    root = (base_url or f"http://localhost:{settings.app_port}").rstrip("/")
    base = f"{root}/api"
    fixtures = _fixtures_dir(fixtures_dir)

    files = sorted(
        (p for p in fixtures.iterdir() if p.suffix.lower() in {".pdf", ".docx"}),
        key=lambda p: (p.suffix.lower() != ".pdf", p.name),
    )
    if not files:
        print(f"演示夹具目录中没有 PDF/docx：{fixtures}", file=sys.stderr)
        return 1

    try:
        with httpx.Client(timeout=60) as client:
            kb = client.post(
                f"{base}/kbs",
                json={
                    "name": kb_name or f"演示知识库-{time.strftime('%H%M%S')}",
                    "description": "rag-kb demo",
                },
            )
            kb.raise_for_status()
            kb_id = kb.json()["id"]
            print(f"知识库: {kb_id}")

            ok = True
            for path in files:
                with path.open("rb") as handle:
                    response = client.post(
                        f"{base}/kbs/{kb_id}/documents",
                        files={"file": (path.name, handle)},
                    )
                if response.status_code >= 400:
                    detail = response.text
                    with contextlib.suppress(ValueError):
                        detail = response.json().get("detail", detail)
                    print(
                        f"{path.name}: 上传失败 {response.status_code} — {detail}", file=sys.stderr
                    )
                    ok = False
                    continue
                document = response.json()
                ok = (
                    _wait_until_ready(client, base, kb_id, document["id"], path.name, timeout_s)
                    and ok
                )

            if not ok:
                print(
                    "演示数据未全部就绪，可用 `docker compose logs app` 查看原因", file=sys.stderr
                )
                return 1

            print(f"完成。打开 {root}/web/kbs/{kb_id}")
            print("可以试问：年假有几天？怎么申请？ / 报销多久打款？ / 支持哪些文档格式？")
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
    demo.add_argument("--fixtures", default=None, help="夹具目录，默认 tests/fixtures/demo")
    demo.add_argument("--name", default=None, help="知识库名称，默认带时间戳")
    demo.add_argument("--timeout", type=float, default=240.0, help="等待 ready 的秒数")
    reindex = sub.add_parser("reindex", help="重建知识库三路索引")
    reindex.add_argument("--kb", required=True, help="知识库 id")
    reindex.add_argument("--base-url", default=None, help="API 根地址")

    args = parser.parse_args(argv)
    if args.command in (None, "serve"):
        from rag_kb.main import run

        run()
        return 0
    if args.command == "migrate":
        return _migrate()
    if args.command == "demo":
        return load_demo(
            args.base_url,
            Path(args.fixtures) if args.fixtures else None,
            args.name,
            args.timeout,
        )
    if args.command == "reindex":
        return _reindex(args.kb, args.base_url)
    parser.error(f"未知命令: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
