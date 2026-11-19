"""一键装载演示知识库。

用法：

    uv run python scripts/load_demo.py
    uv run python scripts/load_demo.py --base-url http://localhost:8000 --name 演示库

真正的逻辑在 ``rag_kb.cli.load_demo``：这是一个 HTTP 客户端用例，CLI 与脚本共用同一份实现，
避免演示脚本与运维命令各写一套状态轮询。脚本只负责参数解析，便于在仓库根直接执行。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_kb.cli import load_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="建演示知识库并上传 PDF/docx，等到 ready")
    parser.add_argument(
        "--base-url", default=None, help="API 根地址，默认 http://localhost:<APP_PORT>"
    )
    parser.add_argument("--fixtures", default=None, help="夹具目录，默认 tests/fixtures/demo")
    parser.add_argument("--name", default=None, help="知识库名称，默认带时间戳")
    parser.add_argument("--timeout", type=float, default=240.0, help="等待 ready 的秒数")
    args = parser.parse_args(argv)
    return load_demo(
        args.base_url,
        Path(args.fixtures) if args.fixtures else None,
        args.name,
        args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
