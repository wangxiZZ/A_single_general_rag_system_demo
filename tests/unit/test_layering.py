"""分层自检：用 AST 静态扫描 import 图，防止依赖方向腐化。

规则来自 AGENTS.md §2 与 docs/项目目录结构说明.md §3：

    domain      只能依赖标准库与 rag_kb.domain
    application 只能依赖标准库、rag_kb.domain、rag_kb.application
    api         不依赖 infra / web / worker / 存储客户端
    web         不依赖 infra / api / worker
    worker      不依赖 infra / api / web
    infra       不反向依赖 api / web / worker / application
    mcp_server  不 import rag_kb（只走 HTTP）

测试本身也验证扫描器能抓到注入的违规，避免"永远为空"的假通过。
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
MCP = REPO_ROOT / "mcp_server"
STDLIB = set(sys.stdlib_module_names)

Rule = Callable[[str], str | None]


def _is(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _module_name(path: Path, base: Path) -> str:
    parts = list(path.relative_to(base).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: Path, module: str) -> Iterator[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                for _ in range(node.level - 1):
                    parts = parts[:-1]
                if node.module:
                    parts += node.module.split(".")
                if parts:
                    yield ".".join(parts)
            elif node.module:
                yield node.module


def _scan(directory: Path, base: Path, rule: Rule) -> list[str]:
    violations: list[str] = []
    for path in sorted(directory.rglob("*.py")):
        module = _module_name(path, base)
        for name in _imports(path, module):
            reason = rule(name)
            if reason is not None:
                violations.append(f"{path.relative_to(base)}: {name} — {reason}")
    return violations


def _domain_rule(name: str) -> str | None:
    if _is(name, "rag_kb.domain"):
        return None
    if name.split(".")[0] in STDLIB:
        return None
    return "domain 只能依赖标准库与 rag_kb.domain"


def _application_rule(name: str) -> str | None:
    if _is(name, "rag_kb.domain") or _is(name, "rag_kb.application"):
        return None
    if name.split(".")[0] in STDLIB:
        return None
    return "application 只能依赖标准库与 domain 端口"


def _api_rule(name: str) -> str | None:
    if name.split(".")[0] in STDLIB:
        return None
    if name == "rag_kb" or _is(name, "rag_kb.api"):
        return None
    if _is(name, "rag_kb.application") or _is(name, "rag_kb.domain"):
        return None
    if _is(name, "rag_kb.config"):
        return None
    if name.split(".")[0] in {"fastapi", "starlette", "pydantic"}:
        return None
    return "api 不允许依赖 infra、其他入口层或存储客户端"


def _web_rule(name: str) -> str | None:
    if name.split(".")[0] in STDLIB:
        return None
    if name == "rag_kb" or _is(name, "rag_kb.web"):
        return None
    if _is(name, "rag_kb.application") or _is(name, "rag_kb.domain"):
        return None
    if _is(name, "rag_kb.config"):
        return None
    if name.split(".")[0] in {"fastapi", "starlette", "jinja2"}:
        return None
    return "web 不允许依赖 infra、api 或 worker"


def _worker_rule(name: str) -> str | None:
    if name.split(".")[0] in STDLIB:
        return None
    if name == "rag_kb" or _is(name, "rag_kb.worker"):
        return None
    if _is(name, "rag_kb.application") or _is(name, "rag_kb.domain"):
        return None
    if _is(name, "rag_kb.config"):
        return None
    return "worker 不允许依赖 infra、api 或 web"


def _infra_rule(name: str) -> str | None:
    if not _is(name, "rag_kb"):
        return None  # 标准库与第三方客户端由 infra 实现端口，允许
    if _is(name, "rag_kb.infra") or _is(name, "rag_kb.domain") or _is(name, "rag_kb.config"):
        return None
    return "infra 不允许反向依赖 api / web / worker / application"


def _mcp_rule(name: str) -> str | None:
    return "mcp_server 只能走 HTTP，禁止 import rag_kb" if _is(name, "rag_kb") else None


def test_domain_depends_only_on_stdlib_and_domain() -> None:
    assert _scan(SRC / "rag_kb" / "domain", SRC, _domain_rule) == []


def test_application_does_not_import_web_or_infra() -> None:
    assert _scan(SRC / "rag_kb" / "application", SRC, _application_rule) == []


def test_api_does_not_import_infra_or_storage_clients() -> None:
    assert _scan(SRC / "rag_kb" / "api", SRC, _api_rule) == []


def test_web_does_not_import_infra() -> None:
    assert _scan(SRC / "rag_kb" / "web", SRC, _web_rule) == []


def test_worker_does_not_import_entry_layers_or_infra() -> None:
    assert _scan(SRC / "rag_kb" / "worker", SRC, _worker_rule) == []


def test_infra_does_not_import_upward() -> None:
    assert _scan(SRC / "rag_kb" / "infra", SRC, _infra_rule) == []


def test_mcp_server_does_not_import_rag_kb() -> None:
    assert _scan(MCP, REPO_ROOT, _mcp_rule) == []


def test_scanner_detects_domain_violation(tmp_path: Path) -> None:
    bad = tmp_path / "rag_kb" / "domain" / "bad.py"
    bad.parent.mkdir(parents=True)
    bad.write_text("import psycopg\n", encoding="utf-8")

    violations = _scan(tmp_path / "rag_kb" / "domain", tmp_path, _domain_rule)
    assert len(violations) == 1
    assert "psycopg" in violations[0]


def test_scanner_detects_api_storage_violation(tmp_path: Path) -> None:
    bad = tmp_path / "rag_kb" / "api" / "bad.py"
    bad.parent.mkdir(parents=True)
    bad.write_text("from pymilvus import MilvusClient\n", encoding="utf-8")

    violations = _scan(tmp_path / "rag_kb" / "api", tmp_path, _api_rule)
    assert len(violations) == 1
    assert "pymilvus" in violations[0]


def test_scanner_detects_mcp_violation(tmp_path: Path) -> None:
    bad = tmp_path / "mcp_server" / "bad.py"
    bad.parent.mkdir(parents=True)
    bad.write_text("import rag_kb.infra.milvus\n", encoding="utf-8")

    violations = _scan(tmp_path / "mcp_server", tmp_path, _mcp_rule)
    assert len(violations) == 1
    assert "rag_kb" in violations[0]
