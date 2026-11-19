"""MCP stdio 端到端：真实拉起 server，走 REST 建库入库，再通过 MCP 工具检索。

为什么不用 fake HTTP：
这一层要验证的是「Claude Code 实际会经历的那条链路」——子进程启动、stdio 握手、工具
schema、结构化返回。任何一段用 mock 替代都会漏掉真实失败（例如入口找不到模块、
环境变量没透传、返回值不是对象）。

依赖：`docker compose up -d` 已就绪、模型 Key 已配置、已 `uv sync --extra mcp`。
运行：`uv run pytest -m e2e tests/e2e/test_mcp_stdio.py -q`
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_URL = os.environ.get("RAG_KB_BASE_URL", "http://localhost:8000").rstrip("/")
PDF_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "demo" / "employee_handbook.pdf"


async def _call(session: ClientSession, name: str, arguments: dict) -> dict:
    result = await session.call_tool(name, arguments)
    assert not result.is_error, f"{name} 返回错误：{result.content}"
    assert result.structured_content is not None, f"{name} 没有结构化结果"
    return result.structured_content


def _require_ready_service() -> None:
    try:
        response = httpx.get(f"{BASE_URL}/readyz", timeout=5)
    except httpx.HTTPError as exc:
        pytest.skip(f"RAG 服务未启动（{BASE_URL}）：{exc}")
    if response.status_code != 200 or not response.json().get("ready"):
        pytest.skip(f"RAG 服务未就绪：{response.text[:200]}")


def _create_kb_with_pdf(client: httpx.Client) -> tuple[str, str]:
    kb = client.post(
        f"{BASE_URL}/api/kbs",
        json={"name": f"mcp-e2e-{int(time.time())}", "description": "e2e"},
    )
    kb.raise_for_status()
    kb_id = kb.json()["id"]

    response = client.post(
        f"{BASE_URL}/api/kbs/{kb_id}/documents",
        files={"file": (PDF_FIXTURE.name, PDF_FIXTURE.read_bytes(), "application/pdf")},
    )
    response.raise_for_status()
    doc_id = response.json()["id"]

    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        detail = client.get(f"{BASE_URL}/api/kbs/{kb_id}/documents/{doc_id}").json()
        if detail["status"] == "ready":
            return kb_id, doc_id
        if detail["status"] == "failed":
            pytest.fail(f"摄取失败：{detail.get('error')}")
        time.sleep(1)
    pytest.fail("文档在 240s 内未 ready，请检查 worker 与模型配置")


async def test_mcp_stdio_list_then_search() -> None:
    _require_ready_service()
    with httpx.Client(timeout=60) as client:
        kb_id, _ = _create_kb_with_pdf(client)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(REPO_ROOT),
        env={**os.environ, "RAG_KB_BASE_URL": BASE_URL},
    )
    try:
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()

            tools = {tool.name for tool in (await session.list_tools()).tools}
            assert tools == {
                "list_knowledge_bases",
                "search_knowledge_base",
                "ask_knowledge_base",
            }

            listed = await _call(session, "list_knowledge_bases", {})
            kb_ids = [kb["id"] for kb in listed["knowledge_bases"]]
            assert kb_id in kb_ids

            found = await _call(
                session,
                "search_knowledge_base",
                {"kb_id": kb_id, "query": "年假怎么申请", "mode": "hybrid", "top_k": 5},
            )
            assert found["count"] >= 1
            hit = found["hits"][0]
            assert hit["doc_name"] == PDF_FIXTURE.name
            assert hit["page_no"] == 1
            assert hit["snippet"]
            # 结构化结果必须能原样 JSON 序列化，避免 MCP 客户端收到不可解析对象
            json.dumps(found, ensure_ascii=False)

            missing = await session.call_tool(
                "search_knowledge_base",
                {"kb_id": "00000000-0000-0000-0000-000000000000", "query": "年假"},
            )
            assert missing.is_error
            assert "404" in missing.content[0].text
    finally:
        with httpx.Client(timeout=30) as client:
            client.delete(f"{BASE_URL}/api/kbs/{kb_id}")
