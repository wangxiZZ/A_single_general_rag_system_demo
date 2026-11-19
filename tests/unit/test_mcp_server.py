"""MCP server 单测：只验证 HTTP 转发与错误翻译，不起服务、不连数据库。

依赖 `mcp` 可选依赖；未执行 `uv sync --extra mcp` 时整组跳过，
避免主单测套件因为缺少协议库而失败。
"""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("mcp")

from mcp_server import server


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_KB_BASE_URL", "http://rag.test")
    monkeypatch.setattr(server, "_transport", None)


def _use(handler) -> None:
    server._transport = httpx.MockTransport(handler)


async def test_list_knowledge_bases_forwards_get() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"id": "kb-1", "name": "手册"}])

    _use(handler)
    result = await server.list_knowledge_bases()

    assert result == {"knowledge_bases": [{"id": "kb-1", "name": "手册"}]}
    assert seen[0].method == "GET"
    assert str(seen[0].url) == "http://rag.test/api/kbs"


async def test_search_forwards_mode_and_top_k() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"chunk_id": "c1", "snippet": "年假"}])

    _use(handler)
    result = await server.search_knowledge_base("kb-1", "年假", mode="keyword", top_k=5)

    assert result["count"] == 1
    assert result["hits"][0]["chunk_id"] == "c1"
    assert str(seen[0].url) == "http://rag.test/api/kbs/kb-1/search"
    assert json.loads(seen[0].content) == {"query": "年假", "mode": "keyword", "top_k": 5}


async def test_ask_forwards_question() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"answer": "5 天", "citations": [], "no_evidence": False})

    _use(handler)
    result = await server.ask_knowledge_base("kb-1", "年假几天？")

    assert result["answer"] == "5 天"
    assert json.loads(seen[0].content) == {
        "question": "年假几天？",
        "mode": "hybrid",
        "top_k": 8,
    }


async def test_error_detail_is_surfaced() -> None:
    _use(lambda request: httpx.Response(404, json={"detail": "知识库不存在"}))

    with pytest.raises(RuntimeError, match="404：知识库不存在"):
        await server.search_knowledge_base("missing", "年假")


async def test_connect_error_has_actionable_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _use(handler)
    with pytest.raises(RuntimeError, match="无法连接 RAG 知识库服务"):
        await server.list_knowledge_bases()


async def test_timeout_has_actionable_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    _use(handler)
    with pytest.raises(RuntimeError, match="请求超时"):
        await server.list_knowledge_bases()
