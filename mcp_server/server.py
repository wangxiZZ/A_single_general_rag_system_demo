"""MCP stdio server：把知识库能力暴露给 Claude Code / 桌面端。

为什么这里只有 HTTP 转发：
MCP 是**协议适配层**，不是第二个业务实现。如果在这里拼检索、做融合或读数据库，
就会出现两套口径（REST 一套、MCP 一套），任何排序或引用规则的修改都要改两处。
所以本模块只做「工具参数 → REST 请求 → JSON 结果」的翻译，且**不 import rag_kb**。

为什么地址从环境变量读而不是复用 src/rag_kb/config.py：
复用会违反仓库的分层约束（mcp_server 不得依赖 rag_kb 内部实现），而且 MCP 可能被装到
另一台机器上只连远端服务。因此这里只认 `RAG_KB_BASE_URL` / `RAG_KB_TIMEOUT` 两个变量，
默认值对齐本机 `docker compose up` 后的 `http://localhost:8000`。
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30.0

# 单测注入点：httpx.MockTransport 可替换真实网络，同时不改变工具函数的调用路径。
# 生产代码不读取它，保持「工具 → _request → HTTP」这一条链路可被完整覆盖。
_transport: httpx.AsyncBaseTransport | None = None

# mcp SDK 2.x 把 v1 的 FastMCP 更名为 MCPServer；依赖已固定 >=2,<3，
# 避免同一份代码在 1.x/2.x 两套 API 之间漂移。
mcp = MCPServer("rag-kb")


def _base_url() -> str:
    return os.environ.get("RAG_KB_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _timeout() -> float:
    raw = os.environ.get("RAG_KB_TIMEOUT")
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT


async def _request(method: str, path: str, *, json: dict[str, Any] | None = None) -> Any:
    """调用 REST 并把失败翻译成模型能读懂的文本。

    不吞异常：错误信息里保留 HTTP 状态码与服务端 detail，让模型（或用户）能判断
    是「知识库不存在」「服务没启动」还是「参数不合法」，而不是看到一句 "request failed"。
    """
    url = f"{_base_url()}{path}"
    try:
        async with httpx.AsyncClient(timeout=_timeout(), transport=_transport) as client:
            response = await client.request(method, url, json=json)
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"无法连接 RAG 知识库服务（{url}）。请确认服务已启动，"
            "或通过 RAG_KB_BASE_URL 指定正确地址。"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"请求超时：{url}（{_timeout()}s）") from exc

    if response.status_code >= 400:
        detail = response.text
        with contextlib.suppress(ValueError):
            detail = response.json().get("detail", detail)
        raise RuntimeError(f"RAG 服务返回 {response.status_code}：{detail}")

    if response.status_code == 204:
        return None
    return response.json()


# 返回值统一用对象而不是裸数组：MCP 的结构化输出要求顶层是对象，裸 list 会被 SDK
# 包成 {"result": [...]}，空结果时连文本块都没有。显式命名既稳定又让模型知道字段含义。
@mcp.tool()
async def list_knowledge_bases() -> dict[str, Any]:
    """列出所有知识库。

    返回 `knowledge_bases`，每项含 `id` / `name` / `description`。调用方应先列出再选择，
    不要猜测 kb_id；检索与提问工具都要求真实的 id。
    """
    return {"knowledge_bases": await _request("GET", "/api/kbs")}


@mcp.tool()
async def search_knowledge_base(
    kb_id: str,
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
) -> dict[str, Any]:
    """在指定知识库中检索片段，返回带来源的命中列表。

    参数：
    - kb_id：先用 `list_knowledge_bases` 获取，不要猜。
    - mode：`vector` / `keyword` / `graph` / `hybrid`（默认）/ `hybrid_weighted`。
    - top_k：1–50。

    返回 `hits` 与 `count`；每条命中含 `chunk_id`、`doc_name`、`page_no`、`heading_path`、
    `snippet`、`score`。这是「只看证据」的入口；需要成段回答时用 `ask_knowledge_base`。
    """
    hits = await _request(
        "POST",
        f"/api/kbs/{kb_id}/search",
        json={"query": query, "mode": mode, "top_k": top_k},
    )
    return {"hits": hits, "count": len(hits)}


@mcp.tool()
async def ask_knowledge_base(
    kb_id: str,
    question: str,
    mode: str = "hybrid",
    top_k: int = 8,
) -> dict[str, Any]:
    """基于知识库证据回答问题，返回答案与引用。

    返回 `answer`、`citations`、`no_evidence`、`attempted_channels`。
    `no_evidence=true` 表示证据不足、服务已拒答；此时应换关键词或换库重试，
    而不是把拒答文本当作答案转述。
    """
    return await _request(
        "POST",
        f"/api/kbs/{kb_id}/ask",
        json={"question": question, "mode": mode, "top_k": top_k},
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
