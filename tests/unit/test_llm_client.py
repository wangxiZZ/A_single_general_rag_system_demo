"""OpenAI 兼容客户端单测：用 httpx.MockTransport，不发真实网络请求。"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from rag_kb.domain.errors import EmbeddingError, LlmError
from rag_kb.domain.models import ChatMessage
from rag_kb.infra.llm.openai_compat import (
    OpenAICompatEmbedder,
    OpenAICompatLLM,
    OpenAICompatVLM,
)

BASE = "https://example.test/v1"


def _http(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _llm(http: httpx.AsyncClient, **kwargs: Any) -> OpenAICompatLLM:
    return OpenAICompatLLM(
        base_url=BASE,
        api_key="key",
        model="model",
        timeout_s=5,
        max_concurrency=2,
        backoff_base_s=0,
        client=http,
        **kwargs,
    )


async def test_llm_chat_payload_and_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    http = _http(handler)
    llm = _llm(http)
    text = await llm.chat([ChatMessage(role="user", content="hi")], max_tokens=10)

    assert text == "ok"
    assert seen["body"]["model"] == "model"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert llm.last_usage == {"prompt_tokens": 2, "completion_tokens": 1}
    await http.aclose()


async def test_llm_retries_5xx_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="temporary")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    http = _http(handler)
    llm = _llm(http, max_retries=1)
    assert await llm.chat([ChatMessage(role="user", content="hi")]) == "ok"
    assert calls["n"] == 2
    await http.aclose()


async def test_llm_does_not_retry_4xx() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad model")

    http = _http(handler)
    llm = _llm(http, max_retries=3)
    with pytest.raises(LlmError, match="HTTP 400"):
        await llm.chat([ChatMessage(role="user", content="hi")])
    assert calls["n"] == 1
    await http.aclose()


async def test_structured_json_strips_code_fence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '```json\n{"a": 1}\n```'}}]},
        )

    http = _http(handler)
    llm = _llm(http)
    assert await llm.structured_json([ChatMessage(role="user", content="x")], {}) == {"a": 1}
    await http.aclose()


async def test_structured_json_rejects_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    http = _http(handler)
    llm = _llm(http)
    with pytest.raises(LlmError, match="不是合法 JSON"):
        await llm.structured_json([ChatMessage(role="user", content="x")], {})
    await http.aclose()


async def test_embedder_batches_and_preserves_order() -> None:
    requests: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        requests.append(inputs)
        data = [
            {"index": index, "embedding": [float(text[-1])] * 4}
            for index, text in enumerate(inputs)
        ]
        return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 3}})

    http = _http(handler)
    embedder = OpenAICompatEmbedder(
        base_url=BASE,
        api_key="key",
        model="embed",
        timeout_s=5,
        max_concurrency=2,
        backoff_base_s=0,
        dimension=4,
        batch_size=2,
        client=http,
    )
    vectors = await embedder.embed_documents(["a1", "b2", "c3"])

    assert len(requests) == 2
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]
    assert embedder.dimension == 4
    assert (await embedder.embed_query("q9"))[0] == 9.0
    await http.aclose()


async def test_embedder_rejects_dimension_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0, 1.0]}]})

    http = _http(handler)
    embedder = OpenAICompatEmbedder(
        base_url=BASE,
        api_key="key",
        model="embed",
        timeout_s=5,
        max_concurrency=1,
        backoff_base_s=0,
        dimension=4,
        client=http,
    )
    with pytest.raises(EmbeddingError, match="维度不一致"):
        await embedder.embed_documents(["text"])
    await http.aclose()


async def test_embedder_rejects_empty_text() -> None:
    http = _http(lambda request: httpx.Response(500))
    embedder = OpenAICompatEmbedder(
        base_url=BASE,
        api_key="key",
        model="embed",
        timeout_s=5,
        max_concurrency=1,
        backoff_base_s=0,
        dimension=4,
        client=http,
    )
    with pytest.raises(EmbeddingError, match="空文本"):
        await embedder.embed_documents(["  "])
    await http.aclose()


async def test_embedder_rejects_overlong_text() -> None:
    http = _http(lambda request: httpx.Response(500))
    embedder = OpenAICompatEmbedder(
        base_url=BASE,
        api_key="key",
        model="embed",
        timeout_s=5,
        max_concurrency=1,
        backoff_base_s=0,
        dimension=4,
        max_chars=5,
        client=http,
    )
    with pytest.raises(EmbeddingError, match="超过 5 字符"):
        await embedder.embed_documents(["123456"])
    await http.aclose()


async def test_vlm_sends_data_url_and_returns_text() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "一张红色图片"}}]})

    http = _http(handler)
    vlm = OpenAICompatVLM(
        base_url=BASE,
        api_key="key",
        model="vl",
        timeout_s=5,
        max_concurrency=1,
        backoff_base_s=0,
        client=http,
    )
    text = await vlm.describe_image(b"\x89PNG", mime="image/png", prompt="描述")

    assert text == "一张红色图片"
    content = seen["body"]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["text"] == "描述"
    await http.aclose()
