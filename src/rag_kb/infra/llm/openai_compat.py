"""OpenAI 兼容模型客户端。

三类能力共用一套 HTTP 重试/超时/并发逻辑，但模型、端点与 Key 各自独立：

- ``OpenAICompatLLM``      生成与图谱抽取（DeepSeek）
- ``OpenAICompatEmbedder`` 文本向量化（Qwen MaaS，固定维度校验）
- ``OpenAICompatVLM``      图片描述（Qwen MaaS）

重试只针对 429、5xx、超时与传输错误；4xx（模型名错误、参数错误）立即失败，避免把
配置错误重试成几十秒的等待。JSON 解析失败抛 ``LlmError``，由调用方决定丢弃该批，
而不是在客户端里无限重试。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
from collections.abc import Sequence
from typing import Any

import httpx

from rag_kb.domain.errors import EmbeddingError, LlmError
from rag_kb.domain.models import ChatMessage

logger = logging.getLogger(__name__)

DEFAULT_VLM_PROMPT = (
    "请用简洁的中文描述这张图片的内容，重点说明图表、表格或流程中的关键信息；看不清的内容不要编造。"
)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _chat_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError(f"模型响应缺少 choices/content: {payload}") from exc

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        raise LlmError("模型响应 content 类型不支持")
    if not text.strip():
        raise LlmError("模型返回空内容")
    return text


class _OpenAICompatBase:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float,
        max_concurrency: int,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency 必须 >= 1")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_retries = max(0, max_retries)
        self._backoff_base_s = max(0.0, backoff_base_s)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self.last_usage: dict[str, int] | None = None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(
        self, path: str, payload: dict[str, Any], error_cls: type[Exception]
    ) -> dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.post(url, json=payload, headers=headers)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    last_error = error_cls(
                        f"模型服务暂不可用 HTTP {response.status_code}: {response.text[:300]}"
                    )
                    if attempt < self._max_retries:
                        await self._sleep(attempt)
                        continue
                    raise last_error
                if response.status_code >= 400:
                    raise error_cls(
                        f"模型请求失败 HTTP {response.status_code}: {response.text[:300]}"
                    )
                try:
                    return response.json()
                except ValueError as exc:
                    raise error_cls("模型返回的不是合法 JSON") from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = error_cls(f"模型请求失败: {exc}")
                if attempt < self._max_retries:
                    await self._sleep(attempt)
                    continue
                raise last_error from exc

        raise last_error or error_cls("模型请求失败")

    async def _sleep(self, attempt: int) -> None:
        delay = self._backoff_base_s * (2**attempt)
        if delay > 0:
            await asyncio.sleep(delay + random.uniform(0, self._backoff_base_s))

    def _chat_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload


class OpenAICompatLLM(_OpenAICompatBase):
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        data = await self._post(
            "chat/completions",
            self._chat_payload(messages, temperature=temperature, max_tokens=max_tokens),
            LlmError,
        )
        self.last_usage = data.get("usage")
        logger.info("llm usage=%s", self.last_usage)
        return _chat_content(data)

    async def structured_json(
        self, messages: Sequence[ChatMessage], schema: dict[str, Any]
    ) -> dict[str, Any]:
        system = ChatMessage(
            role="system",
            content=(
                "只输出一个 JSON 对象，不要 markdown 代码块或解释文字。"
                f"JSON Schema: {json.dumps(schema, ensure_ascii=False)}"
            ),
        )
        content = await self.chat([system, *messages], temperature=0.0)
        try:
            data = json.loads(_strip_code_fence(content))
        except json.JSONDecodeError as exc:
            raise LlmError(f"模型输出不是合法 JSON: {content[:500]}") from exc
        if not isinstance(data, dict):
            raise LlmError("模型输出的 JSON 不是对象")
        return data


class OpenAICompatEmbedder(_OpenAICompatBase):
    # bge-m3 上限约 8192 tokens；按 1 token≈1.5 汉字折算，8000 字符留出安全余量。
    DEFAULT_MAX_CHARS = 8000

    def __init__(
        self,
        *,
        dimension: int,
        batch_size: int = 16,
        max_chars: int = DEFAULT_MAX_CHARS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if dimension < 1:
            raise ValueError("dimension 必须 >= 1")
        if batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")
        if max_chars < 1:
            raise ValueError("max_chars 必须 >= 1")
        self._dimension = dimension
        self._batch_size = batch_size
        self._max_chars = max_chars

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("embedding 输入包含空文本")
        if any(len(text) > self._max_chars for text in texts):
            raise EmbeddingError(f"embedding 输入超过 {self._max_chars} 字符，请先分块或截断")

        batches = [
            list(texts[index : index + self._batch_size])
            for index in range(0, len(texts), self._batch_size)
        ]
        results = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))
        return [vector for batch in results for vector in batch]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        data = await self._post(
            "embeddings", {"model": self._model, "input": texts}, EmbeddingError
        )
        self.last_usage = data.get("usage")
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        if len(items) != len(texts):
            raise EmbeddingError(
                f"embedding 返回数量不一致: expected={len(texts)} actual={len(items)}"
            )
        vectors: list[list[float]] = []
        for item in items:
            vector = item.get("embedding")
            if not isinstance(vector, list) or len(vector) != self._dimension:
                raise EmbeddingError(
                    f"embedding 维度不一致: expected={self._dimension} "
                    f"actual={len(vector) if isinstance(vector, list) else 'n/a'}"
                )
            vectors.append([float(value) for value in vector])
        return vectors


class OpenAICompatVLM(_OpenAICompatBase):
    async def describe_image(
        self, image: bytes, *, mime: str = "image/png", prompt: str = ""
    ) -> str:
        if not image:
            raise LlmError("图片为空")
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt or DEFAULT_VLM_PROMPT},
                    ],
                }
            ],
            "max_tokens": 300,
        }
        data = await self._post("chat/completions", payload, LlmError)
        self.last_usage = data.get("usage")
        logger.info("vlm usage=%s", self.last_usage)
        return _chat_content(data)
