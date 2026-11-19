"""向量化用例薄封装单测。"""

from __future__ import annotations

import pytest

from rag_kb.application.embedding import embed_texts
from rag_kb.domain.errors import ConfigurationError


class FakeEmbedder:
    dimension = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


async def test_requires_embedder_when_texts_present() -> None:
    with pytest.raises(ConfigurationError, match="EMBED_API_KEY"):
        await embed_texts(None, ["text"])


async def test_empty_texts_do_not_require_embedder() -> None:
    assert await embed_texts(None, []) == []


async def test_delegates_to_embedder() -> None:
    embedder = FakeEmbedder()
    vectors = await embed_texts(embedder, ["a", "b"])
    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert embedder.calls == [["a", "b"]]
