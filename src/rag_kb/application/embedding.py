"""向量化用例的薄封装。

把"没有配置 EMBED_API_KEY 时在写索引前失败"集中在一处：M3 的索引流程只要调用
``embed_texts``，就不会出现"先写了一半 chunk_lex/Milvus，再发现没 key"的脏状态。
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_kb.domain.errors import ConfigurationError
from rag_kb.domain.ports import Embedder


async def embed_texts(embedder: Embedder | None, texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    if embedder is None:
        raise ConfigurationError("EMBED_API_KEY 未配置，无法建立向量索引")
    return await embedder.embed_documents(texts)
