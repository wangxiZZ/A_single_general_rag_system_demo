"""LLM listwise 重排单测。"""

from __future__ import annotations

from uuid import uuid4

from rag_kb.domain.models import ChunkLevel, SearchHit
from rag_kb.infra.llm.rerank import LLMListwiseReranker

DOC_ID = uuid4()


class FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def structured_json(self, messages, schema):
        return self.payload

    async def chat(self, messages, *, temperature: float = 0.0, max_tokens=None) -> str:
        return ""


def _hit(chunk_id: str, snippet: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=DOC_ID,
        doc_name="a.pdf",
        level=ChunkLevel.CHILD,
        score=0.5,
        channels=("vector",),
        snippet=snippet,
    )


async def test_llm_rerank_reorders_and_keeps_metadata() -> None:
    candidates = [_hit("a" * 32, "A"), _hit("b" * 32, "B"), _hit("c" * 32, "C")]
    reranker = LLMListwiseReranker(FakeLLM({"order": [3, 1, 2]}))

    result = await reranker.rerank("q", candidates, top_k=2)

    assert [hit.chunk_id for hit in result] == ["c" * 32, "a" * 32]
    assert result[0].debug["rerank_rank"] == 1
    assert result[0].doc_name == "a.pdf"


async def test_invalid_indices_are_ignored_and_missing_appended() -> None:
    candidates = [_hit("a" * 32, "A"), _hit("b" * 32, "B")]
    reranker = LLMListwiseReranker(FakeLLM({"order": [9, 2]}))

    result = await reranker.rerank("q", candidates, top_k=5)

    assert [hit.chunk_id for hit in result] == ["b" * 32, "a" * 32]


async def test_empty_candidates() -> None:
    assert await LLMListwiseReranker(FakeLLM({"order": []})).rerank("q", [], 5) == []
