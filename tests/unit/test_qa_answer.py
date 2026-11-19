"""回答生成单测：fake repo + fake LLM。"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from rag_kb.application.qa_service import ANSWER_SCHEMA, QAService
from rag_kb.domain.errors import ConfigurationError, LlmError
from rag_kb.domain.models import (
    Chunk,
    ChunkKind,
    ChunkLevel,
    SearchHit,
)

KB_ID = uuid4()
DOC_ID = uuid4()


class FakeRepo:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}

    async def get_chunks(self, kb_id: UUID, chunk_ids) -> list[Chunk]:
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]


class FakeLLM:
    def __init__(self, payload: dict | Exception) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    async def structured_json(self, messages, schema):
        self.calls.append((messages[0].content, schema))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    async def chat(self, messages, *, temperature: float = 0.0, max_tokens=None) -> str:
        return ""


def _chunks() -> tuple[Chunk, Chunk]:
    parent = Chunk(
        id="p" * 32,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        level=ChunkLevel.PARENT,
        kind=ChunkKind.TEXT,
        content="父块：年假政策规定员工每年享有带薪年假。",
        seq=0,
        char_start=0,
        char_end=20,
        token_count=20,
        chunker_version="v1",
        heading_path=("Manual", "3.2"),
        page_no=3,
    )
    child = Chunk(
        id="c" * 32,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        parent_id=parent.id,
        level=ChunkLevel.CHILD,
        kind=ChunkKind.TEXT,
        content="子块",
        seq=1,
        char_start=0,
        char_end=2,
        token_count=2,
        chunker_version="v1",
        heading_path=("Manual", "3.2"),
        page_no=3,
    )
    return parent, child


def _hit(child_id: str) -> SearchHit:
    return SearchHit(
        chunk_id=child_id,
        doc_id=DOC_ID,
        doc_name="a.pdf",
        level=ChunkLevel.CHILD,
        score=0.9,
        channels=("vector", "keyword"),
        heading_path=("Manual", "3.2"),
        page_no=3,
    )


def _service(repo: FakeRepo, llm=None) -> QAService:
    return QAService(
        repo,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        max_context_tokens=100,
        parent_max_tokens=10,
        chars_per_token=1.0,
    )


async def test_answer_uses_backfilled_evidence_and_returns_citations() -> None:
    parent, child = _chunks()
    llm = FakeLLM({"answer": "年假为带薪假 [1]。", "citations": [1]})
    service = _service(FakeRepo([parent, child]), llm)

    result = await service.answer(KB_ID, "年假怎么休？", [_hit(child.id)])

    assert result.no_evidence is False
    assert result.answer == "年假为带薪假 [1]。"
    assert result.citations == [1]
    assert result.attempted_channels == ["keyword", "vector"]
    prompt, schema = llm.calls[0]
    assert "年假怎么休？" in prompt
    assert parent.content in prompt
    assert "《a.pdf》第 3 页 · Manual > 3.2" in prompt
    assert schema == ANSWER_SCHEMA


async def test_no_hits_returns_refusal_without_llm() -> None:
    llm = FakeLLM({"answer": "x", "citations": []})
    result = await _service(FakeRepo([]), llm).answer(KB_ID, "问题", [])
    assert result.no_evidence is True
    assert result.citations == []
    assert llm.calls == []


async def test_missing_evidence_returns_refusal() -> None:
    llm = FakeLLM({"answer": "x", "citations": []})
    result = await _service(FakeRepo([]), llm).answer(KB_ID, "问题", [_hit("z" * 32)])
    assert result.no_evidence is True
    assert llm.calls == []


async def test_missing_llm_raises_when_evidence_exists() -> None:
    parent, child = _chunks()
    service = _service(FakeRepo([parent, child]), None)
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        await service.answer(KB_ID, "问题", [_hit(child.id)])


async def test_empty_answer_raises() -> None:
    parent, child = _chunks()
    service = _service(FakeRepo([parent, child]), FakeLLM({"answer": " ", "citations": []}))
    with pytest.raises(LlmError, match="空回答"):
        await service.answer(KB_ID, "问题", [_hit(child.id)])


async def test_non_integer_citations_are_dropped() -> None:
    parent, child = _chunks()
    service = _service(
        FakeRepo([parent, child]),
        FakeLLM({"answer": "答案 [1]", "citations": [1, "2", None]}),
    )
    result = await service.answer(KB_ID, "问题", [_hit(child.id)])
    assert result.citations == [1]
