"""问答 API/页面单测：注入 fake SearchService 与 QAService。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.application.qa_service import NO_EVIDENCE_ANSWER, AnswerResult, Evidence
from rag_kb.config import Settings
from rag_kb.domain.models import (
    Chunk,
    ChunkKind,
    ChunkLevel,
    SearchHit,
    SearchMode,
)
from rag_kb.main import create_app
from tests.unit.fakes import FakeMetadataRepository

DOC_ID = uuid4()


class FakeSearchService:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    async def search(
        self, kb_id: UUID, query: str, *, mode=SearchMode.HYBRID, top_k=20, rerank=False
    ):
        self.calls.append({"query": query, "mode": mode, "top_k": top_k})
        return self.hits


class FakeQAService:
    def __init__(self, result: AnswerResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def answer(
        self,
        kb_id: UUID,
        question: str,
        hits,
        *,
        attempted_channels=None,
        query_rewrites=None,
    ) -> AnswerResult:
        self.calls.append(
            {
                "question": question,
                "hits": hits,
                "attempted_channels": attempted_channels,
                "query_rewrites": query_rewrites,
            }
        )
        return self.result


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id="c" * 32,
        doc_id=DOC_ID,
        doc_name="manual.pdf",
        level=ChunkLevel.CHILD,
        score=0.9,
        channels=("vector", "keyword"),
        heading_path=("Manual", "3.2"),
        page_no=3,
    )


def _body() -> Chunk:
    return Chunk(
        id="p" * 32,
        kb_id=uuid4(),
        doc_id=DOC_ID,
        level=ChunkLevel.PARENT,
        kind=ChunkKind.TEXT,
        content="年假政策规定员工每年享有带薪年假。",
        seq=0,
        char_start=0,
        char_end=20,
        token_count=20,
        chunker_version="v1",
        heading_path=("Manual", "3.2"),
        page_no=3,
    )


def _answer_result() -> AnswerResult:
    hit = _hit()
    body = _body()
    return AnswerResult(
        answer="年假为带薪假 [1]。",
        citations=[1],
        evidence=[Evidence(index=1, hit=hit, body=body, text=body.content, token_count=20)],
        attempted_channels=["vector", "keyword", "graph"],
    )


@pytest.fixture
def services():
    search = FakeSearchService([_hit()])
    qa = FakeQAService(_answer_result())
    return search, qa


@pytest.fixture
def app(tmp_path: Path, services):
    search, qa = services
    settings = Settings(_env_file=None, data_dir=tmp_path)
    return create_app(
        settings=settings,
        repository=FakeMetadataRepository(),
        start_worker=False,
        search_service=search,  # type: ignore[arg-type]
        qa_service=qa,  # type: ignore[arg-type]
    )


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        yield http


async def test_ask_returns_answer_and_citations(client: AsyncClient, services) -> None:
    search, qa = services
    kb = await client.post("/api/kbs", json={"name": "ask-kb"})
    kb_id = kb.json()["id"]

    response = await client.post(
        f"/api/kbs/{kb_id}/ask",
        json={"question": "年假怎么休？", "mode": "hybrid", "top_k": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "年假为带薪假 [1]。"
    assert payload["no_evidence"] is False
    assert payload["attempted_channels"] == ["vector", "keyword", "graph"]
    assert payload["citations"][0]["doc_name"] == "manual.pdf"
    assert payload["citations"][0]["page_no"] == 3
    assert search.calls[0]["mode"] is SearchMode.HYBRID
    assert qa.calls[0]["attempted_channels"] == ["vector", "keyword", "graph"]


async def test_ask_no_evidence(client: AsyncClient, services) -> None:
    _, qa = services
    qa.result = AnswerResult(
        answer=NO_EVIDENCE_ANSWER,
        citations=[],
        evidence=[],
        no_evidence=True,
        attempted_channels=["vector"],
        query_rewrites=["年假 申请"],
    )
    kb_id = (await client.post("/api/kbs", json={"name": "ask-kb"})).json()["id"]

    payload = (
        await client.post(
            f"/api/kbs/{kb_id}/ask",
            json={"question": "不存在的政策", "mode": "vector"},
        )
    ).json()
    assert payload["no_evidence"] is True
    assert payload["citations"] == []
    assert payload["query_rewrites"] == ["年假 申请"]


async def test_ask_web_page(client: AsyncClient, services) -> None:
    kb_id = (await client.post("/api/kbs", json={"name": "ask-web"})).json()["id"]
    page = await client.get(f"/web/kbs/{kb_id}/ask")
    assert page.status_code == 200
    assert "提问" in page.text

    answer = await client.post(
        f"/web/kbs/{kb_id}/ask",
        data={"question": "年假怎么休？", "mode": "hybrid"},
    )
    assert answer.status_code == 200
    assert "年假为带薪假" in answer.text
    assert "manual.pdf" in answer.text
