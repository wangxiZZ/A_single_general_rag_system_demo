"""检索路由单测：注入 fake SearchService，不连任何存储。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.config import Settings
from rag_kb.domain.models import ChunkLevel, SearchHit, SearchMode
from rag_kb.main import create_app
from tests.unit.fakes import FakeMetadataRepository


class FakeSearchService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def search(
        self,
        kb_id: UUID,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 20,
        rerank: bool = False,
    ) -> list[SearchHit]:
        self.calls.append(
            {"kb_id": kb_id, "query": query, "mode": mode, "top_k": top_k, "rerank": rerank}
        )
        return [
            SearchHit(
                chunk_id="a" * 32,
                doc_id=uuid4(),
                doc_name="manual.pdf",
                level=ChunkLevel.CHILD,
                score=0.75,
                channels=("vector", "keyword"),
                snippet="年假政策规定……",
                heading_path=("Manual", "3.2"),
                page_no=3,
                debug={"rank_per_channel": {"vector": 1, "keyword": 1}},
            )
        ]


@pytest.fixture
def fake_service() -> FakeSearchService:
    return FakeSearchService()


@pytest.fixture
def app(tmp_path: Path, fake_service: FakeSearchService):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    return create_app(
        settings=settings,
        repository=FakeMetadataRepository(),
        start_worker=False,
        search_service=fake_service,  # type: ignore[arg-type]
    )


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        yield http


async def test_search_returns_hits_with_debug(
    client: AsyncClient, fake_service: FakeSearchService
) -> None:
    kb_id = uuid4()
    response = await client.post(
        f"/api/kbs/{kb_id}/search",
        json={"query": "年假", "mode": "hybrid", "top_k": 5, "rerank": True},
    )
    assert response.status_code == 200
    hit = response.json()[0]
    assert hit["chunk_id"] == "a" * 32
    assert hit["doc_name"] == "manual.pdf"
    assert hit["channels"] == ["vector", "keyword"]
    assert hit["level"] == "child"
    assert hit["debug"]["rank_per_channel"]["vector"] == 1

    call = fake_service.calls[0]
    assert call["kb_id"] == kb_id
    assert call["mode"] is SearchMode.HYBRID
    assert call["top_k"] == 5
    assert call["rerank"] is True


async def test_search_defaults_and_validation(
    client: AsyncClient, fake_service: FakeSearchService
) -> None:
    kb_id = uuid4()
    response = await client.post(f"/api/kbs/{kb_id}/search", json={"query": "年假"})
    assert response.status_code == 200
    assert fake_service.calls[0]["mode"] is SearchMode.HYBRID
    assert fake_service.calls[0]["top_k"] == 20

    assert (
        await client.post(f"/api/kbs/{kb_id}/search", json={"query": "x", "mode": "bad"})
    ).status_code == 422
    assert (
        await client.post(f"/api/kbs/{kb_id}/search", json={"query": "x", "top_k": 100})
    ).status_code == 422
