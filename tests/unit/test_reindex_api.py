"""reindex 路由单测：fake ReindexService。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.config import Settings
from rag_kb.main import create_app
from tests.unit.fakes import FakeMetadataRepository


class FakeReindexService:
    def __init__(self) -> None:
        self.calls: list[UUID] = []
        self.started = asyncio.Event()

    async def reindex_kb(self, kb_id: UUID) -> list:
        self.calls.append(kb_id)
        self.started.set()
        return []


@pytest.fixture
def fake_reindex() -> FakeReindexService:
    return FakeReindexService()


@pytest.fixture
def app(tmp_path: Path, fake_reindex: FakeReindexService):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    return create_app(
        settings=settings,
        repository=FakeMetadataRepository(),
        start_worker=False,
        reindex_service=fake_reindex,  # type: ignore[arg-type]
    )


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        yield http


async def test_reindex_returns_202_and_runs_task(
    client: AsyncClient, fake_reindex: FakeReindexService
) -> None:
    created = await client.post("/api/kbs", json={"name": "reindex-kb"})
    kb_id = created.json()["id"]

    response = await client.post(f"/api/kbs/{kb_id}/reindex")
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    assert task_id

    await asyncio.wait_for(fake_reindex.started.wait(), timeout=1)
    assert [str(call) for call in fake_reindex.calls] == [kb_id]

    status_response = await client.get(f"/api/kbs/{kb_id}/reindex/{task_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] in {"running", "done"}
    assert (await client.get(f"/api/kbs/{kb_id}/reindex/unknown")).status_code == 404


async def test_reindex_missing_kb_returns_404(client: AsyncClient) -> None:
    from uuid import uuid4

    response = await client.post(f"/api/kbs/{uuid4()}/reindex")
    assert response.status_code == 404
