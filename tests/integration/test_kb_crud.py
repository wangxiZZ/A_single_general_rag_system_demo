"""知识库/文档 CRUD 集成测试：真实 PostgreSQL + 临时 data_dir。"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.config import Settings, get_settings
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    dsn = get_settings().pg_dsn
    apply_migrations(dsn)
    return Settings(
        _env_file=None,
        pg_dsn=dsn,
        data_dir=tmp_path,
        max_upload_mb=1,
    )


@pytest.fixture
def app(settings: Settings):
    return create_app(settings=settings, start_worker=False)


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        yield http


async def _create_kb(client: AsyncClient) -> str:
    response = await client.post("/api/kbs", json={"name": f"it-{uuid4().hex[:8]}"})
    assert response.status_code == 201
    return response.json()["id"]


async def test_kb_document_crud(client: AsyncClient, app, tmp_path: Path) -> None:
    kb_id = await _create_kb(client)
    try:
        listed = await client.get("/api/kbs")
        assert kb_id in [item["id"] for item in listed.json()]

        upload = await client.post(
            f"/api/kbs/{kb_id}/documents",
            files={"file": ("a.pdf", b"%PDF-1.7\nintegration", "application/pdf")},
        )
        assert upload.status_code == 201
        doc = upload.json()
        assert doc["status"] == "pending"
        assert (tmp_path / "uploads" / kb_id / doc["id"]).is_dir()

        detail = await client.get(f"/api/kbs/{kb_id}/documents/{doc['id']}")
        assert detail.status_code == 200
        assert detail.json()["filename"] == "a.pdf"

        assert (await client.delete(f"/api/kbs/{kb_id}/documents/{doc['id']}")).status_code == 204
        assert (await client.get(f"/api/kbs/{kb_id}/documents")).json() == []
        assert not (tmp_path / "uploads" / kb_id / doc["id"]).exists()
    finally:
        with contextlib.suppress(Exception):
            await client.delete(f"/api/kbs/{kb_id}")

    assert (await client.get(f"/api/kbs/{kb_id}")).status_code == 404
    assert await app.state.repository.get_kb(kb_id) is None


async def test_duplicate_upload_returns_409(client: AsyncClient) -> None:
    kb_id = await _create_kb(client)
    payload = b"%PDF-1.7\nduplicate"
    try:
        first = await client.post(
            f"/api/kbs/{kb_id}/documents",
            files={"file": ("a.pdf", payload, "application/pdf")},
        )
        second = await client.post(
            f"/api/kbs/{kb_id}/documents",
            files={"file": ("b.pdf", payload, "application/pdf")},
        )
        assert second.status_code == 409
        assert second.json()["existing_document_id"] == first.json()["id"]
        assert len((await client.get(f"/api/kbs/{kb_id}/documents")).json()) == 1
    finally:
        with contextlib.suppress(Exception):
            await client.delete(f"/api/kbs/{kb_id}")


async def test_oversize_and_fake_docx_are_rejected(client: AsyncClient) -> None:
    kb_id = await _create_kb(client)
    try:
        oversize = await client.post(
            f"/api/kbs/{kb_id}/documents",
            files={"file": ("big.pdf", b"%PDF-" + b"x" * (1024 * 1024), "application/pdf")},
        )
        assert oversize.status_code == 413
        assert oversize.json()["code"] == "file_too_large"

        fake = await client.post(
            f"/api/kbs/{kb_id}/documents",
            files={"file": ("fake.docx", b"not-a-zip", "application/octet-stream")},
        )
        assert fake.status_code == 415
        assert fake.json()["code"] == "unsupported_file_type"

        assert (await client.get(f"/api/kbs/{kb_id}/documents")).json() == []
    finally:
        with contextlib.suppress(Exception):
            await client.delete(f"/api/kbs/{kb_id}")


async def test_delete_kb_cascades_db_and_files(client: AsyncClient, app, tmp_path: Path) -> None:
    kb_id = await _create_kb(client)
    upload = await client.post(
        f"/api/kbs/{kb_id}/documents",
        files={"file": ("a.pdf", b"%PDF-1.7\ncascade", "application/pdf")},
    )
    doc = upload.json()

    assert (await client.delete(f"/api/kbs/{kb_id}")).status_code == 204
    assert await app.state.repository.get_kb(kb_id) is None
    assert await app.state.repository.get_document(kb_id, doc["id"]) is None
    assert await app.state.repository.count_chunks(kb_id, doc["id"]) == 0
    assert not (tmp_path / "uploads" / kb_id).exists()
    assert not (tmp_path / "images" / kb_id).exists()
