"""HTTP 路由单测：fake 仓储 + 临时目录，不连数据库。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.config import Settings
from rag_kb.domain.models import BBox, Chunk, ChunkKind, ChunkLevel, DocumentStatus
from rag_kb.main import create_app
from tests.unit.fakes import FakeMetadataRepository


@pytest.fixture
def repo() -> FakeMetadataRepository:
    return FakeMetadataRepository()


@pytest.fixture
def app(tmp_path: Path, repo: FakeMetadataRepository):
    settings = Settings(_env_file=None, data_dir=tmp_path, max_upload_mb=1)
    return create_app(settings=settings, repository=repo, start_worker=False)


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        yield http


async def _create_kb(client: AsyncClient, name: str = "测试库") -> dict:
    response = await client.post("/api/kbs", json={"name": name})
    assert response.status_code == 201
    return response.json()


async def test_kb_crud(client: AsyncClient) -> None:
    created = await _create_kb(client, "产品手册")
    kb_id = created["id"]
    assert created["name"] == "产品手册"

    listed = await client.get("/api/kbs")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [kb_id]

    fetched = await client.get(f"/api/kbs/{kb_id}")
    assert fetched.status_code == 200
    assert fetched.json()["description"] == ""

    deleted = await client.delete(f"/api/kbs/{kb_id}")
    assert deleted.status_code == 204
    assert (await client.get(f"/api/kbs/{kb_id}")).status_code == 404


async def test_create_kb_rejects_blank_name(client: AsyncClient) -> None:
    response = await client.post("/api/kbs", json={"name": "   "})
    assert response.status_code == 422
    assert response.json()["code"] == "validation"


async def test_upload_list_get_delete_document(client: AsyncClient) -> None:
    kb = await _create_kb(client)
    response = await client.post(
        f"/api/kbs/{kb['id']}/documents",
        files={"file": ("a.pdf", b"%PDF-1.7\nhello", "application/pdf")},
    )
    assert response.status_code == 201
    doc = response.json()
    assert doc["status"] == "pending"
    assert doc["filename"] == "a.pdf"

    listed = await client.get(f"/api/kbs/{kb['id']}/documents")
    assert [item["id"] for item in listed.json()] == [doc["id"]]

    detail = await client.get(f"/api/kbs/{kb['id']}/documents/{doc['id']}")
    assert detail.status_code == 200
    assert detail.json()["id"] == doc["id"]

    deleted = await client.delete(f"/api/kbs/{kb['id']}/documents/{doc['id']}")
    assert deleted.status_code == 204
    assert (await client.get(f"/api/kbs/{kb['id']}/documents")).json() == []


async def test_list_chunks_returns_chunk_fields(
    client: AsyncClient, repo: FakeMetadataRepository
) -> None:
    kb = await _create_kb(client)
    upload = await client.post(
        f"/api/kbs/{kb['id']}/documents",
        files={"file": ("a.pdf", b"%PDF-1.7\nchunks", "application/pdf")},
    )
    doc_id = UUID(upload.json()["id"])
    repo.chunks[doc_id] = [
        Chunk(
            id="c" * 32,
            kb_id=UUID(kb["id"]),
            doc_id=doc_id,
            level=ChunkLevel.CHILD,
            kind=ChunkKind.IMAGE,
            content="【图片】一张图\n【上下文】前后文",
            seq=0,
            char_start=10,
            char_end=10,
            token_count=8,
            chunker_version="v1",
            heading_path=("Manual", "3.2"),
            page_no=1,
            page_to=1,
            bbox=BBox(1, 2, 101, 202),
            meta={"low_weight": True},
        )
    ]

    response = await client.get(f"/api/kbs/{kb['id']}/documents/{doc_id}/chunks")
    assert response.status_code == 200
    chunk = response.json()[0]
    assert chunk["level"] == "child"
    assert chunk["kind"] == "image"
    assert chunk["heading_path"] == ["Manual", "3.2"]
    assert chunk["bbox"] == {"x0": 1.0, "y0": 2.0, "x1": 101.0, "y1": 202.0}
    assert chunk["meta"] == {"low_weight": True}

    assert (await client.get(f"/api/kbs/{kb['id']}/documents/{uuid4()}/chunks")).status_code == 404


async def test_retry_failed_document(client: AsyncClient, repo: FakeMetadataRepository) -> None:
    kb = await _create_kb(client)
    upload = await client.post(
        f"/api/kbs/{kb['id']}/documents",
        files={"file": ("a.pdf", b"%PDF-1.7\nretry", "application/pdf")},
    )
    doc_id = UUID(upload.json()["id"])
    doc = repo.documents[doc_id]
    repo.documents[doc_id] = replace(doc, status=DocumentStatus.FAILED, error="boom")

    response = await client.post(f"/api/kbs/{kb['id']}/documents/{doc_id}/retry")
    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert response.json()["error"] is None

    # 已回到 pending，不能再次重试
    assert (await client.post(f"/api/kbs/{kb['id']}/documents/{doc_id}/retry")).status_code == 422


async def test_fake_docx_is_rejected_by_magic(client: AsyncClient) -> None:
    kb = await _create_kb(client)
    response = await client.post(
        f"/api/kbs/{kb['id']}/documents",
        files={
            "file": (
                "fake.docx",
                b"not a zip",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_file_type"


async def test_duplicate_upload_returns_409_with_existing_id(client: AsyncClient) -> None:
    kb = await _create_kb(client)
    payload = b"%PDF-1.7\nsame"
    first = await client.post(
        f"/api/kbs/{kb['id']}/documents",
        files={"file": ("a.pdf", payload, "application/pdf")},
    )
    second = await client.post(
        f"/api/kbs/{kb['id']}/documents",
        files={"file": ("b.pdf", payload, "application/pdf")},
    )
    assert second.status_code == 409
    assert second.json()["existing_document_id"] == first.json()["id"]


async def test_oversized_upload_returns_413(client: AsyncClient) -> None:
    kb = await _create_kb(client)
    response = await client.post(
        f"/api/kbs/{kb['id']}/documents",
        files={"file": ("big.pdf", b"%PDF-" + b"x" * (1024 * 1024), "application/pdf")},
    )
    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"


async def test_upload_to_missing_kb_returns_404(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/kbs/{uuid4()}/documents",
        files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")},
    )
    assert response.status_code == 404


async def test_invalid_uuid_returns_422(client: AsyncClient) -> None:
    assert (await client.get("/api/kbs/not-a-uuid")).status_code == 422
