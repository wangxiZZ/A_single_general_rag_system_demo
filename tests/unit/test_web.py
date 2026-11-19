"""Web 管理页单测：fake 仓储 + 临时目录，不连数据库。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.config import Settings
from rag_kb.domain.models import DocumentStatus
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


async def _create_kb(client: AsyncClient, name: str = "Web 测试库") -> str:
    response = await client.post("/web/kbs", data={"name": name, "description": "demo"})
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/web/kbs/")
    return location.rsplit("/", 1)[-1]


async def test_root_redirects_to_web(client: AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 303
    assert response.headers["location"] == "/web/kbs"


async def test_index_and_create_kb(client: AsyncClient) -> None:
    index = await client.get("/web/kbs")
    assert index.status_code == 200
    assert "知识库" in index.text

    kb_id = await _create_kb(client, "产品手册")
    detail = await client.get(f"/web/kbs/{kb_id}")
    assert detail.status_code == 200
    assert "产品手册" in detail.text
    assert "上传文档" in detail.text


async def test_create_kb_shows_validation_error(client: AsyncClient) -> None:
    response = await client.post("/web/kbs", data={"name": "   "})
    assert response.status_code == 400
    assert "名称不能为空" in response.text


async def test_upload_document_and_status_partial(
    client: AsyncClient, repo: FakeMetadataRepository
) -> None:
    kb_id = await _create_kb(client)
    response = await client.post(
        f"/web/kbs/{kb_id}/documents",
        files={"file": ("a.pdf", b"%PDF-1.4\nweb test", "application/pdf")},
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/web/kbs/{kb_id}"

    detail = await client.get(f"/web/kbs/{kb_id}")
    assert "a.pdf" in detail.text
    assert "等待处理" in detail.text

    doc = next(iter(repo.documents.values()))
    status = await client.get(f"/web/kbs/{kb_id}/documents/{doc.id}/status")
    assert status.status_code == 200
    assert 'hx-trigger="every 2s"' in status.text
    assert "a.pdf" in status.text

    # 终态后不再带轮询属性
    repo.documents[doc.id] = replace(doc, status=DocumentStatus.READY)
    ready = await client.get(f"/web/kbs/{kb_id}/documents/{doc.id}/status")
    assert 'hx-trigger="every 2s"' not in ready.text
    assert "可检索" in ready.text


async def test_duplicate_upload_shows_error(client: AsyncClient) -> None:
    kb_id = await _create_kb(client)
    files = {"file": ("a.pdf", b"%PDF-1.4\nsame", "application/pdf")}
    first = await client.post(f"/web/kbs/{kb_id}/documents", files=files)
    assert first.status_code == 303

    second = await client.post(
        f"/web/kbs/{kb_id}/documents",
        files={"file": ("b.pdf", b"%PDF-1.4\nsame", "application/pdf")},
    )
    assert second.status_code == 400
    assert "已存在相同文件" in second.text


async def test_delete_document_returns_documents_partial(
    client: AsyncClient, repo: FakeMetadataRepository
) -> None:
    kb_id = await _create_kb(client)
    await client.post(
        f"/web/kbs/{kb_id}/documents",
        files={"file": ("a.pdf", b"%PDF-1.4\nx", "application/pdf")},
    )
    doc = next(iter(repo.documents.values()))

    response = await client.post(f"/web/kbs/{kb_id}/documents/{doc.id}/delete")
    assert response.status_code == 200
    assert 'id="documents"' in response.text
    assert "a.pdf" not in response.text
    assert repo.documents == {}


async def test_delete_kb(client: AsyncClient, repo: FakeMetadataRepository) -> None:
    kb_id = await _create_kb(client)
    response = await client.post(f"/web/kbs/{kb_id}/delete")
    assert response.status_code == 303
    assert response.headers["location"] == "/web/kbs"
    assert repo.kbs == {}


async def test_static_assets_are_served(client: AsyncClient) -> None:
    css = await client.get("/static/app.css")
    js = await client.get("/static/htmx.min.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert "svg" not in js.text[:20]  # 确认返回的是脚本而不是 404 页面
