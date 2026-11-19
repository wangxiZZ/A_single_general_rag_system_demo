"""Web 管理页路由（Jinja2 + HTMX）。

与 REST 共用 application 用例；页面只负责渲染与表单转发。状态轮询由 HTMX 每 2 秒请求
一行 ``<tr>``，终态（ready/failed）返回的行不再带 ``hx-get``，轮询自然停止。
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile as StarletteUploadFile

from rag_kb.application.document_service import DocumentService
from rag_kb.application.kb_service import KbService
from rag_kb.application.upload_validation import validate_upload
from rag_kb.domain.errors import DomainError, NotFoundError

router = APIRouter(prefix="/web", tags=["web"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

STATUS_LABELS = {
    "pending": "等待处理",
    "parsing": "解析中",
    "chunked": "已分块",
    "indexing": "索引中",
    "ready": "可检索",
    "failed": "失败",
}


def get_kb_service(request: Request) -> KbService:
    return request.app.state.kb_service


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


KbDep = Annotated[KbService, Depends(get_kb_service)]
DocumentDep = Annotated[DocumentService, Depends(get_document_service)]


def _render(
    request: Request, name: str, context: dict[str, object], *, status_code: int = 200
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, name, {"status_labels": STATUS_LABELS, **context}, status_code=status_code
    )


@router.get("/kbs", response_class=HTMLResponse)
async def index(request: Request, kb_service: KbDep) -> HTMLResponse:
    return _render(request, "index.html", {"kbs": await kb_service.list_kbs()})


@router.post("/kbs")
async def create_kb(request: Request, kb_service: KbDep) -> Response:
    form = await request.form()
    try:
        kb = await kb_service.create_kb(str(form.get("name", "")), str(form.get("description", "")))
    except DomainError as exc:
        return _render(
            request,
            "index.html",
            {"kbs": await kb_service.list_kbs(), "error": exc.message},
            status_code=400,
        )
    return RedirectResponse(f"/web/kbs/{kb.id}", status_code=303)


@router.get("/kbs/{kb_id}", response_class=HTMLResponse)
async def kb_detail(
    request: Request, kb_id: UUID, kb_service: KbDep, document_service: DocumentDep
) -> Response:
    try:
        kb = await kb_service.get_kb(kb_id)
        documents = await document_service.list_documents(kb_id)
    except NotFoundError:
        return RedirectResponse("/web/kbs", status_code=303)
    return _render(
        request,
        "kb_detail.html",
        {"kb": kb, "kb_id": kb_id, "docs": documents},
    )


@router.post("/kbs/{kb_id}/documents")
async def upload_document(
    request: Request,
    kb_id: UUID,
    kb_service: KbDep,
    document_service: DocumentDep,
) -> Response:
    form = await request.form()
    upload = form.get("file")
    error: str | None = None

    if not isinstance(upload, StarletteUploadFile) or not upload.filename:
        error = "请选择 PDF 或 docx 文件"
    else:
        data = await upload.read()
        try:
            mime = validate_upload(upload.filename, upload.content_type, data)
            await document_service.upload_document(
                kb_id, filename=upload.filename, mime=mime, data=data
            )
        except DomainError as exc:
            error = exc.message

    if error is None:
        return RedirectResponse(f"/web/kbs/{kb_id}", status_code=303)

    try:
        kb = await kb_service.get_kb(kb_id)
        documents = await document_service.list_documents(kb_id)
    except NotFoundError:
        return RedirectResponse("/web/kbs", status_code=303)
    return _render(
        request,
        "kb_detail.html",
        {"kb": kb, "kb_id": kb_id, "docs": documents, "error": error},
        status_code=400,
    )


@router.get("/kbs/{kb_id}/documents/{doc_id}/status", response_class=HTMLResponse)
async def document_status(
    request: Request,
    kb_id: UUID,
    doc_id: UUID,
    document_service: DocumentDep,
) -> HTMLResponse:
    try:
        doc = await document_service.get_document(kb_id, doc_id)
    except NotFoundError:
        return HTMLResponse("", status_code=404)
    return _render(request, "_document_row.html", {"doc": doc, "kb_id": kb_id})


@router.post("/kbs/{kb_id}/documents/{doc_id}/delete", response_class=HTMLResponse)
async def delete_document(
    request: Request,
    kb_id: UUID,
    doc_id: UUID,
    document_service: DocumentDep,
) -> HTMLResponse:
    try:
        await document_service.delete_document(kb_id, doc_id)
        documents = await document_service.list_documents(kb_id)
    except NotFoundError:
        return HTMLResponse("", status_code=404)
    return _render(request, "_documents.html", {"docs": documents, "kb_id": kb_id})


@router.post("/kbs/{kb_id}/delete")
async def delete_kb(kb_id: UUID, kb_service: KbDep) -> Response:
    with contextlib.suppress(NotFoundError):
        await kb_service.delete_kb(kb_id)
    return RedirectResponse("/web/kbs", status_code=303)
