"""文档上传、列表、详情与删除路由。

上传校验分三层，缺一不可：

1. 扩展名白名单（``.pdf`` / ``.docx``）；
2. magic bytes（PDF ``%PDF-``、docx 是 ZIP ``PK\\x03\\x04``）——只看 Content-Type 会被
   伪造的扩展名骗过；
3. 大小上限在 ``DocumentService`` 里统一判断，抛 ``FileTooLargeError`` 后映射 413。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, UploadFile, status

from rag_kb.api.deps import DocumentServiceDep
from rag_kb.api.schemas import DocumentRead
from rag_kb.domain.errors import UnsupportedFileTypeError

router = APIRouter(tags=["documents"])

_MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _validate_magic(ext: str, data: bytes, filename: str) -> None:
    # docx 本质是 zip；只接受 PK\x03\x04（含实际内容），空 zip 不算 docx。
    ok = data.startswith(b"%PDF-") if ext == ".pdf" else data.startswith(b"PK\x03\x04")
    if not ok:
        raise UnsupportedFileTypeError(filename, _MIME_BY_EXT[ext])


@router.post(
    "/kbs/{kb_id}/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    kb_id: UUID,
    service: DocumentServiceDep,
    file: Annotated[UploadFile, File()],
) -> DocumentRead:
    filename = file.filename or "upload.bin"
    ext = Path(filename).suffix.lower()
    if ext not in _MIME_BY_EXT:
        raise UnsupportedFileTypeError(filename, file.content_type or "")

    data = await file.read()
    _validate_magic(ext, data, filename)

    doc = await service.upload_document(
        kb_id,
        filename=filename,
        mime=file.content_type or _MIME_BY_EXT[ext],
        data=data,
    )
    return DocumentRead.model_validate(doc)


@router.get("/kbs/{kb_id}/documents", response_model=list[DocumentRead])
async def list_documents(kb_id: UUID, service: DocumentServiceDep) -> list[DocumentRead]:
    return [DocumentRead.model_validate(doc) for doc in await service.list_documents(kb_id)]


@router.get("/kbs/{kb_id}/documents/{doc_id}", response_model=DocumentRead)
async def get_document(kb_id: UUID, doc_id: UUID, service: DocumentServiceDep) -> DocumentRead:
    return DocumentRead.model_validate(await service.get_document(kb_id, doc_id))


@router.delete(
    "/kbs/{kb_id}/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(kb_id: UUID, doc_id: UUID, service: DocumentServiceDep) -> None:
    await service.delete_document(kb_id, doc_id)
