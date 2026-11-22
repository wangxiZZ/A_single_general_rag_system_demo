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

from fastapi import APIRouter, File, Response, UploadFile, status

from rag_kb.api.deps import DocumentServiceDep
from rag_kb.api.schemas import ChunkRead, DocumentRead
from rag_kb.application.upload_validation import validate_upload

router = APIRouter(tags=["documents"])

# 图片扩展名 → 媒体类型。放在 api 层是因为它只影响 HTTP 响应头，不属于业务规则；
# 未登记的扩展名回退到 octet-stream，让浏览器下载而不是猜成文本。
_MEDIA_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


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
    data = await file.read()
    mime = validate_upload(filename, file.content_type, data)

    doc = await service.upload_document(
        kb_id,
        filename=filename,
        mime=mime,
        data=data,
    )
    return DocumentRead.model_validate(doc)


@router.get("/kbs/{kb_id}/documents", response_model=list[DocumentRead])
async def list_documents(kb_id: UUID, service: DocumentServiceDep) -> list[DocumentRead]:
    return [DocumentRead.model_validate(doc) for doc in await service.list_documents(kb_id)]


@router.get("/kbs/{kb_id}/documents/{doc_id}", response_model=DocumentRead)
async def get_document(kb_id: UUID, doc_id: UUID, service: DocumentServiceDep) -> DocumentRead:
    return DocumentRead.model_validate(await service.get_document(kb_id, doc_id))


@router.post(
    "/kbs/{kb_id}/documents/{doc_id}/retry",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_document(kb_id: UUID, doc_id: UUID, service: DocumentServiceDep) -> DocumentRead:
    return DocumentRead.model_validate(await service.retry_document(kb_id, doc_id))


@router.get(
    "/kbs/{kb_id}/documents/{doc_id}/chunks",
    response_model=list[ChunkRead],
)
async def list_chunks(kb_id: UUID, doc_id: UUID, service: DocumentServiceDep) -> list[ChunkRead]:
    return [ChunkRead.model_validate(chunk) for chunk in await service.list_chunks(kb_id, doc_id)]


@router.get(
    "/kbs/{kb_id}/documents/{doc_id}/images/{image_id}",
    response_class=Response,
    responses={200: {"content": {"image/*": {}}}},
)
async def get_document_image(
    kb_id: UUID,
    doc_id: UUID,
    image_id: UUID,
    service: DocumentServiceDep,
) -> Response:
    """返回抽取出的原图，并用响应头带出它在原文中的位置。

    位置信息为什么走响应头而不是再开一个 JSON 接口：前端拿到 ``<img>`` 时同步就要知道
    "这张图属于哪一页、落在哪个 bbox、阅读序号是多少"，两处查询会出现时间窗不一致；
    响应头与字节在同一响应里，也方便 curl 直接验证。
    """
    image, data = await service.get_image(kb_id, doc_id, image_id)
    headers = {"X-Seq": str(image.seq)}
    if image.page_no is not None:
        headers["X-Page-No"] = str(image.page_no)
    if image.bbox is not None:
        headers["X-Bbox"] = ",".join(f"{value:.2f}" for value in image.bbox.as_list())
    media_type = _MEDIA_BY_EXT.get(Path(image.file_path).suffix.lower(), "application/octet-stream")
    return Response(content=data, media_type=media_type, headers=headers)


@router.delete(
    "/kbs/{kb_id}/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(kb_id: UUID, doc_id: UUID, service: DocumentServiceDep) -> None:
    await service.delete_document(kb_id, doc_id)
