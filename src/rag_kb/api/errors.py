"""领域异常 → HTTP 状态码。

路由里不写 try/except：映射集中在这里，保证 REST 与未来 CLI/MCP 看到的是同一套错误码。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_kb.domain.errors import (
    DomainError,
    DuplicateDocumentError,
    ErrorCode,
)

_STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.VALIDATION: 422,
    ErrorCode.UNSUPPORTED_FILE_TYPE: 415,
    ErrorCode.FILE_TOO_LARGE: 413,
    ErrorCode.PARSE: 422,
    ErrorCode.NO_EXTRACTABLE_TEXT: 422,
    ErrorCode.EMBEDDING: 500,
    ErrorCode.LLM: 500,
    ErrorCode.STORAGE: 500,
    ErrorCode.DEPENDENCY_UNAVAILABLE: 503,
    ErrorCode.CONFIGURATION: 503,
    ErrorCode.INGEST_ABORTED: 409,
    ErrorCode.INTERNAL: 500,
}


async def _domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    payload: dict[str, object] = {
        "detail": exc.message,
        "code": exc.code.value,
    }
    if isinstance(exc, DuplicateDocumentError):
        # 让客户端能直接跳到已存在的文档，而不是只看到一句"重复"。
        payload["existing_document_id"] = str(exc.existing_document_id)
    return JSONResponse(
        status_code=_STATUS_BY_CODE.get(exc.code, 500),
        content=payload,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, _domain_error_handler)
