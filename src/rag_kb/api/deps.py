"""FastAPI 依赖：从 app.state 取用例服务。

服务和端口在 lifespan 里构造，路由只拿现成对象；测试注入 fake 仓储时不需要改路由。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from rag_kb.application.document_service import DocumentService
from rag_kb.application.kb_service import KbService
from rag_kb.application.qa_service import QAService
from rag_kb.application.reindex import ReindexService
from rag_kb.application.search_service import SearchService


def get_kb_service(request: Request) -> KbService:
    return request.app.state.kb_service


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_reindex_service(request: Request) -> ReindexService:
    return request.app.state.reindex_service


def get_qa_service(request: Request) -> QAService:
    return request.app.state.qa_service


KbServiceDep = Annotated[KbService, Depends(get_kb_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
ReindexServiceDep = Annotated[ReindexService, Depends(get_reindex_service)]
QAServiceDep = Annotated[QAService, Depends(get_qa_service)]
