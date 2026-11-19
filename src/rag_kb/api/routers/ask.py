"""问答路由：检索 → 证据拼装 → 生成 → 引用回校验。"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from rag_kb.api.deps import QAServiceDep, SearchServiceDep
from rag_kb.api.schemas import AskRequest, AskResponse, CitationRead
from rag_kb.domain.models import SearchMode

router = APIRouter(tags=["ask"])

_ALL_CHANNELS = ["vector", "keyword", "graph"]


@router.post("/kbs/{kb_id}/ask", response_model=AskResponse)
async def ask(
    kb_id: UUID,
    payload: AskRequest,
    search_service: SearchServiceDep,
    qa_service: QAServiceDep,
) -> AskResponse:
    hits = await search_service.search(
        kb_id,
        payload.question,
        mode=payload.mode,
        top_k=payload.top_k,
    )
    attempted = (
        list(_ALL_CHANNELS)
        if payload.mode in {SearchMode.HYBRID, SearchMode.HYBRID_WEIGHTED}
        else [payload.mode.value]
    )
    result = await qa_service.answer(
        kb_id,
        payload.question,
        hits,
        attempted_channels=attempted,
    )

    citations: list[CitationRead] = []
    for number in result.citations:
        evidence = result.evidence[number - 1]
        citations.append(
            CitationRead(
                index=number,
                chunk_id=evidence.hit.chunk_id,
                doc_id=evidence.hit.doc_id,
                doc_name=evidence.hit.doc_name,
                page_no=evidence.body.page_no,
                seq=evidence.body.seq,
                heading_path=evidence.body.heading_path,
                snippet=evidence.text[:300],
            )
        )

    return AskResponse(
        answer=result.answer,
        citations=citations,
        no_evidence=result.no_evidence,
        attempted_channels=result.attempted_channels,
        query_rewrites=result.query_rewrites,
    )
