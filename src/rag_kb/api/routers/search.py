"""检索路由：四/五种模式共用一个入口。"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from rag_kb.api.deps import SearchServiceDep
from rag_kb.api.schemas import SearchHitRead, SearchRequest

router = APIRouter(tags=["search"])


@router.post("/kbs/{kb_id}/search", response_model=list[SearchHitRead])
async def search(
    kb_id: UUID, payload: SearchRequest, service: SearchServiceDep
) -> list[SearchHitRead]:
    hits = await service.search(
        kb_id,
        payload.query,
        mode=payload.mode,
        top_k=payload.top_k,
        rerank=payload.rerank,
    )
    return [SearchHitRead.model_validate(hit) for hit in hits]
