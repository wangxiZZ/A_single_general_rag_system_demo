"""知识库 CRUD 路由。

路由层只做参数校验、调用用例、组装响应；不写 SQL、不碰文件系统。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from rag_kb.api.deps import KbServiceDep
from rag_kb.api.schemas import KBCreate, KBRead

router = APIRouter(prefix="/kbs", tags=["knowledge bases"])


@router.post("", response_model=KBRead, status_code=status.HTTP_201_CREATED)
async def create_kb(payload: KBCreate, service: KbServiceDep) -> KBRead:
    kb = await service.create_kb(payload.name, payload.description)
    return KBRead.model_validate(kb)


@router.get("", response_model=list[KBRead])
async def list_kbs(service: KbServiceDep) -> list[KBRead]:
    return [KBRead.model_validate(kb) for kb in await service.list_kbs()]


@router.get("/{kb_id}", response_model=KBRead)
async def get_kb(kb_id: UUID, service: KbServiceDep) -> KBRead:
    return KBRead.model_validate(await service.get_kb(kb_id))


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb(kb_id: UUID, service: KbServiceDep) -> None:
    await service.delete_kb(kb_id)
