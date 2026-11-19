"""知识库 CRUD 路由。

路由层只做参数校验、调用用例、组装响应；不写 SQL、不碰文件系统。
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, status

from rag_kb.api.deps import KbServiceDep, ReindexServiceDep
from rag_kb.api.schemas import KBCreate, KBRead, ReindexAccepted, ReindexStatus

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


@router.post("/{kb_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_kb(
    kb_id: UUID,
    request: Request,
    kb_service: KbServiceDep,
    reindex_service: ReindexServiceDep,
) -> ReindexAccepted:
    # 先确认知识库存在，避免后台任务静默跑空。
    await kb_service.get_kb(kb_id)
    task_id = uuid4().hex
    tasks: dict[str, asyncio.Task[object]] = request.app.state.reindex_tasks
    task = asyncio.create_task(reindex_service.reindex_kb(kb_id), name=f"reindex-{task_id}")
    tasks[task_id] = task
    return ReindexAccepted(task_id=task_id)


@router.get("/{kb_id}/reindex/{task_id}", response_model=ReindexStatus)
async def reindex_status(kb_id: UUID, task_id: str, request: Request) -> ReindexStatus:
    task = request.app.state.reindex_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="reindex task 不存在")
    return ReindexStatus(task_id=task_id, status="done" if task.done() else "running")
