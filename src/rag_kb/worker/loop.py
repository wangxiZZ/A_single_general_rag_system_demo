"""摄取 worker。

设计要点：

1. **领取任务用 ``FOR UPDATE SKIP LOCKED``**（在仓储实现里），多个 worker 也不会抢同一条；
2. **交给 processor 处理具体阶段**：worker 只负责job 状态、计时与失败兜底，解析/分块/
   索引逻辑在 ``application/ingest_pipeline.py``，便于单测替换；
3. **tombstone**：processor 在阶段间检查文档是否已删并抛 ``IngestAbortedError``；worker
   收到后直接放弃，不把任务标成 failed；
4. **阶段耗时**：processor 返回 ``ProcessResult``，worker 合并 ``timings`` 并写 ``total_ms``；
5. worker 自身在 app lifespan 内作为 asyncio 任务运行，stop_event 触发退出。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

from rag_kb.domain.errors import IngestAbortedError, NotFoundError
from rag_kb.domain.models import (
    DocumentStatus,
    IngestJob,
    IngestJobStatus,
    can_transition,
)
from rag_kb.domain.ports import MetadataRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    stage: str
    timings: dict[str, object] = field(default_factory=dict)


Processor = Callable[[IngestJob], Awaitable[ProcessResult]]


class IngestWorker:
    def __init__(
        self,
        repository: MetadataRepository,
        processor: Processor,
        *,
        poll_interval_s: float = 1.0,
    ) -> None:
        self._repo = repository
        self._processor = processor
        self._poll_interval_s = max(0.05, poll_interval_s)

    async def run(self, stop_event: asyncio.Event) -> None:
        logger.info("ingest worker started")
        while not stop_event.is_set():
            try:
                job = await self._repo.claim_ingest_job()
            except Exception:
                # 存储短暂不可用时 worker 应继续存活；记录完整堆栈便于排查。
                logger.exception("claim ingest job failed")
                await self._wait(stop_event)
                continue

            if job is None:
                await self._wait(stop_event)
                continue
            await self._process(job)
        logger.info("ingest worker stopped")

    async def _wait(self, stop_event: asyncio.Event) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=self._poll_interval_s)

    async def _process(self, job: IngestJob) -> None:
        started = time.perf_counter()
        timings: dict[str, object] = dict(job.timings)

        try:
            result = await self._processor(job)
        except IngestAbortedError:
            logger.info("document deleted, ingest aborted job=%s", job.id)
            return
        except Exception as exc:
            timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
            await self._mark_failed(job, exc, timings)
            return

        timings.update(result.timings)
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        succeeded = replace(
            job,
            stage=result.stage,
            status=IngestJobStatus.SUCCEEDED,
            error=None,
            timings=timings,
        )
        with contextlib.suppress(NotFoundError):
            await self._repo.update_ingest_job(succeeded)

    async def _mark_failed(
        self, job: IngestJob, exc: Exception, timings: dict[str, object]
    ) -> None:
        message = f"{type(exc).__name__}: {exc}"[:2000]
        failed = replace(
            job,
            stage="failed",
            status=IngestJobStatus.FAILED,
            error=message,
            timings=timings,
        )
        with contextlib.suppress(NotFoundError):
            await self._repo.update_ingest_job(failed)

        try:
            document = await self._repo.get_document(job.kb_id, job.doc_id)
        except Exception:
            logger.exception("load document for failure marking failed doc=%s", job.doc_id)
            return
        if document is None:
            return
        if document.status is DocumentStatus.FAILED:
            with contextlib.suppress(Exception):
                await self._repo.update_document(job.kb_id, replace(document, error=message))
            return
        if can_transition(document.status, DocumentStatus.FAILED):
            await self._repo.update_document(
                job.kb_id,
                replace(document, status=DocumentStatus.FAILED, error=message),
            )
        else:
            logger.warning(
                "cannot mark document failed from status=%s doc=%s",
                document.status.value,
                job.doc_id,
            )
