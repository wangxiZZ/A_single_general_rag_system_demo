"""摄取 worker 单测：fake 仓储 + fake processor。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

import pytest

from rag_kb.domain.errors import IngestAbortedError
from rag_kb.domain.models import (
    KB,
    Document,
    DocumentStatus,
    IngestJob,
    IngestJobStatus,
    ProcessResult,
)
from rag_kb.worker.loop import IngestWorker
from tests.unit.fakes import FakeMetadataRepository


async def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("等待条件超时")
        await asyncio.sleep(0.01)


async def _seed(repo: FakeMetadataRepository) -> tuple[KB, Document, IngestJob]:
    kb = await repo.create_kb(KB(id=uuid4(), name="kb"))
    doc = await repo.create_document(
        Document(
            id=uuid4(),
            kb_id=kb.id,
            filename="a.pdf",
            mime="application/pdf",
            sha256="a" * 64,
            size=1,
        )
    )
    job = await repo.enqueue_ingest_job(kb.id, doc.id)
    return kb, doc, job


async def test_worker_marks_job_succeeded_with_timings() -> None:
    repo = FakeMetadataRepository()
    _, _, job = await _seed(repo)
    processed = asyncio.Event()

    async def processor(current: IngestJob) -> ProcessResult:
        assert current.id == job.id
        processed.set()
        return ProcessResult(stage="chunked", timings={"parse_ms": 1.0})

    worker = IngestWorker(repo, processor, poll_interval_s=0.01)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))

    await asyncio.wait_for(processed.wait(), timeout=1)
    await _wait_until(lambda: repo.jobs[0].status is IngestJobStatus.SUCCEEDED)
    stop.set()
    await asyncio.wait_for(task, timeout=1)

    finished = repo.jobs[0]
    assert finished.attempts == 1
    assert finished.stage == "chunked"
    assert finished.error is None
    assert finished.timings["parse_ms"] == 1.0
    assert "total_ms" in finished.timings


async def test_worker_marks_failure_on_processor_exception() -> None:
    repo = FakeMetadataRepository()
    kb, doc, _ = await _seed(repo)

    async def processor(current: IngestJob) -> ProcessResult:
        raise ValueError("boom")

    worker = IngestWorker(repo, processor, poll_interval_s=0.01)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))

    await _wait_until(lambda: repo.jobs[0].status is IngestJobStatus.FAILED)
    stop.set()
    await asyncio.wait_for(task, timeout=1)

    failed = repo.jobs[0]
    assert "ValueError: boom" in (failed.error or "")
    document = repo.documents[doc.id]
    assert document.status is DocumentStatus.FAILED
    assert "boom" in (document.error or "")
    assert document.kb_id == kb.id


async def test_worker_does_not_mark_deleted_document_failed() -> None:
    repo = FakeMetadataRepository()
    kb, doc, _ = await _seed(repo)

    async def processor(current: IngestJob) -> ProcessResult:
        await repo.delete_document(kb.id, doc.id)
        raise IngestAbortedError(doc.id)

    worker = IngestWorker(repo, processor, poll_interval_s=0.01)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))

    await _wait_until(lambda: not repo.documents and not repo.jobs)
    stop.set()
    await asyncio.wait_for(task, timeout=1)
    assert repo.jobs == []


async def test_worker_stops_without_jobs() -> None:
    repo = FakeMetadataRepository()

    async def processor(current: IngestJob) -> ProcessResult:
        raise AssertionError("不应该被调用")

    worker = IngestWorker(repo, processor, poll_interval_s=0.01)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    await asyncio.sleep(0.02)
    stop.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.parametrize("initial", [DocumentStatus.PENDING, DocumentStatus.PARSING])
async def test_worker_failure_marks_document_from_valid_states(
    initial: DocumentStatus,
) -> None:
    repo = FakeMetadataRepository()
    kb, doc, _ = await _seed(repo)
    repo.documents[doc.id] = Document(
        id=doc.id,
        kb_id=doc.kb_id,
        filename=doc.filename,
        mime=doc.mime,
        sha256=doc.sha256,
        size=doc.size,
        status=initial,
    )

    async def processor(current: IngestJob) -> ProcessResult:
        raise RuntimeError("bad")

    worker = IngestWorker(repo, processor, poll_interval_s=0.01)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    await _wait_until(lambda: repo.documents[doc.id].status is DocumentStatus.FAILED)
    stop.set()
    await asyncio.wait_for(task, timeout=1)
    assert repo.documents[doc.id].status is DocumentStatus.FAILED
    assert kb.id == repo.documents[doc.id].kb_id
