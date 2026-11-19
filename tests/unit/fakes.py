"""测试用的内存端口实现。

单测不得连数据库；这里用最少的字典状态模拟仓储与文件存储，行为与协议保持一致。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from rag_kb.domain.errors import NotFoundError
from rag_kb.domain.models import (
    KB,
    Chunk,
    Document,
    IngestJob,
    IngestJobStatus,
    QueryLog,
)


class FakeMetadataRepository:
    def __init__(self) -> None:
        self.kbs: dict[UUID, KB] = {}
        self.documents: dict[UUID, Document] = {}
        self.chunks: dict[UUID, list[Chunk]] = {}
        self.jobs: list[IngestJob] = []
        self.query_logs: list[QueryLog] = []

    # ---- KB ----
    async def create_kb(self, kb: KB) -> KB:
        self.kbs[kb.id] = kb
        return kb

    async def get_kb(self, kb_id: UUID) -> KB | None:
        return self.kbs.get(kb_id)

    async def list_kbs(self) -> list[KB]:
        return list(self.kbs.values())

    async def delete_kb(self, kb_id: UUID) -> None:
        if kb_id not in self.kbs:
            raise NotFoundError("knowledge base", kb_id)
        del self.kbs[kb_id]
        doc_ids = [doc.id for doc in self.documents.values() if doc.kb_id == kb_id]
        for doc_id in doc_ids:
            self.documents.pop(doc_id, None)
            self.chunks.pop(doc_id, None)
        self.jobs = [job for job in self.jobs if job.doc_id not in doc_ids]

    # ---- 文档 ----
    async def create_document(self, doc: Document) -> Document:
        self.documents[doc.id] = doc
        return doc

    async def get_document(self, kb_id: UUID, doc_id: UUID) -> Document | None:
        doc = self.documents.get(doc_id)
        return doc if doc and doc.kb_id == kb_id else None

    async def list_documents(self, kb_id: UUID) -> list[Document]:
        return [doc for doc in self.documents.values() if doc.kb_id == kb_id]

    async def update_document(self, kb_id: UUID, doc: Document) -> Document:
        if await self.get_document(kb_id, doc.id) is None:
            raise NotFoundError("document", doc.id)
        self.documents[doc.id] = doc
        return doc

    async def delete_document(self, kb_id: UUID, doc_id: UUID) -> None:
        if await self.get_document(kb_id, doc_id) is None:
            raise NotFoundError("document", doc_id)
        del self.documents[doc_id]
        self.chunks.pop(doc_id, None)
        self.jobs = [job for job in self.jobs if job.doc_id != doc_id]

    async def find_document_by_sha(self, kb_id: UUID, sha256: str) -> Document | None:
        for doc in self.documents.values():
            if doc.kb_id == kb_id and doc.sha256 == sha256:
                return doc
        return None

    # ---- 分块 ----
    async def replace_chunks(self, kb_id: UUID, doc_id: UUID, chunks: list[Chunk]) -> None:
        self.chunks[doc_id] = list(chunks)

    async def list_chunks(self, kb_id: UUID, doc_id: UUID) -> list[Chunk]:
        return [chunk for chunk in self.chunks.get(doc_id, []) if chunk.kb_id == kb_id]

    async def get_chunks(self, kb_id: UUID, chunk_ids: list[str]) -> list[Chunk]:
        wanted = set(chunk_ids)
        return [
            chunk
            for chunks in self.chunks.values()
            for chunk in chunks
            if chunk.kb_id == kb_id and chunk.id in wanted
        ]

    async def delete_chunks(self, kb_id: UUID, doc_id: UUID) -> int:
        removed = len(self.chunks.pop(doc_id, []))
        return removed

    async def count_chunks(self, kb_id: UUID, doc_id: UUID) -> int:
        return len(await self.list_chunks(kb_id, doc_id))

    # ---- 任务 ----
    async def enqueue_ingest_job(self, doc_id: UUID) -> IngestJob:
        for job in self.jobs:
            if job.doc_id == doc_id and job.status in (
                IngestJobStatus.PENDING,
                IngestJobStatus.RUNNING,
            ):
                return job
        job = IngestJob(id=uuid4(), doc_id=doc_id, stage="parse")
        self.jobs.append(job)
        return job

    async def claim_ingest_job(self) -> IngestJob | None:
        for index, job in enumerate(self.jobs):
            if job.status is IngestJobStatus.PENDING:
                claimed = replace(
                    job,
                    status=IngestJobStatus.RUNNING,
                    claimed_at=datetime.now(UTC),
                )
                self.jobs[index] = claimed
                return claimed
        return None

    async def update_ingest_job(self, job: IngestJob) -> IngestJob:
        for index, current in enumerate(self.jobs):
            if current.id == job.id:
                self.jobs[index] = job
                return job
        raise NotFoundError("ingest job", job.id)

    async def add_query_log(self, log: QueryLog) -> None:
        self.query_logs.append(log)


class FakeStorage:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def save_upload(self, kb_id: UUID, doc_id: UUID, filename: str, data: bytes) -> str:
        path = f"uploads/{kb_id}/{doc_id}/{filename}"
        self.files[path] = data
        return path

    async def save_image(
        self, kb_id: UUID, doc_id: UUID, image_id: str, data: bytes, *, ext: str
    ) -> str:
        path = f"images/{kb_id}/{doc_id}/{image_id}{ext}"
        self.files[path] = data
        return path

    async def read(self, path: str) -> bytes:
        return self.files[path]

    async def delete(self, path: str) -> None:
        self.files.pop(path, None)

    async def delete_document(self, kb_id: UUID, doc_id: UUID) -> None:
        prefixes = (f"uploads/{kb_id}/{doc_id}/", f"images/{kb_id}/{doc_id}/")
        for path in list(self.files):
            if path.startswith(prefixes):
                del self.files[path]
