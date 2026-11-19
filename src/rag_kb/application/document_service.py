"""文档用例：上传、状态机、删除级联与重试。

事务边界在这里：一次上传要么留下"文档行 + 任务 + 文件"，要么全部回滚（文件尽力清理）。
状态迁移只允许走 ADR-017 的表，不允许调用方直接写 ``status``。
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import replace
from uuid import UUID, uuid4

from rag_kb.domain.errors import (
    ConflictError,
    DuplicateDocumentError,
    FileTooLargeError,
    NotFoundError,
    ValidationError,
)
from rag_kb.domain.models import Chunk, Document, DocumentStatus
from rag_kb.domain.ports import (
    GraphStore,
    LexicalIndex,
    MetadataRepository,
    Storage,
    VectorStore,
)


class DocumentService:
    def __init__(
        self,
        repository: MetadataRepository,
        storage: Storage,
        *,
        max_upload_mb: int = 50,
        vector_store: VectorStore | None = None,
        lexical_index: LexicalIndex | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self._repo = repository
        self._storage = storage
        self._max_upload_mb = max_upload_mb
        self._vector = vector_store
        self._lexical = lexical_index
        self._graph = graph_store

    async def upload_document(
        self,
        kb_id: UUID,
        *,
        filename: str,
        mime: str,
        data: bytes,
    ) -> Document:
        if not filename.strip():
            raise ValidationError("文件名不能为空")
        if len(data) > self._max_upload_mb * 1024 * 1024:
            raise FileTooLargeError(filename, len(data), self._max_upload_mb)

        # kb_id 必须存在；否则后续外键失败会变成 500，而不是可读的 404。
        if await self._repo.get_kb(kb_id) is None:
            raise NotFoundError("knowledge base", kb_id)

        sha256 = hashlib.sha256(data).hexdigest()
        existing = await self._repo.find_document_by_sha(kb_id, sha256)
        if existing is not None:
            raise DuplicateDocumentError(existing.id, filename)

        doc_id = uuid4()
        relative_path = await self._storage.save_upload(kb_id, doc_id, filename, data)
        doc = Document(
            id=doc_id,
            kb_id=kb_id,
            filename=filename,
            mime=mime,
            sha256=sha256,
            size=len(data),
            file_path=relative_path,
            status=DocumentStatus.PENDING,
        )

        try:
            created = await self._repo.create_document(doc)
        except ConflictError as exc:
            # 并发上传同一文件时唯一约束会先触发；清理文件后回查已有文档。
            await self._storage.delete(relative_path)
            existing = await self._repo.find_document_by_sha(kb_id, sha256)
            if existing is not None:
                raise DuplicateDocumentError(existing.id, filename) from exc
            raise

        try:
            await self._repo.enqueue_ingest_job(kb_id, doc_id)
        except Exception:
            # 文档行已创建但任务没入队：删除行与文件，避免留下永远不会被处理的文档。
            with contextlib.suppress(Exception):
                await self._repo.delete_document(kb_id, doc_id)
            with contextlib.suppress(Exception):
                await self._storage.delete(relative_path)
            raise

        return created

    async def get_document(self, kb_id: UUID, doc_id: UUID) -> Document:
        doc = await self._repo.get_document(kb_id, doc_id)
        if doc is None:
            raise NotFoundError("document", doc_id)
        return doc

    async def list_documents(self, kb_id: UUID) -> list[Document]:
        if await self._repo.get_kb(kb_id) is None:
            raise NotFoundError("knowledge base", kb_id)
        return await self._repo.list_documents(kb_id)

    async def list_chunks(self, kb_id: UUID, doc_id: UUID) -> list[Chunk]:
        await self.get_document(kb_id, doc_id)
        return await self._repo.list_chunks(kb_id, doc_id)

    async def transition_document(
        self, kb_id: UUID, doc_id: UUID, target: DocumentStatus
    ) -> Document:
        doc = await self.get_document(kb_id, doc_id)
        if not doc.can_transition_to(target):
            raise ValidationError(f"非法状态迁移: {doc.status.value} -> {target.value}")
        return await self._repo.update_document(kb_id, replace(doc, status=target))

    async def mark_failed(self, kb_id: UUID, doc_id: UUID, error: str) -> Document:
        doc = await self.get_document(kb_id, doc_id)
        if doc.status is DocumentStatus.FAILED:
            return doc
        if not doc.can_transition_to(DocumentStatus.FAILED):
            raise ValidationError(f"文档状态 {doc.status.value} 不能标记为 failed")
        # 错误信息会展示给用户，截断到 2000 字符，避免把整段堆栈写进 API。
        return await self._repo.update_document(
            kb_id,
            replace(doc, status=DocumentStatus.FAILED, error=error[:2000]),
        )

    async def retry_document(self, kb_id: UUID, doc_id: UUID) -> Document:
        doc = await self.get_document(kb_id, doc_id)
        if doc.status is not DocumentStatus.FAILED:
            raise ValidationError(f"只有 failed 文档可以重试，当前为 {doc.status.value}")

        await self._repo.delete_chunks(kb_id, doc_id)
        retried = await self._repo.update_document(
            kb_id,
            replace(doc, status=DocumentStatus.PENDING, error=None, chunk_count=0),
        )
        await self._repo.enqueue_ingest_job(kb_id, doc_id)
        return retried

    async def delete_document(self, kb_id: UUID, doc_id: UUID) -> None:
        await self.get_document(kb_id, doc_id)
        if self._vector is not None:
            await self._vector.delete_by_doc(kb_id, doc_id)
        if self._lexical is not None:
            await self._lexical.delete_by_doc(kb_id, doc_id)
        if self._graph is not None:
            await self._graph.delete_by_doc(kb_id, doc_id)
        # DB 是权威数据；文件清理失败会向上抛，但不会让文档"复活"。
        await self._repo.delete_document(kb_id, doc_id)
        await self._storage.delete_document(kb_id, doc_id)
