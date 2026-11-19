"""摄取管线：解析 → 分块 → 图片描述 → 落库。

M2 只做到 ``chunked``；向量/词法/图谱索引在 M3 的 reindex/indexing 阶段完成。这样
M2 的验收对象是"分块质量"，而不是"检索质量"（ADR-017）。

事务与失败：

- 解析/分块/图片全部在内存中完成后才写库；
- 先写 ``doc_image``（chunk.image_id 依赖它），再写 ``chunk``；写 chunk 失败时清空图片行；
- 每个阶段前检查文档是否仍存在（tombstone），删除则抛 ``IngestAbortedError``；
- 文档状态 ``pending → parsing → chunked``；失败由 worker 统一置 ``failed``。
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import replace

from rag_kb.application.chunking import chunk_document
from rag_kb.application.image_chunks import build_image_chunks
from rag_kb.domain.errors import IngestAbortedError, StorageError, ValidationError
from rag_kb.domain.models import (
    Document,
    DocumentStatus,
    IngestJob,
    ProcessResult,
)
from rag_kb.domain.ports import VLM, MetadataRepository, Parser, Storage


class IngestPipeline:
    def __init__(
        self,
        repository: MetadataRepository,
        storage: Storage,
        parser: Parser,
        vlm: VLM | None,
        *,
        chunker_version: str,
        target_tokens: int,
        overlap_ratio: float,
        parent_max_tokens: int,
        chars_per_token: float,
        embed_model: str = "",
    ) -> None:
        self._repo = repository
        self._storage = storage
        self._parser = parser
        self._vlm = vlm
        self._chunker_version = chunker_version
        self._target_tokens = target_tokens
        self._overlap_ratio = overlap_ratio
        self._parent_max_tokens = parent_max_tokens
        self._chars_per_token = chars_per_token
        self._embed_model = embed_model

    async def process(self, job: IngestJob) -> ProcessResult:
        timings: dict[str, object] = {}
        document = await self._require_document(job)
        if document.status is DocumentStatus.PENDING:
            document = await self._repo.update_document(
                job.kb_id,
                replace(document, status=DocumentStatus.PARSING, error=None),
            )
        elif document.status not in {
            DocumentStatus.PARSING,
            DocumentStatus.CHUNKED,
        }:
            raise ValidationError(
                f"文档状态 {document.status.value} 不允许摄取，期望 pending/parsing/chunked"
            )

        if not document.file_path:
            raise StorageError(f"文档缺少上传路径: {document.id}")
        data = await self._storage.read(document.file_path)

        started = time.perf_counter()
        parsed = self._parser.parse(data, filename=document.filename, mime=document.mime)
        timings["parse_ms"] = round((time.perf_counter() - started) * 1000, 1)

        # 解析可能耗时较长，落库前再检查一次 tombstone。
        document = await self._require_document(job)
        started = time.perf_counter()
        text_chunks = chunk_document(
            list(parsed.blocks),
            kb_id=job.kb_id,
            doc_id=job.doc_id,
            doc_sha256=document.sha256,
            chunker_version=self._chunker_version,
            target_tokens=self._target_tokens,
            overlap_ratio=self._overlap_ratio,
            parent_max_tokens=self._parent_max_tokens,
            chars_per_token=self._chars_per_token,
        )
        timings["chunk_ms"] = round((time.perf_counter() - started) * 1000, 1)

        started = time.perf_counter()
        image_chunks, images = await build_image_chunks(
            list(parsed.blocks),
            text_chunks,
            kb_id=job.kb_id,
            doc_id=job.doc_id,
            doc_sha256=document.sha256,
            chunker_version=self._chunker_version,
            storage=self._storage,
            vlm=self._vlm,
            chars_per_token=self._chars_per_token,
        )
        timings["image_ms"] = round((time.perf_counter() - started) * 1000, 1)

        document = await self._require_document(job)
        chunks = [*text_chunks, *image_chunks]
        started = time.perf_counter()
        try:
            await self._repo.replace_images(job.kb_id, job.doc_id, images)
            await self._repo.replace_chunks(job.kb_id, job.doc_id, chunks)
        except Exception:
            # 图片行可能已写入；清掉，保证失败文档没有脏索引（文件由后续清理兜底）。
            with contextlib.suppress(Exception):
                await self._repo.replace_images(job.kb_id, job.doc_id, [])
            raise
        timings["persist_ms"] = round((time.perf_counter() - started) * 1000, 1)

        updated = replace(
            document,
            status=DocumentStatus.CHUNKED,
            page_count=parsed.page_count,
            chunk_count=len(chunks),
            chunker_version=self._chunker_version,
            embed_model=self._embed_model,
            error=None,
        )
        await self._repo.update_document(job.kb_id, updated)
        return ProcessResult(stage="chunked", timings=timings)

    async def _require_document(self, job: IngestJob) -> Document:
        document = await self._repo.get_document(job.kb_id, job.doc_id)
        if document is None:
            raise IngestAbortedError(job.doc_id)
        return document
