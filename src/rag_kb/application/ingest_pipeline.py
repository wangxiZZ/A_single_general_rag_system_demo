"""摄取管线：解析 → 分块 → 图片描述 → 落库。

M2 只做到 ``chunked``；向量/词法/图谱索引在 M3 的 reindex/indexing 阶段完成。这样
M2 的验收对象是"分块质量"，而不是"检索质量"（ADR-017）。

事务与失败：

- 解析/分块/图片全部在内存中完成后才写库；
- 先写 ``doc_image``（chunk.image_id 依赖它），再写 ``chunk``；写 chunk 失败时清空图片行；
- 每个阶段前检查文档是否仍存在（tombstone），删除则抛 ``IngestAbortedError``；
- 文档状态 ``pending → parsing → chunked``；
- **任何阶段失败都在本层清理**（chunk / 图片行 / 图片文件 / 状态），worker 只负责记 job
  结果：失败文档若仍留着 chunk 行，``/chunks`` 还能读到内容、``chunk_count`` 也不为 0，
  "失败无残留"的验收口径就不成立了。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import replace
from functools import partial

from rag_kb.application.chunking import chunk_document
from rag_kb.application.image_chunks import build_image_chunks
from rag_kb.application.indexing import IndexingService
from rag_kb.domain.errors import (
    IngestAbortedError,
    NotFoundError,
    StorageError,
    ValidationError,
)
from rag_kb.domain.models import (
    Document,
    DocumentStatus,
    IngestJob,
    ProcessResult,
    can_transition,
)
from rag_kb.domain.ports import VLM, MetadataRepository, Parser, Storage

logger = logging.getLogger(__name__)


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
        indexer: IndexingService | None = None,
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
        self._indexer = indexer

    async def process(self, job: IngestJob) -> ProcessResult:
        try:
            return await self._process(job)
        except IngestAbortedError:
            # 文档在摄取过程中被删除：清理责任在删除路径，任务不算失败。
            raise
        except Exception as exc:
            await self._cleanup_failed(job, exc)
            raise

    async def _cleanup_failed(self, job: IngestJob, error: Exception) -> None:
        """失败文档的收尾：清 chunk、图片行、图片文件，并把状态置 failed。

        索引三路的清理已经在 ``IndexingService`` 内部完成（它知道哪些端口真的写过），
        这里不重复调用；重复调用会把"端口缺失"误报成清理失败。
        """
        steps = (
            ("chunks", partial(self._repo.delete_chunks, job.kb_id, job.doc_id)),
            ("image rows", partial(self._repo.replace_images, job.kb_id, job.doc_id, [])),
            ("image files", partial(self._storage.delete_images, job.kb_id, job.doc_id)),
        )
        for name, step in steps:
            try:
                await step()
            except Exception:
                # 清理失败不能掩盖原始错误，也不能静默：留日志供排查残留。
                logger.exception("cleanup %s failed for doc %s", name, job.doc_id)

        try:
            document = await self._repo.get_document(job.kb_id, job.doc_id)
        except Exception:
            logger.exception("load document for cleanup failed doc=%s", job.doc_id)
            return
        if document is None or document.status is DocumentStatus.FAILED:
            return
        if not can_transition(document.status, DocumentStatus.FAILED):
            logger.warning(
                "cannot mark document failed from status=%s doc=%s",
                document.status.value,
                job.doc_id,
            )
            return
        message = f"{type(error).__name__}: {error}"[:2000]
        with contextlib.suppress(Exception):
            await self._repo.update_document(
                job.kb_id,
                replace(
                    document,
                    status=DocumentStatus.FAILED,
                    error=message,
                    chunk_count=0,
                ),
            )

    async def _process(self, job: IngestJob) -> ProcessResult:
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
        # parse 与分块都是同步 CPU 操作：直接在协程里调用会把事件循环（健康检查、
        # 其他请求、worker 心跳）一起卡住，大 PDF 上能卡几十秒，所以丢线程池。
        # 用标准库的 asyncio.to_thread 而不是 anyio：application 只允许依赖标准库与
        # domain（AGENTS §2），anyio 属于适配器层的实现细节（infra 用它包装同步 SDK）。
        parsed = await asyncio.to_thread(
            self._parser.parse,
            data,
            filename=document.filename,
            mime=document.mime,
        )
        timings["parse_ms"] = round((time.perf_counter() - started) * 1000, 1)

        # 解析可能耗时较长，落库前再检查一次 tombstone。
        document = await self._require_document(job)
        started = time.perf_counter()
        text_chunks = await asyncio.to_thread(
            chunk_document,
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

        try:
            document = await self._require_document(job)
        except IngestAbortedError:
            # 图片描述阶段可能已经写下新文件；删除请求可能发生在保存之后，补一次清理。
            with contextlib.suppress(Exception):
                await self._storage.delete_document(job.kb_id, job.doc_id)
            raise

        chunks = [*text_chunks, *image_chunks]
        started = time.perf_counter()
        try:
            await self._repo.replace_images(job.kb_id, job.doc_id, images)
            await self._repo.replace_chunks(job.kb_id, job.doc_id, chunks)
        except Exception:
            # 图片行可能已写入；清掉，保证失败文档没有脏索引。
            with contextlib.suppress(Exception):
                await self._repo.replace_images(job.kb_id, job.doc_id, [])
            # 如果失败原因是文档在写入过程中被删除，再清理一次文件；普通 DB 故障不动文件。
            with contextlib.suppress(Exception):
                if await self._repo.get_document(job.kb_id, job.doc_id) is None:
                    await self._storage.delete_document(job.kb_id, job.doc_id)
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
        try:
            await self._repo.update_document(job.kb_id, updated)
        except NotFoundError as exc:
            # 删除发生在本阶段与写状态之间：DB 行已被级联清理，补清一次文件。
            with contextlib.suppress(Exception):
                await self._storage.delete_document(job.kb_id, job.doc_id)
            raise IngestAbortedError(job.doc_id) from exc

        if self._indexer is not None:
            await self._indexer.index_document(job.kb_id, job.doc_id)
            return ProcessResult(stage="ready", timings=timings)
        return ProcessResult(stage="chunked", timings=timings)

    async def _require_document(self, job: IngestJob) -> Document:
        document = await self._repo.get_document(job.kb_id, job.doc_id)
        if document is None:
            raise IngestAbortedError(job.doc_id)
        return document
