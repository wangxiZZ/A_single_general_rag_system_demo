"""索引编排：把已分块的文档写入向量/词法/图谱三路，并推进到 ready。

- 输入是 PostgreSQL 里的 ``chunk`` 行，不重新解析上传件；因此 M2 存量文档也能回填；
- 重建前先删除该文档在三路中的旧数据，保证可重复执行、不残留旧 chunk；
- 任一路失败：清掉已写入的三路数据，文档置 failed；
- ``ready`` 的语义由此处兑现：三路都完成才算可检索。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import partial
from uuid import UUID

from rag_kb.application.deletion import purge_indexes
from rag_kb.application.graph_extract import GraphExtractor
from rag_kb.application.vector_index import index_chunks
from rag_kb.domain.errors import (
    ConfigurationError,
    IngestAbortedError,
    LlmError,
    ValidationError,
)
from rag_kb.domain.models import ChunkKind, ChunkLevel, Document, DocumentStatus
from rag_kb.domain.ports import (
    LLM,
    Embedder,
    GraphStore,
    LexicalIndex,
    MetadataRepository,
    VectorStore,
)

logger = logging.getLogger(__name__)


class IndexingService:
    def __init__(
        self,
        repository: MetadataRepository,
        *,
        embedder: Embedder | None,
        vector_store: VectorStore | None,
        lexical_index: LexicalIndex | None,
        graph_store: GraphStore | None,
        llm: LLM | None,
        chunker_version: str,
        embed_model: str,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._graph_store = graph_store
        self._llm = llm
        self._chunker_version = chunker_version
        self._embed_model = embed_model

    def _require_ports(self) -> tuple[Embedder, VectorStore, LexicalIndex, GraphStore, LLM]:
        if self._embedder is None or self._vector_store is None:
            raise ConfigurationError("EMBED_API_KEY 未配置，无法建立向量索引")
        if self._lexical_index is None:
            raise ConfigurationError("词法索引未就绪")
        if self._graph_store is None:
            raise ConfigurationError("图谱存储未就绪")
        if self._llm is None:
            raise ConfigurationError("LLM_API_KEY 未配置，无法抽取图谱")
        return (
            self._embedder,
            self._vector_store,
            self._lexical_index,
            self._graph_store,
            self._llm,
        )

    async def index_document(self, kb_id: UUID, doc_id: UUID) -> Document:
        document = await self._repo.get_document(kb_id, doc_id)
        if document is None:
            raise IngestAbortedError(doc_id)
        if document.chunker_version and document.chunker_version != self._chunker_version:
            raise ValidationError(
                f"chunker_version 不一致（doc={document.chunker_version} "
                f"current={self._chunker_version}），需要重新摄取"
            )
        if document.status not in {DocumentStatus.CHUNKED, DocumentStatus.READY}:
            raise ValidationError(
                f"文档状态 {document.status.value} 不允许建索引，期望 chunked/ready"
            )

        chunks = await self._repo.list_chunks(kb_id, doc_id)
        children = [chunk for chunk in chunks if chunk.level is ChunkLevel.CHILD]
        if not children:
            raise ValidationError("文档没有子块，无法建索引")

        # 端口校验也放进 try：缺少 EMBED_API_KEY 时索引还没写，但管线此前已经落了
        # chunk 行与图片；失败口径要求这些也清掉，否则文档 failed 却仍能被 /chunks 读到。
        ports: tuple[Embedder, VectorStore, LexicalIndex, GraphStore, LLM] | None = None
        try:
            ports = self._require_ports()
            embedder, vector_store, lexical_index, graph_store, llm = ports
            indexing = replace(document, status=DocumentStatus.INDEXING, error=None)
            indexing = await self._repo.update_document(kb_id, indexing)

            await vector_store.delete_by_doc(kb_id, doc_id)
            await lexical_index.delete_by_doc(kb_id, doc_id)
            await graph_store.delete_by_doc(kb_id, doc_id)

            await index_chunks(
                children,
                kb_id=kb_id,
                embedder=embedder,
                vector_store=vector_store,
            )
            await lexical_index.upsert(kb_id, children)

            text_children = [chunk for chunk in children if chunk.kind is ChunkKind.TEXT]
            extracted = await GraphExtractor(llm).extract(children, kb_id=kb_id)
            if text_children and extracted.failed_chunks == len(text_children):
                raise LlmError("图谱抽取全部失败")
            await graph_store.upsert(
                kb_id,
                doc_id,
                extracted.entities,
                extracted.relations,
                extracted.mentions,
            )

            ready = replace(
                indexing,
                status=DocumentStatus.READY,
                embed_model=self._embed_model,
                error=None,
            )
            return await self._repo.update_document(kb_id, ready)
        except Exception as exc:
            # 失败文档不能留下半套索引：三路索引之外，chunk 与图片行也要清掉。
            # 否则 ADR-022 的验收口径（chunk 行数 = 0）不成立，retry 也会基于旧块重建。
            # 端口都没拿到时没有索引可删，但 chunk 仍需清（见 include_indexes 的说明）。
            await self.cleanup_failed(kb_id, doc_id, include_indexes=ports is not None)
            try:
                await self._repo.update_document(
                    kb_id,
                    replace(
                        document,
                        status=DocumentStatus.FAILED,
                        error=str(exc)[:2000],
                        chunk_count=0,
                    ),
                )
            except Exception:
                # 状态更新失败也不能掩盖原始异常，留日志交给 worker 的失败兜底。
                logger.exception("mark document failed failed doc=%s", doc_id)
            raise

    async def cleanup_failed(
        self, kb_id: UUID, doc_id: UUID, *, include_indexes: bool = True
    ) -> None:
        """清理某文档的三路索引与 chunk/图片行（幂等，可重入）。

        ``include_indexes=False`` 用于"端口校验阶段就失败"的调用方：那时三路一条都没写，
        再去调 delete 只会把"端口缺失"污染成清理失败，也会让单测里
        "端口缺失时不应有删除调用"的断言失去意义。

        清理本身失败只记日志不抛出：调用方已经在处理原始异常，这里再抛会把真正的
        失败原因盖掉（AGENTS §3.8 要求"不吞异常"，但吞掉的是**清理**异常并留下日志，
        与静默丢弃业务异常是两回事）。
        """
        if include_indexes:
            failures = await purge_indexes(
                kb_id,
                doc_id,
                vector_store=self._vector_store,
                lexical_index=self._lexical_index,
                graph_store=self._graph_store,
            )
            if failures:
                logger.error("cleanup indexes incomplete doc=%s: %s", doc_id, "；".join(failures))
        rows = (
            ("chunks", partial(self._repo.delete_chunks, kb_id, doc_id)),
            ("image rows", partial(self._repo.replace_images, kb_id, doc_id, [])),
        )
        for name, step in rows:
            try:
                await step()
            except Exception:
                logger.exception("cleanup %s failed doc=%s", name, doc_id)
