"""索引编排：把已分块的文档写入向量/词法/图谱三路，并推进到 ready。

- 输入是 PostgreSQL 里的 ``chunk`` 行，不重新解析上传件；因此 M2 存量文档也能回填；
- 重建前先删除该文档在三路中的旧数据，保证可重复执行、不残留旧 chunk；
- 任一路失败：清掉已写入的三路数据，文档置 failed；
- ``ready`` 的语义由此处兑现：三路都完成才算可检索。
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from uuid import UUID

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

        embedder, vector_store, lexical_index, graph_store, llm = self._require_ports()
        indexing = replace(document, status=DocumentStatus.INDEXING, error=None)
        indexing = await self._repo.update_document(kb_id, indexing)

        try:
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
            # 失败文档不能留下半套索引。
            with contextlib.suppress(Exception):
                await vector_store.delete_by_doc(kb_id, doc_id)
            with contextlib.suppress(Exception):
                await lexical_index.delete_by_doc(kb_id, doc_id)
            with contextlib.suppress(Exception):
                await graph_store.delete_by_doc(kb_id, doc_id)
            with contextlib.suppress(Exception):
                await self._repo.update_document(
                    kb_id,
                    replace(indexing, status=DocumentStatus.FAILED, error=str(exc)[:2000]),
                )
            raise
