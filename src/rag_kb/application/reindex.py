"""知识库级索引回填。

只处理已有 chunk 的文档；解析失败/无 chunk 的文档跳过。单个文档失败不影响其他文档，
失败状态由 ``IndexingService`` 写入文档行。
"""

from __future__ import annotations

import logging
from uuid import UUID

from rag_kb.application.indexing import IndexingService
from rag_kb.domain.models import Document, DocumentStatus
from rag_kb.domain.ports import MetadataRepository

logger = logging.getLogger(__name__)


class ReindexService:
    def __init__(self, repository: MetadataRepository, indexer: IndexingService) -> None:
        self._repo = repository
        self._indexer = indexer

    async def reindex_kb(self, kb_id: UUID) -> list[Document]:
        documents = await self._repo.list_documents(kb_id)
        results: list[Document] = []
        for document in documents:
            if document.chunk_count <= 0:
                continue
            if document.status not in {DocumentStatus.CHUNKED, DocumentStatus.READY}:
                continue
            try:
                results.append(await self._indexer.index_document(kb_id, document.id))
            except Exception:
                logger.exception("reindex failed doc=%s", document.id)
        return results
