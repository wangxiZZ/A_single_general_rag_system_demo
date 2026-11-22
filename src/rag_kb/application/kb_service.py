"""知识库用例。

只依赖 domain 端口，不 import FastAPI / psycopg；HTTP、CLI、测试可以复用同一套逻辑。
"""

from __future__ import annotations

from uuid import UUID, uuid4

from rag_kb.application.deletion import ensure_purged, purge_indexes
from rag_kb.domain.errors import NotFoundError, ValidationError
from rag_kb.domain.models import KB
from rag_kb.domain.ports import (
    GraphStore,
    LexicalIndex,
    MetadataRepository,
    Storage,
    VectorStore,
)


class KbService:
    def __init__(
        self,
        repository: MetadataRepository,
        storage: Storage,
        *,
        vector_store: VectorStore | None = None,
        lexical_index: LexicalIndex | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self._repo = repository
        self._storage = storage
        self._vector = vector_store
        self._lexical = lexical_index
        self._graph = graph_store

    async def create_kb(self, name: str, description: str = "") -> KB:
        clean_name = name.strip()
        if not clean_name:
            raise ValidationError("知识库名称不能为空")
        kb = KB(id=uuid4(), name=clean_name, description=description.strip())
        return await self._repo.create_kb(kb)

    async def get_kb(self, kb_id: UUID) -> KB:
        kb = await self._repo.get_kb(kb_id)
        if kb is None:
            raise NotFoundError("knowledge base", kb_id)
        return kb

    async def list_kbs(self) -> list[KB]:
        return await self._repo.list_kbs()

    async def delete_kb(self, kb_id: UUID) -> None:
        await self.get_kb(kb_id)
        documents = await self._repo.list_documents(kb_id)
        # 先清三路索引，再删 DB 与文件；任一步失败都向上抛，避免"接口 200 但仍有召回"。
        # 所有文档的失败一起收集：中途停下会留下"库还在、部分通道已空"的半删状态，
        # 而 delete 是幂等的，整体重试即可收敛。
        failures: list[str] = []
        for document in documents:
            failures.extend(
                await purge_indexes(
                    kb_id,
                    document.id,
                    vector_store=self._vector,
                    lexical_index=self._lexical,
                    graph_store=self._graph,
                )
            )
        ensure_purged(failures, action="删除知识库索引")
        await self._repo.delete_kb(kb_id)
        for document in documents:
            await self._storage.delete_document(kb_id, document.id)
