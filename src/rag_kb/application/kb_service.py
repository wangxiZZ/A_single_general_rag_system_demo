"""知识库用例。

只依赖 domain 端口，不 import FastAPI / psycopg；HTTP、CLI、测试可以复用同一套逻辑。
"""

from __future__ import annotations

from uuid import UUID, uuid4

from rag_kb.domain.errors import NotFoundError, ValidationError
from rag_kb.domain.models import KB
from rag_kb.domain.ports import MetadataRepository, Storage


class KbService:
    def __init__(self, repository: MetadataRepository, storage: Storage) -> None:
        self._repo = repository
        self._storage = storage

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
        # 先删 DB（权威数据），再清文件；文件清理失败会向上抛，便于运维重试。
        documents = await self._repo.list_documents(kb_id)
        await self._repo.delete_kb(kb_id)
        for document in documents:
            await self._storage.delete_document(kb_id, document.id)
