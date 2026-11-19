"""PostgreSQL 词法索引集成测试。"""

from __future__ import annotations

import contextlib
from uuid import UUID, uuid4

import pytest

from rag_kb.config import get_settings
from rag_kb.domain.errors import ValidationError
from rag_kb.domain.models import KB, Chunk, ChunkKind, ChunkLevel, Document
from rag_kb.infra.postgres.lexical import PgLexicalIndex
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.infra.postgres.repository import PostgresMetadataRepository

pytestmark = pytest.mark.integration


def _chunk(kb_id: UUID, doc_id: UUID, chunk_id: str, text: str, seq: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        kb_id=kb_id,
        doc_id=doc_id,
        level=ChunkLevel.CHILD,
        kind=ChunkKind.TEXT,
        content=text,
        seq=seq,
        char_start=0,
        char_end=len(text),
        token_count=len(text),
        chunker_version="v1",
    )


async def test_lexical_search_isolation_delete() -> None:
    settings = get_settings()
    apply_migrations(settings.pg_dsn)
    repo = PostgresMetadataRepository(settings.pg_dsn, min_size=1, max_size=2)
    lexical = PgLexicalIndex(settings.pg_dsn, min_size=1, max_size=2)
    await repo.open()
    await lexical.open()
    kb_id = uuid4()
    doc_id = uuid4()
    other_kb = uuid4()

    try:
        await repo.create_kb(KB(id=kb_id, name=f"lex-{uuid4().hex[:6]}"))
        await repo.create_document(
            Document(
                id=doc_id,
                kb_id=kb_id,
                filename="a.pdf",
                mime="application/pdf",
                sha256=uuid4().hex * 2,
                size=1,
            )
        )
        chunks = [
            _chunk(kb_id, doc_id, "a" * 32, "年假政策规定员工每年享有带薪年假。", 0),
            _chunk(kb_id, doc_id, "b" * 32, "请假流程需要提前三天提交申请。", 1),
        ]
        await repo.replace_chunks(kb_id, doc_id, chunks)
        await lexical.upsert(kb_id, chunks)

        hits = await lexical.search(kb_id, "年假", top_k=5)
        assert [hit.chunk_id for hit in hits] == ["a" * 32]

        hits = await lexical.search(kb_id, "请假 流程", top_k=5)
        assert [hit.chunk_id for hit in hits] == ["b" * 32]

        assert await lexical.search(other_kb, "年假", top_k=5) == []
        assert await lexical.count(kb_id) == 2
        assert await lexical.count(kb_id, doc_id) == 2

        # 父块不允许写入词法索引
        parent = Chunk(
            id="p" * 32,
            kb_id=kb_id,
            doc_id=doc_id,
            level=ChunkLevel.PARENT,
            kind=ChunkKind.TEXT,
            content="父块",
            seq=10,
            char_start=0,
            char_end=2,
            token_count=2,
            chunker_version="v1",
        )
        with pytest.raises(ValidationError, match="child"):
            await lexical.upsert(kb_id, [parent])

        assert await lexical.delete_by_doc(kb_id, doc_id) == 2
        assert await lexical.count(kb_id, doc_id) == 0
        assert await lexical.search(kb_id, "年假", top_k=5) == []
    finally:
        with contextlib.suppress(Exception):
            await repo.delete_kb(kb_id)
        await lexical.close()
        await repo.close()
