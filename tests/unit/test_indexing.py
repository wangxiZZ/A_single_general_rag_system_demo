"""索引编排单测：fake 三路索引 + fake LLM。"""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID, uuid4

import pytest

from rag_kb.application.indexing import IndexingService
from rag_kb.domain.errors import ConfigurationError, LlmError, ValidationError
from rag_kb.domain.models import (
    Chunk,
    ChunkKind,
    ChunkLevel,
    Document,
    DocumentStatus,
    Entity,
    Mention,
    Relation,
    VectorRecord,
)

KB_ID = uuid4()
DOC_ID = uuid4()


class FakeEmbedder:
    async def embed_documents(self, texts):
        return [[float(len(text)), 0.0] for text in texts]

    async def embed_query(self, text: str):
        return [0.0, 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted = 0
        self.records: list[VectorRecord] = []

    async def delete_by_doc(self, kb_id: UUID, doc_id: UUID) -> int:
        self.deleted += 1
        return 1

    async def upsert(self, kb_id: UUID, records) -> None:
        self.records = list(records)


class FakeLexicalIndex:
    def __init__(self) -> None:
        self.deleted = 0
        self.chunks: list[Chunk] = []

    async def delete_by_doc(self, kb_id: UUID, doc_id: UUID) -> int:
        self.deleted += 1
        return 1

    async def upsert(self, kb_id: UUID, chunks) -> None:
        self.chunks = list(chunks)


class FakeGraphStore:
    def __init__(self) -> None:
        self.deleted = 0
        self.entities: list[Entity] = []
        self.relations: list[Relation] = []
        self.mentions: list[Mention] = []

    async def delete_by_doc(self, kb_id: UUID, doc_id: UUID) -> int:
        self.deleted += 1
        return 1

    async def upsert(self, kb_id, doc_id, entities, relations, mentions) -> None:
        self.entities = list(entities)
        self.relations = list(relations)
        self.mentions = list(mentions)


class FakeLLM:
    def __init__(self, payload: dict | Exception) -> None:
        self.payload = payload

    async def structured_json(self, messages, schema):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    async def chat(self, messages, *, temperature: float = 0.0, max_tokens=None) -> str:
        return ""


class FakeRepository:
    def __init__(self, chunks: list[Chunk], document: Document) -> None:
        self.chunks = chunks
        self.document = document
        self.statuses: list[DocumentStatus] = []

    async def get_document(self, kb_id: UUID, doc_id: UUID) -> Document | None:
        return self.document

    async def list_chunks(self, kb_id: UUID, doc_id: UUID) -> list[Chunk]:
        return list(self.chunks)

    async def update_document(self, kb_id: UUID, doc: Document) -> Document:
        self.document = doc
        self.statuses.append(doc.status)
        return doc

    async def delete_chunks(self, kb_id: UUID, doc_id: UUID) -> int:
        count = len(self.chunks)
        self.chunks.clear()
        return count

    async def replace_images(self, kb_id: UUID, doc_id: UUID, images) -> None:
        self.images = list(images)


def _chunk() -> Chunk:
    text = "年假政策规定员工。"
    return Chunk(
        id="a" * 32,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        level=ChunkLevel.CHILD,
        kind=ChunkKind.TEXT,
        content=text,
        seq=0,
        char_start=0,
        char_end=len(text),
        token_count=len(text),
        chunker_version="v1",
    )


def _document(status: DocumentStatus = DocumentStatus.CHUNKED) -> Document:
    return Document(
        id=DOC_ID,
        kb_id=KB_ID,
        filename="a.pdf",
        mime="application/pdf",
        sha256="a" * 64,
        size=1,
        status=status,
        chunker_version="v1",
        chunk_count=1,
    )


def _payload() -> dict:
    return {
        "entities": [
            {"name": "年假政策", "type": "条款", "span": "年假政策"},
            {"name": "员工", "type": "人物", "span": "员工"},
        ],
        "relations": [
            {
                "head": "年假政策",
                "tail": "员工",
                "relation": "适用于",
                "evidence_span": "年假政策规定员工",
            }
        ],
    }


def _service(
    repo, *, llm=None, use_llm: bool = True
) -> tuple[IndexingService, FakeVectorStore, FakeLexicalIndex, FakeGraphStore]:
    vector = FakeVectorStore()
    lexical = FakeLexicalIndex()
    graph = FakeGraphStore()
    effective_llm = llm if llm is not None else (FakeLLM(_payload()) if use_llm else None)
    service = IndexingService(
        repo,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        vector_store=vector,  # type: ignore[arg-type]
        lexical_index=lexical,  # type: ignore[arg-type]
        graph_store=graph,  # type: ignore[arg-type]
        llm=effective_llm,  # type: ignore[arg-type]
        chunker_version="v1",
        embed_model="qwen-test",
    )
    return service, vector, lexical, graph


async def test_builds_all_three_indexes_and_marks_ready() -> None:
    repo = FakeRepository([_chunk()], _document())
    service, vector, lexical, graph = _service(repo)

    result = await service.index_document(KB_ID, DOC_ID)

    assert result.status is DocumentStatus.READY
    assert result.embed_model == "qwen-test"
    assert repo.statuses == [DocumentStatus.INDEXING, DocumentStatus.READY]
    assert vector.deleted == 1
    assert lexical.deleted == 1
    assert graph.deleted == 1
    assert [record.chunk_id for record in vector.records] == ["a" * 32]
    assert [chunk.id for chunk in lexical.chunks] == ["a" * 32]
    assert {entity.name for entity in graph.entities} == {"年假政策", "员工"}
    assert len(graph.relations) == 1
    assert len(graph.mentions) == 2


async def test_missing_llm_marks_failed_and_cleans_chunks() -> None:
    repo = FakeRepository([_chunk()], _document())
    service, _, _, _ = _service(repo, use_llm=False)

    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        await service.index_document(KB_ID, DOC_ID)

    # 配置缺失同样是可重试的失败：文档 failed、chunk 清空，修好 Key 后 retry 即可。
    assert repo.document.status is DocumentStatus.FAILED
    assert repo.document.chunk_count == 0
    assert repo.chunks == []
    assert repo.statuses == [DocumentStatus.FAILED]


async def test_chunker_version_mismatch_is_rejected() -> None:
    document = replace(_document(), chunker_version="v0")
    repo = FakeRepository([_chunk()], document)
    service, _, _, _ = _service(repo)

    with pytest.raises(ValidationError, match="chunker_version"):
        await service.index_document(KB_ID, DOC_ID)
    assert repo.statuses == []


async def test_graph_failure_cleans_indexes_and_marks_failed() -> None:
    repo = FakeRepository([_chunk()], _document())
    service, vector, lexical, graph = _service(repo, llm=FakeLLM(LlmError("boom")))

    with pytest.raises(LlmError):
        await service.index_document(KB_ID, DOC_ID)

    assert repo.document.status is DocumentStatus.FAILED
    assert repo.document.chunk_count == 0
    assert repo.chunks == []
    # 初次清理 + 失败清理
    assert vector.deleted == 2
    assert lexical.deleted == 2
    assert graph.deleted == 2


async def test_index_failure_without_embedder_counts_as_failed_and_cleans() -> None:
    repo = FakeRepository([_chunk()], _document())
    service, vector, lexical, graph = _service(repo, use_llm=False)
    service._embedder = None  # type: ignore[attr-defined]

    with pytest.raises(ConfigurationError, match="EMBED_API_KEY"):
        await service.index_document(KB_ID, DOC_ID)

    assert repo.document.status is DocumentStatus.FAILED
    assert repo.chunks == []
    # 端口校验失败时三路都没有写入，因此不需要（也不应该）产生删除调用。
    assert vector.deleted == 0
    assert lexical.deleted == 0
    assert graph.deleted == 0
