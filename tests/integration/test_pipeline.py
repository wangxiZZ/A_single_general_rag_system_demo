"""全链路集成：上传 → 解析分块 → 三路索引 → 检索 → 问答 → reindex。

与 `tests/e2e` 的区别：这里不经过 HTTP，直接装配 application 服务与真实三存储，模型换成
确定性 fake。这样能精确断言「解析出的 chunk 是否真的进了三条通道」，又不会因为网络或
模型波动让测试变红。真实模型链路由 `tests/e2e/test_mcp_stdio.py` 覆盖。
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from rag_kb.application.document_service import DocumentService
from rag_kb.application.indexing import IndexingService
from rag_kb.application.ingest_pipeline import IngestPipeline
from rag_kb.application.kb_service import KbService
from rag_kb.application.qa_service import QAService
from rag_kb.application.reindex import ReindexService
from rag_kb.application.search_service import SearchService
from rag_kb.config import get_settings
from rag_kb.domain.models import DocumentStatus, SearchMode
from rag_kb.infra.milvus.store import MilvusVectorStore
from rag_kb.infra.neo4j.store import Neo4jGraphStore
from rag_kb.infra.parsers import build_parser_registry
from rag_kb.infra.postgres.lexical import PgLexicalIndex
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.infra.postgres.repository import PostgresMetadataRepository
from rag_kb.infra.storage.local import LocalStorage

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "demo"


class FakeEmbedder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed_documents(self, texts):
        return [[1.0] * self.dimension for _ in texts]

    async def embed_query(self, text: str):
        return [1.0] * self.dimension


class FakeVLM:
    async def describe_image(
        self, image: bytes, *, mime: str = "image/png", prompt: str = ""
    ) -> str:
        return "一张流程图"


class FakeLLM:
    """图谱抽取与问答共用；用 schema 区分调用方。"""

    async def structured_json(self, messages, schema):
        if "answer" in schema.get("properties", {}):
            return {"answer": "年假为 5 个工作日 [1]。", "citations": [1]}

        text = messages[0].content
        if "年假" in text:
            return {
                "entities": [
                    {"name": "年假政策", "type": "条款", "span": "年假"},
                    {"name": "员工", "type": "人物", "span": "员工"},
                ],
                "relations": [
                    {
                        "head": "年假政策",
                        "tail": "员工",
                        "relation": "关联",
                        "evidence_span": "年假",
                    }
                ],
            }
        if "知识库服务" in text:
            return {
                "entities": [
                    {"name": "知识库服务", "type": "产品", "span": "知识库服务"},
                    {"name": "部署", "type": "流程", "span": "部署"},
                ],
                "relations": [
                    {
                        "head": "知识库服务",
                        "tail": "部署",
                        "relation": "使用",
                        "evidence_span": "部署",
                    }
                ],
            }
        return {"entities": [], "relations": []}

    async def chat(self, messages, *, temperature: float = 0.0, max_tokens=None) -> str:
        return ""


async def test_upload_index_search_ask_and_reindex(tmp_path: Path) -> None:
    settings = get_settings()
    apply_migrations(settings.pg_dsn)

    repo = PostgresMetadataRepository(settings.pg_dsn, min_size=1, max_size=2)
    lexical = PgLexicalIndex(settings.pg_dsn, min_size=1, max_size=2)
    vector = MilvusVectorStore(settings.milvus_uri, collection_name=settings.milvus_collection)
    graph = Neo4jGraphStore(
        settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    storage = LocalStorage(tmp_path / "data")
    llm = FakeLLM()
    embedder = FakeEmbedder(settings.embedding_dim)
    parser = build_parser_registry()

    await repo.open()
    await lexical.open()
    await vector.ensure_collection(settings.embedding_dim)
    await graph.ensure_schema()

    indexer = IndexingService(
        repo,
        embedder=embedder,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
        llm=llm,
        chunker_version=settings.chunker_version,
        embed_model="fake-embed",
    )
    pipeline = IngestPipeline(
        repo,
        storage,
        parser,
        FakeVLM(),
        chunker_version=settings.chunker_version,
        target_tokens=settings.chunk_target_tokens,
        overlap_ratio=settings.chunk_overlap_ratio,
        parent_max_tokens=settings.parent_max_tokens,
        chars_per_token=settings.chars_per_token,
        embed_model="fake-embed",
        indexer=indexer,
    )
    documents = DocumentService(
        repo,
        storage,
        max_upload_mb=settings.max_upload_mb,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
    )
    kbs = KbService(
        repo,
        storage,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
    )
    search = SearchService(
        repo,
        embedder=embedder,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
    )
    qa = QAService(
        repo,
        llm=llm,
        max_context_tokens=settings.qa_max_context_tokens,
        parent_max_tokens=settings.parent_max_tokens,
        chars_per_token=settings.chars_per_token,
    )
    reindex = ReindexService(repo, indexer)

    kb = await kbs.create_kb(f"pipeline-{uuid4().hex[:8]}")
    try:
        uploaded = []
        for name in ("employee_handbook.pdf", "product_guide.docx"):
            data = (FIXTURES / name).read_bytes()
            mime = (
                "application/pdf"
                if name.endswith(".pdf")
                else ("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            )
            document = await documents.upload_document(kb.id, filename=name, mime=mime, data=data)
            job = await repo.claim_ingest_job()
            assert job is not None and job.doc_id == document.id
            result = await pipeline.process(job)
            assert result.stage == "ready"
            uploaded.append(await documents.get_document(kb.id, document.id))

        assert all(doc.status is DocumentStatus.READY for doc in uploaded)
        assert all(doc.chunk_count > 0 for doc in uploaded)

        modes = [
            SearchMode.VECTOR,
            SearchMode.KEYWORD,
            SearchMode.GRAPH,
            SearchMode.HYBRID,
        ]
        for mode in modes:
            hits = await search.search(kb.id, "年假", mode=mode, top_k=5)
            assert hits, f"{mode.value} 通道没有命中"
            assert all(hit.doc_id in {doc.id for doc in uploaded} for hit in hits)

        answer = await qa.answer(kb.id, "年假有几天？", hits)
        assert answer.no_evidence is False
        assert answer.citations == [1]
        cited = answer.evidence[answer.citations[0] - 1]
        assert cited.body.doc_id in {doc.id for doc in uploaded}
        assert cited.hit.doc_id == cited.body.doc_id

        reindexed = await reindex.reindex_kb(kb.id)
        assert {doc.id for doc in reindexed} == {doc.id for doc in uploaded}
        after = await search.search(kb.id, "年假", mode=SearchMode.HYBRID, top_k=5)
        assert after, "reindex 后 hybrid 仍应有命中"
    finally:
        await kbs.delete_kb(kb.id)
        await vector.close()
        await graph.close()
        await lexical.close()
        await repo.close()
