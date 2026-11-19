"""应用装配入口。

职责只有三件：读配置、构造端口实现、挂载路由。业务逻辑一律不放在这里，也不放在
路由里——路由只做参数校验与调用 application 层用例。

M0 阶段只装配健康检查；M1 起按里程碑依次挂载 kb / document / search / ask 路由与
Web 页面，装配顺序与 ``docs/实施计划.md`` 的里程碑一一对应。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from rag_kb import __version__
from rag_kb.api.errors import register_exception_handlers
from rag_kb.api.routers import ask, documents, health, kbs, search
from rag_kb.application.document_service import DocumentService
from rag_kb.application.indexing import IndexingService
from rag_kb.application.ingest_pipeline import IngestPipeline
from rag_kb.application.kb_service import KbService
from rag_kb.application.qa_service import QAService
from rag_kb.application.reindex import ReindexService
from rag_kb.application.search_service import SearchService
from rag_kb.config import Settings, get_settings
from rag_kb.domain.ports import MetadataRepository
from rag_kb.infra.health import collect_readiness
from rag_kb.infra.llm.openai_compat import (
    OpenAICompatEmbedder,
    OpenAICompatLLM,
    OpenAICompatVLM,
)
from rag_kb.infra.llm.rerank import LLMListwiseReranker
from rag_kb.infra.milvus.store import MilvusVectorStore
from rag_kb.infra.neo4j.store import Neo4jGraphStore
from rag_kb.infra.parsers import build_parser_registry
from rag_kb.infra.postgres.lexical import PgLexicalIndex
from rag_kb.infra.postgres.repository import PostgresMetadataRepository
from rag_kb.infra.storage.local import LocalStorage
from rag_kb.web import routes as web_routes
from rag_kb.worker.loop import IngestWorker

logger = logging.getLogger(__name__)

WEB_STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    # 目录创建放在 lifespan 而非 import 期：import 期做副作用会让"只想拿 app 对象"
    # 的测试与工具在无写权限的环境里直接失败。
    settings.ensure_dirs()

    # 测试可以注入 fake 仓储；只有应用自己创建时才由应用负责关闭。
    created_repository: PostgresMetadataRepository | None = None
    if app.state.repository is None:
        created_repository = PostgresMetadataRepository(
            settings.pg_dsn,
            min_size=settings.pg_pool_min,
            max_size=settings.pg_pool_max,
        )
        await created_repository.open()
        app.state.repository = created_repository

    repository: MetadataRepository = app.state.repository
    storage = LocalStorage(settings.data_dir)

    vector_store: MilvusVectorStore | None = None
    if created_repository is not None:
        vector_store = MilvusVectorStore(
            settings.milvus_uri,
            collection_name=settings.milvus_collection,
            token=settings.milvus_token,
        )
        await vector_store.ensure_collection(settings.embedding_dim)
        app.state.vector_store = vector_store

    lexical_index: PgLexicalIndex | None = None
    if created_repository is not None:
        lexical_index = PgLexicalIndex(
            settings.pg_dsn,
            min_size=settings.pg_pool_min,
            max_size=settings.pg_pool_max,
        )
        await lexical_index.open()
        app.state.lexical_index = lexical_index

    graph_store: Neo4jGraphStore | None = None
    if created_repository is not None:
        graph_store = Neo4jGraphStore(
            settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        await graph_store.ensure_schema()
        app.state.graph_store = graph_store

    app.state.kb_service = KbService(
        repository,
        storage,
        vector_store=vector_store,
        lexical_index=lexical_index,
        graph_store=graph_store,
    )
    app.state.document_service = DocumentService(
        repository,
        storage,
        max_upload_mb=settings.max_upload_mb,
        vector_store=vector_store,
        lexical_index=lexical_index,
        graph_store=graph_store,
    )

    embedder: OpenAICompatEmbedder | None = None
    if settings.embed_api_key:
        embedder = OpenAICompatEmbedder(
            base_url=settings.embed_base_url,
            api_key=settings.embed_api_key,
            model=settings.embed_model,
            timeout_s=settings.embed_timeout_s,
            max_concurrency=settings.embed_max_concurrency,
            dimension=settings.embedding_dim,
            batch_size=settings.embed_batch_size,
        )
    app.state.embedder = embedder

    llm: OpenAICompatLLM | None = None
    if settings.llm_api_key:
        llm = OpenAICompatLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            max_concurrency=settings.llm_max_concurrency,
        )
    if app.state.search_service is None:
        app.state.search_service = SearchService(
            repository,
            embedder=embedder,
            vector_store=vector_store,
            lexical_index=lexical_index,
            graph_store=graph_store,
            reranker=LLMListwiseReranker(llm) if llm is not None else None,
            top_k=settings.retrieval_top_k,
            rrf_k=settings.fusion_rrf_k,
            graph_seed_limit=settings.graph_seed_limit,
            graph_max_hops=settings.graph_max_hops,
            graph_neighbor_limit=settings.graph_neighbor_limit,
            hybrid_weights={
                "vector": settings.hybrid_weight_vector,
                "keyword": settings.hybrid_weight_keyword,
                "graph": settings.hybrid_weight_graph,
            },
        )

    if app.state.qa_service is None:
        app.state.qa_service = QAService(
            repository,
            llm=llm,
            max_context_tokens=settings.qa_max_context_tokens,
            parent_max_tokens=settings.parent_max_tokens,
            chars_per_token=settings.chars_per_token,
        )

    indexer = IndexingService(
        repository,
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
        graph_store=graph_store,
        llm=llm,
        chunker_version=settings.chunker_version,
        embed_model=settings.embed_model,
    )
    if app.state.reindex_service is None:
        app.state.reindex_service = ReindexService(repository, indexer)

    worker_task: asyncio.Task[None] | None = None
    stop_event = asyncio.Event()
    vlm: OpenAICompatVLM | None = None

    if app.state.start_worker:
        if settings.vlm_enabled and settings.vlm_api_key:
            vlm = OpenAICompatVLM(
                base_url=settings.vlm_base_url,
                api_key=settings.vlm_api_key,
                model=settings.vlm_model,
                timeout_s=settings.vlm_timeout_s,
                max_concurrency=settings.vlm_max_concurrency,
            )
        pipeline = IngestPipeline(
            repository,
            storage,
            build_parser_registry(),
            vlm,
            chunker_version=settings.chunker_version,
            target_tokens=settings.chunk_target_tokens,
            overlap_ratio=settings.chunk_overlap_ratio,
            parent_max_tokens=settings.parent_max_tokens,
            chars_per_token=settings.chars_per_token,
            embed_model=settings.embed_model,
            indexer=indexer,
        )
        worker = IngestWorker(repository, pipeline.process, poll_interval_s=1.0)
        worker_task = asyncio.create_task(worker.run(stop_event), name="ingest-worker")
        app.state.worker_task = worker_task

    configured = settings.model_api_configured
    logger.info(
        "rag-kb %s 启动 env=%s data_dir=%s 模型配置 llm=%s embedding=%s vlm=%s",
        __version__,
        settings.app_env,
        settings.data_dir,
        configured["llm"],
        configured["embedding"],
        configured["vlm"],
    )
    if not all(configured.values()):
        logger.warning("存在未配置的模型能力，相关功能将不可用；请检查 .env")

    # TODO(M3): 建立 Milvus 客户端与 Neo4j driver，做向量维度 fail-fast 校验。
    yield

    if worker_task is not None:
        stop_event.set()
        try:
            await asyncio.wait_for(worker_task, timeout=10)
        except TimeoutError:
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task
        except Exception:
            logger.exception("ingest worker stopped with error")
    if embedder is not None:
        await embedder.aclose()
    if llm is not None:
        await llm.aclose()
    if vlm is not None:
        await vlm.aclose()
    if vector_store is not None:
        await vector_store.close()
    if lexical_index is not None:
        await lexical_index.close()
    if graph_store is not None:
        await graph_store.close()
    if created_repository is not None:
        await created_repository.close()


def create_app(
    settings: Settings | None = None,
    repository: MetadataRepository | None = None,
    *,
    start_worker: bool = True,
    search_service: SearchService | None = None,
    reindex_service: ReindexService | None = None,
    qa_service: QAService | None = None,
) -> FastAPI:
    """工厂函数：测试可注入自定义 Settings / fake 仓储，并关闭后台 worker。"""
    resolved = settings or get_settings()

    app = FastAPI(
        title="通用 RAG 知识库服务",
        version=__version__,
        description=(
            "文档解析 → 向量/关键词/图谱三路召回 → 基于证据回答。"
            "MCP 与 Skill 均复用本服务的 REST 能力。"
        ),
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.repository = repository
    app.state.search_service = search_service
    app.state.reindex_service = reindex_service
    app.state.qa_service = qa_service
    app.state.reindex_tasks = {}
    app.state.start_worker = start_worker
    app.state.readiness_probe = lambda: collect_readiness(resolved)

    app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")
    app.include_router(health.router)
    app.include_router(kbs.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(ask.router, prefix="/api")
    app.include_router(web_routes.router)
    register_exception_handlers(app)

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/web/kbs", status_code=303)

    return app


app = create_app()


def run() -> None:
    """``rag-kb`` 命令行入口，等价于 uvicorn 启动。"""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "rag_kb.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
    )
