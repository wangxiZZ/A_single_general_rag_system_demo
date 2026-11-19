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
from rag_kb.api.routers import documents, health, kbs
from rag_kb.application.document_service import DocumentService
from rag_kb.application.ingest_pipeline import IngestPipeline
from rag_kb.application.kb_service import KbService
from rag_kb.config import Settings, get_settings
from rag_kb.domain.ports import MetadataRepository
from rag_kb.infra.health import collect_readiness
from rag_kb.infra.llm.openai_compat import OpenAICompatVLM
from rag_kb.infra.milvus.store import MilvusVectorStore
from rag_kb.infra.parsers import build_parser_registry
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
    app.state.kb_service = KbService(repository, storage)
    app.state.document_service = DocumentService(
        repository,
        storage,
        max_upload_mb=settings.max_upload_mb,
    )

    vector_store: MilvusVectorStore | None = None
    if created_repository is not None:
        vector_store = MilvusVectorStore(
            settings.milvus_uri,
            collection_name=settings.milvus_collection,
            token=settings.milvus_token,
        )
        await vector_store.ensure_collection(settings.embedding_dim)
        app.state.vector_store = vector_store

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
    if vlm is not None:
        await vlm.aclose()
    if vector_store is not None:
        await vector_store.close()
    if created_repository is not None:
        await created_repository.close()


def create_app(
    settings: Settings | None = None,
    repository: MetadataRepository | None = None,
    *,
    start_worker: bool = True,
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
    app.state.start_worker = start_worker
    app.state.readiness_probe = lambda: collect_readiness(resolved)

    app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")
    app.include_router(health.router)
    app.include_router(kbs.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
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
