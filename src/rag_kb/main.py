"""应用装配入口。

职责只有三件：读配置、构造端口实现、挂载路由。业务逻辑一律不放在这里，也不放在
路由里——路由只做参数校验与调用 application 层用例。

M0 阶段只装配健康检查；M1 起按里程碑依次挂载 kb / document / search / ask 路由与
Web 页面，装配顺序与 ``docs/实施计划.md`` 的里程碑一一对应。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_kb import __version__
from rag_kb.api.routers import health
from rag_kb.config import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    # 目录创建放在 lifespan 而非 import 期：import 期做副作用会让"只想拿 app 对象"
    # 的测试与工具在无写权限的环境里直接失败。
    settings.ensure_dirs()

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

    # TODO(M3): 在此建立 Postgres 连接池、Milvus 客户端、Neo4j driver，做向量维度
    # fail-fast 校验，并把 ingest worker 作为后台任务拉起。
    yield

    # TODO(M3): 关闭连接池与 driver，等待 worker 退出，保证 Docker stop 时优雅收尾。


def create_app(settings: Settings | None = None) -> FastAPI:
    """工厂函数：测试可注入自定义 Settings，无需改写环境变量。"""
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

    app.include_router(health.router)

    @app.get("/", include_in_schema=False)
    def index() -> dict[str, object]:
        # M1 会替换为 Web 管理页；在此之前给出可用的自描述入口，避免访问根路径 404。
        return {
            "name": resolved.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/healthz",
            "ready": "/readyz",
        }

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
