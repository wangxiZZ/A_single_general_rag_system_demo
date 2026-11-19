"""健康与就绪检查路由。

与业务路由分开：探针由编排系统高频调用，不应与业务共用鉴权、限流或数据库连接池，
避免依赖故障时探针也一起挂掉而失去诊断信息。
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from rag_kb import __version__
from rag_kb.infra.health import collect_readiness

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="进程存活探针")
def healthz() -> dict[str, str]:
    """只回答"进程是否在跑"，不碰任何外部依赖。"""
    return {"status": "ok", "version": __version__}


@router.get("/readyz", summary="依赖就绪探针")
def readyz(request: Request, response: Response) -> dict[str, object]:
    """逐项报告 Postgres / Milvus / Neo4j 连通性，以及模型配置状态。

    任一 required 项失败返回 503，便于编排系统据此不把流量打进来。
    """
    settings = request.app.state.settings
    checks, ready = collect_readiness(settings)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ready, "checks": [item.as_dict() for item in checks]}
