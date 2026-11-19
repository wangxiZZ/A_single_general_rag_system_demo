"""依赖健康探测。

设计原因
--------
``/readyz`` 探测的是**真实连通性**而不是启动时缓存的结果。容器编排（compose、
k8s）只看进程存活会把"进程在跑但数据库已断"误判为就绪；而探测频率很低（编排
探针 + 人工排查），不值得为它加缓存与失效策略。

两条刻意的取舍：

1. **不发起推理请求**。LLM/Embedding 是外部付费 API，用真实调用做探针既烧 token，
   也会把第三方的瞬时抖动算成本服务未就绪。这里只校验配置是否填写，标记
   ``required=False``。
2. **客户端库惰性导入**。任一存储客户端缺失时应表现为"该项检查失败"，而不是
   import 阶段整个进程起不来——否则部署排查会丢掉最有用的错误信息。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from rag_kb.config import Settings

PROBE_TIMEOUT_S = 5


@dataclass(frozen=True, slots=True)
class CheckResult:
    """单项依赖检查结果。

    ``required=False`` 的项失败不影响整体就绪，只在响应里如实展示。
    """

    name: str
    ok: bool
    required: bool
    latency_ms: float
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "required": self.required,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
        }


def _run(name: str, required: bool, probe: Callable[[], str]) -> CheckResult:
    started = time.perf_counter()
    ok = True
    detail = ""
    try:
        detail = probe()
    except Exception as exc:  # 探测层必须兜住一切异常并转成报告，不能让它冒泡
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    return CheckResult(name=name, ok=ok, required=required, latency_ms=latency_ms, detail=detail)


def check_postgres(settings: Settings) -> CheckResult:
    def probe() -> str:
        import psycopg

        with psycopg.connect(settings.pg_dsn, connect_timeout=PROBE_TIMEOUT_S) as conn:
            row = conn.execute("select current_setting('server_version')").fetchone()
        return f"postgres {row[0]}" if row else "postgres ok"

    return _run("postgres", True, probe)


def check_milvus(settings: Settings) -> CheckResult:
    def probe() -> str:
        from pymilvus import MilvusClient

        kwargs: dict[str, object] = {
            "uri": settings.milvus_uri,
            "timeout": PROBE_TIMEOUT_S,
        }
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        client = MilvusClient(**kwargs)  # type: ignore[arg-type]
        try:
            collections = client.list_collections()
        finally:
            client.close()
        return f"milvus ok, collections={len(collections)}"

    return _run("milvus", True, probe)


def check_neo4j(settings: Settings) -> CheckResult:
    def probe() -> str:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=PROBE_TIMEOUT_S,
        )
        try:
            driver.verify_connectivity()
            with driver.session(database=settings.neo4j_database) as session:
                value = session.run("RETURN 1 AS v").single(strict=True)["v"]
        finally:
            driver.close()
        return f"neo4j ok (RETURN {value})"

    return _run("neo4j", True, probe)


def collect_readiness(settings: Settings) -> tuple[list[CheckResult], bool]:
    """返回（逐项结果，是否整体就绪）。整体就绪只看 required 项。"""
    results = [
        check_postgres(settings),
        check_milvus(settings),
        check_neo4j(settings),
    ]

    configured = settings.model_api_configured
    for name, ready in configured.items():
        results.append(
            CheckResult(
                name=f"model:{name}",
                ok=ready,
                required=False,
                latency_ms=0.0,
                detail="已配置" if ready else "未配置（可启动，但相关功能不可用）",
            )
        )

    ready_overall = all(item.ok for item in results if item.required)
    return results, ready_overall
