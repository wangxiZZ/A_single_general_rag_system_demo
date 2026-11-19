"""批量跑分：`mode × rerank`，输出 markdown 表与 JSON 原始结果。

为什么在进程内跑而不是打 HTTP：
指标需要命中块的 ``char_start/char_end`` 才能与金标区间比对，而 REST 返回的是截断后的
snippet。进程内直接装配同一套 application 服务，既复用真实检索逻辑，又能从仓储取回
完整 Chunk。金标里的 ``doc_id`` 每次装载都不同，这里按 ``doc_name`` 解析。

维度只跑 ``mode × rerank``；不跑 chunk_size——换尺寸要重建索引并重算金标，
不属于本阶段预算。

用法：

    uv run python scripts/bench/run_bench.py --rerank off
    uv run python scripts/bench/run_bench.py --modes hybrid --rerank both \
        --out-md docs/benchmark.md --out-json scripts/bench/results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from rag_kb.application.document_service import DocumentService
from rag_kb.application.indexing import IndexingService
from rag_kb.application.ingest_pipeline import IngestPipeline
from rag_kb.application.kb_service import KbService
from rag_kb.application.search_service import SearchService
from rag_kb.config import get_settings
from rag_kb.domain.models import KB, DocumentStatus, SearchMode
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
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.infra.postgres.repository import PostgresMetadataRepository
from rag_kb.infra.storage.local import LocalStorage
from scripts.bench.metrics import aggregate, score_query

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "demo"
QA_FILE = REPO_ROOT / "tests" / "fixtures" / "bench" / "qa.jsonl"
ALL_MODES = [
    SearchMode.VECTOR,
    SearchMode.KEYWORD,
    SearchMode.GRAPH,
    SearchMode.HYBRID,
    SearchMode.HYBRID_WEIGHTED,
]


def _load_qa(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _build_models(settings):
    llm = None
    if settings.llm_api_key:
        llm = OpenAICompatLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            max_concurrency=settings.llm_max_concurrency,
        )
    embedder = None
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
    vlm = None
    if settings.vlm_enabled and settings.vlm_api_key:
        vlm = OpenAICompatVLM(
            base_url=settings.vlm_base_url,
            api_key=settings.vlm_api_key,
            model=settings.vlm_model,
            timeout_s=settings.vlm_timeout_s,
            max_concurrency=settings.vlm_max_concurrency,
        )
    return llm, embedder, vlm


async def _prepare_kb(repo, docs, pipeline) -> tuple[KB, dict[str, Any]]:
    kb = await repo.create_kb(KB(id=uuid4(), name=f"bench-{int(time.time())}"))
    mapping: dict[str, Any] = {}
    for filename in sorted(
        p.name for p in FIXTURES.iterdir() if p.suffix.lower() in {".pdf", ".docx"}
    ):
        path = FIXTURES / filename
        mime = (
            "application/pdf"
            if filename.endswith(".pdf")
            else ("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        )
        document = await docs.upload_document(
            kb.id, filename=filename, mime=mime, data=path.read_bytes()
        )
        job = await repo.claim_ingest_job()
        if job is None or job.doc_id != document.id:
            raise RuntimeError("任务队列里出现了其他文档的任务，请先清空等待中的 ingest_job")
        await pipeline.process(job)
        loaded = await docs.get_document(kb.id, document.id)
        if loaded.status is not DocumentStatus.READY:
            raise RuntimeError(f"{filename} 未就绪：{loaded.status} {loaded.error}")
        mapping[filename] = loaded
        print(f"  {filename}: ready, chunks={loaded.chunk_count}")
    return kb, mapping


async def _run_config(repo, search, qa_cases, resolved, kb_id, mode: SearchMode, rerank: bool):
    scores, ndcgs, latencies, raw = [], [], [], []
    for case in qa_cases:
        started = time.perf_counter()
        hits = await search.search(kb_id, case["question"], mode=mode, top_k=10, rerank=rerank)
        latencies.append((time.perf_counter() - started) * 1000)
        chunks = {
            chunk.id: chunk for chunk in await repo.get_chunks(kb_id, [h.chunk_id for h in hits])
        }
        score, ndcg = score_query(
            case["id"], case["question"], case["type"], hits, chunks, resolved[case["id"]]
        )
        scores.append(score)
        ndcgs.append(ndcg)
        raw.append(
            {
                "id": case["id"],
                "type": case["type"],
                "first_rank": score.first_rank,
                "ndcg10": round(ndcg, 4),
                "hits": [hit.chunk_id for hit in hits],
            }
        )
    summary = aggregate(scores, ndcgs)
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    return summary, avg_latency, raw


def _markdown(rows: list[dict[str, Any]], qa_count: int) -> str:
    lines = [
        "| mode | rerank | Recall@5 | Recall@10 | MRR | nDCG@10 | 不可回答命中空 | 平均延迟 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        summary = row["summary"]
        lines.append(
            f"| {row['mode']} | {'on' if row['rerank'] else 'off'} "
            f"| {summary['recall5']:.3f} | {summary['recall10']:.3f} "
            f"| {summary['mrr']:.3f} | {summary['ndcg10']:.3f} "
            f"| {summary['unanswerable_empty']}/{summary['unanswerable']} "
            f"| {row['avg_latency_ms']:.0f} |"
        )
    lines.append("")
    lines.append(f"> 样本：{qa_count} 条（不可回答题不参与召回均值）；维度：mode × rerank。")
    return "\n".join(lines)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 检索批量评测")
    parser.add_argument("--qa", type=Path, default=QA_FILE)
    parser.add_argument("--modes", default=",".join(mode.value for mode in ALL_MODES))
    parser.add_argument("--rerank", choices=["off", "on", "both"], default="off")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--keep-kb", action="store_true", help="保留评测知识库便于排查")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not (settings.llm_api_key and settings.embed_api_key):
        print("缺少 LLM_API_KEY / EMBED_API_KEY，无法评测")
        return 2
    apply_migrations(settings.pg_dsn)

    qa_cases = _load_qa(args.qa)
    modes = [SearchMode(value.strip()) for value in args.modes.split(",") if value.strip()]
    rerank_flags = [False, True] if args.rerank == "both" else [args.rerank == "on"]

    repo = PostgresMetadataRepository(settings.pg_dsn, min_size=1, max_size=4)
    lexical = PgLexicalIndex(settings.pg_dsn, min_size=1, max_size=4)
    vector = MilvusVectorStore(settings.milvus_uri, collection_name=settings.milvus_collection)
    graph = Neo4jGraphStore(
        settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    await repo.open()
    await lexical.open()
    await vector.ensure_collection(settings.embedding_dim)
    await graph.ensure_schema()

    llm, embedder, vlm = _build_models(settings)
    storage = LocalStorage(settings.data_dir)
    indexer = IndexingService(
        repo,
        embedder=embedder,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
        llm=llm,
        chunker_version=settings.chunker_version,
        embed_model=settings.embed_model,
    )
    pipeline = IngestPipeline(
        repo,
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
    documents_service = DocumentService(
        repo,
        storage,
        max_upload_mb=settings.max_upload_mb,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
    )
    kbs = KbService(repo, storage, vector_store=vector, lexical_index=lexical, graph_store=graph)

    kb = None
    try:
        print("装载评测知识库…")
        kb, mapping = await _prepare_kb(repo, documents_service, pipeline)
        resolved = {
            case["id"]: [
                {**gold, "doc_id": str(mapping[gold["doc_name"]].id)} for gold in case["gold"]
            ]
            for case in qa_cases
        }

        rows: list[dict[str, Any]] = []
        for rerank in rerank_flags:
            search = SearchService(
                repo,
                embedder=embedder,
                vector_store=vector,
                lexical_index=lexical,
                graph_store=graph,
                reranker=LLMListwiseReranker(llm) if rerank and llm is not None else None,
                top_k=10,
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
            for mode in modes:
                label = f"{mode.value}{' + rerank' if rerank else ''}"
                print(f"跑分：{label}")
                summary, latency, raw = await _run_config(
                    repo, search, qa_cases, resolved, kb.id, mode, rerank
                )
                rows.append(
                    {
                        "mode": mode.value,
                        "rerank": rerank,
                        "summary": {
                            "answerable": summary.answerable,
                            "recall5": round(summary.recall5, 4),
                            "recall10": round(summary.recall10, 4),
                            "mrr": round(summary.mrr, 4),
                            "ndcg10": round(summary.ndcg10, 4),
                            "unanswerable": summary.unanswerable,
                            "unanswerable_empty": summary.unanswerable_empty,
                        },
                        "avg_latency_ms": round(latency, 1),
                        "queries": raw,
                    }
                )
                print(
                    f"  R@5={summary.recall5:.3f} R@10={summary.recall10:.3f} "
                    f"MRR={summary.mrr:.3f} nDCG@10={summary.ndcg10:.3f}"
                )

        table = _markdown(rows, len(qa_cases))
        print("\n" + table)
        if args.out_md:
            args.out_md.parent.mkdir(parents=True, exist_ok=True)
            args.out_md.write_text(table + "\n", encoding="utf-8")
            print(f"markdown -> {args.out_md}")
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(
                json.dumps(
                    {
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "chunker_version": settings.chunker_version,
                        "embed_model": settings.embed_model,
                        "qa_file": str(args.qa.relative_to(REPO_ROOT)),
                        "rows": rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"json -> {args.out_json}")
    finally:
        if kb is not None and not args.keep_kb:
            try:
                await kbs.delete_kb(kb.id)
            except Exception as exc:  # 清理失败不影响已产出的分数，但必须提示
                print(f"清理评测知识库失败：{exc}")
        await vector.close()
        await graph.close()
        await lexical.close()
        await repo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
