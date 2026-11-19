"""图谱抽取评测：实体/关系 P/R/F1、类型准确率、evidence_span 有效率。

评测对象是「入库后 Neo4j 里实际留下的图」，不是模型原始输出——被 span 校验丢弃的
实体/关系本来就不该算命中。金标见 `tests/fixtures/graph/gold.json`，夹具由
`scripts/make_graph_fixtures.py` 生成。

用法：

    uv run python scripts/graph_eval.py \
        --out-md docs/graph-eval.md --out-json scripts/bench/results/graph-eval.json
    uv run python scripts/graph_eval.py --kb <已有 kb id>      # 只评不装载
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from neo4j import GraphDatabase

from rag_kb.application.document_service import DocumentService
from rag_kb.application.graph_extract import normalize_name
from rag_kb.application.indexing import IndexingService
from rag_kb.application.ingest_pipeline import IngestPipeline
from rag_kb.application.kb_service import KbService
from rag_kb.config import get_settings
from rag_kb.domain.models import KB, DocumentStatus
from rag_kb.infra.milvus.store import MilvusVectorStore
from rag_kb.infra.neo4j.store import Neo4jGraphStore
from rag_kb.infra.parsers import build_parser_registry
from rag_kb.infra.postgres.lexical import PgLexicalIndex
from rag_kb.infra.postgres.migrations import apply_migrations
from rag_kb.infra.postgres.repository import PostgresMetadataRepository
from rag_kb.infra.storage.local import LocalStorage
from scripts.bench.run_bench import _build_models

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "graph"
GOLD_FILE = FIXTURES / "gold.json"
MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def entity_key(name: str, type_name: str) -> tuple[str, str]:
    return normalize_name(name), type_name


def relation_key(head: str, relation: str, tail: str) -> tuple[str, str, str]:
    return normalize_name(head), relation, normalize_name(tail)


def score_entities(
    extracted: Sequence[dict[str, Any]], gold: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], float]:
    """返回 P/R/F1 与「名称命中时类型一致」的准确率。"""
    extracted_keys = {entity_key(row["name"], row["type"]) for row in extracted}
    gold_keys = {entity_key(row["name"], row["type"]) for row in gold}
    tp = len(extracted_keys & gold_keys)
    result = _prf(tp, len(extracted_keys - gold_keys), len(gold_keys - extracted_keys))

    gold_type_by_name = {normalize_name(row["name"]): row["type"] for row in gold}
    name_matches = [row for row in extracted if normalize_name(row["name"]) in gold_type_by_name]
    correct = sum(
        1 for row in name_matches if row["type"] == gold_type_by_name[normalize_name(row["name"])]
    )
    type_accuracy = _ratio(correct, len(name_matches))
    result["type_accuracy"] = round(type_accuracy, 4)
    result["type_matched"] = len(name_matches)
    return result, type_accuracy


def score_relations(
    extracted: Sequence[dict[str, Any]], gold: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """按 (head, relation, tail) 去重后比较；同一三元组落在多个 chunk 只算一次。"""
    extracted_keys = {relation_key(row["head"], row["relation"], row["tail"]) for row in extracted}
    gold_keys = {relation_key(row["head"], row["relation"], row["tail"]) for row in gold}
    tp = len(extracted_keys & gold_keys)
    return _prf(tp, len(extracted_keys - gold_keys), len(gold_keys - extracted_keys))


def span_is_valid(span: Any, content: str | None) -> bool:
    if content is None or not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    start, end = int(span[0]), int(span[1])
    if start < 0 or end < start or end > len(content):
        return False
    return bool(content[start:end].strip())


def _fetch_graph(settings, kb_id) -> tuple[list[dict], list[dict]]:
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        with driver.session(database=settings.neo4j_database) as session:
            entities = session.run(
                "MATCH (e:Entity {kb_id: $kb}) "
                "RETURN e.name AS name, e.normalized AS normalized, e.type AS type",
                kb=str(kb_id),
            ).data()
            relations = session.run(
                "MATCH (h:Entity {kb_id: $kb})-[r:REL]->(t:Entity {kb_id: $kb}) "
                "RETURN h.name AS head, h.normalized AS head_norm, h.type AS head_type, "
                "r.type AS relation, t.name AS tail, t.normalized AS tail_norm, "
                "t.type AS tail_type, r.chunk_id AS chunk_id, "
                "r.evidence_span AS evidence_span, r.doc_id AS doc_id",
                kb=str(kb_id),
            ).data()
    finally:
        driver.close()
    return entities, relations


async def _ingest_fixtures(settings, repo, documents, pipeline) -> tuple[KB, dict[str, Any]]:
    kb = await repo.create_kb(KB(id=uuid4(), name=f"graph-eval-{int(time.time())}"))
    mapping: dict[str, Any] = {}
    for path in sorted(FIXTURES.iterdir()):
        if path.suffix.lower() not in MIME:
            continue
        document = await documents.upload_document(
            kb.id, filename=path.name, mime=MIME[path.suffix.lower()], data=path.read_bytes()
        )
        job = await repo.claim_ingest_job()
        if job is None or job.doc_id != document.id:
            raise RuntimeError("任务队列里出现了其他文档的任务，请先清空等待中的 ingest_job")
        await pipeline.process(job)
        loaded = await documents.get_document(kb.id, document.id)
        if loaded.status is not DocumentStatus.READY:
            raise RuntimeError(f"{path.name} 未就绪：{loaded.status} {loaded.error}")
        mapping[path.name] = loaded
        print(f"  {path.name}: ready, chunks={loaded.chunk_count}")
    return kb, mapping


def _markdown(
    entity_score: dict[str, Any],
    relation_score: dict[str, Any],
    span: dict[str, Any],
    missing: list[str],
    extra: list[str],
    entities_count: int,
    relations_count: int,
    gold: dict[str, Any],
    kb_name: str,
) -> str:
    lines = [
        "# 图谱抽取评测结果",
        "",
        f"- 知识库：`{kb_name}`",
        f"- 金标：{len(gold['entities'])} 个实体、{len(gold['relations'])} 条关系",
        f"- 实抽：{entities_count} 个实体、{relations_count} 条关系边（去重前）",
        "",
        "## 汇总",
        "",
        "| 指标 | Precision | Recall | F1 | TP/FP/FN |",
        "|---|---|---|---|---|",
        f"| 实体（名称+类型） | {entity_score['precision']:.3f} | "
        f"{entity_score['recall']:.3f} | {entity_score['f1']:.3f} | "
        f"{entity_score['tp']}/{entity_score['fp']}/{entity_score['fn']} |",
        f"| 关系（head+type+tail） | {relation_score['precision']:.3f} | "
        f"{relation_score['recall']:.3f} | {relation_score['f1']:.3f} | "
        f"{relation_score['tp']}/{relation_score['fp']}/{relation_score['fn']} |",
        "",
        f"- 实体类型准确率（名称命中时）：{entity_score['type_accuracy']:.3f} "
        f"（{entity_score['type_matched']} 个名称命中）",
        f"- evidence_span 有效率：{span['valid']}/{span['total']} = {span['rate']:.3f}",
        "",
        "## 漏抽关系（金标有、图里没有）",
        "",
    ]
    if missing:
        lines.extend(f"- {item}" for item in missing[:30])
    else:
        lines.append("- 无")
    lines.extend(["", "## 多抽关系（图里有、金标没有）", ""])
    if extra:
        lines.extend(f"- {item}" for item in extra[:30])
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "> 多抽包含对抗语料故意制造的噪声（否定句、代词、表格、图片、错误事实），",
            "> 因此 Precision 低于 1 不一定是缺陷；重点看金标关系的 Recall 与 span 有效率。",
        ]
    )
    return "\n".join(lines) + "\n"


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="图谱抽取评测")
    parser.add_argument("--kb", default=None, help="已有 kb id；省略则新建并装载夹具")
    parser.add_argument("--gold", type=Path, default=GOLD_FILE)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--keep-kb", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not (settings.llm_api_key and settings.embed_api_key):
        print("缺少 LLM_API_KEY / EMBED_API_KEY，无法评测")
        return 2
    apply_migrations(settings.pg_dsn)
    gold = json.loads(args.gold.read_text(encoding="utf-8"))

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
    documents = DocumentService(
        repo,
        storage,
        max_upload_mb=settings.max_upload_mb,
        vector_store=vector,
        lexical_index=lexical,
        graph_store=graph,
    )
    kbs = KbService(repo, storage, vector_store=vector, lexical_index=lexical, graph_store=graph)

    created_kb = None
    try:
        if args.kb:
            kb_id = UUID(args.kb)
            kb = await repo.get_kb(kb_id)
            if kb is None:
                print(f"知识库不存在：{args.kb}")
                return 1
            kb_name = kb.name
        else:
            print("装载图谱评测夹具…")
            created_kb, _ = await _ingest_fixtures(settings, repo, documents, pipeline)
            kb_id, kb_name = created_kb.id, created_kb.name

        entities, relations = _fetch_graph(settings, kb_id)
        chunks = {
            chunk.id: chunk
            for chunk in await repo.get_chunks(kb_id, [row["chunk_id"] for row in relations])
        }
        valid = sum(
            1
            for row in relations
            if span_is_valid(
                row.get("evidence_span"),
                chunks[row["chunk_id"]].content if row["chunk_id"] in chunks else None,
            )
        )
        span = {
            "valid": valid,
            "total": len(relations),
            "rate": round(_ratio(valid, len(relations)), 4),
        }

        entity_score, _ = score_entities(entities, gold["entities"])
        relation_score = score_relations(relations, gold["relations"])
        gold_keys = {
            relation_key(row["head"], row["relation"], row["tail"]) for row in gold["relations"]
        }
        extracted_keys = {
            relation_key(row["head"], row["relation"], row["tail"]) for row in relations
        }
        missing = [
            f"{row['head']} -[{row['relation']}]-> {row['tail']}"
            for row in gold["relations"]
            if relation_key(row["head"], row["relation"], row["tail"]) not in extracted_keys
        ]
        extra = [
            f"{row['head']} -[{row['relation']}]-> {row['tail']}"
            for row in relations
            if relation_key(row["head"], row["relation"], row["tail"]) not in gold_keys
        ]

        report = _markdown(
            entity_score,
            relation_score,
            span,
            missing,
            extra,
            len(entities),
            len(relations),
            gold,
            kb_name,
        )
        print(report)
        if args.out_md:
            args.out_md.parent.mkdir(parents=True, exist_ok=True)
            args.out_md.write_text(report, encoding="utf-8")
            print(f"markdown -> {args.out_md}")
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "kb_id": str(kb_id),
                "kb_name": kb_name,
                "entities": entity_score,
                "relations": relation_score,
                "span": span,
                "missing_relations": missing,
                "extra_relations": extra,
                "extracted_entities": entities,
                "extracted_relations": relations,
            }
            args.out_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"json -> {args.out_json}")
    finally:
        if created_kb is not None and not args.keep_kb:
            try:
                await kbs.delete_kb(created_kb.id)
            except Exception as exc:
                print(f"清理评测知识库失败：{exc}")
        await vector.close()
        await graph.close()
        await lexical.close()
        await repo.close()
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
