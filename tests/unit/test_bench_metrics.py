"""评测指标单测：命中判定与 Recall/MRR/nDCG 口径。"""

from __future__ import annotations

import math
from uuid import uuid4

from scripts.bench.metrics import aggregate, is_relevant, ndcg_at_k, score_query

from rag_kb.domain.models import Chunk, ChunkKind, ChunkLevel, SearchHit

KB_ID = uuid4()
DOC_ID = uuid4()


def _chunk(chunk_id: str, start: int, end: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        level=ChunkLevel.CHILD,
        kind=ChunkKind.TEXT,
        content="x" * max(end - start, 1),
        seq=0,
        char_start=start,
        char_end=end,
        token_count=1,
        chunker_version="v1",
    )


def _hit(chunk_id: str, *, page: int | None = 1, doc_name: str = "a.pdf") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=DOC_ID,
        doc_name=doc_name,
        level=ChunkLevel.CHILD,
        score=1.0,
        channels=("vector",),
        page_no=page,
    )


GOLD = [{"doc_name": "a.pdf", "doc_id": None, "page_no": 1, "char_start": 100, "char_end": 130}]


def test_overlap_matches_and_document_must_match() -> None:
    chunk = _chunk("c1", 120, 160)
    assert is_relevant(_hit("c1"), chunk, GOLD)
    assert not is_relevant(_hit("c1", doc_name="other.pdf"), chunk, GOLD)
    assert not is_relevant(_hit("c1"), _chunk("c2", 200, 240), GOLD)
    assert not is_relevant(_hit("c1"), None, GOLD)


def test_zero_length_gold_matches_by_page_and_point() -> None:
    image_gold = [
        {"doc_name": "a.pdf", "doc_id": None, "page_no": 1, "char_start": 164, "char_end": 164}
    ]
    assert is_relevant(_hit("c1"), _chunk("c1", 164, 164), image_gold)
    assert not is_relevant(_hit("c1", page=2), _chunk("c1", 164, 164), image_gold)
    assert not is_relevant(_hit("c1"), _chunk("c2", 200, 240), image_gold)


def test_score_query_rank_and_ndcg() -> None:
    chunks = {"c1": _chunk("c1", 0, 50), "c2": _chunk("c2", 100, 140)}
    hits = [_hit("c1"), _hit("c2")]
    score, ndcg = score_query("q1", "问题", "single_hop", hits, chunks, GOLD)

    assert score.first_rank == 2
    assert score.recall5 == 1.0
    assert score.mrr == 0.5
    assert ndcg == 1.0 / math.log2(3)


def test_no_relevant_hit_scores_zero() -> None:
    chunks = {"c1": _chunk("c1", 0, 50)}
    score, ndcg = score_query("q1", "问题", "single_hop", [_hit("c1")], chunks, GOLD)

    assert score.first_rank is None
    assert score.recall10 == 0.0
    assert score.mrr == 0.0
    assert ndcg == 0.0


def test_ndcg_ideal_counts_multiple_gold_spans() -> None:
    gold = [
        {"doc_name": "a.pdf", "page_no": 1, "char_start": 100, "char_end": 130},
        {"doc_name": "a.pdf", "page_no": 1, "char_start": 300, "char_end": 330},
    ]
    chunks = {"c1": _chunk("c1", 100, 130), "c2": _chunk("c2", 300, 330)}
    hits = [_hit("c1"), _hit("c2")]
    assert ndcg_at_k(hits, chunks, gold) == 1.0


def test_ndcg_credits_each_gold_span_once() -> None:
    """重叠分块可能有多条命中指向同一句原文，不能把 nDCG 推到 1 以上。"""
    chunks = {
        "c1": _chunk("c1", 100, 130),
        "c2": _chunk("c2", 95, 140),
    }
    hits = [_hit("c1"), _hit("c2")]
    value = ndcg_at_k(hits, chunks, GOLD)

    assert value == 1.0
    assert value <= 1.0


def test_aggregate_separates_unanswerable() -> None:
    chunks = {"c1": _chunk("c1", 100, 130)}
    scores, ndcgs = [], []
    for query_id, qtype, hits in [
        ("q1", "single_hop", [_hit("c1")]),
        ("q2", "multi_hop", [_hit("c1", page=3)]),
        ("q3", "unanswerable", []),
    ]:
        score, ndcg = score_query(query_id, "问题", qtype, hits, chunks, GOLD)
        scores.append(score)
        ndcgs.append(ndcg)

    result = aggregate(scores, ndcgs)
    assert result.answerable == 2
    assert result.recall5 == 0.5
    assert result.unanswerable == 1
    assert result.unanswerable_empty == 1
    assert result.unanswerable_accuracy == 1.0
