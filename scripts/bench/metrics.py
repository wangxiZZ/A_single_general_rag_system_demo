"""检索评测指标：Recall@K、MRR、nDCG@10。

为什么不用 chunk_id 当金标：
chunk id 随分块参数与 chunker_version 变化，换参数后金标全部失效。这里用「文档 +
页码/字符区间」判定命中——只要命中块与金标区间有交集（或图片按页命中），就算相关，
因此同一份金标可以横比分块与检索配置。

指标口径（写进 benchmark.md）：
- 相关性与名次无关，只看命中块与金标区间是否重叠；
- Recall@K：每条 query 取 top-K，只要有一个命中就算 1，最后求平均；
- MRR：第一个相关命中的倒数名次；
- nDCG@10：二值相关性，IDCG 按金标条目数（上限 10）计算；
- 不可回答题不参与 Recall/MRR/nDCG，单独统计「top-K 为空」的比例。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_kb.domain.models import Chunk, SearchHit


def _doc_matches(hit: SearchHit, gold: dict[str, Any]) -> bool:
    gold_doc_id = gold.get("doc_id")
    if gold_doc_id:
        return str(hit.doc_id) == str(gold_doc_id)
    gold_doc_name = gold.get("doc_name")
    return not gold_doc_name or hit.doc_name == gold_doc_name


def _matches_gold(hit: SearchHit, chunk: Chunk | None, gold: dict[str, Any]) -> bool:
    if chunk is None or not _doc_matches(hit, gold):
        return False
    start = int(gold.get("char_start") or 0)
    end = int(gold.get("char_end") or 0)
    page = gold.get("page_no")
    # 同库可能存在同字符区间的不同页（PDF 每页文本独立计数），页码必须一致；
    # docx 没有页码，两侧都是 None，不参与判断。
    if page is not None and hit.page_no is not None and hit.page_no != page:
        return False
    if start == end:
        # 图片/表格等没有正文区间的金标：按页码 + 字符点命中。
        if page is not None and hit.page_no != page:
            return False
        return chunk.char_start <= start <= chunk.char_end
    return max(0, min(chunk.char_end, end) - max(chunk.char_start, start)) > 0


def is_relevant(hit: SearchHit, chunk: Chunk | None, gold_spans: Sequence[dict[str, Any]]) -> bool:
    """命中块与任一金标区间重叠，或金标是零长度点时落在块范围内。"""
    return any(_matches_gold(hit, chunk, gold) for gold in gold_spans)


def _first_relevant_rank(
    hits: Sequence[SearchHit], chunks: dict[str, Chunk], gold: Sequence[dict[str, Any]]
) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, chunks.get(hit.chunk_id), gold):
            return rank
    return None


def ndcg_at_k(
    hits: Sequence[SearchHit],
    chunks: dict[str, Chunk],
    gold: Sequence[dict[str, Any]],
    k: int = 10,
) -> float:
    dcg = 0.0
    credited: set[int] = set()
    for rank, hit in enumerate(hits[:k], start=1):
        # 同一金标只记一次：重叠分块可能让多条命中指向同一句原文，
        # 若重复计分会得到 nDCG > 1。
        for index, span in enumerate(gold):
            if index in credited:
                continue
            if _matches_gold(hit, chunks.get(hit.chunk_id), span):
                credited.add(index)
                dcg += 1.0 / math.log2(rank + 1)
                break
    ideal_hits = min(len(gold), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


@dataclass(frozen=True, slots=True)
class QueryScore:
    query_id: str
    question: str
    qtype: str
    hits: int
    first_rank: int | None

    @property
    def recall5(self) -> float:
        return 1.0 if self.first_rank is not None and self.first_rank <= 5 else 0.0

    @property
    def recall10(self) -> float:
        return 1.0 if self.first_rank is not None and self.first_rank <= 10 else 0.0

    @property
    def mrr(self) -> float:
        return 0.0 if self.first_rank is None else 1.0 / self.first_rank


def score_query(
    query_id: str,
    question: str,
    qtype: str,
    hits: Sequence[SearchHit],
    chunks: dict[str, Chunk],
    gold: Sequence[dict[str, Any]],
) -> tuple[QueryScore, float]:
    first_rank = _first_relevant_rank(hits, chunks, gold)
    score = QueryScore(
        query_id=query_id,
        question=question,
        qtype=qtype,
        hits=len(hits),
        first_rank=first_rank,
    )
    return score, ndcg_at_k(hits, chunks, gold)


@dataclass(frozen=True, slots=True)
class Aggregate:
    answerable: int
    recall5: float
    recall10: float
    mrr: float
    ndcg10: float
    unanswerable: int
    unanswerable_empty: int

    @property
    def unanswerable_accuracy(self) -> float:
        return 0.0 if self.unanswerable == 0 else self.unanswerable_empty / self.unanswerable


def aggregate(scores: Sequence[QueryScore], ndcgs: Sequence[float]) -> Aggregate:
    # 不可回答题由 qtype 区分，单独统计；不混入召回均值。
    answerable = [score for score in scores if score.qtype != "unanswerable"]
    unanswerable = [score for score in scores if score.qtype == "unanswerable"]
    answerable_ndcgs = [
        ndcg for score, ndcg in zip(scores, ndcgs, strict=True) if score.qtype != "unanswerable"
    ]
    count = len(answerable) or 1
    return Aggregate(
        answerable=len(answerable),
        recall5=sum(score.recall5 for score in answerable) / count,
        recall10=sum(score.recall10 for score in answerable) / count,
        mrr=sum(score.mrr for score in answerable) / count,
        ndcg10=sum(answerable_ndcgs) / count,
        unanswerable=len(unanswerable),
        unanswerable_empty=sum(1 for score in unanswerable if score.hits == 0),
    )
