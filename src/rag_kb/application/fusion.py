"""融合排序。

三路分数不可比：余弦相似度、ts_rank_cd、图路径分。RRF 只使用排名，对量纲免疫；
加权融合作为对照实验，先对每路做 min-max 归一化再加权。
"""

from __future__ import annotations

from dataclasses import replace

from rag_kb.domain.models import SearchHit

Rankings = dict[str, list[SearchHit]]


def _merge_channels(
    existing: SearchHit | None,
    incoming: SearchHit,
    *,
    channel: str,
    rank: int,
    score: float,
) -> SearchHit:
    debug: dict[str, object] = dict(existing.debug) if existing else {}
    rank_map = dict(debug.get("rank_per_channel") or {})  # type: ignore[arg-type]
    raw_map = dict(debug.get("raw_score") or {})  # type: ignore[arg-type]
    rank_map[channel] = rank
    raw_map[channel] = incoming.score
    debug["rank_per_channel"] = rank_map
    debug["raw_score"] = raw_map

    channels = tuple(dict.fromkeys((*(existing.channels if existing else ()), channel)))
    base = existing or incoming
    return replace(base, score=score, channels=channels, debug=debug)


def reciprocal_rank_fusion(
    rankings: Rankings,
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[SearchHit]:
    scores: dict[str, float] = {}
    merged: dict[str, SearchHit] = {}
    weights = weights or {}

    for channel, hits in rankings.items():
        weight = weights.get(channel, 1.0)
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)
            merged[hit.chunk_id] = _merge_channels(
                merged.get(hit.chunk_id),
                hit,
                channel=channel,
                rank=rank,
                score=0.0,
            )

    result: list[SearchHit] = []
    for chunk_id, hit in merged.items():
        result.append(replace(hit, score=round(scores[chunk_id], 6)))
    return sorted(result, key=lambda item: item.score, reverse=True)


def weighted_fusion(
    rankings: Rankings,
    *,
    weights: dict[str, float] | None = None,
) -> list[SearchHit]:
    weights = weights or {}
    scores: dict[str, float] = {}
    merged: dict[str, SearchHit] = {}

    for channel, hits in rankings.items():
        if not hits:
            continue
        values = [hit.score for hit in hits]
        low, high = min(values), max(values)
        span = high - low
        weight = weights.get(channel, 1.0)
        for rank, hit in enumerate(hits, start=1):
            normalized = 1.0 if span == 0 else (hit.score - low) / span
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight * normalized
            merged[hit.chunk_id] = _merge_channels(
                merged.get(hit.chunk_id),
                hit,
                channel=channel,
                rank=rank,
                score=0.0,
            )

    result = [replace(hit, score=round(scores[chunk_id], 6)) for chunk_id, hit in merged.items()]
    return sorted(result, key=lambda item: item.score, reverse=True)
