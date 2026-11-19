"""RRF 与加权融合单测。"""

from __future__ import annotations

from uuid import uuid4

from rag_kb.application.fusion import reciprocal_rank_fusion, weighted_fusion
from rag_kb.domain.models import ChunkLevel, SearchHit

DOC_ID = uuid4()


def _hit(chunk_id: str, score: float, channel: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=DOC_ID,
        doc_name="a.pdf",
        level=ChunkLevel.CHILD,
        score=score,
        channels=(channel,),
        snippet=chunk_id,
    )


def test_rrf_deduplicates_and_merges_channels() -> None:
    fused = reciprocal_rank_fusion(
        {
            "vector": [_hit("a", 0.9, "vector"), _hit("b", 0.8, "vector")],
            "keyword": [_hit("b", 4.0, "keyword"), _hit("c", 3.0, "keyword")],
        },
        k=60,
    )

    assert [hit.chunk_id for hit in fused] == ["b", "a", "c"]
    assert fused[0].channels == ("vector", "keyword")
    assert fused[0].debug["rank_per_channel"] == {"vector": 2, "keyword": 1}
    assert fused[0].score > fused[1].score > fused[2].score


def test_rrf_weights_change_order() -> None:
    rankings = {
        "vector": [_hit("a", 0.9, "vector")],
        "keyword": [_hit("b", 9.0, "keyword")],
    }
    fused = reciprocal_rank_fusion(rankings, k=60, weights={"vector": 10.0})
    assert fused[0].chunk_id == "a"


def test_weighted_fusion_normalizes_per_channel() -> None:
    fused = weighted_fusion(
        {
            "vector": [_hit("a", 10.0, "vector"), _hit("b", 0.0, "vector")],
            "keyword": [_hit("b", 100.0, "keyword"), _hit("c", 50.0, "keyword")],
        }
    )
    scores = {hit.chunk_id: hit.score for hit in fused}
    assert scores["a"] == 1.0
    assert scores["b"] == 1.0
    assert scores["c"] == 0.0
    b_hit = next(hit for hit in fused if hit.chunk_id == "b")
    assert b_hit.channels == ("vector", "keyword")


def test_empty_rankings() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert weighted_fusion({"vector": []}) == []
