"""图谱评测指标单测：归一化、P/R/F1、span 校验。"""

from __future__ import annotations

from scripts.graph_eval import (
    entity_key,
    relation_key,
    score_entities,
    score_relations,
    span_is_valid,
)

GOLD_ENTITIES = [
    {"name": "知识库服务", "type": "产品"},
    {"name": "星云科技", "type": "组织"},
    {"name": "SLA 99.9%", "type": "指标"},
]
GOLD_RELATIONS = [
    {"head": "星云科技", "relation": "提供", "tail": "知识库服务"},
    {"head": "知识库服务", "relation": "包含", "tail": "检索组件"},
]


def test_entity_key_normalizes_spaces_case_and_parentheses() -> None:
    assert entity_key(" 检索组件（XR 组件） ", "产品") == entity_key("检索组件", "产品")
    assert entity_key("SLA 99.9%", "指标") == entity_key("sla99.9%", "指标")


def test_relation_key_normalizes_endpoints() -> None:
    assert relation_key(" 星云科技 ", "提供", "知识库服务（KB）") == (
        "星云科技",
        "提供",
        "知识库服务",
    )


def test_score_entities_perfect_match() -> None:
    score, _ = score_entities(GOLD_ENTITIES, GOLD_ENTITIES)
    assert score["precision"] == score["recall"] == score["f1"] == 1.0
    assert score["type_accuracy"] == 1.0


def test_score_entities_counts_type_mismatch_as_wrong() -> None:
    extracted = [
        {"name": "知识库服务", "type": "组织"},  # 名称对、类型错 → 实体键算 FP+FN
        {"name": "星云科技", "type": "组织"},
        {"name": "SLA99.9%", "type": "指标"},  # 归一化后命中
    ]
    score, accuracy = score_entities(extracted, GOLD_ENTITIES)

    assert score["tp"] == 2
    assert score["fp"] == 1
    assert score["fn"] == 1
    # 名称命中的 3 个里，2 个类型正确
    assert score["type_matched"] == 3
    assert accuracy == 2 / 3


def test_score_relations_deduplicates_same_triple_across_chunks() -> None:
    extracted = [
        {"head": "星云科技", "relation": "提供", "tail": "知识库服务", "chunk_id": "a"},
        {"head": "星云科技", "relation": "提供", "tail": "知识库服务（KB）", "chunk_id": "b"},
        {"head": "知识库服务", "relation": "包含", "tail": "检索组件", "chunk_id": "a"},
        {"head": "星云科技", "relation": "关联", "tail": "检索组件", "chunk_id": "b"},
    ]
    score = score_relations(extracted, GOLD_RELATIONS)

    assert score["tp"] == 2
    assert score["fp"] == 1
    assert score["fn"] == 0
    assert score["precision"] == round(2 / 3, 4)


def test_span_is_valid_checks_bounds_and_non_empty() -> None:
    assert span_is_valid([0, 4], "星云科技")
    assert span_is_valid([2, 2], "abc") is False
    assert span_is_valid([0, 9], "abc") is False
    assert span_is_valid([-1, 2], "abc") is False
    assert span_is_valid(None, "abc") is False
    assert span_is_valid([0, 2], None) is False
