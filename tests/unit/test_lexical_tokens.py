"""词法索引的纯函数测试（不连数据库）。"""

from __future__ import annotations

from rag_kb.infra.postgres.lexical import build_tsquery, tokenize


def test_tokenize_removes_stopwords_and_punctuation() -> None:
    tokens = tokenize("年假 政策，与 请假 流程！")
    assert "年假" in tokens
    assert "政策" in tokens
    assert "请假" in tokens
    assert "与" not in tokens
    assert "，" not in tokens


def test_tokenize_handles_empty_and_symbols() -> None:
    assert tokenize("") == []
    assert tokenize("！！！，，，") == []


def test_build_tsquery_deduplicates_and_ors() -> None:
    query = build_tsquery("年假 年假 政策")
    assert query == "年假 | 政策"


def test_build_tsquery_empty() -> None:
    assert build_tsquery("的 了") == ""
