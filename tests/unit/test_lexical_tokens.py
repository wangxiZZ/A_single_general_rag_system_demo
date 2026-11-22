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


def test_tokenize_keeps_known_word_in_natural_question() -> None:
    """回归：问句里的词被 jieba 切成单字后，不能因为合并而丢掉。

    历史缺陷：`年假有几天` 被切成 年/假/有/几天，连续单字合并成「年假有」，
    这个串在文档里不存在，于是关键词通道对这句查询 0 命中（向量通道正常）。
    """
    tokens = tokenize("年假有几天")
    assert "年假" in tokens
    assert "年假" in build_tsquery("年假有几天")


def test_tokenize_emits_bigrams_for_single_char_run() -> None:
    """连续单字 run 除了整串，还要给出 2-gram，保证真实词仍是候选。"""
    tokens = tokenize("年假龘")  # jieba 把三个字都切成单字
    assert "年假龘" in tokens
    assert "年假" in tokens


def test_stopwords_file_ignores_comment_lines() -> None:
    from rag_kb.infra.postgres.lexical import stopwords

    assert "有" in stopwords()
    assert not any(word.startswith("#") for word in stopwords())
