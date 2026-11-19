"""图谱抽取单测：fake LLM + span/白名单校验。"""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_kb.application.graph_extract import (
    GraphExtractor,
    find_span,
    normalize_name,
)
from rag_kb.domain.errors import ConfigurationError, LlmError
from rag_kb.domain.models import Chunk, ChunkKind, ChunkLevel

KB_ID = uuid4()
DOC_ID = uuid4()


class FakeLLM:
    def __init__(self, payload: dict | Exception, calls: int = 1) -> None:
        self.payload = payload
        self.calls: list[str] = []
        self.remaining = calls

    async def structured_json(self, messages, schema):
        self.calls.append(messages[0].content)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    async def chat(self, messages, *, temperature: float = 0.0, max_tokens=None) -> str:
        return ""


def _chunk(
    text: str,
    *,
    seq: int = 0,
    level: ChunkLevel = ChunkLevel.CHILD,
    kind: ChunkKind = ChunkKind.TEXT,
) -> Chunk:
    return Chunk(
        id=f"{seq:032d}",
        kb_id=KB_ID,
        doc_id=DOC_ID,
        level=level,
        kind=kind,
        content=text,
        seq=seq,
        char_start=0,
        char_end=len(text),
        token_count=len(text),
        chunker_version="v1",
    )


def _payload() -> dict:
    return {
        "entities": [
            {"name": "年假政策", "type": "条款", "span": "年假政策"},
            {"name": "员工", "type": "人物", "span": "员工"},
            {"name": "幽灵", "type": "产品", "span": "不存在的词"},
        ],
        "relations": [
            {
                "head": "年假政策",
                "tail": "员工",
                "relation": "适用于",
                "evidence_span": "年假政策规定员工",
            },
            {
                "head": "幽灵",
                "tail": "员工",
                "relation": "包含",
                "evidence_span": "员工",
            },
        ],
    }


async def test_extracts_entities_mentions_and_falls_back_relation_type() -> None:
    chunk = _chunk("年假政策规定员工每年享有带薪年假。")
    extractor = GraphExtractor(FakeLLM(_payload()))

    result = await extractor.extract([chunk], kb_id=KB_ID)

    assert len(result.entities) == 2
    assert {entity.type for entity in result.entities} == {"条款", "人物"}
    assert result.dropped_spans == 1
    assert result.dropped_relations == 1
    assert len(result.mentions) == 2
    assert all(mention.chunk_id == chunk.id for mention in result.mentions)

    relation = result.relations[0]
    assert relation.type == "OTHER"
    assert relation.chunk_id == chunk.id
    assert relation.evidence_span is not None
    text = chunk.content
    assert text[relation.evidence_span[0] : relation.evidence_span[1]] == "年假政策规定员工"


async def test_same_entity_across_chunks_is_merged() -> None:
    chunks = [
        _chunk("年假政策规定员工。", seq=0),
        _chunk("员工需要遵守年假政策规定员工的条款。", seq=1),
    ]
    result = await GraphExtractor(FakeLLM(_payload())).extract(chunks, kb_id=KB_ID)

    assert len(result.entities) == 2
    assert len({entity.id for entity in result.entities}) == 2
    assert len(result.mentions) == 4
    assert len(result.relations) == 2


async def test_only_child_text_chunks_are_sent_to_llm() -> None:
    llm = FakeLLM(_payload())
    chunks = [
        _chunk("年假政策规定员工。", seq=0),
        _chunk("父块", seq=1, level=ChunkLevel.PARENT),
        _chunk("图片块", seq=2, kind=ChunkKind.IMAGE),
    ]
    await GraphExtractor(llm).extract(chunks, kb_id=KB_ID)
    assert len(llm.calls) == 1


async def test_llm_failure_counts_failed_chunk() -> None:
    result = await GraphExtractor(FakeLLM(LlmError("boom"))).extract([_chunk("文本")], kb_id=KB_ID)
    assert result.failed_chunks == 1
    assert result.entities == []


async def test_missing_llm_raises() -> None:
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        await GraphExtractor(None).extract([_chunk("文本")], kb_id=KB_ID)


def test_normalize_name_and_span_helpers() -> None:
    assert normalize_name(" 年假政策（试行） ") == "年假政策"
    assert find_span("年假 政策", "年假政策") is not None
    assert find_span("年假政策", "不存在的词") is None


async def test_unknown_entity_type_falls_back() -> None:
    payload = {
        "entities": [{"name": "员工", "type": "未知类型", "span": "员工"}],
        "relations": [],
    }
    result = await GraphExtractor(FakeLLM(payload)).extract([_chunk("员工")], kb_id=KB_ID)
    assert result.entities[0].type == "其他"


async def test_malformed_payload_is_counted_as_failed_chunk() -> None:
    """模型返回了 JSON 但结构不对：整块作废，不能把半截数据写进图谱。"""
    result = await GraphExtractor(FakeLLM({"entities": "oops", "relations": []})).extract(
        [_chunk("员工")], kb_id=KB_ID
    )
    assert result.failed_chunks == 1
    assert result.entities == []


async def test_invalid_entity_items_are_dropped_not_crashing() -> None:
    payload = {
        "entities": [
            "不是对象",
            {"name": "缺 span", "type": "人物"},
            {"name": "   ", "type": "人物", "span": "员工"},
            {"name": "员工", "type": "人物", "span": "不存在的词"},
            {"name": "员工", "type": "人物", "span": "员工"},
        ],
        "relations": [],
    }
    result = await GraphExtractor(FakeLLM(payload)).extract([_chunk("员工")], kb_id=KB_ID)

    assert len(result.entities) == 1
    assert result.entities[0].name == "员工"
    assert result.dropped_spans == 4


async def test_aliases_are_normalized_filtered_and_merged() -> None:
    payload = {
        "entities": [
            {
                "name": "年假政策",
                "type": "条款",
                "span": "年假政策",
                # 非字符串、空串、全角括号后缀都要被规范化或过滤
                "aliases": [" 年假政策（试行） ", "", 3, "带薪年假"],
            }
        ],
        "relations": [],
    }
    chunks = [_chunk("年假政策与带薪年假。", seq=0), _chunk("年假政策", seq=1)]
    result = await GraphExtractor(FakeLLM(payload)).extract(chunks, kb_id=KB_ID)

    assert len(result.entities) == 1
    assert result.entities[0].aliases == ("年假政策", "带薪年假")
    assert len(result.mentions) == 2


async def test_relation_requires_known_entities_and_evidence_span() -> None:
    payload = {
        "entities": [{"name": "员工", "type": "人物", "span": "员工"}],
        "relations": [
            # head 不在本块实体表
            {"head": "幽灵", "tail": "员工", "relation": "包含", "evidence_span": "员工"},
            # evidence_span 在原文中找不到
            {"head": "员工", "tail": "员工", "relation": "包含", "evidence_span": "不存在"},
            # 非对象
            "坏数据",
        ],
    }
    result = await GraphExtractor(FakeLLM(payload)).extract([_chunk("员工")], kb_id=KB_ID)

    assert result.relations == []
    assert result.dropped_relations == 3


async def test_one_failed_chunk_does_not_discard_others() -> None:
    class SelectiveLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def structured_json(self, messages, schema):
            self.calls += 1
            if "失败块" in messages[0].content:
                raise LlmError("boom")
            return {
                "entities": [{"name": "员工", "type": "人物", "span": "员工"}],
                "relations": [],
            }

        async def chat(self, messages, *, temperature: float = 0.0, max_tokens=None) -> str:
            return ""

    result = await GraphExtractor(SelectiveLLM()).extract(
        [_chunk("失败块", seq=0), _chunk("员工", seq=1)], kb_id=KB_ID
    )

    assert result.failed_chunks == 1
    assert len(result.entities) == 1


def test_find_span_rejects_empty_needle() -> None:
    assert find_span("年假政策", "") is None
    assert find_span("年假政策", "   ") is None
