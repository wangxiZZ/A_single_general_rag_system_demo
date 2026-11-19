"""图谱抽取：LLM → 受约束 JSON → span 校验 → 领域实体/关系/提及。

为什么校验放在应用层而不是信任模型：

- 模型会编造不存在的实体或无出处的边；``span`` 必须在块原文中找到，否则丢弃；
- 类型与关系走白名单/闭集，未知类型降级为"其他"，未知关系降级为 ``OTHER``；
- 只处理 ``level=child AND kind=text`` 的块，图片 caption 不参与（否则 span 必然找不到）；
- 失败按块丢弃并计数，不因一批失败中断整个文档。
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from uuid import UUID

from rag_kb.domain.errors import ConfigurationError, LlmError
from rag_kb.domain.models import (
    ChatMessage,
    Chunk,
    ChunkKind,
    ChunkLevel,
    Entity,
    Mention,
    Relation,
)
from rag_kb.domain.ports import LLM

ENTITY_TYPES = ("人物", "组织", "产品", "指标", "流程", "条款", "地点", "事件", "其他")
RELATION_TYPES = (
    "属于",
    "包含",
    "依赖",
    "导致",
    "提供",
    "使用",
    "位于",
    "关联",
    "OTHER",
)
FALLBACK_ENTITY_TYPE = "其他"
FALLBACK_RELATION_TYPE = "OTHER"

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "span": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "type", "span"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "head": {"type": "string"},
                    "tail": {"type": "string"},
                    "relation": {"type": "string", "enum": list(RELATION_TYPES)},
                    "evidence_span": {"type": "string"},
                },
                "required": ["head", "tail", "relation", "evidence_span"],
            },
        },
    },
    "required": ["entities", "relations"],
}

PROMPT = """从下面的文档块中抽取实体与关系，只输出 JSON。
要求：
- entities[].span 必须逐字摘自文本；
- relations[].head/tail 必须在 entities 中出现；
- relations[].evidence_span 必须逐字摘自文本；
- 看不清或没有原文依据的内容不要输出。

文本：
{text}
"""


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]
    mentions: list[Mention]
    dropped_spans: int = 0
    dropped_relations: int = 0
    failed_chunks: int = 0


@dataclass(slots=True)
class _ChunkExtraction:
    entities: list[Entity]
    relations: list[Relation]
    mentions: list[Mention]
    dropped_spans: int = 0
    dropped_relations: int = 0


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).strip().lower()
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    return re.sub(r"\s+", "", text)


def _compact(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        chars.append(char)
        positions.append(index)
    return "".join(chars), positions


def find_span(text: str, raw: str) -> tuple[int, int] | None:
    needle = raw.strip()
    if not needle:
        return None
    exact = text.find(needle)
    if exact >= 0:
        return exact, exact + len(needle)

    compact_text, positions = _compact(text)
    compact_needle = "".join(needle.split())
    found = compact_text.find(compact_needle)
    if found < 0 or not compact_needle:
        return None
    start = positions[found]
    end = positions[found + len(compact_needle) - 1] + 1
    return start, end


def entity_id(kb_id: UUID, normalized: str, type_name: str) -> str:
    raw = f"{kb_id}:{normalized}:{type_name}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


class GraphExtractor:
    def __init__(self, llm: LLM | None) -> None:
        self._llm = llm

    async def extract(self, chunks: list[Chunk], *, kb_id: UUID) -> ExtractionResult:
        if self._llm is None:
            raise ConfigurationError("LLM_API_KEY 未配置，无法进行图谱抽取")
        targets = [
            chunk
            for chunk in chunks
            if chunk.level is ChunkLevel.CHILD and chunk.kind is ChunkKind.TEXT
        ]
        results = await asyncio.gather(*(self._extract_chunk(chunk, kb_id) for chunk in targets))

        entities: dict[tuple[str, str], Entity] = {}
        relations: dict[tuple[str, str, str, str], Relation] = {}
        mentions: dict[tuple[str, str, tuple[int, int]], Mention] = {}
        dropped_spans = dropped_relations = failed_chunks = 0

        for result in results:
            if result is None:
                failed_chunks += 1
                continue
            dropped_spans += result.dropped_spans
            dropped_relations += result.dropped_relations
            for entity in result.entities:
                key = (entity.normalized, entity.type)
                existing = entities.get(key)
                if existing is None:
                    entities[key] = entity
                else:
                    aliases = tuple(dict.fromkeys((*existing.aliases, *entity.aliases)))
                    entities[key] = Entity(
                        id=existing.id,
                        kb_id=existing.kb_id,
                        name=existing.name,
                        normalized=existing.normalized,
                        type=existing.type,
                        aliases=aliases,
                    )
            for relation in result.relations:
                key = (relation.head_id, relation.tail_id, relation.type, relation.chunk_id)
                relations.setdefault(key, relation)
            for mention in result.mentions:
                mentions[(mention.chunk_id, mention.entity_id, mention.span)] = mention

        return ExtractionResult(
            entities=list(entities.values()),
            relations=list(relations.values()),
            mentions=list(mentions.values()),
            dropped_spans=dropped_spans,
            dropped_relations=dropped_relations,
            failed_chunks=failed_chunks,
        )

    async def _extract_chunk(self, chunk: Chunk, kb_id: UUID) -> _ChunkExtraction | None:
        try:
            data = await self._llm.structured_json(
                [ChatMessage(role="user", content=PROMPT.format(text=chunk.content))],
                SCHEMA,
            )
        except LlmError:
            return None

        raw_entities = data.get("entities")
        raw_relations = data.get("relations")
        if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
            return None

        local: dict[str, Entity] = {}
        mentions: list[Mention] = []
        dropped_spans = 0

        for item in raw_entities:
            if not isinstance(item, dict):
                dropped_spans += 1
                continue
            name = item.get("name")
            span_raw = item.get("span")
            if not isinstance(name, str) or not isinstance(span_raw, str):
                dropped_spans += 1
                continue
            span = find_span(chunk.content, span_raw)
            normalized = normalize_name(name)
            if span is None or not normalized:
                dropped_spans += 1
                continue

            type_raw = item.get("type")
            type_name = (
                type_raw
                if isinstance(type_raw, str) and type_raw in ENTITY_TYPES
                else FALLBACK_ENTITY_TYPE
            )
            aliases = tuple(
                dict.fromkeys(
                    normalized_alias
                    for alias in _string_list(item.get("aliases"))
                    if (normalized_alias := normalize_name(alias))
                )
            )
            entity = Entity(
                id=entity_id(kb_id, normalized, type_name),
                kb_id=kb_id,
                name=name.strip(),
                normalized=normalized,
                type=type_name,
                aliases=aliases,
            )
            local[normalized] = entity
            mentions.append(Mention(chunk_id=chunk.id, entity_id=entity.id, span=span))

        relations: list[Relation] = []
        dropped_relations = 0
        for item in raw_relations:
            if not isinstance(item, dict):
                dropped_relations += 1
                continue
            head = item.get("head")
            tail = item.get("tail")
            evidence = item.get("evidence_span")
            if not all(isinstance(value, str) for value in (head, tail, evidence)):
                dropped_relations += 1
                continue
            head_entity = local.get(normalize_name(head))
            tail_entity = local.get(normalize_name(tail))
            span = find_span(chunk.content, evidence)
            if head_entity is None or tail_entity is None or span is None:
                dropped_relations += 1
                continue

            relation_raw = item.get("relation")
            relation_type = (
                relation_raw
                if isinstance(relation_raw, str) and relation_raw in RELATION_TYPES
                else FALLBACK_RELATION_TYPE
            )
            relations.append(
                Relation(
                    head_id=head_entity.id,
                    tail_id=tail_entity.id,
                    type=relation_type,
                    chunk_id=chunk.id,
                    weight=1.0,
                    evidence_span=span,
                )
            )

        return _ChunkExtraction(
            entities=list(local.values()),
            relations=relations,
            mentions=mentions,
            dropped_spans=dropped_spans,
            dropped_relations=dropped_relations,
        )
