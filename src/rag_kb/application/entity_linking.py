"""查询期种子实体链接。

三档策略（ADR-020）：精确 → 别名 → 2-gram 模糊。

链接是业务逻辑，放 application；``GraphStore`` 只执行"按条件找实体"。2-gram 用查询
字符串本身生成，不依赖 jieba，避免 application 反向依赖 infra。
"""

from __future__ import annotations

import re
import unicodedata
from uuid import UUID

from rag_kb.domain.models import Entity
from rag_kb.domain.ports import GraphStore

_SEGMENT_RE = re.compile(r"[\w一-鿿]+", re.UNICODE)


def query_segments(query: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", query).lower()
    return [segment for segment in _SEGMENT_RE.findall(normalized) if segment]


def bigrams(segments: list[str]) -> list[str]:
    grams: set[str] = set()
    for segment in segments:
        if len(segment) <= 2:
            grams.add(segment)
        else:
            grams.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return sorted(grams)


async def link_seed_entities(
    graph_store: GraphStore,
    kb_id: UUID,
    query: str,
    *,
    limit: int = 8,
) -> list[Entity]:
    segments = query_segments(query)
    if not segments:
        return []

    exact = await graph_store.candidate_entities(kb_id, segments, limit)
    if exact:
        return exact

    fuzzy = await graph_store.candidate_entities(kb_id, bigrams(segments), limit, contains=True)
    # 多个 2-gram 可能命中同一实体，按 id 去重并保留首次顺序。
    deduped: dict[str, Entity] = {entity.id: entity for entity in fuzzy}
    return list(deduped.values())
