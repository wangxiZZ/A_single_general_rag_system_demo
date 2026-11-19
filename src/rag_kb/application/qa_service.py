"""回答与证据拼装。

M4-01 先实现证据侧：命中子块 → 回填父块 → 受 ``QA_MAX_CONTEXT_TOKENS`` 预算约束。
超预算时按融合分从低到高丢弃整条证据，**不截断单条**——截断会让引用指向不完整的原文。
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from rag_kb.application.chunking import estimate_tokens
from rag_kb.domain.errors import ConfigurationError
from rag_kb.domain.models import Chunk, SearchHit
from rag_kb.domain.ports import MetadataRepository


@dataclass(frozen=True, slots=True)
class Evidence:
    index: int
    hit: SearchHit
    body: Chunk
    text: str
    token_count: int


class QAService:
    def __init__(
        self,
        repository: MetadataRepository,
        *,
        max_context_tokens: int,
        parent_max_tokens: int,
        chars_per_token: float,
    ) -> None:
        if max_context_tokens < parent_max_tokens:
            raise ConfigurationError(
                f"QA_MAX_CONTEXT_TOKENS={max_context_tokens} 小于 "
                f"PARENT_MAX_TOKENS={parent_max_tokens}，单条父块会天然超预算"
            )
        self._repo = repository
        self._max_context_tokens = max_context_tokens
        self._chars_per_token = chars_per_token

    async def assemble_evidence(self, kb_id: UUID, hits: list[SearchHit]) -> list[Evidence]:
        if not hits:
            return []

        ids = {hit.chunk_id for hit in hits}
        # 先按 id 批量取子块，再把需要的父块 id 一并取回。
        children = await self._repo.get_chunks(kb_id, list(ids))
        children_by_id = {chunk.id: chunk for chunk in children}
        parent_ids = {chunk.parent_id for chunk in children if chunk.parent_id is not None}
        parents = await self._repo.get_chunks(kb_id, list(parent_ids)) if parent_ids else []
        parents_by_id = {chunk.id: chunk for chunk in parents}

        candidates: list[Evidence] = []
        seen_bodies: set[str] = set()
        for hit in sorted(hits, key=lambda item: item.score, reverse=True):
            child = children_by_id.get(hit.chunk_id)
            if child is None:
                continue
            body = parents_by_id.get(child.parent_id) if child.parent_id else None
            body = body or child
            if body.id in seen_bodies:
                # 同一个父块下的多个子块命中时只保留分数最高的一条，避免重复消耗预算。
                continue
            text = body.content
            if not text.strip():
                continue
            seen_bodies.add(body.id)
            candidates.append(
                Evidence(
                    index=0,
                    hit=hit,
                    body=body,
                    text=text,
                    token_count=estimate_tokens(text, self._chars_per_token),
                )
            )

        total = sum(item.token_count for item in candidates)
        while total > self._max_context_tokens and candidates:
            # 丢整条最低分证据；不截断单条。
            lowest = min(range(len(candidates)), key=lambda i: candidates[i].hit.score)
            total -= candidates[lowest].token_count
            candidates.pop(lowest)

        return [
            Evidence(
                index=index,
                hit=item.hit,
                body=item.body,
                text=item.text,
                token_count=item.token_count,
            )
            for index, item in enumerate(candidates, start=1)
        ]
