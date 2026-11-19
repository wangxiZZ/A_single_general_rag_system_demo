"""回答与证据拼装。

M4-01 先实现证据侧：命中子块 → 回填父块 → 受 ``QA_MAX_CONTEXT_TOKENS`` 预算约束。
超预算时按融合分从低到高丢弃整条证据，**不截断单条**——截断会让引用指向不完整的原文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from rag_kb.application.chunking import estimate_tokens
from rag_kb.domain.errors import ConfigurationError, LlmError
from rag_kb.domain.models import ChatMessage, Chunk, SearchHit
from rag_kb.domain.ports import LLM, MetadataRepository

NO_EVIDENCE_ANSWER = "知识库中未找到足够依据。"
ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "citations"],
}


@dataclass(frozen=True, slots=True)
class Evidence:
    index: int
    hit: SearchHit
    body: Chunk
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    citations: list[int]
    evidence: list[Evidence]
    no_evidence: bool = False
    attempted_channels: list[str] = field(default_factory=list)


class QAService:
    def __init__(
        self,
        repository: MetadataRepository,
        *,
        llm: LLM | None = None,
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
        self._llm = llm
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

    async def answer(self, kb_id: UUID, question: str, hits: list[SearchHit]) -> AnswerResult:
        attempted_channels = sorted({channel for hit in hits for channel in hit.channels})
        evidence = await self.assemble_evidence(kb_id, hits)
        if not evidence:
            return AnswerResult(
                answer=NO_EVIDENCE_ANSWER,
                citations=[],
                evidence=[],
                no_evidence=True,
                attempted_channels=attempted_channels,
            )
        if self._llm is None:
            raise ConfigurationError("LLM_API_KEY 未配置，无法生成回答")

        data = await self._llm.structured_json(
            [ChatMessage(role="user", content=self._prompt(question, evidence))],
            ANSWER_SCHEMA,
        )
        answer_raw = data.get("answer")
        citations_raw = data.get("citations")
        if not isinstance(answer_raw, str) or not answer_raw.strip():
            raise LlmError("模型返回空回答")
        citations = (
            [item for item in citations_raw if isinstance(item, int)]
            if isinstance(citations_raw, list)
            else []
        )
        # 回校验：只保留落在 [1..n] 的编号并去重；模型不能"创造"证据编号。
        valid_citations = sorted({item for item in citations if 1 <= item <= len(evidence)})
        if not valid_citations:
            return AnswerResult(
                answer=NO_EVIDENCE_ANSWER,
                citations=[],
                evidence=evidence,
                no_evidence=True,
                attempted_channels=attempted_channels,
            )
        return AnswerResult(
            answer=answer_raw.strip(),
            citations=valid_citations,
            evidence=evidence,
            no_evidence=False,
            attempted_channels=attempted_channels,
        )

    @staticmethod
    def _prompt(question: str, evidence: list[Evidence]) -> str:
        lines = [
            "只根据下面的证据回答问题；每个结论后必须标注来源编号 [n]。",
            "证据不足时直接回答“知识库中未找到足够依据”，不要编造。",
            "",
            f"问题：{question}",
            "",
            "证据：",
        ]
        for item in evidence:
            location = (
                f"第 {item.body.page_no} 页"
                if item.body.page_no is not None
                else f"段落 #{item.body.seq}"
            )
            heading = " > ".join(item.body.heading_path)
            lines.append(f"[{item.index}] 《{item.hit.doc_name}》{location} · {heading}")
            lines.append(item.text)
            lines.append("")
        return "\n".join(lines)
