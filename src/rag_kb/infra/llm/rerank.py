"""LLM listwise 重排（显式 opt-in）。

默认检索路径不调用 LLM；只有在 ``rerank=True`` 且注入了本 reranker 时才执行。模型只
返回候选序号，客户端负责校验范围、去重与补齐，避免模型"创造"新的 chunk。
"""

from __future__ import annotations

from dataclasses import replace

from rag_kb.domain.models import ChatMessage, SearchHit
from rag_kb.domain.ports import LLM

_SCHEMA = {
    "type": "object",
    "properties": {
        "order": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["order"],
}


class LLMListwiseReranker:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def rerank(self, query: str, candidates: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not candidates:
            return []

        listing = "\n".join(
            f"[{index}] {hit.snippet[:300]}" for index, hit in enumerate(candidates, start=1)
        )
        prompt = (
            f"问题：{query}\n\n候选片段：\n{listing}\n\n"
            "按与问题的相关性从高到低排序，只输出候选序号，不要解释。"
        )
        data = await self._llm.structured_json([ChatMessage(role="user", content=prompt)], _SCHEMA)
        raw_order = data.get("order")
        order: list[int] = []
        if isinstance(raw_order, list):
            for value in raw_order:
                if isinstance(value, int) and 1 <= value <= len(candidates) and value not in order:
                    order.append(value)
        for index in range(1, len(candidates) + 1):
            if index not in order:
                order.append(index)

        return [
            replace(
                candidates[value - 1],
                debug={**candidates[value - 1].debug, "rerank_rank": rank},
            )
            for rank, value in enumerate(order[:top_k], start=1)
        ]
