"""检索编排：单通道 / RRF / 加权融合 / 可选重排。

- 向量、词法、图谱三路各自返回统一的 ``SearchHit`` 结构；
- 融合按 ``chunk_id`` 去重，``channels`` 记录命中的通道；
- 默认 RRF，加权融合作为对照；
- 重排默认关闭，且只作用于已融合的候选。
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from rag_kb.application.fusion import reciprocal_rank_fusion, weighted_fusion
from rag_kb.application.graph_search import graph_search
from rag_kb.domain.errors import (
    ConfigurationError,
    DependencyUnavailableError,
    DomainError,
    NotFoundError,
)
from rag_kb.domain.models import SearchHit, SearchMode
from rag_kb.domain.ports import (
    Embedder,
    GraphStore,
    LexicalIndex,
    MetadataRepository,
    Reranker,
    VectorStore,
)


class SearchService:
    def __init__(
        self,
        repository: MetadataRepository,
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        lexical_index: LexicalIndex | None = None,
        graph_store: GraphStore | None = None,
        reranker: Reranker | None = None,
        top_k: int = 20,
        rrf_k: int = 60,
        graph_seed_limit: int = 8,
        graph_max_hops: int = 2,
        graph_neighbor_limit: int = 50,
        hybrid_weights: dict[str, float] | None = None,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._graph_store = graph_store
        self._reranker = reranker
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._graph_seed_limit = graph_seed_limit
        self._graph_max_hops = graph_max_hops
        self._graph_neighbor_limit = graph_neighbor_limit
        self._hybrid_weights = hybrid_weights or {
            "vector": 1.0,
            "keyword": 1.0,
            "graph": 1.0,
        }

    async def search(
        self,
        kb_id: UUID,
        query: str,
        *,
        mode: SearchMode | str = SearchMode.HYBRID,
        top_k: int | None = None,
        rerank: bool = False,
    ) -> list[SearchHit]:
        if await self._repo.get_kb(kb_id) is None:
            raise NotFoundError("knowledge base", kb_id)
        resolved_mode = SearchMode(mode)
        limit = top_k or self._top_k

        if (
            resolved_mode is not SearchMode.HYBRID
            and resolved_mode is not SearchMode.HYBRID_WEIGHTED
        ):
            hits = await self._channel(resolved_mode.value, kb_id, query)
        else:
            rankings = await self._collect_channels(kb_id, query)
            if not rankings:
                return []
            if resolved_mode is SearchMode.HYBRID:
                hits = reciprocal_rank_fusion(rankings, k=self._rrf_k, weights=self._hybrid_weights)
            else:
                hits = weighted_fusion(rankings, weights=self._hybrid_weights)

        if rerank:
            if self._reranker is None:
                raise ConfigurationError("RERANK_* 未配置，无法重排")
            hits = await self._reranker.rerank(query, hits[:30], limit)
        return hits[:limit]

    async def _collect_channels(self, kb_id: UUID, query: str) -> dict[str, list[SearchHit]]:
        channels = ("vector", "keyword", "graph")
        results = await asyncio.gather(
            *(self._channel(channel, kb_id, query) for channel in channels),
            return_exceptions=True,
        )
        rankings: dict[str, list[SearchHit]] = {}
        errors: list[BaseException] = []
        for channel, result in zip(channels, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(result)
            elif result:
                rankings[channel] = result
        if not rankings:
            for error in errors:
                if not isinstance(error, DomainError):
                    raise error
            if errors:
                raise errors[0]
        return rankings

    async def _channel(self, channel: str, kb_id: UUID, query: str) -> list[SearchHit]:
        if channel == "vector":
            return await self._vector(kb_id, query)
        if channel == "keyword":
            return await self._keyword(kb_id, query)
        if channel == "graph":
            return await self._graph(kb_id, query)
        raise ValueError(f"未知检索通道: {channel}")

    async def _vector(self, kb_id: UUID, query: str) -> list[SearchHit]:
        if self._embedder is None or self._vector_store is None:
            raise ConfigurationError("EMBED_API_KEY 未配置，向量通道不可用")
        vector = await self._embedder.embed_query(query)
        raw = await self._vector_store.search(kb_id, vector, self._top_k)
        return await self._hydrate(
            kb_id, [(hit.chunk_id, hit.score) for hit in raw], channel="vector"
        )

    async def _keyword(self, kb_id: UUID, query: str) -> list[SearchHit]:
        if self._lexical_index is None:
            raise DependencyUnavailableError("postgres lexical index")
        raw = await self._lexical_index.search(kb_id, query, self._top_k)
        return await self._hydrate(
            kb_id, [(hit.chunk_id, hit.score) for hit in raw], channel="keyword"
        )

    async def _graph(self, kb_id: UUID, query: str) -> list[SearchHit]:
        if self._graph_store is None:
            raise DependencyUnavailableError("neo4j")
        return await graph_search(
            kb_id,
            query,
            graph_store=self._graph_store,
            repository=self._repo,
            seed_limit=self._graph_seed_limit,
            max_hops=self._graph_max_hops,
            neighbor_limit=self._graph_neighbor_limit,
            top_k=self._top_k,
        )

    async def _hydrate(
        self,
        kb_id: UUID,
        ranked: list[tuple[str, float]],
        *,
        channel: str,
    ) -> list[SearchHit]:
        if not ranked:
            return []
        chunk_ids = [chunk_id for chunk_id, _ in ranked]
        chunks = await self._repo.get_chunks(kb_id, chunk_ids)
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        doc_ids = {chunk.doc_id for chunk in chunks}
        documents = await asyncio.gather(
            *(self._repo.get_document(kb_id, doc_id) for doc_id in doc_ids)
        )
        names = {document.id: document.filename for document in documents if document is not None}

        hits: list[SearchHit] = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    doc_name=names.get(chunk.doc_id, ""),
                    level=chunk.level,
                    score=score,
                    channels=(channel,),
                    snippet=chunk.content[:300],
                    heading_path=chunk.heading_path,
                    page_no=chunk.page_no,
                    debug={"rank": rank, "raw_score": score},
                )
            )
        return hits
