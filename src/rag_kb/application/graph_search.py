"""图谱召回：实体链接 → 多跳展开 → 回 PostgreSQL 取原文子块。

图只是"找到 chunk_id 的手段"，最终返回给用户的是原文块（ADR-020）。没有原文可取的
图证据直接丢弃，不把三元组当结果。
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from rag_kb.application.entity_linking import link_seed_entities
from rag_kb.domain.models import SearchHit
from rag_kb.domain.ports import GraphStore, MetadataRepository


async def graph_search(
    kb_id: UUID,
    query: str,
    *,
    graph_store: GraphStore,
    repository: MetadataRepository,
    seed_limit: int = 8,
    max_hops: int = 2,
    neighbor_limit: int = 50,
    top_k: int = 20,
) -> list[SearchHit]:
    seeds = await link_seed_entities(graph_store, kb_id, query, limit=seed_limit)
    if not seeds:
        return []

    evidence = await graph_store.expand(
        kb_id,
        [entity.id for entity in seeds],
        max_hops,
        neighbor_limit,
    )
    if not evidence:
        return []

    chunks = await repository.get_chunks(kb_id, [item.chunk_id for item in evidence])
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    document_ids = {chunk.doc_id for chunk in chunks}
    documents = await asyncio.gather(
        *(repository.get_document(kb_id, doc_id) for doc_id in document_ids)
    )
    names = {document.id: document.filename for document in documents if document is not None}

    hits: list[SearchHit] = []
    for item in evidence:
        chunk = chunks_by_id.get(item.chunk_id)
        if chunk is None:
            continue
        hits.append(
            SearchHit(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                doc_name=names.get(chunk.doc_id, ""),
                level=chunk.level,
                score=item.score,
                channels=("graph",),
                snippet=chunk.content[:300],
                heading_path=chunk.heading_path,
                page_no=chunk.page_no,
                debug={
                    "hops": item.hops,
                    "seed_ids": list(item.seed_ids),
                },
            )
        )
        if len(hits) >= top_k:
            break
    return hits
