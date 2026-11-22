"""删除/重试时的三路索引清理。

为什么单独抽一个模块：删文档、删整库、重试失败文档三处都要"先清三路索引再动 DB"，
而且失败语义必须一致。

关键取舍是**收集全部失败再抛**，而不是遇错即停：如果 vector 删成功、graph 抛错就立刻
返回，调用方既不知道 lexical 的状态，也拿不到"还差哪一路"，最后留下"文档行还在、
某个通道已空"的半删状态——这正是验收口径里最容易被误判为成功的情况。全部尝试完再
抛出，配合可重试的删除（delete 是幂等的），用户再点一次即可收敛。
"""

from __future__ import annotations

import logging
from uuid import UUID

from rag_kb.domain.errors import StorageError
from rag_kb.domain.ports import GraphStore, LexicalIndex, VectorStore

logger = logging.getLogger(__name__)


async def purge_indexes(
    kb_id: UUID,
    doc_id: UUID,
    *,
    vector_store: VectorStore | None,
    lexical_index: LexicalIndex | None,
    graph_store: GraphStore | None,
) -> list[str]:
    """逐路清理，返回失败描述（空列表表示全部成功）。"""
    failures: list[str] = []
    channels = (
        ("vector", vector_store),
        ("lexical", lexical_index),
        ("graph", graph_store),
    )
    for name, port in channels:
        if port is None:
            # 端口未注入说明该通道本来就没数据（例如单测只关心 DB），不算失败。
            continue
        try:
            await port.delete_by_doc(kb_id, doc_id)
        except Exception as exc:
            logger.exception("%s channel delete failed kb=%s doc=%s", name, kb_id, doc_id)
            failures.append(f"{name}: {exc}")
    return failures


def ensure_purged(failures: list[str], *, action: str, doc_id: UUID | None = None) -> None:
    """有失败就抛领域异常，让调用方看到"可重试"，而不是静默留下半删状态。"""
    if not failures:
        return
    target = f"文档 {doc_id} " if doc_id is not None else ""
    raise StorageError(f"{action}{target}未完成（可重试）：" + "；".join(failures))


__all__ = ["ensure_purged", "purge_indexes"]
