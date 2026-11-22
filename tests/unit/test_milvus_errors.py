"""Milvus 适配器的异常翻译（不连 Milvus，直接替换 client）。"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pymilvus.exceptions import MilvusException

from rag_kb.domain.errors import DependencyUnavailableError
from rag_kb.infra.milvus.store import MilvusVectorStore


class _FailingClient:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def search(self, **kwargs: object) -> object:
        raise self._exc

    def delete(self, **kwargs: object) -> object:
        raise self._exc


def _store(exc: Exception) -> MilvusVectorStore:
    # 绕开 __init__：它会在构造期就连 Milvus，而单测不允许连外部依赖。
    store = object.__new__(MilvusVectorStore)
    store._client = _FailingClient(exc)  # type: ignore[attr-defined]
    store._collection = "chunk_vec"  # type: ignore[attr-defined]
    store._dimension = 2  # type: ignore[attr-defined]
    return store


async def test_search_translates_milvus_exception_to_dependency_error() -> None:
    """Milvus 掉线要变成 503 语义，而不是带连接串的 500。"""
    store = _store(MilvusException(message="connection refused"))

    with pytest.raises(DependencyUnavailableError, match="milvus"):
        await store.search(uuid4(), [0.0, 1.0], 5)


async def test_delete_translates_milvus_exception_to_dependency_error() -> None:
    store = _store(MilvusException(message="connection refused"))

    with pytest.raises(DependencyUnavailableError, match="milvus"):
        await store.delete_by_doc(uuid4(), uuid4())


async def test_non_milvus_exception_is_not_masked() -> None:
    """只有 pymilvus 的异常才翻译；其它异常原样抛出，避免掩盖真实 bug。"""
    store = _store(ValueError("unexpected"))

    with pytest.raises(ValueError, match="unexpected"):
        await store.search(uuid4(), [0.0, 1.0], 5)
