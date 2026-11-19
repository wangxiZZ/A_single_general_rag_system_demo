"""端口契约的静态自检：I/O 端口必须异步，Parser 必须同步。"""

from __future__ import annotations

import inspect

from rag_kb.domain.ports import (
    LLM,
    VLM,
    Embedder,
    GraphStore,
    LexicalIndex,
    MetadataRepository,
    Parser,
    Reranker,
    Storage,
    VectorStore,
)

ASYNC_METHODS: dict[type, tuple[str, ...]] = {
    MetadataRepository: (
        "create_kb",
        "get_kb",
        "list_kbs",
        "delete_kb",
        "create_document",
        "get_document",
        "list_documents",
        "update_document",
        "delete_document",
        "find_document_by_sha",
        "replace_chunks",
        "list_chunks",
        "get_chunks",
        "delete_chunks",
        "count_chunks",
        "enqueue_ingest_job",
        "claim_ingest_job",
        "update_ingest_job",
        "add_query_log",
    ),
    VectorStore: ("ensure_collection", "upsert", "search", "delete_by_doc", "count"),
    LexicalIndex: ("upsert", "search", "delete_by_doc", "count"),
    GraphStore: (
        "ensure_schema",
        "upsert",
        "candidate_entities",
        "expand",
        "delete_by_doc",
        "count",
    ),
    Embedder: ("embed_documents", "embed_query"),
    LLM: ("chat", "structured_json"),
    VLM: ("describe_image",),
    Reranker: ("rerank",),
    Storage: ("save_upload", "save_image", "read", "delete", "delete_document"),
}


def test_io_ports_are_async() -> None:
    for port, names in ASYNC_METHODS.items():
        for name in names:
            method = getattr(port, name)
            assert inspect.iscoroutinefunction(method), f"{port.__name__}.{name} 必须是 async"


def test_parser_parse_is_sync() -> None:
    # CPU 密集方法由 worker 放进线程池，端口本身不应声明为 async。
    assert not inspect.iscoroutinefunction(Parser.parse)
