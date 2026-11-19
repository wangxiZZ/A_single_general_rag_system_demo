"""端口协议。

``application`` 只依赖这里的接口，不 import 任何 infra 实现；具体适配器在 ``main.py``
的 lifespan 里构造并注入。这样 API、worker、CLI 与测试可以复用同一组用例，换存储只改
``infra/`` 一个目录。

异步契约（ADR-019）：

- 所有 I/O 端口方法声明为 ``async def``；同步 SDK（psycopg / pymilvus / neo4j / 本地文件）
  必须由适配器内部用 ``anyio.to_thread.run_sync`` 包装，禁止在协程里直接阻塞。
- ``Parser.parse`` 是 CPU 密集的同步方法，由 worker 放进线程池调用。
- 端口只抛 ``domain/errors.py`` 里的领域异常；第三方异常在 infra 层转换。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain.models import (
    KB,
    ChatMessage,
    Chunk,
    DocImage,
    Document,
    Entity,
    GraphEvidence,
    IngestJob,
    LexicalHit,
    ParsedDocument,
    QueryLog,
    Relation,
    SearchHit,
    VectorHit,
    VectorRecord,
)


@runtime_checkable
class MetadataRepository(Protocol):
    """业务元数据、分块、摄取任务与查询日志的持久化。

    所有涉及文档/分块的方法都显式带 ``kb_id``：即使 ``doc_id`` 全局唯一，也不把
    "库内隔离"责任留给调用方。
    """

    # ---- 知识库 ----
    async def create_kb(self, kb: KB) -> KB: ...
    async def get_kb(self, kb_id: UUID) -> KB | None: ...
    async def list_kbs(self) -> list[KB]: ...
    async def delete_kb(self, kb_id: UUID) -> None: ...

    # ---- 文档 ----
    async def create_document(self, doc: Document) -> Document: ...
    async def get_document(self, kb_id: UUID, doc_id: UUID) -> Document | None: ...
    async def list_documents(self, kb_id: UUID) -> list[Document]: ...
    async def update_document(self, kb_id: UUID, doc: Document) -> Document: ...
    async def delete_document(self, kb_id: UUID, doc_id: UUID) -> None: ...
    async def find_document_by_sha(self, kb_id: UUID, sha256: str) -> Document | None: ...

    # ---- 分块 ----
    async def replace_chunks(self, kb_id: UUID, doc_id: UUID, chunks: Sequence[Chunk]) -> None:
        """原子替换某文档的全部分块；重跑摄取时先删后插。"""
        ...

    async def list_chunks(self, kb_id: UUID, doc_id: UUID) -> list[Chunk]: ...
    async def get_chunks(self, kb_id: UUID, chunk_ids: Sequence[str]) -> list[Chunk]: ...
    async def delete_chunks(self, kb_id: UUID, doc_id: UUID) -> int: ...
    async def count_chunks(self, kb_id: UUID, doc_id: UUID) -> int: ...

    # ---- 图片 ----
    async def replace_images(self, kb_id: UUID, doc_id: UUID, images: Sequence[DocImage]) -> None:
        """原子替换某文档的图片记录；重跑摄取时先删后插。"""
        ...

    async def list_images(self, kb_id: UUID, doc_id: UUID) -> list[DocImage]: ...
    async def count_images(self, kb_id: UUID, doc_id: UUID) -> int: ...

    # ---- 摄取任务 ----
    async def enqueue_ingest_job(self, kb_id: UUID, doc_id: UUID) -> IngestJob: ...
    async def claim_ingest_job(self) -> IngestJob | None:
        """领取一个待处理任务（``FOR UPDATE SKIP LOCKED``），无任务返回 ``None``。"""
        ...

    async def update_ingest_job(self, job: IngestJob) -> IngestJob: ...

    # ---- 查询日志 ----
    async def add_query_log(self, log: QueryLog) -> None: ...


@runtime_checkable
class VectorStore(Protocol):
    """稠密向量通道。所有读写必须带 ``kb_id``。"""

    async def ensure_collection(self, dimension: int) -> None:
        """集合不存在则创建；存在则校验维度，不一致抛领域异常。"""
        ...

    async def upsert(self, kb_id: UUID, records: Sequence[VectorRecord]) -> None: ...
    async def search(self, kb_id: UUID, vector: Sequence[float], top_k: int) -> list[VectorHit]: ...
    async def delete_by_doc(self, kb_id: UUID, doc_id: UUID) -> int: ...
    async def count(self, kb_id: UUID, doc_id: UUID | None = None) -> int: ...


@runtime_checkable
class LexicalIndex(Protocol):
    """关键词通道。入库前由适配器完成 jieba 分词与清洗。"""

    async def upsert(self, kb_id: UUID, chunks: Sequence[Chunk]) -> None: ...
    async def search(self, kb_id: UUID, query: str, top_k: int) -> list[LexicalHit]: ...
    async def delete_by_doc(self, kb_id: UUID, doc_id: UUID) -> int: ...
    async def count(self, kb_id: UUID, doc_id: UUID | None = None) -> int: ...


@runtime_checkable
class GraphStore(Protocol):
    """图谱通道。

    只负责执行查询与写入；"精确 → 别名 → 2-gram"的链接策略与路径打分在
    ``application/search_service.py``，不要下沉到适配器。
    """

    async def ensure_schema(self) -> None: ...
    async def upsert(
        self,
        kb_id: UUID,
        doc_id: UUID,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
    ) -> None: ...
    async def candidate_entities(
        self, kb_id: UUID, terms: Sequence[str], limit: int
    ) -> list[Entity]: ...
    async def expand(
        self, kb_id: UUID, seed_ids: Sequence[str], max_hops: int, neighbor_limit: int
    ) -> list[GraphEvidence]: ...
    async def delete_by_doc(self, kb_id: UUID, doc_id: UUID) -> int: ...
    async def count(self, kb_id: UUID, doc_id: UUID | None = None) -> int: ...


@runtime_checkable
class Embedder(Protocol):
    """文本向量化。实现内部按 ``EMBED_BATCH_SIZE`` 分批并限流。"""

    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class LLM(Protocol):
    """生成式模型：回答与图谱抽取共用。"""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str: ...

    async def structured_json(
        self, messages: Sequence[ChatMessage], schema: dict[str, object]
    ) -> dict[str, object]:
        """返回受 schema 约束的 JSON；解析失败抛 ``LlmError``，由调用方决定丢弃或重试。"""
        ...


@runtime_checkable
class VLM(Protocol):
    """图片描述；失败时由摄取管线降级为占位文本，不阻塞文档 ready。"""

    async def describe_image(
        self, image: bytes, *, mime: str = "image/png", prompt: str = ""
    ) -> str: ...


@runtime_checkable
class Reranker(Protocol):
    """重排。默认关闭；候选是融合后的 ``SearchHit`` 列表。"""

    async def rerank(
        self, query: str, candidates: Sequence[SearchHit], top_k: int
    ) -> list[SearchHit]: ...


@runtime_checkable
class Parser(Protocol):
    """文档解析。同步 CPU 方法，由 worker 放线程池调用。"""

    def parse(self, data: bytes, *, filename: str, mime: str) -> ParsedDocument: ...


@runtime_checkable
class Storage(Protocol):
    """上传件与图片的本地/对象存储。"""

    async def save_upload(self, kb_id: UUID, doc_id: UUID, filename: str, data: bytes) -> str: ...

    async def save_image(
        self, kb_id: UUID, doc_id: UUID, image_id: str, data: bytes, *, ext: str
    ) -> str: ...

    async def read(self, path: str) -> bytes: ...
    async def delete(self, path: str) -> None: ...
    async def delete_document(self, kb_id: UUID, doc_id: UUID) -> None: ...


__all__ = [
    "LLM",
    "VLM",
    "Embedder",
    "GraphStore",
    "LexicalIndex",
    "MetadataRepository",
    "Parser",
    "Reranker",
    "Storage",
    "VectorStore",
]
