"""领域模型。

这里是业务事实的唯一定义处：只依赖标准库，不 import 任何存储客户端、Web 框架或
第三方 SDK。字段与 ``docs/技术选型文档.md`` 的 ADR-017/018 保持一致——改字段前先改
ADR，再改这里，避免代码与设计文档各说一套。

两个刻意保留的空值：

1. docx 没有页码与坐标，``page_no`` / ``bbox`` 允许为 ``None``；用 ``char_start/char_end``
   与 ``seq`` 定位。为满足字段而伪造页码会让引用失真。
2. 父块只用于回答阶段回填，``Chunk.is_indexable`` 明确只有子块进向量/词法索引。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class DocumentStatus(StrEnum):
    """文档状态机。

    ``READY`` 的语义是"向量/词法/图谱三路索引全部完成"，不是"解析完"。M2 结束只到
    ``CHUNKED``，M3 建完索引才进 ``READY``——否则 MCP 的 ``ready_count`` 与全链路测试
    都会把"只有 chunk 行"误认为"可检索"。
    """

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKED = "chunked"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


# 允许的状态迁移。READY -> INDEXING 用于 reindex；FAILED -> PENDING 只允许显式重试。
_ALLOWED_TRANSITIONS: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    DocumentStatus.PENDING: frozenset({DocumentStatus.PARSING, DocumentStatus.FAILED}),
    DocumentStatus.PARSING: frozenset({DocumentStatus.CHUNKED, DocumentStatus.FAILED}),
    DocumentStatus.CHUNKED: frozenset({DocumentStatus.INDEXING, DocumentStatus.FAILED}),
    DocumentStatus.INDEXING: frozenset({DocumentStatus.READY, DocumentStatus.FAILED}),
    DocumentStatus.READY: frozenset({DocumentStatus.INDEXING}),
    DocumentStatus.FAILED: frozenset({DocumentStatus.PENDING}),
}


def can_transition(current: DocumentStatus, target: DocumentStatus) -> bool:
    """判断状态迁移是否合法。

    单独暴露成函数，便于 API/worker 在写库前校验；真正的不可逆保证仍要靠仓储层
    的条件更新，不能只依赖调用方自觉。
    """
    return target in _ALLOWED_TRANSITIONS[current]


class ChunkLevel(StrEnum):
    PARENT = "parent"
    CHILD = "child"


class ChunkKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


class SearchMode(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    GRAPH = "graph"
    HYBRID = "hybrid"
    HYBRID_WEIGHTED = "hybrid_weighted"


@dataclass(frozen=True, slots=True)
class BBox:
    """PDF 文本块/图片的包围盒，坐标原点在页面左上角。"""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"bbox 的右下角不能小于左上角: {self}")

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(frozen=True, slots=True)
class KB:
    id: UUID
    name: str
    description: str = ""
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Document:
    id: UUID
    kb_id: UUID
    filename: str
    mime: str
    sha256: str
    size: int
    status: DocumentStatus = DocumentStatus.PENDING
    error: str | None = None
    page_count: int | None = None
    chunk_count: int = 0
    chunker_version: str = ""
    embed_model: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def can_transition_to(self, target: DocumentStatus) -> bool:
        return can_transition(self.status, target)

    @property
    def is_retryable(self) -> bool:
        return self.status is DocumentStatus.FAILED


@dataclass(frozen=True, slots=True)
class Chunk:
    """可回溯的分块。

    ``id`` 由 ``sha256(doc_sha256:seq:chunker_version)`` 确定性生成，保证重跑幂等与
    评测金标可复现；``level`` 决定是否进索引，``meta`` 只放低权重/降级标记等附属信息。
    """

    id: str
    kb_id: UUID
    doc_id: UUID
    level: ChunkLevel
    kind: ChunkKind
    content: str
    seq: int
    char_start: int
    char_end: int
    token_count: int
    chunker_version: str
    parent_id: str | None = None
    heading_path: tuple[str, ...] = ()
    page_no: int | None = None
    page_to: int | None = None
    bbox: BBox | None = None
    image_id: UUID | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.char_end < self.char_start:
            raise ValueError("char_end 不能小于 char_start")
        if self.seq < 0:
            raise ValueError("seq 不能为负数")

    @property
    def is_indexable(self) -> bool:
        """只有子块进向量/词法索引；父块只做回答阶段回填。"""
        return self.level is ChunkLevel.CHILD


@dataclass(frozen=True, slots=True)
class SearchHit:
    """统一的检索返回结构。

    ``chunk_id`` 默认是子块；只有图谱证据无法映射到子块时才回退父块并标
    ``level=parent``。``debug`` 记录各通道排名与原始分，用于排查"某一路是否真的生效"。
    """

    chunk_id: str
    doc_id: UUID
    doc_name: str
    level: ChunkLevel
    score: float
    channels: tuple[str, ...]
    snippet: str = ""
    heading_path: tuple[str, ...] = ()
    page_no: int | None = None
    image_url: str | None = None
    debug: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Citation:
    """回答中的一条来源引用，``index`` 对应正文里的 ``[n]``。"""

    index: int
    doc_id: UUID
    doc_name: str
    chunk_id: str
    heading_path: tuple[str, ...] = ()
    page_no: int | None = None
    seq: int | None = None
    snippet: str = ""

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("引用编号从 1 开始")


class BlockType(StrEnum):
    TEXT = "text"
    HEADING = "heading"
    IMAGE = "image"
    TABLE = "table"


class IngestJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ImageRef:
    """解析器抽出的图片；``data`` 在领域层保持 bytes，落盘由 Storage 端口负责。"""

    id: str
    data: bytes
    mime: str = "image/png"


@dataclass(frozen=True, slots=True)
class Block:
    """解析器的统一输出单元。

    两种格式的能力不同：PDF 有 ``page_no/bbox``，docx 为 ``None``；两者都必须提供
    ``seq`` 与 ``char_start/char_end``，否则图片与文本无法回到原文位置。
    """

    type: BlockType
    text: str
    seq: int
    char_start: int
    char_end: int
    page_no: int | None = None
    bbox: BBox | None = None
    heading_level: int | None = None
    image_ref: ImageRef | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    blocks: tuple[Block, ...]
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class VectorRecord:
    chunk_id: str
    doc_id: UUID
    level: ChunkLevel
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VectorHit:
    chunk_id: str
    score: float
    rank: int | None = None


@dataclass(frozen=True, slots=True)
class LexicalHit:
    chunk_id: str
    score: float
    rank: int | None = None


@dataclass(frozen=True, slots=True)
class Entity:
    """Neo4j 中的实体；``normalized`` 参与唯一约束与种子链接。"""

    id: str
    kb_id: UUID
    name: str
    normalized: str
    type: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Relation:
    """实体关系边。``chunk_id`` 必须指向可检索的子块，否则无法回溯原文。"""

    head_id: str
    tail_id: str
    type: str
    chunk_id: str
    weight: float = 1.0
    evidence_span: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class GraphEvidence:
    chunk_id: str
    score: float
    hops: int
    seed_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestJob:
    id: UUID
    doc_id: UUID
    stage: str
    status: IngestJobStatus = IngestJobStatus.PENDING
    attempts: int = 0
    error: str | None = None
    timings: dict[str, object] = field(default_factory=dict)
    claimed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class QueryLog:
    kb_id: UUID
    mode: SearchMode
    query: str
    latency_ms: float
    hit_ids: tuple[str, ...] = ()
    channels: dict[str, int] = field(default_factory=dict)
    rerank: bool = False
    tokens: int | None = None
    created_at: datetime | None = None
