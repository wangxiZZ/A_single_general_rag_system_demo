"""HTTP 请求/响应 schema。

与领域模型分开：领域模型可能增删内部字段（如 meta/chunker_version），API 只暴露
调用方需要的部分；响应结构一旦公开就要保持稳定。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.domain.models import ChunkKind, ChunkLevel, DocumentStatus, SearchMode


class KBCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class KBRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str
    created_at: datetime | None = None


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    kb_id: UUID
    filename: str
    mime: str
    size: int
    status: DocumentStatus
    error: str | None = None
    page_count: int | None = None
    chunk_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SearchHitRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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
    debug: dict[str, object] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=20, ge=1, le=50)
    rerank: bool = False


class BBoxRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    x0: float
    y0: float
    x1: float
    y1: float


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kb_id: UUID
    doc_id: UUID
    parent_id: str | None = None
    level: ChunkLevel
    kind: ChunkKind
    content: str
    heading_path: tuple[str, ...] = ()
    page_no: int | None = None
    page_to: int | None = None
    bbox: BBoxRead | None = None
    char_start: int
    char_end: int
    seq: int
    token_count: int
    image_id: UUID | None = None
    meta: dict[str, object]
    chunker_version: str
