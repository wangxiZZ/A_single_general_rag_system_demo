"""HTTP 请求/响应 schema。

与领域模型分开：领域模型可能增删内部字段（如 meta/chunker_version），API 只暴露
调用方需要的部分；响应结构一旦公开就要保持稳定。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.domain.models import DocumentStatus


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
