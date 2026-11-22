"""检索与回答结果里的对外链接。

为什么把 URL 拼装放在 application 而不是 api：
``SearchHit.image_url`` 是跨入口的统一契约——REST、MCP（转发 REST 后原样返回）、
Web 都消费同一个值。如果每个入口各拼一次，就会出现"MCP 返回的图片地址 REST 打不开"
这类漂移；放在这里，路径只有一处来源。

取舍：这条路径确实包含 ``/api`` 前缀，等于 application 知道 REST 的挂载点。接受这个
耦合，因为另一端（``api/routers/documents.py`` 的图片路由）本来就由本仓库定义；真正
要避免的是 application 依赖 FastAPI 或 HTTP 语义，而这里只是拼字符串。
"""

from __future__ import annotations

from uuid import UUID

from rag_kb.domain.models import Chunk, ChunkKind


def document_image_path(kb_id: UUID, doc_id: UUID, image_id: UUID) -> str:
    """原图的 REST 路径；对应 ``GET /api/kbs/{kb}/documents/{doc}/images/{image}``。"""
    return f"/api/kbs/{kb_id}/documents/{doc_id}/images/{image_id}"


def chunk_image_path(kb_id: UUID, chunk: Chunk) -> str | None:
    """块对应的原图地址：只有图片块（``kind=image`` 且带 ``image_id``）才有。"""
    if chunk.kind is not ChunkKind.IMAGE or chunk.image_id is None:
        return None
    return document_image_path(kb_id, chunk.doc_id, chunk.image_id)


__all__ = ["chunk_image_path", "document_image_path"]
