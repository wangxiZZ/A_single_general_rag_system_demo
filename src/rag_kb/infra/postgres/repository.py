"""PostgreSQL 版 MetadataRepository。

实现要点：

1. **所有文档/分块查询都带 `kb_id`**，即使 `doc_id` 全局唯一，也不把隔离责任留给调用方。
2. **SQL 手写**，不引 ORM：检索类查询（后续的 tsquery、分页、计数）需要直接看执行计划。
3. **错误转换**：psycopg 异常在这里统一转成领域异常，application 不需要知道驱动类型。
4. **同步池 + 线程池**：psycopg 的异步连接在 Windows 的 ProactorEventLoop 上不可用，
   因此这里使用同步 ``ConnectionPool``，每个方法用 ``anyio.to_thread`` 包装（ADR-019）。
   这样 Linux 与 Windows 行为一致，也不会阻塞事件循环。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial, wraps
from typing import Any
from uuid import UUID

import anyio
import psycopg
from psycopg import Connection
from psycopg import errors as pg_errors
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from rag_kb.domain.errors import (
    ConflictError,
    DependencyUnavailableError,
    NotFoundError,
    StorageError,
)
from rag_kb.domain.models import (
    KB,
    BBox,
    Chunk,
    ChunkKind,
    ChunkLevel,
    DocImage,
    Document,
    DocumentStatus,
    IngestJob,
    IngestJobStatus,
    QueryLog,
)


def _to_thread[**P, R](func: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    """把同步仓储方法包装成端口要求的 async 方法。"""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await anyio.to_thread.run_sync(partial(func, *args, **kwargs))

    return wrapper


def _row_to_kb(row: dict[str, Any]) -> KB:
    return KB(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        created_at=row["created_at"],
    )


def _row_to_document(row: dict[str, Any]) -> Document:
    return Document(
        id=row["id"],
        kb_id=row["kb_id"],
        filename=row["filename"],
        mime=row["mime"],
        sha256=row["sha256"],
        size=row["size"],
        file_path=row["file_path"],
        status=DocumentStatus(row["status"]),
        error=row["error"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        chunker_version=row["chunker_version"],
        embed_model=row["embed_model"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    bbox = row["bbox"]
    return Chunk(
        id=row["id"],
        kb_id=row["kb_id"],
        doc_id=row["doc_id"],
        parent_id=row["parent_id"],
        level=ChunkLevel(row["level"]),
        kind=ChunkKind(row["kind"]),
        content=row["content"],
        heading_path=tuple(row["heading_path"] or ()),
        page_no=row["page_no"],
        page_to=row["page_to"],
        bbox=BBox(**bbox) if bbox else None,
        char_start=row["char_start"],
        char_end=row["char_end"],
        seq=row["seq"],
        token_count=row["token_count"],
        image_id=row["image_id"],
        meta=row["meta"] or {},
        chunker_version=row["chunker_version"],
    )


def _row_to_job(row: dict[str, Any]) -> IngestJob:
    return IngestJob(
        id=row["id"],
        kb_id=row["kb_id"],
        doc_id=row["doc_id"],
        stage=row["stage"],
        status=IngestJobStatus(row["status"]),
        attempts=row["attempts"],
        error=row["error"],
        timings=row["timings"] or {},
        claimed_at=row["claimed_at"],
    )


def _row_to_image(row: dict[str, Any]) -> DocImage:
    bbox = row["bbox"]
    return DocImage(
        id=row["id"],
        kb_id=row["kb_id"],
        doc_id=row["doc_id"],
        seq=row["seq"],
        page_no=row["page_no"],
        bbox=BBox(**bbox) if bbox else None,
        file_path=row["file_path"],
        caption=row["caption"],
    )


def _bbox_json(bbox: BBox | None) -> Jsonb | None:
    if bbox is None:
        return None
    return Jsonb({"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1})


class PostgresMetadataRepository:
    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 8) -> None:
        self._pool = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"row_factory": dict_row},
        )

    async def open(self) -> None:
        await anyio.to_thread.run_sync(self._pool.open)
        await anyio.to_thread.run_sync(self._pool.wait)

    async def close(self) -> None:
        await anyio.to_thread.run_sync(self._pool.close)

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        """统一把驱动异常转成领域异常；UniqueViolation 先于泛化的 psycopg.Error 捕获。"""
        try:
            with self._pool.connection() as conn:
                yield conn
        except pg_errors.UniqueViolation as exc:
            raise ConflictError(f"唯一约束冲突: {exc}") from exc
        except (psycopg.OperationalError, PoolTimeout) as exc:
            raise DependencyUnavailableError("postgres") from exc
        except psycopg.Error as exc:
            raise StorageError(f"PostgreSQL 操作失败: {exc}") from exc

    # ---------------------------------------------------------------- 知识库

    @_to_thread
    def create_kb(self, kb: KB) -> KB:
        with self._connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO kb (id, name, description, created_at)
                VALUES (%s, %s, %s, COALESCE(%s, now()))
                RETURNING *
                """,
                (kb.id, kb.name, kb.description, kb.created_at),
            )
            row = cur.fetchone()
        return _row_to_kb(row)

    @_to_thread
    def get_kb(self, kb_id: UUID) -> KB | None:
        with self._connection() as conn:
            cur = conn.execute("SELECT * FROM kb WHERE id = %s", (kb_id,))
            row = cur.fetchone()
        return _row_to_kb(row) if row else None

    @_to_thread
    def list_kbs(self) -> list[KB]:
        with self._connection() as conn:
            cur = conn.execute("SELECT * FROM kb ORDER BY created_at")
            rows = cur.fetchall()
        return [_row_to_kb(row) for row in rows]

    @_to_thread
    def delete_kb(self, kb_id: UUID) -> None:
        with self._connection() as conn:
            cur = conn.execute("DELETE FROM kb WHERE id = %s", (kb_id,))
            if cur.rowcount == 0:
                raise NotFoundError("knowledge base", kb_id)

    # ---------------------------------------------------------------- 文档

    @_to_thread
    def create_document(self, doc: Document) -> Document:
        with self._connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO document (
                    id, kb_id, filename, mime, sha256, size, file_path, status, error,
                    page_count, chunk_count, chunker_version, embed_model,
                    created_at, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    COALESCE(%s, now()), COALESCE(%s, now())
                )
                RETURNING *
                """,
                (
                    doc.id,
                    doc.kb_id,
                    doc.filename,
                    doc.mime,
                    doc.sha256,
                    doc.size,
                    doc.file_path,
                    doc.status.value,
                    doc.error,
                    doc.page_count,
                    doc.chunk_count,
                    doc.chunker_version,
                    doc.embed_model,
                    doc.created_at,
                    doc.updated_at,
                ),
            )
            row = cur.fetchone()
        return _row_to_document(row)

    @_to_thread
    def get_document(self, kb_id: UUID, doc_id: UUID) -> Document | None:
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT * FROM document WHERE kb_id = %s AND id = %s", (kb_id, doc_id)
            )
            row = cur.fetchone()
        return _row_to_document(row) if row else None

    @_to_thread
    def list_documents(self, kb_id: UUID) -> list[Document]:
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT * FROM document WHERE kb_id = %s ORDER BY created_at DESC", (kb_id,)
            )
            rows = cur.fetchall()
        return [_row_to_document(row) for row in rows]

    @_to_thread
    def update_document(self, kb_id: UUID, doc: Document) -> Document:
        with self._connection() as conn:
            cur = conn.execute(
                """
                UPDATE document SET
                    file_path = %s, status = %s, error = %s, page_count = %s,
                    chunk_count = %s, chunker_version = %s, embed_model = %s,
                    updated_at = now()
                WHERE kb_id = %s AND id = %s
                RETURNING *
                """,
                (
                    doc.file_path,
                    doc.status.value,
                    doc.error,
                    doc.page_count,
                    doc.chunk_count,
                    doc.chunker_version,
                    doc.embed_model,
                    kb_id,
                    doc.id,
                ),
            )
            row = cur.fetchone()
        if row is None:
            raise NotFoundError("document", doc.id)
        return _row_to_document(row)

    @_to_thread
    def delete_document(self, kb_id: UUID, doc_id: UUID) -> None:
        with self._connection() as conn:
            cur = conn.execute("DELETE FROM document WHERE kb_id = %s AND id = %s", (kb_id, doc_id))
            if cur.rowcount == 0:
                raise NotFoundError("document", doc_id)

    @_to_thread
    def find_document_by_sha(self, kb_id: UUID, sha256: str) -> Document | None:
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT * FROM document WHERE kb_id = %s AND sha256 = %s", (kb_id, sha256)
            )
            row = cur.fetchone()
        return _row_to_document(row) if row else None

    # ---------------------------------------------------------------- 分块

    @_to_thread
    def replace_chunks(self, kb_id: UUID, doc_id: UUID, chunks: Sequence[Chunk]) -> None:
        # 先删后插：重跑摄取时不会留下旧版本分块；整个操作在一个事务内完成。
        with self._connection() as conn, conn.transaction():
            conn.execute("DELETE FROM chunk WHERE kb_id = %s AND doc_id = %s", (kb_id, doc_id))
            if not chunks:
                return

            # 父块先写，配合 DEFERRABLE 外键，避免批量插入时顺序问题。
            ordered = sorted(chunks, key=lambda c: 0 if c.level is ChunkLevel.PARENT else 1)
            rows = [
                (
                    chunk.id,
                    chunk.kb_id,
                    chunk.doc_id,
                    chunk.parent_id,
                    chunk.level.value,
                    chunk.kind.value,
                    chunk.content,
                    list(chunk.heading_path),
                    chunk.page_no,
                    chunk.page_to,
                    _bbox_json(chunk.bbox),
                    chunk.char_start,
                    chunk.char_end,
                    chunk.seq,
                    chunk.token_count,
                    chunk.image_id,
                    Jsonb(chunk.meta),
                    chunk.chunker_version,
                )
                for chunk in ordered
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunk (
                        id, kb_id, doc_id, parent_id, level, kind, content,
                        heading_path, page_no, page_to, bbox, char_start, char_end,
                        seq, token_count, image_id, meta, chunker_version
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s
                    )
                    """,
                    rows,
                )

    @_to_thread
    def list_chunks(self, kb_id: UUID, doc_id: UUID) -> list[Chunk]:
        with self._connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM chunk
                WHERE kb_id = %s AND doc_id = %s
                ORDER BY seq
                """,
                (kb_id, doc_id),
            )
            rows = cur.fetchall()
        return [_row_to_chunk(row) for row in rows]

    @_to_thread
    def get_chunks(self, kb_id: UUID, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        with self._connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM chunk
                WHERE kb_id = %s AND id = ANY(%s)
                ORDER BY seq
                """,
                (kb_id, list(chunk_ids)),
            )
            rows = cur.fetchall()
        return [_row_to_chunk(row) for row in rows]

    @_to_thread
    def delete_chunks(self, kb_id: UUID, doc_id: UUID) -> int:
        with self._connection() as conn:
            cur = conn.execute(
                "DELETE FROM chunk WHERE kb_id = %s AND doc_id = %s", (kb_id, doc_id)
            )
            return cur.rowcount

    @_to_thread
    def count_chunks(self, kb_id: UUID, doc_id: UUID) -> int:
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT count(*) AS n FROM chunk WHERE kb_id = %s AND doc_id = %s",
                (kb_id, doc_id),
            )
            row = cur.fetchone()
        return int(row["n"])

    # ---------------------------------------------------------------- 图片

    @_to_thread
    def replace_images(self, kb_id: UUID, doc_id: UUID, images: Sequence[DocImage]) -> None:
        with self._connection() as conn, conn.transaction():
            conn.execute("DELETE FROM doc_image WHERE kb_id = %s AND doc_id = %s", (kb_id, doc_id))
            if not images:
                return
            rows = [
                (
                    image.id,
                    image.kb_id,
                    image.doc_id,
                    image.seq,
                    image.page_no,
                    _bbox_json(image.bbox),
                    image.file_path,
                    image.caption,
                )
                for image in sorted(images, key=lambda item: item.seq)
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO doc_image (
                        id, kb_id, doc_id, seq, page_no, bbox, file_path, caption
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )

    @_to_thread
    def list_images(self, kb_id: UUID, doc_id: UUID) -> list[DocImage]:
        with self._connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM doc_image
                WHERE kb_id = %s AND doc_id = %s
                ORDER BY seq
                """,
                (kb_id, doc_id),
            )
            rows = cur.fetchall()
        return [_row_to_image(row) for row in rows]

    @_to_thread
    def count_images(self, kb_id: UUID, doc_id: UUID) -> int:
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT count(*) AS n FROM doc_image WHERE kb_id = %s AND doc_id = %s",
                (kb_id, doc_id),
            )
            row = cur.fetchone()
        return int(row["n"])

    # ---------------------------------------------------------------- 摄取任务

    @_to_thread
    def enqueue_ingest_job(self, kb_id: UUID, doc_id: UUID) -> IngestJob:
        with self._connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO ingest_job (doc_id, stage, status)
                VALUES (%s, 'parse', 'pending')
                ON CONFLICT DO NOTHING
                RETURNING id
                """,
                (doc_id,),
            )
            inserted = cur.fetchone()
            job_id = inserted["id"] if inserted else None
            if job_id is None:
                # 已有活跃任务：返回它，避免同一文档重复摄取（幂等）。
                cur = conn.execute(
                    """
                    SELECT id FROM ingest_job
                    WHERE doc_id = %s AND status IN ('pending', 'running')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (doc_id,),
                )
                active = cur.fetchone()
                job_id = active["id"] if active else None
            if job_id is None:
                raise ConflictError(f"文档已有活跃摄取任务: {doc_id}")

            cur = conn.execute(
                """
                SELECT j.*, d.kb_id
                FROM ingest_job j JOIN document d ON d.id = j.doc_id
                WHERE j.id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise NotFoundError("ingest job", job_id)
        return _row_to_job(row)

    @_to_thread
    def claim_ingest_job(self) -> IngestJob | None:
        with self._connection() as conn, conn.transaction():
            cur = conn.execute(
                """
                WITH claimed AS (
                    UPDATE ingest_job SET
                        status = 'running',
                        attempts = attempts + 1,
                        claimed_at = now(),
                        updated_at = now()
                    WHERE id = (
                        SELECT id FROM ingest_job
                        WHERE status = 'pending'
                        ORDER BY created_at
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    RETURNING *
                )
                SELECT c.*, d.kb_id
                FROM claimed c JOIN document d ON d.id = c.doc_id
                """
            )
            row = cur.fetchone()
        return _row_to_job(row) if row else None

    @_to_thread
    def update_ingest_job(self, job: IngestJob) -> IngestJob:
        with self._connection() as conn:
            cur = conn.execute(
                """
                UPDATE ingest_job SET
                    stage = %s, status = %s, attempts = %s, error = %s,
                    timings = %s, claimed_at = %s, updated_at = now()
                WHERE id = %s
                RETURNING id
                """,
                (
                    job.stage,
                    job.status.value,
                    job.attempts,
                    job.error,
                    Jsonb(job.timings),
                    job.claimed_at,
                    job.id,
                ),
            )
            updated = cur.fetchone()
            if updated is None:
                raise NotFoundError("ingest job", job.id)
            cur = conn.execute(
                """
                SELECT j.*, d.kb_id
                FROM ingest_job j JOIN document d ON d.id = j.doc_id
                WHERE j.id = %s
                """,
                (job.id,),
            )
            row = cur.fetchone()
        if row is None:
            raise NotFoundError("ingest job", job.id)
        return _row_to_job(row)

    # ---------------------------------------------------------------- 查询日志

    @_to_thread
    def add_query_log(self, log: QueryLog) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO query_log (
                    kb_id, mode, query, latency_ms, hit_ids, channels, rerank, tokens
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    log.kb_id,
                    log.mode.value,
                    log.query,
                    log.latency_ms,
                    Jsonb(list(log.hit_ids)),
                    Jsonb(log.channels),
                    log.rerank,
                    log.tokens,
                ),
            )
