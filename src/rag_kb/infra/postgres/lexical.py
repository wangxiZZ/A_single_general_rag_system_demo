"""PostgreSQL 词法索引。

中文不会因为装了分词器就自动可搜：入库前 jieba 分词、清洗、去停用词，把结果空格连接
写入 ``chunk_lex.lex_tokens``；查询侧走同一套清洗后拼 ``to_tsquery('simple', 'a | b')``。
``tsv`` 是生成列，索引由 GIN 承担，``ts_rank_cd`` 只做排序。

只索引 child 块；查询必须 join ``chunk`` 并带 ``kb_id``。
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import anyio
import jieba
import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from rag_kb.domain.errors import (
    DependencyUnavailableError,
    NotFoundError,
    StorageError,
    ValidationError,
)
from rag_kb.domain.models import Chunk, ChunkLevel, LexicalHit

_STOPWORDS_FILE = Path(__file__).resolve().parent / "stopwords_zh.txt"
_TOKEN_RE = re.compile(r"^[\w一-鿿]+$", re.UNICODE)


def _to_thread[**P, R](func: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await anyio.to_thread.run_sync(lambda: func(*args, **kwargs))

    return wrapper


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    if not _STOPWORDS_FILE.exists():
        return frozenset()
    return frozenset(
        line.strip()
        for line in _STOPWORDS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _is_single_cjk(token: str) -> bool:
    return len(token) == 1 and "一" <= token <= "鿿"


def tokenize(text: str) -> list[str]:
    words = stopwords()
    tokens: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer:
            tokens.append(buffer)
            buffer = ""

    # jieba 对未登录复合词（如"年假"）常切成单字；把连续单字合并回复合词，
    # 否则"年假"查询会匹配任何包含"年"或"假"的块。空白、标点与停用词是词边界，
    # 必须 flush，不能把"年假 年假"合成为一个词。
    for raw in jieba.lcut(text):
        token = raw.strip().lower()
        if not token or not _TOKEN_RE.fullmatch(token) or token in words:
            flush()
            continue
        if _is_single_cjk(token):
            buffer += token
            continue
        flush()
        tokens.append(token)
    flush()
    return tokens


def build_tsquery(text: str) -> str:
    # 去重后拼 OR；to_tsquery 对未转义的特殊字符会直接报语法错误。
    tokens = list(dict.fromkeys(tokenize(text)))
    return " | ".join(tokens)


class PgLexicalIndex:
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
        try:
            with self._pool.connection() as conn:
                yield conn
        except (psycopg.OperationalError, PoolTimeout) as exc:
            raise DependencyUnavailableError("postgres") from exc
        except psycopg.Error as exc:
            raise StorageError(f"PostgreSQL 词法索引操作失败: {exc}") from exc

    @_to_thread
    def upsert(self, kb_id: UUID, chunks: Sequence[Chunk]) -> None:
        rows: list[tuple[str, str, UUID]] = []
        for chunk in chunks:
            if chunk.level is not ChunkLevel.CHILD:
                raise ValidationError("只有 level=child 的块可以写入词法索引")
            rows.append((chunk.id, " ".join(tokenize(chunk.content)), kb_id))
        if not rows:
            return

        with self._connection() as conn, conn.transaction(), conn.cursor() as cur:
            for chunk_id, lex_tokens, kb in rows:
                cur.execute(
                    """
                        INSERT INTO chunk_lex (chunk_id, lex_tokens)
                        SELECT id, %s FROM chunk WHERE id = %s AND kb_id = %s
                        ON CONFLICT (chunk_id) DO UPDATE
                        SET lex_tokens = EXCLUDED.lex_tokens
                        """,
                    (lex_tokens, chunk_id, kb),
                )
                if cur.rowcount == 0:
                    raise NotFoundError("chunk", chunk_id)

    @_to_thread
    def search(self, kb_id: UUID, query: str, top_k: int) -> list[LexicalHit]:
        tsquery = build_tsquery(query)
        if not tsquery:
            return []
        with self._connection() as conn:
            cur = conn.execute(
                """
                SELECT c.id AS chunk_id, ts_rank_cd(l.tsv, q) AS rank
                FROM chunk_lex l
                JOIN chunk c ON c.id = l.chunk_id
                CROSS JOIN to_tsquery('simple', %s) AS q
                WHERE c.kb_id = %s AND l.tsv @@ q
                ORDER BY rank DESC, c.id
                LIMIT %s
                """,
                (tsquery, kb_id, top_k),
            )
            rows = cur.fetchall()
        return [
            LexicalHit(chunk_id=row["chunk_id"], score=float(row["rank"]), rank=index)
            for index, row in enumerate(rows)
        ]

    @_to_thread
    def delete_by_doc(self, kb_id: UUID, doc_id: UUID) -> int:
        with self._connection() as conn:
            cur = conn.execute(
                """
                DELETE FROM chunk_lex l
                USING chunk c
                WHERE l.chunk_id = c.id AND c.kb_id = %s AND c.doc_id = %s
                """,
                (kb_id, doc_id),
            )
            return cur.rowcount

    @_to_thread
    def count(self, kb_id: UUID, doc_id: UUID | None = None) -> int:
        expr = "c.kb_id = %s"
        params: list[object] = [kb_id]
        if doc_id is not None:
            expr += " AND c.doc_id = %s"
            params.append(doc_id)
        with self._connection() as conn:
            cur = conn.execute(
                f"""
                SELECT count(*) AS n
                FROM chunk_lex l JOIN chunk c ON c.id = l.chunk_id
                WHERE {expr}
                """,
                tuple(params),
            )
            row = cur.fetchone()
        return int(row["n"])
