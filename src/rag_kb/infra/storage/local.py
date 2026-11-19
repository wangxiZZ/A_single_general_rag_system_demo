"""本地文件存储。

路径约定（数据库里存相对路径，宿主与容器可移植）：

    data/uploads/<kb_id>/<doc_id>/<safe_filename>
    data/images/<kb_id>/<doc_id>/<image_id>.<ext>

三个刻意的设计：

1. **路径带 kb_id**：即使 doc_id 全局唯一，删除/排查时也不会跨库误伤；与检索层
   "一切带 kb_id" 的约束保持一致。
2. **原子写**：先写同目录临时文件再 ``os.replace``，避免进程被杀时留下半个上传件。
3. **路径越界检查**：``read``/``delete`` 只接受解析后仍位于 data_dir 内的路径，
   防止数据库被污染后读到任意文件。

同步文件 I/O 由 ``anyio.to_thread`` 包装，避免阻塞事件循环（ADR-019）。
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from collections.abc import Awaitable, Callable
from functools import partial, wraps
from pathlib import Path
from uuid import UUID, uuid4

import anyio

from rag_kb.domain.errors import StorageError

_EXT_RE = re.compile(r"^\.[a-z0-9]{1,5}$")
_UNSAFE_RE = re.compile(r"[^\w.\-]")


def _to_thread[**P, R](func: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await anyio.to_thread.run_sync(partial(func, *args, **kwargs))

    return wrapper


def safe_filename(filename: str, *, fallback: str = "upload.bin") -> str:
    """只保留基名并过滤不安全字符；防止 ``../`` 与 Windows 反斜杠穿越。"""
    base = filename.replace("\\", "/").split("/")[-1].replace("\x00", "").strip()
    base = _UNSAFE_RE.sub("_", base).lstrip("._")
    if not base:
        return fallback
    return base[:180]


def safe_ext(ext: str) -> str:
    normalized = ext.strip().lower()
    if normalized and not normalized.startswith("."):
        normalized = "." + normalized
    return normalized if _EXT_RE.fullmatch(normalized) else ".bin"


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self._root / relative_path).resolve()
        if not candidate.is_relative_to(self._root):
            raise StorageError(f"路径越界: {relative_path}")
        return candidate

    def _write_atomic(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _document_dirs(self, kb_id: UUID, doc_id: UUID) -> tuple[Path, Path]:
        return (
            self._root / "uploads" / str(kb_id) / str(doc_id),
            self._root / "images" / str(kb_id) / str(doc_id),
        )

    @_to_thread
    def save_upload(self, kb_id: UUID, doc_id: UUID, filename: str, data: bytes) -> str:
        upload_dir, _ = self._document_dirs(kb_id, doc_id)
        path = upload_dir / safe_filename(filename)
        self._write_atomic(path, data)
        return path.relative_to(self._root).as_posix()

    @_to_thread
    def save_image(self, kb_id: UUID, doc_id: UUID, image_id: str, data: bytes, *, ext: str) -> str:
        _, image_dir = self._document_dirs(kb_id, doc_id)
        path = image_dir / f"{safe_filename(image_id, fallback='image')}{safe_ext(ext)}"
        self._write_atomic(path, data)
        return path.relative_to(self._root).as_posix()

    @_to_thread
    def read(self, path: str) -> bytes:
        resolved = self._resolve(path)
        try:
            return resolved.read_bytes()
        except FileNotFoundError as exc:
            raise StorageError(f"文件不存在: {path}") from exc

    @_to_thread
    def delete(self, path: str) -> None:
        self._resolve(path).unlink(missing_ok=True)

    @_to_thread
    def delete_document(self, kb_id: UUID, doc_id: UUID) -> None:
        upload_dir, image_dir = self._document_dirs(kb_id, doc_id)
        for directory in (upload_dir, image_dir):
            shutil.rmtree(directory, ignore_errors=True)
            # 删除空的知识库目录，避免 data/ 下积累空目录；目录非空时 rmdir 会失败，忽略即可。
            with contextlib.suppress(OSError):
                directory.parent.rmdir()


__all__ = ["LocalStorage", "safe_ext", "safe_filename"]
