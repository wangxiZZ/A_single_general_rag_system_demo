"""上传文件类型校验（纯函数，API 与 Web 共用）。

只信任扩展名与 magic bytes，不信任浏览器上报的 Content-Type；后者仅作为缺省 MIME 使用。
"""

from __future__ import annotations

from pathlib import Path

from rag_kb.domain.errors import UnsupportedFileTypeError

MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def validate_upload(filename: str, content_type: str | None, data: bytes) -> str:
    """返回应使用的 MIME；扩展名或 magic bytes 不合法时抛领域异常。"""
    ext = Path(filename).suffix.lower()
    if ext not in MIME_BY_EXT:
        raise UnsupportedFileTypeError(filename, content_type or "")

    # docx 本质是 zip；只接受 PK\x03\x04（含实际内容），空 zip 不算 docx。
    ok = data.startswith(b"%PDF-") if ext == ".pdf" else data.startswith(b"PK\x03\x04")
    if not ok:
        raise UnsupportedFileTypeError(filename, MIME_BY_EXT[ext])
    return content_type or MIME_BY_EXT[ext]
