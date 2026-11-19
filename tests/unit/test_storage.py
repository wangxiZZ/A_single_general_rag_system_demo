"""本地存储单测：使用临时目录，不连数据库。"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from rag_kb.domain.errors import StorageError
from rag_kb.infra.storage.local import LocalStorage, safe_ext, safe_filename


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path)


async def test_save_and_read_upload(storage: LocalStorage, tmp_path: Path) -> None:
    kb_id, doc_id = uuid4(), uuid4()
    relative = await storage.save_upload(kb_id, doc_id, "说明.pdf", b"%PDF-1.7 test")

    assert relative == f"uploads/{kb_id}/{doc_id}/说明.pdf"
    assert (tmp_path / relative).read_bytes() == b"%PDF-1.7 test"
    assert await storage.read(relative) == b"%PDF-1.7 test"


async def test_save_overwrites_atomically(storage: LocalStorage) -> None:
    kb_id, doc_id = uuid4(), uuid4()
    first = await storage.save_upload(kb_id, doc_id, "a.pdf", b"first")
    second = await storage.save_upload(kb_id, doc_id, "a.pdf", b"second")
    assert first == second
    assert await storage.read(second) == b"second"


async def test_filename_traversal_is_sanitized(storage: LocalStorage) -> None:
    kb_id, doc_id = uuid4(), uuid4()
    relative = await storage.save_upload(kb_id, doc_id, "../../evil.pdf", b"x")
    assert relative.endswith("/evil.pdf")
    assert ".." not in relative
    assert await storage.read(relative) == b"x"


async def test_read_rejects_path_traversal(storage: LocalStorage) -> None:
    with pytest.raises(StorageError, match="路径越界"):
        await storage.read("../outside.txt")


async def test_save_image_and_delete_document(storage: LocalStorage, tmp_path: Path) -> None:
    kb_id, doc_id = uuid4(), uuid4()
    upload = await storage.save_upload(kb_id, doc_id, "a.pdf", b"pdf")
    image = await storage.save_image(kb_id, doc_id, "img-1", b"png", ext=".PNG")

    assert image == f"images/{kb_id}/{doc_id}/img-1.png"
    assert (tmp_path / upload).exists()
    assert (tmp_path / image).exists()

    await storage.delete_document(kb_id, doc_id)
    assert not (tmp_path / upload).exists()
    assert not (tmp_path / image).exists()


async def test_delete_is_idempotent(storage: LocalStorage) -> None:
    await storage.delete("uploads/missing/file.pdf")
    await storage.delete_document(uuid4(), uuid4())


def test_safe_filename_and_ext() -> None:
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("..\\..\\x.pdf") == "x.pdf"
    assert safe_filename("   ") == "upload.bin"
    assert safe_ext("PNG") == ".png"
    assert safe_ext("../../x") == ".bin"
