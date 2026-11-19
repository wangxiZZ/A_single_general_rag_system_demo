"""KB / Document 用例的纯单测：使用内存 fake 端口。"""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from rag_kb.application.document_service import DocumentService
from rag_kb.application.kb_service import KbService
from rag_kb.domain.errors import (
    DuplicateDocumentError,
    FileTooLargeError,
    NotFoundError,
    ValidationError,
)
from rag_kb.domain.models import DocumentStatus, IngestJobStatus
from tests.unit.fakes import FakeMetadataRepository, FakeStorage


@pytest.fixture
def repo() -> FakeMetadataRepository:
    return FakeMetadataRepository()


@pytest.fixture
def storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def kb_service(repo: FakeMetadataRepository, storage: FakeStorage) -> KbService:
    return KbService(repo, storage)


@pytest.fixture
def doc_service(repo: FakeMetadataRepository, storage: FakeStorage) -> DocumentService:
    return DocumentService(repo, storage, max_upload_mb=1)


async def test_create_kb_rejects_empty_name(kb_service: KbService) -> None:
    with pytest.raises(ValidationError):
        await kb_service.create_kb("   ")


async def test_upload_creates_pending_document_and_job(
    kb_service: KbService, doc_service: DocumentService, repo: FakeMetadataRepository
) -> None:
    kb = await kb_service.create_kb("测试库")
    doc = await doc_service.upload_document(
        kb.id, filename="a.pdf", mime="application/pdf", data=b"%PDF-1.7"
    )

    assert doc.status is DocumentStatus.PENDING
    assert doc.kb_id == kb.id
    assert len(doc.sha256) == 64
    assert repo.documents[doc.id] == doc
    assert len(repo.jobs) == 1
    assert repo.jobs[0].doc_id == doc.id
    assert repo.jobs[0].status is IngestJobStatus.PENDING


async def test_duplicate_sha_raises_with_existing_id(
    kb_service: KbService, doc_service: DocumentService, repo: FakeMetadataRepository
) -> None:
    kb = await kb_service.create_kb("测试库")
    first = await doc_service.upload_document(
        kb.id, filename="a.pdf", mime="application/pdf", data=b"same"
    )

    with pytest.raises(DuplicateDocumentError) as exc_info:
        await doc_service.upload_document(
            kb.id, filename="b.pdf", mime="application/pdf", data=b"same"
        )

    assert exc_info.value.existing_document_id == first.id
    assert len(repo.documents) == 1


async def test_oversized_upload_has_no_side_effects(
    kb_service: KbService, doc_service: DocumentService, repo: FakeMetadataRepository
) -> None:
    kb = await kb_service.create_kb("测试库")
    with pytest.raises(FileTooLargeError):
        await doc_service.upload_document(
            kb.id,
            filename="big.pdf",
            mime="application/pdf",
            data=b"x" * (1024 * 1024 + 1),
        )
    assert repo.documents == {}
    assert repo.jobs == []


async def test_upload_to_missing_kb_raises(doc_service: DocumentService) -> None:
    with pytest.raises(NotFoundError):
        await doc_service.upload_document(
            uuid4(), filename="a.pdf", mime="application/pdf", data=b"x"
        )


async def test_delete_document_removes_row_and_files(
    kb_service: KbService,
    doc_service: DocumentService,
    repo: FakeMetadataRepository,
    storage: FakeStorage,
) -> None:
    kb = await kb_service.create_kb("测试库")
    doc = await doc_service.upload_document(
        kb.id, filename="a.pdf", mime="application/pdf", data=b"x"
    )
    assert storage.files

    await doc_service.delete_document(kb.id, doc.id)
    assert doc.id not in repo.documents
    assert storage.files == {}
    assert repo.jobs == []


async def test_delete_kb_cascades_documents_and_files(
    kb_service: KbService,
    doc_service: DocumentService,
    repo: FakeMetadataRepository,
    storage: FakeStorage,
) -> None:
    kb = await kb_service.create_kb("测试库")
    first = await doc_service.upload_document(
        kb.id, filename="a.pdf", mime="application/pdf", data=b"a"
    )
    second = await doc_service.upload_document(
        kb.id, filename="b.pdf", mime="application/pdf", data=b"b"
    )

    await kb_service.delete_kb(kb.id)
    assert await repo.get_kb(kb.id) is None
    assert first.id not in repo.documents
    assert second.id not in repo.documents
    assert storage.files == {}
    assert repo.jobs == []


class RecordingIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    async def delete_by_doc(self, kb_id, doc_id) -> int:
        self.calls.append((kb_id, doc_id))
        return 1


async def test_delete_document_cleans_all_indexes(
    kb_service: KbService,
    repo: FakeMetadataRepository,
    storage: FakeStorage,
) -> None:
    kb = await kb_service.create_kb("测试库")
    vector, lexical, graph = RecordingIndex(), RecordingIndex(), RecordingIndex()
    service = DocumentService(
        repo,
        storage,
        vector_store=vector,  # type: ignore[arg-type]
        lexical_index=lexical,  # type: ignore[arg-type]
        graph_store=graph,  # type: ignore[arg-type]
    )
    doc = await service.upload_document(kb.id, filename="a.pdf", mime="application/pdf", data=b"x")

    await service.delete_document(kb.id, doc.id)

    for port in (vector, lexical, graph):
        assert port.calls == [(kb.id, doc.id)]


async def test_delete_kb_cleans_all_indexes(
    repo: FakeMetadataRepository, storage: FakeStorage
) -> None:
    vector, lexical, graph = RecordingIndex(), RecordingIndex(), RecordingIndex()
    service = KbService(
        repo,
        storage,
        vector_store=vector,  # type: ignore[arg-type]
        lexical_index=lexical,  # type: ignore[arg-type]
        graph_store=graph,  # type: ignore[arg-type]
    )
    kb = await service.create_kb("测试库")
    doc = await DocumentService(repo, storage).upload_document(
        kb.id, filename="a.pdf", mime="application/pdf", data=b"x"
    )

    await service.delete_kb(kb.id)

    for port in (vector, lexical, graph):
        assert port.calls == [(kb.id, doc.id)]


async def test_transition_rules_are_enforced(
    kb_service: KbService, doc_service: DocumentService, repo: FakeMetadataRepository
) -> None:
    kb = await kb_service.create_kb("测试库")
    doc = await doc_service.upload_document(
        kb.id, filename="a.pdf", mime="application/pdf", data=b"x"
    )

    # pending 不能直接到 ready
    with pytest.raises(ValidationError):
        await doc_service.transition_document(kb.id, doc.id, DocumentStatus.READY)

    parsing = await doc_service.transition_document(kb.id, doc.id, DocumentStatus.PARSING)
    assert parsing.status is DocumentStatus.PARSING
    assert repo.documents[doc.id].status is DocumentStatus.PARSING


async def test_retry_failed_document(
    kb_service: KbService, doc_service: DocumentService, repo: FakeMetadataRepository
) -> None:
    kb = await kb_service.create_kb("测试库")
    doc = await doc_service.upload_document(
        kb.id, filename="a.pdf", mime="application/pdf", data=b"x"
    )
    repo.jobs[0] = replace(repo.jobs[0], status=IngestJobStatus.FAILED)

    failed = await doc_service.mark_failed(kb.id, doc.id, "解析失败")
    assert failed.status is DocumentStatus.FAILED
    assert failed.error == "解析失败"

    retried = await doc_service.retry_document(kb.id, doc.id)
    assert retried.status is DocumentStatus.PENDING
    assert retried.error is None
    assert len(repo.jobs) == 2
    assert repo.jobs[-1].status is IngestJobStatus.PENDING


async def test_mark_failed_truncates_long_error(
    kb_service: KbService, doc_service: DocumentService
) -> None:
    kb = await kb_service.create_kb("测试库")
    doc = await doc_service.upload_document(
        kb.id, filename="a.pdf", mime="application/pdf", data=b"x"
    )
    failed = await doc_service.mark_failed(kb.id, doc.id, "x" * 3000)
    assert failed.error is not None
    assert len(failed.error) == 2000
