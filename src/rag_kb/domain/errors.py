"""领域异常。

为什么单独定义一套，而不是让 psycopg / pymilvus / httpx 的异常冒到上层：

1. ``application`` 需要决定"重试还是标记失败"，它不应该知道第三方异常的类名；
2. ``api`` 需要把异常映射成 HTTP 状态码，映射表应该基于稳定的错误码，而不是字符串；
3. worker 写 ``document.error`` 时要给用户可读信息，不能把堆栈或密钥带出去。

约定：``code`` 是稳定的对外错误码，``str(exc)`` 是给人看的中文说明。infra 层负责把
第三方异常转换成这里的类型，禁止把裸 ``Exception`` 抛到路由或 worker 之外。
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    INTERNAL = "internal"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    VALIDATION = "validation"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    FILE_TOO_LARGE = "file_too_large"
    PARSE = "parse_error"
    NO_EXTRACTABLE_TEXT = "no_extractable_text"
    EMBEDDING = "embedding_error"
    LLM = "llm_error"
    STORAGE = "storage_error"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    CONFIGURATION = "configuration_error"
    INGEST_ABORTED = "ingest_aborted"


class DomainError(Exception):
    """所有领域异常的基类。"""

    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(self, message: str, *, code: ErrorCode | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class NotFoundError(DomainError):
    code = ErrorCode.NOT_FOUND

    def __init__(self, resource: str, resource_id: object) -> None:
        super().__init__(f"{resource} 不存在: {resource_id}")
        self.resource = resource
        self.resource_id = resource_id


class ConflictError(DomainError):
    """资源冲突，例如同库重复上传同一份文件。"""

    code = ErrorCode.CONFLICT


class ValidationError(DomainError):
    code = ErrorCode.VALIDATION


class UnsupportedFileTypeError(DomainError):
    code = ErrorCode.UNSUPPORTED_FILE_TYPE

    def __init__(self, filename: str, mime: str = "") -> None:
        detail = f"{filename} ({mime})" if mime else filename
        super().__init__(f"不支持的文件类型: {detail}")
        self.filename = filename
        self.mime = mime


class FileTooLargeError(DomainError):
    code = ErrorCode.FILE_TOO_LARGE

    def __init__(self, filename: str, size: int, limit_mb: int) -> None:
        super().__init__(f"文件超过 {limit_mb} MB 限制: {filename} ({size} bytes)")
        self.filename = filename
        self.size = size
        self.limit_mb = limit_mb


class ParseError(DomainError):
    code = ErrorCode.PARSE


class NoExtractableTextError(ParseError):
    """扫描件/纯图片 PDF：题目不要求 OCR，因此明确失败而不是产出空 chunk。"""

    code = ErrorCode.NO_EXTRACTABLE_TEXT

    def __init__(self, filename: str) -> None:
        super().__init__(f"未提取到文本（不支持扫描件 OCR）: {filename}")
        self.filename = filename


class EmbeddingError(DomainError):
    code = ErrorCode.EMBEDDING


class LlmError(DomainError):
    code = ErrorCode.LLM


class StorageError(DomainError):
    code = ErrorCode.STORAGE


class DependencyUnavailableError(DomainError):
    """存储/模型依赖不可用；api 据此返回 503，而不是 500。"""

    code = ErrorCode.DEPENDENCY_UNAVAILABLE

    def __init__(self, dependency: str) -> None:
        super().__init__(f"依赖不可用: {dependency}")
        self.dependency = dependency


class ConfigurationError(DomainError):
    """缺少必要配置（如 EMBED_API_KEY）；应在写任何索引前抛出。"""

    code = ErrorCode.CONFIGURATION


class IngestAbortedError(DomainError):
    """摄取过程中文档被删除（tombstone），用于中止任务而不是标记失败。"""

    code = ErrorCode.INGEST_ABORTED

    def __init__(self, document_id: object) -> None:
        super().__init__(f"文档已被删除，摄取中止: {document_id}")
        self.document_id = document_id
