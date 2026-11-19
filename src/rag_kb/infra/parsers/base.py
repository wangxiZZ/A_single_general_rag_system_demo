"""解析器基类与注册表。

两个职责分开：

- ``BaseParser`` 负责"统一入口 + 错误归一"：扩展名校验、空文件拒绝、把第三方解析异常
  转成 ``ParseError``；子类只实现 ``_parse``，不需要重复这些边界处理。
- ``ParserRegistry`` 负责按扩展名选择解析器。worker 只依赖注册表，不关心具体格式；
  新增格式只需注册一个新 subclass。

``parse`` 是同步 CPU 方法（端口契约见 ADR-019），由 worker 放进线程池调用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from rag_kb.domain.errors import DomainError, ParseError, UnsupportedFileTypeError
from rag_kb.domain.models import ParsedDocument


class BaseParser(ABC):
    extensions: ClassVar[frozenset[str]] = frozenset()

    def supports(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.extensions

    def parse(self, data: bytes, *, filename: str, mime: str = "") -> ParsedDocument:
        if not self.supports(filename):
            raise UnsupportedFileTypeError(filename, mime)
        if not data:
            raise ParseError(f"空文件: {filename}")
        try:
            return self._parse(data, filename=filename, mime=mime)
        except DomainError:
            raise
        except Exception as exc:
            # 不把第三方库的异常类型泄漏到 application；保留原始异常链便于日志排查。
            raise ParseError(f"{filename} 解析失败: {exc}") from exc

    @abstractmethod
    def _parse(self, data: bytes, *, filename: str, mime: str) -> ParsedDocument: ...


class ParserRegistry:
    def __init__(self, parsers: Sequence[BaseParser] = ()) -> None:
        self._by_ext: dict[str, BaseParser] = {}
        for parser in parsers:
            self.register(parser)

    def register(self, parser: BaseParser) -> None:
        for ext in parser.extensions:
            existing = self._by_ext.get(ext)
            if existing is not None and existing is not parser:
                raise ValueError(f"扩展名 {ext} 已注册给 {type(existing).__name__}")
            self._by_ext[ext] = parser

    def parser_for(self, filename: str) -> BaseParser:
        ext = Path(filename).suffix.lower()
        parser = self._by_ext.get(ext)
        if parser is None:
            raise UnsupportedFileTypeError(filename)
        return parser

    def parse(self, data: bytes, *, filename: str, mime: str = "") -> ParsedDocument:
        return self.parser_for(filename).parse(data, filename=filename, mime=mime)
