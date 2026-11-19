"""解析器基类与注册表的纯单测。"""

from __future__ import annotations

import pytest

from rag_kb.domain.errors import ParseError, UnsupportedFileTypeError
from rag_kb.domain.models import Block, BlockType, ParsedDocument
from rag_kb.infra.parsers.base import BaseParser, ParserRegistry


class TextParser(BaseParser):
    extensions = frozenset({".txt"})

    def _parse(self, data: bytes, *, filename: str, mime: str) -> ParsedDocument:
        text = data.decode("utf-8")
        return ParsedDocument(
            blocks=(
                Block(
                    type=BlockType.TEXT,
                    text=text,
                    seq=0,
                    char_start=0,
                    char_end=len(text),
                ),
            )
        )


class BrokenParser(BaseParser):
    extensions = frozenset({".bad"})

    def _parse(self, data: bytes, *, filename: str, mime: str) -> ParsedDocument:
        raise ValueError("第三方解析器炸了")


def test_base_parser_returns_parsed_document() -> None:
    parsed = TextParser().parse(b"hello", filename="a.txt", mime="text/plain")
    assert parsed.blocks[0].text == "hello"
    assert parsed.blocks[0].type is BlockType.TEXT


def test_base_parser_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        TextParser().parse(b"x", filename="a.pdf", mime="application/pdf")


def test_base_parser_rejects_empty_file() -> None:
    with pytest.raises(ParseError, match="空文件"):
        TextParser().parse(b"", filename="a.txt")


def test_base_parser_wraps_unexpected_exception() -> None:
    with pytest.raises(ParseError, match="解析失败") as exc_info:
        BrokenParser().parse(b"x", filename="a.bad")
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_base_parser_preserves_domain_exception() -> None:
    class NoTextParser(BaseParser):
        extensions = frozenset({".empty"})

        def _parse(self, data: bytes, *, filename: str, mime: str) -> ParsedDocument:
            raise ParseError("no text")

    with pytest.raises(ParseError, match="no text"):
        NoTextParser().parse(b"x", filename="a.empty")


def test_registry_selects_parser_by_extension() -> None:
    registry = ParserRegistry([TextParser()])
    parsed = registry.parse(b"ok", filename="a.TXT", mime="text/plain")
    assert parsed.blocks[0].text == "ok"


def test_registry_rejects_unknown_extension() -> None:
    registry = ParserRegistry([TextParser()])
    with pytest.raises(UnsupportedFileTypeError):
        registry.parse(b"x", filename="a.pdf")


def test_registry_rejects_duplicate_extension() -> None:
    registry = ParserRegistry([TextParser()])
    with pytest.raises(ValueError, match="已注册"):
        registry.register(TextParser())
