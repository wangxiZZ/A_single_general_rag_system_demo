"""解析器注册表工厂。"""

from __future__ import annotations

from rag_kb.infra.parsers.base import ParserRegistry
from rag_kb.infra.parsers.docx import DocxParser
from rag_kb.infra.parsers.pdf import PdfParser


def build_parser_registry() -> ParserRegistry:
    return ParserRegistry([PdfParser(), DocxParser()])


__all__ = ["build_parser_registry"]
