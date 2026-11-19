"""生成图谱抽取评测夹具（金标语料 + 对抗语料）。

金标语料用 PDF，方便验证页码；对抗语料用 docx，才能构造真正的表格块（PDF 表格会被
当普通文本抽取）。内容与 `tests/fixtures/graph/gold.json` 一一对应，修改事实时两边一起改。

用法：

    uv run python scripts/make_graph_fixtures.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
from docx import Document

from scripts.make_fixtures import _png

OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "graph"

GOLD_PAGES: list[list[str]] = [
    [
        "图谱评测金标语料",
        "1. 公司与产品",
        "星云科技提供知识库服务。",
        "知识库服务包含检索组件（XR 组件）。",
        "检索组件依赖向量数据库。",
        "向量数据库位于华东机房。",
        "华东机房属于华东数据中心。",
        "华东数据中心位于上海。",
        "张伟属于星云科技。",
        "张伟负责上线审批流程。",
        "上线审批流程依赖数据安全条款。",
        "数据安全条款包含数据脱敏要求。",
    ],
    [
        "2. 合作方与流程",
        "蓝海数据提供数据网关。",
        "数据网关使用数据同步流程。",
        "数据同步流程关联单次批量 500 条。",
        "单次批量 500 条属于服务等级协议。",
        "服务等级协议包含 SLA 99.9%。",
        "李娜属于蓝海数据。",
        "李娜负责数据同步流程。",
        "数据网关包含鉴权模块。",
        "鉴权模块使用数据安全条款。",
        "知识库服务使用数据网关。",
        "2026 年 3 月迁移关联知识库服务。",
        "蓝海数据位于北京。",
    ],
]


def make_gold_pdf(path: Path) -> None:
    doc = pymupdf.open()
    for page_lines in GOLD_PAGES:
        page = doc.new_page(width=595, height=842)
        y = 90
        for index, line in enumerate(page_lines):
            size = 20 if index == 0 else 11
            page.insert_text((72, y), line, fontname="china-s", fontsize=size)
            y += 30 if index == 0 else 26
    doc.save(path)
    doc.close()


def make_adversarial_docx(path: Path) -> None:
    document = Document()

    document.add_heading("对抗语料（用于统计误抽）", level=1)
    document.add_paragraph("星云科技可能在 2027 年推出新版本。")
    document.add_paragraph("知识库服务不依赖旧版检索组件。")
    document.add_paragraph("它负责所有上线审批。")
    document.add_paragraph("蓝海数据与星云科技是同一家公司。")

    document.add_heading("2. 表格中的关系（应被排除）", level=2)
    table = document.add_table(rows=3, cols=2)
    table.cell(0, 0).text = "系统"
    table.cell(0, 1).text = "提供方"
    table.cell(1, 0).text = "审计日志"
    table.cell(1, 1).text = "数据网关"
    table.cell(2, 0).text = "报表服务"
    table.cell(2, 1).text = "蓝海数据"

    document.add_heading("3. 图片（caption 不参与图抽取）", level=2)
    document.add_picture(io.BytesIO(_png(120, seed=21)))

    header = document.sections[0].header.paragraphs[0]
    header.text = "内部资料，请勿外传"
    document.save(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    make_gold_pdf(OUTPUT / "gold_doc.pdf")
    make_adversarial_docx(OUTPUT / "adversarial_doc.docx")
    print(f"graph fixtures written to {OUTPUT}")


if __name__ == "__main__":
    main()
