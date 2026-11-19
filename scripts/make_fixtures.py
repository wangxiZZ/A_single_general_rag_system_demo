"""生成解析测试夹具。

夹具是确定性的（固定随机种子），生成后提交到 ``tests/fixtures/documents/``；
测试直接读取它们，避免每次运行依赖 Pillow/Word 等外部工具。

用法：

    uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

import io
import random
import struct
import zlib
from pathlib import Path

import pymupdf
from docx import Document

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "documents"


def _png(size: int, *, seed: int) -> bytes:
    rng = random.Random(seed)
    raw = b"".join(b"\x00" + rng.randbytes(size * 3) for _ in range(size))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=0))
        + chunk(b"IEND", b"")
    )


def make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Fixture Manual", fontsize=24)
    page.insert_text((72, 130), "This is body text for the fixture manual.", fontsize=10)
    page.insert_text((72, 170), "3.2 Dependencies", fontsize=16)
    page.insert_text((72, 200), "Install dependencies before running the service.", fontsize=10)
    page.insert_image(pymupdf.Rect(72, 240, 272, 440), stream=_png(200, seed=7))

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 80), "Appendix", fontsize=24)
    page2.insert_text((72, 130), "Additional fixture notes.", fontsize=10)

    doc.save(path)
    doc.close()


def make_docx(path: Path) -> None:
    document = Document()
    document.add_heading("Fixture Manual", level=1)
    document.add_paragraph("This is body text for the fixture manual.")
    document.add_heading("3.2 Dependencies", level=2)
    document.add_paragraph("Install dependencies before running the service.")
    document.add_picture(io.BytesIO(_png(80, seed=9)))

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "A"
    table.cell(1, 1).text = "B"

    document.sections[0].header.paragraphs[0].text = "HEADER SHOULD NOT APPEAR"
    document.save(path)


def make_scanned(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(72, 72, 240, 240), color=(1, 0, 0), fill=(1, 0, 0))
    doc.save(path)
    doc.close()


def make_demo_pdf(path: Path) -> None:
    """演示用《员工手册》：中文、两页、含一张图片。

    用 PyMuPDF 内置的 CJK 字体 ``china-s``，避免把字体文件也提交进仓库；
    内容覆盖「多条款 + 数字 + 流程」这类最容易被问到的检索场景。
    """
    doc = pymupdf.open()

    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 90), "员工手册", fontname="china-s", fontsize=24)
    page.insert_text((72, 130), "版本 2026.1 · 人力资源部", fontname="china-s", fontsize=10)
    page.insert_text((72, 180), "1. 年假政策", fontname="china-s", fontsize=16)
    lines = [
        "员工入职满一年后，可享受带薪年假。",
        "满一年不满五年：每年 5 个工作日；满五年不满十年：每年 10 个工作日；",
        "满十年及以上：每年 15 个工作日。",
        "年假以自然年为周期计算，未休完的部分可顺延至次年 3 月 31 日。",
        "申请年假需提前 3 个工作日在人事系统提交，由直属主管审批。",
    ]
    y = 210
    for line in lines:
        page.insert_text((72, y), line, fontname="china-s", fontsize=11)
        y += 24
    page.insert_image(pymupdf.Rect(72, y + 10, 272, y + 210), stream=_png(200, seed=11))

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 90), "2. 费用报销流程", fontname="china-s", fontsize=18)
    lines2 = [
        "差旅费用应在行程结束后 15 日内提交报销申请，并附上合规发票。",
        "单笔报销金额超过 5000 元时，需部门负责人签字确认。",
        "财务在收到完整材料后 5 个工作日内完成审核，审核通过后 3 个工作日内打款。",
        "发票抬头必须为公司全称，个人抬头的发票不予报销。",
        "3. 联系我们",
        "人事问题请联系 hr@example.com；",
        "报销问题请联系 finance@example.com。",
    ]
    y = 130
    for line in lines2:
        page2.insert_text((72, y), line, fontname="china-s", fontsize=11)
        y += 28

    doc.save(path)
    doc.close()


def make_demo_docx(path: Path) -> None:
    """演示用《产品手册》：标题、正文、表格、图片，验证 docx 的无页码引用。"""
    document = Document()
    document.add_heading("产品手册", level=1)
    document.add_paragraph("本手册介绍知识库服务的部署方式与常见问题。")

    document.add_heading("1. 部署方式", level=2)
    document.add_paragraph("支持 Docker Compose 单机部署，也支持本机直接运行 uvicorn。")
    document.add_paragraph("生产环境建议将 PostgreSQL、Milvus 与 Neo4j 独立部署并配置备份。")

    document.add_heading("2. 支持的文档格式", level=2)
    table = document.add_table(rows=3, cols=2)
    table.cell(0, 0).text = "格式"
    table.cell(0, 1).text = "说明"
    table.cell(1, 0).text = "PDF"
    table.cell(1, 1).text = "保留页码与坐标，支持内嵌图片"
    table.cell(2, 0).text = "DOCX"
    table.cell(2, 1).text = "保留段落顺序与表格，引用使用段落号"
    document.add_picture(io.BytesIO(_png(120, seed=13)))

    document.add_heading("3. 常见问题", level=2)
    document.add_paragraph("问：上传后一直显示 pending 怎么办？")
    document.add_paragraph("答：检查 worker 是否启动，以及模型 API Key 是否配置正确。")
    document.add_paragraph("问：为什么扫描件无法检索？")
    document.add_paragraph("答：当前版本不做 OCR，扫描件没有可提取文本，会标记为失败。")

    document.save(path)


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    make_pdf(FIXTURE_DIR / "sample.pdf")
    make_docx(FIXTURE_DIR / "sample.docx")
    make_scanned(FIXTURE_DIR / "scanned.pdf")

    demo_dir = FIXTURE_DIR.parent / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    make_demo_pdf(demo_dir / "employee_handbook.pdf")
    make_demo_docx(demo_dir / "product_guide.docx")
    print(f"fixtures written to {FIXTURE_DIR} and {demo_dir}")


if __name__ == "__main__":
    main()
