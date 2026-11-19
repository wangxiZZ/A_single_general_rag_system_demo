"""生成评测金标 `tests/fixtures/bench/qa.jsonl`。

金标不用手写字符偏移，而是从解析器输出里定位原句：解析器稳定、偏移可复算，
换分块参数后金标仍然有效。`doc_id` 在文件里为 null——它是每次装载 KB 时生成的 uuid，
`run_bench.py` 在运行时按 `doc_name` 解析，避免把某次演示的 id 固化进仓库。

用法：

    uv run python scripts/bench/make_qa.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_kb.infra.parsers import build_parser_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "demo"
OUTPUT = REPO_ROOT / "tests" / "fixtures" / "bench" / "qa.jsonl"
MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _load_blocks(filename: str):
    data = (FIXTURES / filename).read_bytes()
    parsed = build_parser_registry().parse(
        data, filename=filename, mime=MIME[Path(filename).suffix.lower()]
    )
    return parsed.blocks


def _span(filename: str, phrase: str) -> dict:
    """在解析结果中定位一句话，返回与 chunk 相同坐标系的字符区间。"""
    for block in _load_blocks(filename):
        index = block.text.find(phrase)
        if index >= 0:
            return {
                "doc_name": filename,
                "doc_id": None,
                "page_no": block.page_no,
                "char_start": block.char_start + index,
                "char_end": block.char_start + index + len(phrase),
            }
    raise ValueError(f"{filename} 中找不到金标句子：{phrase}")


def _image_span(filename: str, page_no: int | None) -> dict:
    """图片块没有正文区间，用页码级命中；char_start == char_end 表示按页匹配。"""
    for block in _load_blocks(filename):
        if block.type.value == "image" and block.page_no == page_no:
            return {
                "doc_name": filename,
                "doc_id": None,
                "page_no": block.page_no,
                "char_start": block.char_start,
                "char_end": block.char_start,
            }
    raise ValueError(f"{filename} 第 {page_no} 页找不到图片块")


PDF = "employee_handbook.pdf"
DOCX = "product_guide.docx"

# (id, type, question, reference_answer, gold)
CASES = [
    (
        "q-001",
        "single_hop",
        "入职满五年但不满十年的员工，每年有几天年假？",
        "10 个工作日。",
        [_span(PDF, "满五年不满十年：每年 10 个工作日")],
    ),
    (
        "q-002",
        "single_hop",
        "申请年假需要提前几个工作日？",
        "提前 3 个工作日在人事系统提交。",
        [_span(PDF, "申请年假需提前 3 个工作日在人事系统提交，由直属主管审批。")],
    ),
    (
        "q-003",
        "single_hop",
        "当年没休完的年假可以顺延到什么时候？",
        "可顺延至次年 3 月 31 日。",
        [_span(PDF, "未休完的部分可顺延至次年 3 月 31 日")],
    ),
    (
        "q-004",
        "single_hop",
        "差旅费用最晚要在行程结束后多少天内提交报销？",
        "15 日内。",
        [_span(PDF, "差旅费用应在行程结束后 15 日内提交报销申请")],
    ),
    (
        "q-005",
        "single_hop",
        "单笔报销超过多少钱需要部门负责人签字？",
        "超过 5000 元。",
        [_span(PDF, "单笔报销金额超过 5000 元时，需部门负责人签字确认。")],
    ),
    (
        "q-006",
        "single_hop",
        "财务审核通过后多久打款？",
        "审核通过后 3 个工作日内打款。",
        [_span(PDF, "审核通过后 3 个工作日内打款")],
    ),
    (
        "q-007",
        "single_hop",
        "报销对发票抬头有什么要求？",
        "必须是公司全称，个人抬头不予报销。",
        [_span(PDF, "发票抬头必须为公司全称，个人抬头的发票不予报销。")],
    ),
    (
        "q-008",
        "term",
        "人事问题应该联系哪个邮箱？",
        "hr@example.com。",
        [_span(PDF, "hr@example.com")],
    ),
    (
        "q-009",
        "term",
        "报销问题应该联系哪个邮箱？",
        "finance@example.com。",
        [_span(PDF, "finance@example.com")],
    ),
    (
        "q-010",
        "multi_hop",
        "年假申请由谁审批？报销超过 5000 元又由谁签字？",
        "年假由直属主管审批；超过 5000 元的报销需部门负责人签字。",
        [
            _span(PDF, "由直属主管审批"),
            _span(PDF, "需部门负责人签字确认"),
        ],
    ),
    (
        "q-011",
        "multi_hop",
        "年假天数按什么分档？报销到账需要多久？",
        "年假按入职年限分 5/10/15 天；报销审核通过后 3 个工作日内打款。",
        [
            _span(PDF, "满一年不满五年：每年 5 个工作日"),
            _span(PDF, "审核通过后 3 个工作日内打款"),
        ],
    ),
    (
        "q-012",
        "image",
        "员工手册第 1 页的图片展示了什么？",
        "一张彩色噪点图（当前素材是合成占位图，用于验证图片链路）。",
        [_image_span(PDF, 1)],
    ),
    ("q-013", "unanswerable", "公司的股票代码是多少？", "知识库中没有相关信息。", []),
    ("q-014", "unanswerable", "未休年假可以折现吗？", "知识库中没有相关信息。", []),
    (
        "q-015",
        "single_hop",
        "这个服务支持哪些部署方式？",
        "支持 Docker Compose 单机部署，也支持本机直接运行 uvicorn。",
        [_span(DOCX, "支持 Docker Compose 单机部署，也支持本机直接运行 uvicorn。")],
    ),
    (
        "q-016",
        "single_hop",
        "生产环境对三个存储有什么建议？",
        "建议独立部署并配置备份。",
        [_span(DOCX, "生产环境建议将 PostgreSQL、Milvus 与 Neo4j 独立部署并配置备份。")],
    ),
    (
        "q-017",
        "term",
        "PDF 格式支持内嵌图片吗？",
        "支持，PDF 保留页码与坐标并支持内嵌图片。",
        [_span(DOCX, "PDF | 保留页码与坐标，支持内嵌图片")],
    ),
    (
        "q-018",
        "term",
        "DOCX 的引用使用什么定位？",
        "保留段落顺序与表格，引用使用段落号。",
        [_span(DOCX, "DOCX | 保留段落顺序与表格，引用使用段落号")],
    ),
    (
        "q-019",
        "single_hop",
        "上传后一直显示 pending 应该检查什么？",
        "检查 worker 是否启动、模型 API Key 是否配置正确。",
        [_span(DOCX, "答：检查 worker 是否启动，以及模型 API Key 是否配置正确。")],
    ),
    (
        "q-020",
        "single_hop",
        "为什么扫描件无法检索？",
        "当前版本不做 OCR，扫描件没有可提取文本，会标记为失败。",
        [_span(DOCX, "答：当前版本不做 OCR，扫描件没有可提取文本，会标记为失败。")],
    ),
    (
        "q-021",
        "image",
        "产品手册中的图片块对应什么内容？",
        "一张合成占位图，用于验证 docx 图片抽取。",
        [_image_span(DOCX, None)],
    ),
    (
        "q-022",
        "multi_hop",
        "部署方式与支持的文档格式分别是什么？",
        "Docker Compose 或本机 uvicorn；支持 PDF 与 DOCX。",
        [
            _span(DOCX, "支持 Docker Compose 单机部署，也支持本机直接运行 uvicorn。"),
            _span(DOCX, "格式 | 说明"),
        ],
    ),
    (
        "q-023",
        "multi_hop",
        "年假申请需要提前几天？上传后 pending 又该如何排查？",
        "提前 3 个工作日；pending 时检查 worker 与模型 Key。",
        [
            _span(PDF, "申请年假需提前 3 个工作日在人事系统提交"),
            _span(DOCX, "答：检查 worker 是否启动，以及模型 API Key 是否配置正确。"),
        ],
    ),
    ("q-024", "unanswerable", "这个产品卖多少钱？", "知识库中没有相关信息。", []),
    ("q-025", "unanswerable", "是否提供手机 App？", "知识库中没有相关信息。", []),
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for case_id, qtype, question, reference, gold in CASES:
            record = {
                "id": case_id,
                "question": question,
                "type": qtype,
                "gold": gold,
                "reference_answer": reference,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    types = sorted({case[1] for case in CASES})
    print(f"wrote {len(CASES)} cases to {OUTPUT}; types={types}")


if __name__ == "__main__":
    main()
