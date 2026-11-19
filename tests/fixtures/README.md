# 测试夹具

`documents/` 下的二进制文件由 `scripts/make_fixtures.py` 生成，固定随机种子，内容稳定：

| 文件 | 用途 |
|---|---|
| `sample.pdf` | 两页、标题层级、正文、一张 200×200 图片 |
| `sample.docx` | 标题、正文、图片、表格、页眉（用于验证页眉被排除） |
| `scanned.pdf` | 只有图形、无可提取文本，用于验证 `NoExtractableTextError` |

`demo/` 下的文件用于 `scripts/load_demo.py` 的一键演示：

| 文件 | 内容 | 可问的问题 |
|---|---|---|
| `demo/employee_handbook.pdf` | 中文《员工手册》：年假政策、报销流程，含一张图片 | 年假有几天？怎么申请？报销多久打款？ |
| `demo/product_guide.docx` | 中文《产品手册》：部署方式、支持格式表格、常见问题，含图片 | 支持哪些文档格式？扫描件为什么检索不到？ |

重新生成：

```bash
uv run python scripts/make_fixtures.py
```

不要手工编辑这些文件；需要改内容时改生成脚本并重新生成。
