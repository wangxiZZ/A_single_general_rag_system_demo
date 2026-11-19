# 测试夹具

`documents/` 下的二进制文件由 `scripts/make_fixtures.py` 生成，固定随机种子，内容稳定：

| 文件 | 用途 |
|---|---|
| `sample.pdf` | 两页、标题层级、正文、一张 200×200 图片 |
| `sample.docx` | 标题、正文、图片、表格、页眉（用于验证页眉被排除） |
| `scanned.pdf` | 只有图形、无可提取文本，用于验证 `NoExtractableTextError` |

重新生成：

```bash
uv run python scripts/make_fixtures.py
```

不要手工编辑这些文件；需要改内容时改生成脚本并重新生成。
