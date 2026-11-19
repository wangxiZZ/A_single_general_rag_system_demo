# 演示脚本

演示目标：一条命令装载知识库，然后在 **Web / REST / Claude Code（MCP + Skill）** 三个入口
完成「列库 → 检索 → 带引用作答 → 无证据拒答」。文档中的输出是实际跑通的结果，模型措辞可能
略有差异，但引用编号、页码/段落号与拒答行为应保持一致。

## 0. 前置与装载

```bash
docker compose up -d --build
docker compose ps                 # app / postgres / milvus / neo4j 均为 healthy
uv run rag-kb migrate             # 首次
uv run python scripts/load_demo.py
```

预期输出（耗时取决于模型，通常 30–90 秒）：

```
知识库: c098d7ab-...
employee_handbook.pdf: pending
employee_handbook.pdf: parsing
employee_handbook.pdf: indexing
employee_handbook.pdf: ready, chunks=9
product_guide.docx: pending
product_guide.docx: parsing
product_guide.docx: indexing
product_guide.docx: ready, chunks=9
完成。打开 http://localhost:8000/web/kbs/c098d7ab-...
可以试问：年假有几天？怎么申请？ / 报销多久打款？ / 支持哪些文档格式？
```

脚本只在 **`ready`** 后返回；停在 `chunked` 说明索引还没写完，此时检索会 0 命中。
失败时脚本打印服务端 `error`（如「PDF 无可提取文本」）并以状态码 1 退出。

素材说明：

| 文档 | 关键内容 |
|---|---|
| `tests/fixtures/demo/employee_handbook.pdf` | 年假天数与申请流程、报销时限与审批，含一张图片 |
| `tests/fixtures/demo/product_guide.docx` | 部署方式、支持格式表格、FAQ，含图片；无页码，引用用段落号 |

## 1. Web 管理页

打开 <http://localhost:8000/web/kbs>：

1. 进入「演示知识库」→ 文档列表应显示两个 `ready` 文档与 chunk 数；
2. 点「提问」，输入「年假有几天？怎么申请？」；
3. 页面显示答案与来源卡片，来源应指向 `employee_handbook.pdf` 第 1 页。

## 2. Claude Code（MCP + Skill）

按 [`mcp-setup.md`](mcp-setup.md) 注册 `rag-kb` 并安装 Skill。新会话中输入：

> 帮我查一下知识库里年假有几天，怎么申请。

预期行为（工具调用顺序）：

1. `mcp__rag-kb__list_knowledge_bases` → 选中唯一的演示库；
2. `mcp__rag-kb__ask_knowledge_base`，参数 `kb_id=<演示库 id>`、`question="年假有几天？怎么申请？"`；
3. 回答形如：

   ```
   年假天数按入职年限确定：满一年不满五年每年 5 个工作日，满五年不满十年每年 10 个工作日，
   满十年及以上每年 15 个工作日 [1]。申请需提前 3 个工作日在人事系统提交，并由直属主管审批 [1]。

   来源：
   [1] 《employee_handbook.pdf》第 1 页
   ```

再问一个库里没有的问题，验证拒答：

> 公司的股票代码是多少？

预期：`no_evidence=true`，回答「知识库中未找到足够依据」，**不会**编造股票代码。
Skill 要求此时最多换关键词重试 2 次，仍无结果就明确告知用户。

再验证 docx 的引用定位：

> 支持哪些文档格式？

预期引用来自 `product_guide.docx`，没有页码，显示为「第 N 段 / 2. 支持的文档格式」，
而不是伪造页码。

## 3. REST

检索（`mode` 可换 `vector` / `keyword` / `graph` / `hybrid_weighted`）：

```bash
curl -s -X POST localhost:8000/api/kbs/<kb_id>/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"报销多久打款","mode":"hybrid","top_k":3}'
```

预期命中 `employee_handbook.pdf` 第 2 页，`snippet` 含「5 个工作日内完成审核」。

问答：

```bash
curl -s -X POST localhost:8000/api/kbs/<kb_id>/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"报销多久打款？","mode":"hybrid"}'
```

实际返回：

```json
{
  "answer": "财务在收到完整材料后 5 个工作日内完成审核 [1]；审核通过后 3 个工作日内打款 [1]。",
  "citations": [
    {"index": 1, "doc_name": "employee_handbook.pdf", "page_no": 2, "seq": 4}
  ],
  "no_evidence": false
}
```

> Windows Git Bash 下直接在命令行写中文 JSON 可能被代码页破坏（服务端报
> `There was an error parsing the body`）。把请求体存成 UTF-8 文件后用
> `-d @body.json`，或改用 Python/httpx 发请求。

## 4. 无证据与重试

```bash
curl -s -X POST localhost:8000/api/kbs/<kb_id>/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"公司的股票代码是多少？","mode":"hybrid"}'
```

预期：

```json
{
  "answer": "知识库中未找到足够依据。",
  "citations": [],
  "no_evidence": true,
  "attempted_channels": ["vector", "keyword", "graph"]
}
```

此时重试应改变查询表达：把「股票代码」换成「证券代码 / 上市」；仍为空则向用户说明
「知识库中没有相关内容」，不要用通用知识作答后贴来源。

## 5. 边界验证（可选）

```bash
# 知识库隔离：B 库检索 A 库内容应 0 命中
curl -s -X POST localhost:8000/api/kbs/<other_kb_id>/search \
  -H 'Content-Type: application/json' -d '{"query":"年假","mode":"hybrid"}'
# 预期 []

# 删除后不可召回：删除文档，再查三通道
curl -s -X DELETE localhost:8000/api/kbs/<kb_id>/documents/<doc_id>
curl -s -X POST localhost:8000/api/kbs/<kb_id>/search \
  -H 'Content-Type: application/json' -d '{"query":"年假","mode":"vector"}'
# 预期 []
```

## 6. 清理

```bash
# 删除演示知识库（会级联清理三路索引与落盘文件）
curl -s -X DELETE localhost:8000/api/kbs/<kb_id>

# 需要完全重置时
docker compose down -v
```
