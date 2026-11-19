# 通用 RAG 知识库服务（DEMO）

创建知识库 → 上传 PDF/docx（含内嵌图片）→ 解析分块 → 向量/关键词/图谱三路召回 →
基于证据回答。以 Server 形式运行，同时提供 REST、Web 管理页、MCP Server 与 Claude Code Skill。

> **进度**：M0–M7 全部完成（含 MCP/Skill/演示、测试与文档交付、检索评测）。
> 里程碑与验收标准见 [实施计划](docs/实施计划.md)，任务状态见 [项目任务清单](docs/项目任务清单.md)。

## 能做什么

- **知识库与文档**：建库/删库、上传/查看/重试/删除 PDF 与 docx，同库 sha256 去重；
- **摄取**：标题树 + 父子分块、图片 VLM 描述、三路索引，状态 `pending → … → ready`；
- **检索**：`vector` / `keyword` / `graph` / `hybrid`（RRF）/ `hybrid_weighted`，可选重排；
- **回答**：证据拼装、父块回填、引用编号回校验、无证据拒答；
- **图谱**：实体/关系抽取、实体链接、多跳展开，每条边带 `chunk_id` 与证据区间；
- **接入**：REST、Web、MCP 三工具（只转发 REST）、Skill；
- **边界**：所有检索强制 `kb_id`；删除级联清理三路索引、DB 行与磁盘文件；失败可重试。

## 5 分钟跑通

前置：Docker Desktop（建议 ≥ 8 GB 内存）、Python 3.12 + [uv](https://docs.astral.sh/uv/)、
DeepSeek 与 Qwen MaaS 的 API Key。

```bash
# 1) 配置（约 1 分钟）
cp .env.example .env
#    填写 LLM_API_KEY / EMBED_API_KEY / VLM_API_KEY

# 2) 启动（首次构建 + Milvus 就绪约 2–3 分钟）
docker compose up -d --build
docker compose ps                    # 等到 app/postgres/milvus/neo4j 都 healthy

# 3) 迁移与自检（约 10 秒）
uv run rag-kb migrate
curl -s localhost:8000/readyz | python -m json.tool   # ready=true

# 4) 装载演示知识库（约 1 分钟，含模型调用）
uv run python scripts/load_demo.py
```

打开：

- Web 管理页 <http://localhost:8000/web/kbs>
- OpenAPI <http://localhost:8000/docs>

`/readyz` 的 `model:*` 三项为 `required=false`：未配置 Key 不阻塞启动，但检索/问答不可用，
不要把它当成"服务坏了"。

## 演示步骤

完整脚本与预期输出见 **[docs/demo.md](docs/demo.md)**。最短路径：

1. `load_demo.py` 会建库并上传《员工手册》PDF 与《产品手册》docx，等到两个文档 `ready`；
2. Web 或 `POST /api/kbs/<id>/ask` 问「年假有几天？怎么申请？」→ 返回答案与 `[1]` 引用；
3. 问「公司的股票代码是多少？」→ `no_evidence=true`，明确拒答；
4. 继续问「支持哪些文档格式？」→ 引用来自 docx，显示段落号与标题路径而不是页码。

## MCP 与 Skill

```bash
uv sync --extra mcp
claude mcp add -s user --transport stdio \
  rag-kb -e RAG_KB_BASE_URL=http://localhost:8000 \
  -- uv run --extra mcp rag-kb-mcp
claude mcp list        # rag-kb: ✓ Connected
```

- 三个工具：`list_knowledge_bases`、`search_knowledge_base`、`ask_knowledge_base`；
- Skill 源文件 `skills/rag-kb/SKILL.md`；安装副本已放在 `.claude/skills/rag-kb/`，
  在仓库内打开 Claude Code 即可生效；复制到其他项目见安装说明；
- 桌面端 JSON 配置、Windows 下 `uv` 绝对路径、故障排查见 **[docs/mcp-setup.md](docs/mcp-setup.md)**。

MCP 只转发 REST，不连接数据库，配置里不含模型 Key（Key 属于服务端 `.env`）。

## 配置

服务端配置集中在 `src/rag_kb/config.py`，由 `.env` / `docker-compose.yml` 透传。
容器不直接读取 `.env`，compose 的 `environment` 是唯一通道；
`tests/unit/test_config_compose_sync.py` 校验三处一致。

| 变量 | 说明 |
|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | 生成与图谱抽取，默认 DeepSeek `deepseek-flash` |
| `EMBED_BASE_URL` / `EMBED_MODEL` / `EMBED_API_KEY` | 向量化，默认 Qwen MaaS `qwen3.7-text-embedding-flash`（1024 维） |
| `VLM_BASE_URL` / `VLM_MODEL` / `VLM_API_KEY` | 图片描述，默认 Qwen MaaS `qwen3-vl-plus` |
| `EMBEDDING_DIM` | 必须与模型输出一致，默认 `1024`；换模型需重建 collection |
| `PG_DSN` / `MILVUS_URI` / `NEO4J_URI` | 三个存储的连接串 |
| `CHUNK_TARGET_TOKENS` / `PARENT_MAX_TOKENS` | 子块目标 320 / 父块上限 1200 |
| `RERANK_ENABLED` | 默认关闭；开启需同时配置 `RERANK_*` |
| `MAX_UPLOAD_MB` | 单文件上限，默认 50 |

MCP 进程只认 `RAG_KB_BASE_URL`（默认 `http://localhost:8000`）与 `RAG_KB_TIMEOUT`。

## REST 示例

```bash
# 建库
curl -s -X POST localhost:8000/api/kbs \
  -H 'Content-Type: application/json' \
  -d '{"name":"产品手册","description":"demo"}'

# 上传 PDF
curl -s -X POST localhost:8000/api/kbs/<kb_id>/documents \
  -F 'file=@./demo.pdf;type=application/pdf'

# 列表 / 详情 / 分块 / 重试 / 删除
curl -s localhost:8000/api/kbs/<kb_id>/documents
curl -s localhost:8000/api/kbs/<kb_id>/documents/<doc_id>
curl -s localhost:8000/api/kbs/<kb_id>/documents/<doc_id>/chunks
curl -s -X POST localhost:8000/api/kbs/<kb_id>/documents/<doc_id>/retry
curl -s -X DELETE localhost:8000/api/kbs/<kb_id>/documents/<doc_id>

# 检索与问答
curl -s -X POST localhost:8000/api/kbs/<kb_id>/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"年假政策","mode":"hybrid","top_k":10}'
curl -s -X POST localhost:8000/api/kbs/<kb_id>/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"年假怎么申请？","mode":"hybrid"}'

# 重建索引（异步任务）
curl -s -X POST localhost:8000/api/kbs/<kb_id>/reindex
```

错误口径：重复文件 `409`（附 `existing_document_id`）；伪造扩展名/假 docx `415`；
超限 `413`；解析或索引失败为 `status=failed`，`error` 可读，可重试。

> Windows Git Bash 下命令行中文 JSON 可能被代码页破坏，改用 `-d @body.json`（UTF-8 文件）
> 或 Python/httpx 发请求。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `docker compose ps` 里 milvus 一直 starting | 首次启动约 60–90s；`start_period` 已设 90s，再等即可 |
| 上传后一直 `pending` | worker 未启动或模型 Key 不对；`docker compose logs -f app` |
| 文档 `failed`，error 提到 OCR | 扫描件没有可提取文本，本版本不做 OCR；换文本型 PDF |
| `role "rag" does not exist` | 本机 5432 被本地 PostgreSQL 占用；设置 `PG_PORT=5433` 并同步 `PG_DSN` |
| 检索恒为空 | 检查 `EMBEDDING_DIM` 与模型一致；换模型需 `docker compose down -v` 重建 collection |
| MCP 显示 Connected 但无工具 | 没执行 `uv sync --extra mcp`，或启动目录不是仓库根 |
| `/readyz` 中 `model:llm` 为 false | 正常：未配置 Key 不阻塞启动，但问答不可用 |
| 中文 JSON curl 报解析错误 | Windows 代码页问题；用 UTF-8 文件或 httpx |
| 修改 Skill 不生效 | 重新复制到 `.claude/skills/rag-kb/` 并重开会话 |

## 目录导览

```
src/rag_kb/
├── domain/       纯模型、端口协议、领域异常（不依赖任何存储）
├── application/  用例：知识库、文档、分块、索引、检索、融合、问答、reindex
├── infra/        PostgreSQL / Milvus / Neo4j / 解析器 / 模型客户端 / 本地存储
├── api/          REST 路由、schema、异常映射
├── web/          Jinja2 + 本地 HTMX 管理页
└── main.py       装配：连接池、服务注入、路由挂载、worker
mcp_server/       MCP stdio server（只转发 REST，不 import rag_kb）
skills/rag-kb/    Skill 源文件；.claude/skills/rag-kb/ 为安装副本
scripts/          load_demo.py、make_fixtures.py、bench/（M7）
docs/             计划、ADR、架构、设计、清单、MCP、演示
tests/            unit（fake 端口）/ integration（真容器）/ e2e（MCP stdio）
```

模块边界与依赖方向见 [`AGENTS.md`](AGENTS.md) 与
[架构说明](docs/architecture.md)；目录判定流程见 [目录结构说明](docs/项目目录结构说明.md)。

## 开发与测试

```bash
uv sync
uv run pytest tests/unit -q               # 纯单测，无需容器
uv run pytest -m integration -q           # 集成测试，需容器已启动
uv sync --extra mcp
uv run pytest -m e2e -q                   # 端到端（MCP stdio），需服务 + 模型 Key
uv run ruff check src tests mcp_server scripts
uv run ruff format --check src tests
```

运维命令：

```bash
uv run rag-kb migrate                     # 执行迁移
uv run rag-kb demo                        # 等价于 scripts/load_demo.py
uv run rag-kb reindex --kb <kb_id>        # 从 chunk 表重建三路索引
docker compose down                       # 停止（保留数据）
docker compose down -v                    # 清空数据卷（重建索引时用）
```

## 已知限制

- 不支持扫描件 OCR、手写内容与旧版 `.doc`；
- 复杂表格、公式、多栏版式不保证还原；
- 无鉴权、无流式输出，定位为本机/内网 DEMO；
- docx 无页码与 bbox，引用使用段落号 + 标题路径，不伪造页码；
- 重排默认关闭；默认检索路径不调用 LLM。

更完整的取舍与扩展方向见 [设计文档](docs/design.md)。

## 文档

- [实施计划](docs/实施计划.md)
- [技术选型（ADR）](docs/技术选型文档.md)
- [架构与数据流](docs/architecture.md)
- [详细设计与取舍](docs/design.md)
- [项目任务清单](docs/项目任务清单.md)
- [目录结构说明](docs/项目目录结构说明.md)
- [MCP 接入](docs/mcp-setup.md)
- [演示脚本](docs/demo.md)
- [检索评测报告](docs/benchmark.md)
- [待实现方案](docs/design-future.md)
