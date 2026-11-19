# 通用 RAG 知识库服务（DEMO）

创建知识库 → 上传 PDF/docx → 解析分块 → 向量/关键词/图谱三路召回 → 基于证据回答。
以 Server 形式运行，同时提供 REST、Web 管理页、MCP Server 与 Claude Code Skill。

> **当前进度**：M0–M4 已完成（骨架、知识库/文档、解析分块、三路召回、回答与引用）。
> MCP 与 Skill 在 M5 实现；本文档会随里程碑更新。

## 当前已实现

- Web 管理页：建库/删库、上传/查看状态/删除文档；
- REST：知识库 CRUD、文档上传/列表/详情/删除、检索、问答；
- 上传校验：扩展名 + magic bytes、大小限制、同库 sha256 去重（409）；
- 摄取 worker：解析 PDF/docx → 标题树/父子分块 → 图片描述与落盘 → 三路索引，状态推进到 `ready`；
- 检索：向量（Milvus）/ 关键词（PG tsvector）/ 图谱（Neo4j）+ RRF 融合，可选重排；
- 回答：证据拼装、父块回填、引用编号回校验、无证据拒答；
- 文档状态机：`pending → parsing → chunked → indexing → ready`，失败可重试；
- 删除级联：三路索引、DB 行、分块、任务与磁盘文件一起清理；
- 健康检查：`/healthz`、`/readyz`（Postgres / Milvus / Neo4j + 模型配置）。

> MCP Server 与 Skill 尚未实现（M5）；REST 与 Web 已可完成检索与问答。

## 快速开始

前置条件：

- Docker Desktop（建议分配 ≥ 8 GB 内存）；
- Python 3.12 + [uv](https://docs.astral.sh/uv/)；
- 模型 Key：DeepSeek（生成）、Qwen MaaS（Embedding + VLM），在 `.env` 中填写。

```bash
# 1. 准备配置
cp .env.example .env
# 填写 LLM_API_KEY / EMBED_API_KEY / VLM_API_KEY

# 2. 启动依赖与应用
docker compose up -d --build
docker compose ps          # 等全部 healthy（Milvus 首次约 60-90s）

# 3. 初始化数据库（首次或迁移后）
uv run rag-kb migrate

# 4. 检查就绪
curl -s localhost:8000/readyz | python -m json.tool

# 5.（可选）装载演示知识库与固定夹具
uv run rag-kb demo
```

打开：

- Web 管理页：<http://localhost:8000/web/kbs>
- OpenAPI 文档：<http://localhost:8000/docs>

### 本机 5432 已被占用

若宿主机已有 PostgreSQL 服务占用 5432，Docker 映射会"显示成功但 localhost 连到本地实例"
（表现为 role `rag` 不存在）。在 `.env` 中设置：

```dotenv
PG_PORT=5433
PG_DSN=postgresql://rag:rag_local_pw@localhost:5433/ragkb
```

然后执行 `docker compose up -d postgres` 重建端口映射；容器内部仍使用 `postgres:5432`。

## 配置

所有配置集中在 `src/rag_kb/config.py`，并由 `.env` / `docker-compose.yml` 透传。
容器不直接读取 `.env`，compose 的 `environment` 是唯一通道；`tests/unit/test_config_compose_sync.py`
会校验三处一致。

| 变量 | 说明 |
|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | 生成与图谱抽取，默认 DeepSeek `deepseek-flash` |
| `EMBED_BASE_URL` / `EMBED_MODEL` / `EMBED_API_KEY` | 向量化，默认 Qwen MaaS `qwen3.7-text-embedding-flash`（1024 维） |
| `VLM_BASE_URL` / `VLM_MODEL` / `VLM_API_KEY` | 图片描述，默认 Qwen MaaS `qwen3-vl-plus` |
| `EMBEDDING_DIM` | 必须与 embedding 模型输出一致，默认 `1024` |
| `PG_DSN` / `MILVUS_URI` / `NEO4J_URI` | 三个存储的连接串 |
| `RERANK_ENABLED` | 重排默认关闭；`RERANK_*` 端点为占位 |
| `MAX_UPLOAD_MB` | 单文件大小上限，默认 `50` |

## REST 示例

```bash
# 建库
curl -s -X POST localhost:8000/api/kbs \
  -H 'Content-Type: application/json' \
  -d '{"name":"产品手册","description":"demo"}'

# 上传 PDF
curl -s -X POST localhost:8000/api/kbs/<kb_id>/documents \
  -F 'file=@./demo.pdf;type=application/pdf'

# 列表 / 详情 / 删除文档
curl -s localhost:8000/api/kbs/<kb_id>/documents
curl -s localhost:8000/api/kbs/<kb_id>/documents/<doc_id>
curl -s -X DELETE localhost:8000/api/kbs/<kb_id>/documents/<doc_id>

# 删除知识库
curl -s -X DELETE localhost:8000/api/kbs/<kb_id>

# 检索：vector / keyword / graph / hybrid / hybrid_weighted
curl -s -X POST localhost:8000/api/kbs/<kb_id>/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"年假政策","mode":"hybrid","top_k":10}'

# 基于证据回答（返回引用）
curl -s -X POST localhost:8000/api/kbs/<kb_id>/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"年假怎么申请？","mode":"hybrid"}'

# 重建三路索引
curl -s -X POST localhost:8000/api/kbs/<kb_id>/reindex
```

上传重复文件返回 `409` 与 `existing_document_id`；假 docx/伪造扩展名返回 `415`；
超限返回 `413`。

## 开发

```bash
uv sync
uv run pytest tests/unit -q               # 纯单测，无需容器
uv run pytest -m integration -q           # 集成测试，需容器已启动
uv run ruff check src tests mcp_server scripts
uv run ruff format --check src tests
```

数据库迁移与演示：

```bash
uv run rag-kb migrate                     # 宿主直跑，使用 .env 的 PG_DSN
uv run rag-kb demo                        # 通过 REST 建演示库并上传固定夹具
uv run rag-kb reindex --kb <kb_id>        # 从 chunk 表重建三路索引
```

## 目录导览

```
src/rag_kb/
├── domain/       纯模型与端口协议（不依赖任何存储）
├── application/  用例编排：知识库、文档、上传校验
├── infra/        Postgres / Milvus / Neo4j / 解析器 / 模型客户端 / 存储
├── api/          REST 路由、schema、异常映射
├── web/          Jinja2 + 本地 HTMX 管理页
└── main.py       装配：连接池、服务注入、路由挂载
docs/             题目、实施计划、技术选型（ADR）、任务清单、目录说明
tests/            unit（fake 端口）/ integration（真容器）/ e2e
```

更详细的模块边界见 [`AGENTS.md`](AGENTS.md) 与 [`docs/项目目录结构说明.md`](docs/项目目录结构说明.md)。

## 已知限制

- 不支持扫描件 OCR、手写内容与旧版 `.doc`；
- 复杂表格、公式与版式不保证还原；
- 当前无鉴权、无流式输出，定位为本机/内网 DEMO；
- docx 无页码与 bbox，引用将使用段落号而非页码；
- MCP Server 与 Skill（M5）尚未实现。

## 文档

- [实施计划](docs/实施计划.md)
- [技术选型（ADR）](docs/技术选型文档.md)
- [项目任务清单](docs/项目任务清单.md)
- [目录结构说明](docs/项目目录结构说明.md)
