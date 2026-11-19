# AGENTS.md — 给 AI 编程 Agent 的项目约束

> 本文件是**强制契约**。任何修改本仓库的 Agent（Claude Code / Codex / 其他）在动手前
> 必须先读完本文件与 `docs/实施计划.md`。与本文件冲突的实现一律视为缺陷。

## 0. 项目是什么

通用 RAG 知识库服务：创建知识库 → 上传 PDF/docx（含内嵌图片）→ 解析分块 → 向量/关键词/图谱
三路召回 → 基于证据回答。以 Server 形式运行，对外同时提供 REST、Web 管理页、MCP Server
与配套 Skill。

**当前阶段**：M0–M4 已完成（骨架、知识库/文档、解析分块、三路召回、回答与引用）；M5（MCP/Skill/
演示）进行中，MCP 三工具已可用。里程碑与验收标准见 `docs/实施计划.md` §7，任务状态见
`docs/项目任务清单.md`。

## 1. 必读顺序

1. `docs/实施计划.md` — 选型、数据模型、分块与检索策略、里程碑
2. `docs/architecture.md` — 实际架构与数据流（随代码更新，M6 成型）
3. `docs/design.md` — 设计决策与取舍、已知限制
4. 本文件 — 约束与命令

## 2. 目录结构与模块边界

```
src/rag_kb/
├── main.py          应用装配：读配置、lifespan 建连接池、注入端口、挂载路由、拉起 worker
├── config.py        唯一的环境变量读取点（pydantic-settings）
├── cli.py           rag-kb migrate / demo / reindex，同一用例的 CLI 包装
├── domain/          纯业务模型与端口协议。不依赖任何具体存储、不 import infra
│   ├── models.py    KB / Document / Chunk / SearchHit / Citation 等
│   ├── ports.py     MetadataRepository / VectorStore / LexicalIndex / GraphStore /
│   │                Embedder / LLM / VLM / Reranker / Parser / Storage
│   └── errors.py    领域异常，供上层映射为 HTTP 状态码
├── application/     用例编排（事务边界在这里），只依赖 domain 端口
├── infra/           端口的具体实现：postgres / milvus / neo4j / parsers / llm / storage
├── api/             HTTP 适配层：路由 + 请求/响应 schema。禁止写检索、分块等业务逻辑
├── web/             Jinja2 模板与静态资源（HTMX 本地文件，不引 CDN）
└── worker/          摄取任务循环（DB 任务表 + asyncio，随 lifespan 启动）
mcp_server/          MCP stdio server，仅做 HTTP 转发；启动 `uv run --extra mcp rag-kb-mcp`
skills/rag-kb/       Skill 源文件；安装到 .claude/skills/rag-kb/
scripts/             一次性/运维脚本；migrate/demo 由 rag-kb CLI 承接
tests/               unit（fake 端口） / integration（真容器） / e2e
```

### 依赖方向（硬约束）

```
api / web / worker  →  application  →  domain
infra  ──实现──→  domain 里定义的端口（由 main.py 注入）
```

**禁止**：

- `domain/` 里 `import rag_kb.infra.*` 或任何第三方存储客户端
- `api/` 里直接 `import psycopg / pymilvus / neo4j` 或写 SQL / Cypher
- `application/` 里 `import fastapi`（用例应能被 CLI、worker、MCP 复用）
- `mcp_server/` 里实现检索逻辑——它只能调用 REST

自检方式：`uv run pytest tests/unit/test_layering.py`（用 AST 检查 import 图，M1 落地）。

## 3. 开发约束

1. **禁止**引入 LangChain、LangGraph、LlamaIndex 等 RAG/Agent 编排框架（题目硬要求）。
   参考 RAGFlow/WeKnora 的设计可以，但不得直接部署或包装现成 RAG 产品。
2. **不引入 ORM 与迁移框架**。SQL 手写在 `infra/postgres/sql/`，迁移脚本顺序执行并记录在
   `schema_migration` 表。理由：本项目的查询以检索为主，ORM 抽象带不来收益却会遮蔽执行计划。
3. **一切检索必须带 `kb_id` 过滤**。Milvus 用 `partition_key=kb_id` **并强制等值过滤**
   （分区键只负责路由，不带过滤仍会扫描全部分区），PG 用 `where kb_id = %s`，Neo4j 用
   `{kb_id: $kb_id}`。任何缺少 `kb_id` 约束的检索代码视为缺陷，必须有测试覆盖；集成断言
   前先 `flush()` 或使用 Strong 一致性。
4. **每个 chunk 必须可回溯，按格式能力区分**：PDF 必须带 `page_no + bbox + char_start/char_end`；
   docx 至少带 `char_start/char_end + seq`（没有页码与坐标，`page_no/bbox` 允许为 NULL）——
   不允许为了满足字段而伪造页码。图谱的每条关系边必须带 `chunk_id` 与 `evidence_span`，
   否则无法返回原文证据。
5. **注释解释"为什么"，不解释"是什么"**。重点写：设计原因、边界条件、被否方案与取舍。
   示例见 `src/rag_kb/infra/health.py` 的模块 docstring。
6. **密钥与地址只从 `Settings` 读**，不得硬编码。新增配置项必须同步更新 `config.py`、
   `.env.example`、`docker-compose.yml` 三处，缺一不可；随后跑
   `tests/unit/test_config_compose_sync.py`。注意容器不读 `.env`，compose 的
   `environment` 是唯一透传通道。
7. **删除必须级联且可验证**：删文档要清 PG 行、Milvus 向量、Neo4j 节点/边、落盘图片。
   验收口径是"删后三通道均 0 命中"，不是接口返回 200。
8. **异常不吞**。infra 层把第三方异常转成 `domain/errors.py` 里的领域异常；application 层
   决定重试或标记失败；api 层映射 HTTP 状态码。禁止把裸 `Exception` 抛到路由外。
9. Python 3.12 语法：优先 `X | None`、`match`、`from __future__ import annotations`。
10. 提交前必须通过 `ruff check` 与 `ruff format --check`。
11. **端口异步契约**：I/O 端口一律 `async def`；同步 SDK（psycopg / pymilvus / neo4j /
    本地文件）由适配器用 `anyio.to_thread.run_sync` 包装；`Parser.parse()` 是同步 CPU 方法，
    由 worker 放线程池。禁止在协程里直接调用阻塞函数。
12. **只有子块进索引**：向量库与 `chunk_lex` 只写 `level=child`；父块只用于回答阶段回填。
    检索命中与 `/search` 返回的应是子块 id，图谱证据也优先映射到子块。

## 4. 运行命令

```bash
# 首次：准备环境
cp .env.example .env      # 然后填写 LLM/EMBED/VLM 三个 API Key
docker compose up -d --build
docker compose ps         # 等健康状态变为 healthy（milvus 首次约 60-90s）
curl -s localhost:8000/readyz | python -m json.tool
```

**就绪判定标准**：`/readyz` 返回 200 且 `ready=true`。`model:*` 三项为 `required=false`，
未配置不阻塞启动，但对应功能不可用——不要把它当作"服务坏了"。

```bash
docker compose logs -f app        # 看应用日志
docker compose down               # 停止（保留数据卷）
docker compose down -v            # 停止并清空数据（重建索引时用）

# 运维命令统一走 rag-kb CLI（Makefile 只是薄封装）
docker compose exec app rag-kb migrate            # 执行迁移
uv run rag-kb demo                                # 装载演示数据（需宿主侧 tests/fixtures）
uv run rag-kb reindex --kb <id>                   # 从 chunk 表重建三路索引
```

本机不经容器直跑（需要三个存储已在 localhost 暴露）：

```bash
uv sync
# .env 里把 MILVUS_URI / NEO4J_URI / PG_DSN 的 host 改成 localhost
uv run uvicorn rag_kb.main:app --reload --port 8000
```

> 本机无 `make` 时忽略 Makefile，直接用上面的 `docker compose` 命令。

## 5. 测试命令

```bash
uv run pytest -q                          # 全部（integration 需容器已起）
uv run pytest tests/unit -q               # 仅单测，用 fake 端口，秒级
uv run pytest tests/unit/test_layering.py tests/unit/test_config_compose_sync.py -q
uv run pytest -m integration -q           # 集成测试，需 docker compose up
uv run pytest -m e2e -q                   # 端到端（含 MCP stdio）
uv run ruff check src tests mcp_server scripts
uv run ruff format --check src tests
```

**必须覆盖的边界**（题目点名，缺一即扣分）：

| 边界 | 断言口径 |
|---|---|
| 知识库隔离 | A 库文档在 B 库三通道检索均 0 命中 |
| 处理失败 | status=failed、error 可读、不产生脏索引、可重试 |
| 删除后不可召回 | 删除后三通道 0 命中 + 底层计数已清零 |

单测中出现数据库连接即错误（除非标了 `@pytest.mark.integration`）。

## 6. Agent 工作方式

1. **先读后写**：改任何模块前，先读该模块与 `domain/ports.py`，确认接口契约；不确定时先提问，
   不要靠猜测补全接口。
2. **按里程碑推进**：一次只做 `docs/实施计划.md` §7 中一个阶段的事，不要跨阶段顺手改。
3. **小步提交**：一个 commit 一件事，message 用 `feat|fix|refactor|test|docs: 说明`；
   文档改动同样一任务一 commit。
4. **改接口要同步文档**：修改端口签名、数据表结构、API 契约，必须同一 commit 内更新
   `docs/architecture.md`、`docs/design.md`，并在 `docs/技术选型文档.md` 追加或修订 ADR，
   否则视为未完成。
5. **不要擅自扩大范围**：不给自己加题目没要求的功能；加分项只在 M7 做，且标注"待实现"即可。
6. **交付前自检**：① `ruff check` 通过 ② 相关测试通过 ③ 新增配置三处同步 ④ 检索代码带
   `kb_id` ⑤ 注释解释的是原因而非复述代码。
7. **不得提交**：`.env`、`data/` 下的运行期文件、任何真实密钥。
8. **性能与成本意识**：ingest 期的 LLM 调用要并发受控（`llm_max_concurrency` /
   `embed_max_concurrency` / `vlm_max_concurrency`）；检索与融合默认零 LLM 调用，图谱抽取
   只在入库时做一次。`rerank=llm` 必须由调用方显式传参，不属于默认检索路径。

## 7. 常见坑（踩过就写进来）

- Milvus 首次启动要 60-90s，compose healthcheck 的 `start_period` 已设为 90s；`depends_on`
  用的是 `service_healthy`，不要在应用侧再加无意义的 sleep 重试。
- **quay.io 在国内网络下会在下载中途 TLS 断连**（实测拉 etcd v3.5.16 连续失败，先下完几个层
  再 EOF）。因此 compose 固定 `etcd:v3.5.5`（本地已有、3.5 线等价）。调整任何镜像 tag 前，
  先单独 `docker pull` 验证可达，不要直接 `compose up` 才发现。
- 拉大镜像不要用 `timeout N docker pull` 之类的外层超时：被掐断的层会残留成半成品，
  下次 `compose up` 仍会重新拉。让它在后台跑完即可。
- 中文词法召回不是"装个分词器就能用"：本方案入库前 jieba 分词、PG 侧用 `simple` 词典建
  `tsvector`，查询侧同样分词后拼 `to_tsquery`。不要改用 `english` 配置。
- `EMBEDDING_DIM` 与 embedding 模型输出维度不一致会导致"能启动但检索恒空"，启动期已做
  fail-fast 校验，换模型必须同时重建 collection；collection 不存在时应创建而不是报错。
- **Milvus 写入/删除是最终一致的**：集成测试断言"删除后 0 命中"之前先 `flush()` 或使用
  Strong 一致性，否则会偶发失败。
- **`to_tsquery` 对特殊字符敏感**：查询侧 jieba 结果必须清洗标点与 `& | ! ( ) : * '`
  后再拼接，并应用停用词表；空 token 直接返回空结果，不要拼出非法 tsquery。
- **docx 没有页码与 bbox**：不要为它伪造 `page_no`；用 `char_start/char_end + seq` 定位，
  引用展示段落号。PDF 才要求 `page_no + bbox`。
- **pymilvus 2.6.18 ↔ Milvus 2.5.4 只验证了连通性**：M3-01 必须做一次
  `insert → search → delete` 冒烟；不兼容时优先升服务端到 2.6.x（TD-08）。
- Windows 上 `Path("/data")` 会解析到当前盘符根目录，本机直跑用相对路径 `./data`，容器内
  由 compose 覆盖为 `/data`。
- 本机若已有 PostgreSQL 服务占用 5432，Docker 的端口映射会"显示成功但 localhost 实际连到
  本地实例"（表现为 role `rag` 不存在）。在 `.env` 设 `PG_PORT=5433` 并同步 `PG_DSN`，
  然后 `docker compose up -d postgres` 重建端口映射；容器内部仍用 `postgres:5432`。
