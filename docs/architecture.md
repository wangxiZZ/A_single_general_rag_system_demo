# 架构与数据流

> 本文描述**当前代码实际如何工作**。设计取舍与备选方案见 [`design.md`](design.md)，
> 选型背景见 [`技术选型文档.md`](技术选型文档.md)。实现与本文冲突时，以代码为准并立即修本文。

## 1. 总览

单个 FastAPI 进程同时承担：

- REST `/api/*` 与 Web `/web/*`；
- 摄取 worker（asyncio 后台任务，随 lifespan 启动）；
- 检索、融合、回答等用例编排。

MCP Server 是**独立进程**（`uv run --extra mcp rag-kb-mcp`），只通过 HTTP 调用本服务；
Claude Code Skill 只负责提示模型怎么使用这些工具。三者关系：

```
Claude Code ── Skill（规则）──► MCP stdio 进程 ──HTTP──► RAG 服务 ──► PG / Milvus / Neo4j
                                                    ▲
Web 管理页 / REST 客户端 ────────────────────────────┘
```

**没有引入 LangChain/LangGraph/LlamaIndex**：整条链路只有「解析 → 分块 → 三路索引 →
召回 → 证据拼装 → LLM 生成」这些明确的步骤，自己编排比引入框架更容易控制
事务、失败清理与 `kb_id` 隔离。

## 2. 分层与依赖方向

```
api/      web/      worker/            入口层：HTTP 参数、页面、任务循环
                 │
                 ▼
            application/               用例编排；事务边界、清理与重试决策
                 │        ▲
                 ▼        │ 实现
             domain/ ◄────┴────────── infra/   适配器：PG / Milvus / Neo4j / 解析 / 模型
                              （由 main.py lifespan 注入）
```

| 层 | 允许依赖 | 典型内容 |
|---|---|---|
| `domain` | 仅标准库 | `models.py`、`ports.py`、`errors.py` |
| `application` | `domain` | 分块、索引、检索、融合、问答、删除级联 |
| `infra` | `domain` + 第三方库 | 三个存储适配器、解析器、OpenAI 兼容客户端、本地存储 |
| `api` / `web` / `worker` | `application`、`domain` | 路由、模板、worker 循环 |
| `mcp_server` | 仅 `httpx` + MCP SDK | 三工具转发；**不得 import `rag_kb`** |

约束由 `tests/unit/test_layering.py` 用 AST 扫描 import 图强制。

## 3. 端口契约

端口定义在 `src/rag_kb/domain/ports.py`，全部 `async def`；同步 SDK（psycopg / pymilvus /
neo4j / 本地文件）由适配器用 `anyio.to_thread.run_sync` 包装。`Parser.parse()` 例外：
它是同步 CPU 方法，由 worker 放进线程池。

| 端口 | 关键方法 | 实现 |
|---|---|---|
| `MetadataRepository` | KB/Document/Chunk/Image/Job/QueryLog 的 CRUD；`claim_ingest_job` | `infra/postgres/repository.py` |
| `VectorStore` | `ensure_collection`、`upsert`、`search`、`delete_by_doc`、`count` | `infra/milvus/store.py` |
| `LexicalIndex` | `upsert`、`search`、`delete_by_doc`、`count` | `infra/postgres/lexical.py` |
| `GraphStore` | `ensure_schema`、`upsert`、`candidate_entities`、`expand`、`delete_by_doc`、`count` | `infra/neo4j/store.py` |
| `Embedder` | `embed_documents`、`embed_query` | `infra/llm/openai_compat.py` |
| `LLM` | `chat`、`structured_json` | 同上 |
| `VLM` | `describe_image` | 同上 |
| `Reranker` | `rerank` | `infra/llm/rerank.py` |
| `Parser` | `parse(data, filename, mime)` | `infra/parsers/` |
| `Storage` | 上传/图片落盘、读取、删除 | `infra/storage/local.py` |

端口只表达能力，不暴露第三方类型；例如 `SearchHit`、`Entity`、`GraphEvidence` 都是
`domain/models.py` 里的 dataclass。

## 4. 数据模型与存储

### 4.1 PostgreSQL（元数据 + 词法）

手写 DDL，迁移按文件名排序执行并记录在 `schema_migration`；不使用 ORM/迁移框架。

| 表 | 关键列 | 说明 |
|---|---|---|
| `kb` | `id`, `name`, `description` | 知识库 |
| `document` | `kb_id`, `filename`, `mime`, `sha256`, `status`, `error`, `chunk_count`, `chunker_version`, `embed_model`, `file_path` | 状态机 + 去重（`UNIQUE(kb_id, sha256)`） |
| `doc_image` | `kb_id`, `doc_id`, `seq`, `page_no`, `bbox`, `file_path`, `caption` | 抽取出的图片与 VLM 描述 |
| `chunk` | `id varchar(32)`, `kb_id`, `doc_id`, `parent_id`, `level`, `kind`, `content`, `heading_path`, `page_no/page_to`, `bbox`, `char_start/char_end`, `seq`, `token_count`, `image_id`, `meta` | 父子块；`UNIQUE(doc_id, seq)` |
| `chunk_lex` | `chunk_id`, `lex_tokens`, 生成列 `tsv` | 只存子块；`to_tsvector('simple', ...)` + GIN |
| `ingest_job` | `doc_id`, `stage`, `status`, `attempts`, `error`, `timings` | 同文档最多一个 pending/running（部分唯一索引） |
| `query_log` | `kb_id`, `mode`, `query`, `latency_ms`, `hit_ids`, `channels`, `rerank` | 检索可观测性 |

查询一律带 `kb_id`；`chunk` 的 `parent_id` 自引用外键 `DEFERRABLE`，允许先插父块再插子块。

### 4.2 Milvus

collection 由 `ensure_collection(dimension)` 幂等创建/校验：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | VARCHAR(64) PK | chunk id |
| `kb_id` | VARCHAR(64), **partition key** | 分区路由，**仍必须带等值过滤** |
| `doc_id` | VARCHAR(64) | 删除与计数 |
| `level` | VARCHAR(16) | 只有 `child` 会写入 |
| `vector` | FLOAT_VECTOR(dim) | HNSW + COSINE |

`consistency_level="Strong"`：摄取后立即检索、删除后立即断言 0 命中的测试不需要 sleep。
向量维度在启动期与 `EMBEDDING_DIM` 校验，不一致直接失败而不是"能启动但检索恒空"。

### 4.3 Neo4j

| 元素 | 键 | 说明 |
|---|---|---|
| `(:Doc {kb_id, doc_id})` | 复合唯一 | 文档节点 |
| `(:Chunk {kb_id, chunk_id, doc_id})` | 复合唯一 | 文本子块节点 |
| `(:Entity {kb_id, normalized, type, name, id})` | 复合唯一 | 跨块合并的同实体 |
| `(:Doc)-[:HAS_CHUNK]->(:Chunk)` | — | 文档结构 |
| `(:Chunk)-[:MENTIONS]->(:Entity)` | — | 提及关系 |
| `(:Entity)-[:REL {kb_id, doc_id, chunk_id, evidence_span, type}]->(:Entity)` | — | 关系边必须带证据 |

所有查询/写入的 MATCH/MERGE 都带 `kb_id`；删除按 `doc_id` 清边与节点，再清理无引用的
Entity。`expand` 做 BFS 多跳展开，邻居数与跳数有上限。

## 5. 请求时序

### 5.1 上传与摄取

```
POST /documents
  → API 校验扩展名/magic bytes/大小
  → DocumentService：查 kb → sha256 去重 → 落盘 → 建 document(pending) → 入队 ingest_job
  → 202/201 返回

worker (lifespan 内 asyncio 任务)
  → claim_ingest_job (FOR UPDATE SKIP LOCKED)
  → IngestPipeline：
       parsing   : Parser.parse → blocks（PDF 带 page/bbox；docx 无页码）
       chunked   : 标题树 → 父块(≤1200 token) → 子块(目标 320，overlap 15%)
       图片       : 落盘 + VLM caption → 带 low_weight 的 image chunk
       persist   : replace_images / replace_chunks → document=chunked
       indexing  : IndexingService：
                     Milvus upsert（仅 child）→ chunk_lex upsert → GraphExtractor + Neo4j upsert
                     → document=ready
```

失败：`IngestWorker` 把 `ingest_job` 置 `failed`（记录 stage/error/timings），文档置
`failed`；索引层先清理三路、chunk、图片行再置失败，保证"无脏索引"。重试走
`POST /documents/{id}/retry`：清 chunk、状态回 `pending`、重新入队。

### 5.2 检索

```
POST /search {query, mode, top_k, rerank}
  → SearchService 确认 kb 存在
  → vector  : embed_query → Milvus (kb_id ==)
  → keyword : jieba 分词/清洗 → chunks_lex (kb_id ==, to_tsquery)
  → graph   : 实体链接(精确→别名→2-gram) → expand → PG 回查子块
  → hybrid  : RRF(k=60) 或 weighted(min-max)
  → 可选 rerank（默认关闭，需显式传参且配置了端点）
  → 按 chunk_id 批量回填 doc/snippet/heading/page
  → 写 query_log（延迟、命中 id、各通道候选数）
```

任一通道不可用（如未配置 EMBED_API_KEY）时 hybrid 降级到其余通道，而不是整体失败；
显式请求单一通道则抛配置错误。

### 5.3 问答

```
POST /ask {question, mode}
  → SearchService 检索
  → 取子块 → 回填父块（按 token 预算丢整条证据，不截断半条）
  → 全部证据组成带编号的 prompt → LLM structured_json {answer, citations}
  → 引用回校验：只保留 [1..n] 且去重的编号；无有效引用 → 拒答
  → 返回 answer + citations（doc/page/seq/heading/snippet）
```

无命中或证据为空时不调用 LLM，直接返回 `no_evidence=true`，避免"无证据仍编答案"。

### 5.4 重建索引

`POST /kbs/{id}/reindex` 创建后台 asyncio 任务（`task_id`），逐个把 `chunked/ready`
且有 chunk 的文档重新交给 `IndexingService`；已删除、失败、无 chunk 的文档跳过。
`GET /kbs/{id}/reindex/{task_id}` 返回 `running/done`，CLI 依赖它避免读旧状态。

## 6. 删除与并发

删除顺序是**先索引、后 DB、再文件**，任一步失败都向上抛，不返回"假成功"：

```
delete_document: vector.delete_by_doc → lexical.delete_by_doc → graph.delete_by_doc
                 → repo.delete_document（级联 chunk/image/job）
                 → storage.delete_document（uploads/ 与 images/ 目录）
delete_kb      : 对每个文档执行三路删除 → repo.delete_kb → 删除文件目录
```

摄取与删除并发时靠 **tombstone**：管线在解析后、图片后、落库前各检查一次文档是否仍存在；
不存在则抛 `IngestAbortedError`，worker 放弃任务（不标 failed），管线补清已写文件。
删除产生的残留数据由删除路径负责，不依赖 worker。

## 7. 关键不变量

1. **所有检索必须带 `kb_id`**：Milvus 分区键只负责路由，仍需等值过滤；PG 用
   `where kb_id = %s`；Neo4j 用 `{kb_id: $kb_id}`。
2. **只有子块进索引**：Milvus 与 `chunk_lex` 只写 `level=child`；父块只用于回答回填。
3. **chunk id 确定**：`sha256(kb_id:doc_id:seq:chunker_version)[:32]`；同一文件进不同库
   不会主键冲突。
4. **可回溯**：PDF 块带 `page_no + bbox + char_start/char_end`；docx 带
   `char_start/char_end + seq`，不伪造页码；图谱边带 `chunk_id + evidence_span`。
5. **状态即能力**：`ready` 表示三路索引全部完成；`chunked` 只代表解析分块完成。
6. **失败无残留**：三路索引、chunk、图片行与 `chunk_count` 一起清理。
7. **配置单一来源**：`config.py` 的 `Settings`，compose `environment` 是容器唯一透传通道。

## 8. 启动装配（lifespan）

`main.py` 的 lifespan 依次：读 Settings → `ensure_dirs` → 建 PG 连接池 → Milvus
`ensure_collection`（维度 fail-fast）→ 词法索引 → Neo4j `ensure_schema` → 构造端口与
用例服务 → 启动 worker → 注册路由。测试用 `create_app(settings=..., repository=...,
start_worker=False)` 注入 fake/真实仓储，路由从 `app.state` 取服务（`api/deps.py`）。

`/readyz` 逐项检查 PG/Milvus/Neo4j 与模型配置；`model:*` 项 `required=false`，
未配置 Key 不阻塞启动，但依赖模型的功能不可用。

## 9. 测试分层

| 层 | 位置 | 依赖 | 覆盖 |
|---|---|---|---|
| 单测 | `tests/unit` | fake 端口 | 分块、融合、引用校验、状态机、路由 |
| 集成 | `tests/integration` | 真实三存储 | 仓储、索引、隔离、失败、删除、全链路 |
| 端到端 | `tests/e2e` | 容器 + 模型 Key + MCP extra | MCP stdio `list → search` |

架构约束本身也有测试：`test_layering.py`（import 图）、`test_config_compose_sync.py`
（配置三处一致）、`test_skill_sync.py`（Skill 源与安装副本一致）。
