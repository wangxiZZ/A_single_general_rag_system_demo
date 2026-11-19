# 详细设计与取舍

> 本文回答"为什么这样做、代价是什么、哪些情况不保证"。实际结构见
> [`architecture.md`](architecture.md)，完整选型记录见 [`技术选型文档.md`](技术选型文档.md)
> 的 ADR-001–023。

## 1. 设计目标

- 一个可解释、可测试的通用 RAG 服务，而不是"接上一个框架就能跑"的演示；
- 每个回答都能回溯到原文：页码/坐标（PDF）或段落号（docx），图谱关系也要有证据区间；
- 失败、删除、跨库隔离这些边界有明确语义和自动断言；
- 检索与融合默认不调用 LLM，控制延迟与成本；
- 以 Server 形式对外同时提供 REST、Web、MCP 与 Skill，但只维护一套业务实现。

明确不做（本阶段）：OCR、鉴权/多租户、流式输出、在线增量图数据库、复杂表格与公式还原。

## 2. 关键决策一览

| 决策 | 选择 | 主要理由 | 代价 | ADR |
|---|---|---|---|---|
| 编排框架 | 不引入 LangChain/LangGraph/LlamaIndex | 步骤固定；自己控制事务、清理与隔离 | 需要自己写编排与测试 | 题目硬要求 |
| 元数据存储 | PostgreSQL + 手写 SQL | 检索型查询更看重执行计划；避免 ORM 遮蔽 | 手动维护 DDL 与迁移顺序 | ADR-002 |
| 向量库 | Milvus Standalone | 分区键 + HNSW 满足库隔离与规模 | 运维多一个组件；最终一致性需注意 | ADR-004 |
| 图谱 | Neo4j Community | 多跳路径表达自然；复合唯一键自带库隔离 | 图抽取质量依赖 LLM | ADR-020 |
| 词法 | PG `tsvector('simple')` + jieba | 中文分词与停用词在应用侧可控 | 不支持同义词/拼音 | ADR-005 |
| 架构 | 端口 + 适配器，`application` 只依赖 `domain` | 用例可被 API/worker/CLI/MCP 复用，可用 fake 单测 | 多一层抽象 | ADR-019 |
| 分块 | 标题树 + 父子块，父块不进索引 | 检索精度用子块，回答上下文用父块 | 存储与回填逻辑更复杂 | ADR-018 |
| chunk id | `sha256(kb_id:doc_id:seq:version)[:32]` | 可重入、可跨库、可重算 | 改分块参数必须升版本并 reindex | ADR-018 |
| 融合 | RRF（默认）+ weighted（对照） | 三路分数量纲不可比；RRF 对量纲免疫 | 丢弃原始分差信息 | ADR-014 |
| 重排 | 默认关闭，显式 opt-in | 检索路径零 LLM 调用 | 默认精度不是最优 | ADR-014 |
| 图谱抽取 | 只在入库时做一次 | 查询期零 LLM；成本可控 | 新文档必须重新抽取 | ADR-020 |
| 回答 | 证据编号 + 引用回校验 + 无证据拒答 | 降低幻觉与"假引用" | 有效引用缺失时宁可拒答 | ADR-013 |
| 打包 | hatchling 双包（`src/rag_kb` + `mcp_server`） | 保持 MCP 在仓库根且单一入口 | 构建依赖变化 | ADR-023 |

## 3. 分块策略

### 3.1 结构

1. 解析器输出有序 `Block`（标题/正文/图片/表格），PDF 带 `page_no + bbox`，docx 只有
   `char_start/char_end + seq`；
2. 按标题构建层级路径（`heading_path`），标题变化即开启新 section；
3. 每个 section 内先按 `parent_max_tokens=1200` 聚成父块，再切成
   `chunk_target_tokens=320`、`overlap_ratio=0.15` 的子块；
4. 子块不跨父块；重叠按句子边界回退，避免从半句话开始；
5. 图片单独成块（`kind=image`），内容为 VLM 描述 + 前后文；`meta.low_weight=true`，
   避免图片描述压过正文；
6. token 用 `chars_per_token=1.5` 粗估——中文 ≈1.5 字/token，仅用于长度控制，不用于计费。

### 3.2 为什么父块不进索引

- 父块（≤1200 token）语义完整但粒度粗，直接检索会"命中一大段、答非所问"；
- 子块（320）命中更准，但单独交给 LLM 可能缺上下文；
- 因此检索只针对子块，回答阶段按 `parent_id` 回填父块，预算不足时整条证据丢弃，
  不截断半条（截断会破坏引用区间）。

### 3.3 可回溯要求

| 格式 | 必带字段 | 引用展示 |
|---|---|---|
| PDF | `page_no`, `bbox`, `char_start/char_end` | 第 N 页 |
| docx | `char_start/char_end`, `seq` | 第 N 段 + 标题路径 |
| 图片 | `image_id`, `page_no`（PDF） | 图片描述 + 页码 |
| 图谱边 | `chunk_id`, `evidence_span` | 回到原文片段 |

为 docx 伪造页码是明确禁止的：没有的信息就不展示。

## 4. 检索策略

### 4.1 三路召回

| 通道 | 做法 | 适用 |
|---|---|---|
| 向量 | `embed_query` → Milvus `kb_id ==` + COSINE，HNSW(M=16, efConstruction=200) | 语义相近、措辞不同 |
| 关键词 | 查询侧 jieba 分词 → 清洗标点/停用词 → `to_tsquery('simple', 'a | b')` → `ts_rank_cd` | 术语、编号、精确词 |
| 图谱 | 查询实体链接（精确 → 别名 → 2-gram contains）→ BFS 展开 ≤2 跳、邻居 ≤50 → 回查子块 | 多跳、实体关系 |

中文词法不是"装个分词器"就行：入库前分词写入 `lex_tokens`，出查询同样分词；
`to_tsquery` 对 `& | ! ( ) : * '` 敏感，空 token 直接返回空结果，不能拼出非法查询。

### 4.2 融合

- **RRF**：`score = Σ weight / (k + rank)`，`k=60`；只使用排名，天然免疫三路分数量纲；
- **weighted**：每路先 min-max 归一化再加权，保留作对照实验；同分时排序不稳定，
  不作为默认；
- 去重按 `chunk_id` 合并，`channels` 记录命中来源，`debug.rank_per_channel/raw_score`
  保留排查信息；
- 某一路不可用（如未配 Embedding Key）时 hybrid 降级到其余通道；显式指定单通道则报配置错误。

### 4.3 图谱链接与打分

1. 精确匹配 `normalized`；
2. 别名匹配；
3. 2-gram `contains` 模糊兜底，多个 gram 命中同一实体时按 id 去重；
4. `expand` 用 BFS，路径分数按跳数衰减；结果映射回子块（`map_evidence_to_child`），
   映射失败才退回父块。

### 4.4 重排

- 默认关闭：检索与融合路径不产生 LLM 调用；
- `rerank=true` 且配置了端点时才走 LLM listwise 重排，只对前 30 条候选；
- 端点/模型为空不等于服务故障，只有显式开启且缺配置才报错。

## 5. 回答与引用

1. 取检索命中的子块 id，批量回填父块与文档信息；
2. 按 `qa_max_context_tokens=6000` 拼证据，父块整体加入或整体丢弃；
3. prompt 中每条证据带 `[n]`、文档名、页码/段落与标题路径，要求"只根据证据回答"；
4. LLM 返回结构化 `{answer, citations}`；
5. **回校验**：只保留 `1 ≤ n ≤ 证据数` 的整数编号并去重；没有有效引用时返回
   `no_evidence=true` 的拒答，而不是接受一个没有出处的答案；
6. 引用详情携带 `doc_name / page_no / seq / heading_path / snippet`，前端与 MCP 直接展示。

无命中时直接拒答，不调用 LLM——既省成本，也避免模型"自由发挥"。

## 6. 失败与一致性

### 6.1 失败分类

| 场景 | 结果 | 副作用 |
|---|---|---|
| 超过 `MAX_UPLOAD_MB` | HTTP 413 | 不建行、不落盘、不入队 |
| 扩展名/magic bytes 不符 | HTTP 415 | 同上 |
| 同库 sha256 重复 | HTTP 409 + 已有 doc id | 不重复落盘 |
| 损坏 PDF / 扫描件 / 解析异常 | `document.status=failed` | 清 chunk/图片，三路无写入 |
| 模型 Key 缺失 / embedding 失败 | `failed`，`error` 含可操作原因 | 清三路、chunk、图片行 |
| 图抽取全部失败 | `failed` | 同上 |
| 个别 span/关系校验失败 | 不算失败 | 计入 `dropped_spans/dropped_relations` |
| 摄取中被删除 | tombstone 中止 | 不标 failed，补清文件 |

`error` 截断到 2000 字符，不含堆栈与密钥。

### 6.2 状态机

```
pending → parsing → chunked → indexing → ready
    └──────────────┴──────────┴────────────► failed → (retry) pending
```

`ready` 的语义是**三路索引完成**；`chunked` 只代表解析分块完成。重试会清空 chunk、
状态回 `pending` 并重新入队；`chunker_version` 变化时旧文档需要 reindex。

### 6.3 删除与并发

- 删除顺序：三路索引 → PG（级联 chunk/image/job）→ 磁盘；
- `ingest_job` 用 `FOR UPDATE SKIP LOCKED` 领取，同文档最多一个活跃任务（部分唯一索引）；
- 管线在解析后、图片后、落库前检查文档是否仍存在，删除则抛 `IngestAbortedError`；
- 验收口径是"删后三通道 0 命中且底层计数为 0"，不是接口 204；
- Milvus 写入/删除最终一致，因此统一用 `Strong` 一致性，测试无需 sleep。

## 7. 成本与性能

- 检索与融合默认 0 次 LLM 调用；只有 `ask` 调用一次生成，`rerank=llm` 需显式开启；
- 图抽取只在入库时做一次，按块并发，失败按块丢弃；
- 模型调用有并发上限（`llm_max_concurrency` / `embed_max_concurrency` /
  `vlm_max_concurrency`）与超时、429/5xx 重试；
- embedding 按 batch 发送，单条文本超长会截断到模型上限；
- 查询日志记录延迟、命中 id、各通道候选数与是否重排，便于定位是"召不回"还是"答不好"。

## 8. 已知限制

| 限制 | 影响 | 可能的处理 |
|---|---|---|
| 无 OCR | 扫描件入库失败（明确报错） | 接 OCR 或外部文本层 |
| 复杂表格/公式/多栏 | 还原质量有限，阅读顺序可能错 | 专用版面解析器 |
| docx 无页码/bbox | 只能按段落号引用 | 转 PDF 后入库 |
| 中文分词依赖 jieba 词典 | 新词/专有名词可能切错 | 自定义词典 + 同义词 |
| 图抽取质量依赖 LLM | 漏抽或类型降级为"其他" | 二次校验/人工复核 |
| 无鉴权 | 仅适合本机/内网 | 网关鉴权或多租户改造 |
| 无流式输出 | 长回答等待时间长 | SSE/WebSocket |
| Milvus 单 collection | 规模与多租户隔离靠过滤 | 按租户分 collection |
| 无增量图谱更新 | 文档更新需整篇重建 | 版本化实体/边 |

## 9. 扩展方向

1. **评测闭环**：固定 20–30 条金标，按 `mode × rerank` 跑 Recall@5/10、MRR、nDCG，
   用数据决定默认参数（M7）；
2. **混合重排**：向量召回 + cross-encoder/LLM listwise 的级联，先小候选再精排；
3. **增量索引**：文档版本化，按 chunk diff 更新三路，避免整篇重建；
4. **多租户**：`tenant_id` 进入所有键与过滤条件，或按租户分 collection/数据库；
5. **可观测性**：把 `query_log` 接入指标（零命中率、通道贡献、引用拒答率）；
6. **内容增强**：OCR、表格结构化、图片 caption 回写正文，提升图片与表格召回。

## 10. 被否方案（摘要）

| 方案 | 否决原因 |
|---|---|
| 父块直接进向量库 | 命中粒度粗、上下文噪声大；回答阶段回填已够用 |
| 三路分数直接相加 | 余弦、ts_rank、路径分不可比，权重无法解释 |
| 每次检索都做 LLM 重排 | 延迟与成本不可接受，且不是所有查询都需要 |
| 图谱在查询期实时抽取 | 每次提问都调 LLM，成本与延迟失控 |
| 用 chunk_id 做评测金标 | 换分块参数即失效，无法横向比较 |
| ORM/迁移框架 | 检索型 SQL 需要看执行计划，抽象收益低 |
| MCP 内直接连数据库 | 两套检索实现，排序与引用规则会漂移 |
