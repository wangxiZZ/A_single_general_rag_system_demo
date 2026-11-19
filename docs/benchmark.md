# 检索评测报告

> 日期：2026-11-19 · chunker `v1` · 嵌入 `qwen3.7-text-embedding-flash` (1024) ·
> 生成/图抽取 `deepseek-flash` · 语料：`tests/fixtures/demo/` 两个文档（各 9 个 chunk）。
> 原始数据：[`rerank-off.json`](../scripts/bench/results/rerank-off.json)、
> [`rerank-on-hybrid.json`](../scripts/bench/results/rerank-on-hybrid.json)。

## 1. 方法与口径

- 金标 25 条，覆盖 `single_hop` / `multi_hop` / `term` / `image` / `unanswerable`；
  金标定位到文档与字符区间（图片为零长度点 + 页码），不绑定 chunk id；
- 命中判定：命中块与金标区间有交集，或零长度点落在块范围内且页码一致；
- Recall@K：top-K 内至少一个命中记 1；MRR 取首个命中的倒数名次；
  nDCG@10 用二值相关性、同一金标只记一次；
- 不可回答题不参与召回均值，单独统计“top-10 是否为空”；
- 维度只跑 `mode × rerank`，不跑 `chunk_size`（换尺寸需重建索引并重算金标）；
- 金标里的 `doc_id` 每次装载不同，跑分时按 `doc_name` 解析（见 `run_bench.py`）。

复现：

```bash
uv run python scripts/bench/make_qa.py                 # 重新计算金标区间
uv run python scripts/bench/run_bench.py --rerank off \
  --out-md scripts/bench/results/rerank-off.md \
  --out-json scripts/bench/results/rerank-off.json
uv run python scripts/bench/run_bench.py --modes hybrid --rerank on \
  --out-md scripts/bench/results/rerank-on-hybrid.md \
  --out-json scripts/bench/results/rerank-on-hybrid.json
```

## 2. 总体结果

### 2.1 各通道（rerank 关闭）

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | 不可回答 top-10 空 | 平均延迟 ms |
|---|---|---|---|---|---|---|
| vector | 1.000 | 1.000 | 0.917 | 0.937 | 0/4 | 373 |
| keyword | 1.000 | 1.000 | 0.929 | 0.944 | 0/4 | **10** |
| graph | 0.857 | 0.857 | 0.833 | 0.803 | 0/4 | 17 |
| **hybrid** | **1.000** | **1.000** | **0.968** | **0.972** | 0/4 | 376 |
| hybrid_weighted | 1.000 | 1.000 | 0.952 | 0.965 | 0/4 | 358 |

结论：在这份语料上三路都能召回，差距体现在**排序**而不是“能不能找到”。
Hybrid 的 MRR/nDCG 最高；keyword 用 10ms 达到接近的效果；graph 单独使用明显更弱。

### 2.2 重排是否值得

| 配置 | Recall@5 | Recall@10 | MRR | nDCG@10 | 平均延迟 ms |
|---|---|---|---|---|---|
| hybrid（off） | 1.000 | 1.000 | 0.968 | 0.972 | 376 |
| hybrid + LLM 重排（on） | 1.000 | 1.000 | **1.000** | **1.000** | **6054** |

- 收益：把 3 条“第二名才对”的查询全部提到第一（MRR 0.968 → 1.000）；
- 代价：平均延迟 **约 16 倍**（0.38s → 6.05s），每次查询多一次 LLM 调用；
- 结论：重排应保持**显式 opt-in**，不作为默认检索路径；对“必须给第一条就对”的场景
  （如直接展示 top-1 的证据卡片）才值得打开。

> 本次重排只跑了 `hybrid`（默认模式）；`run_bench.py` 支持 `--rerank both` 跑全模式，
> 代价是 5 倍 LLM 调用量，未纳入本次报告。

## 3. 分题型分析（Top-1 准确率）

“Top-1 准确”= 第一条命中就覆盖金标；`multi_hop` 只要命中任一金标条目即算。

| mode | single_hop | multi_hop | term | image |
|---|---|---|---|---|
| vector | 0.955 | 0.875 | 1.000 | 0.500 |
| keyword | 1.000 | 1.000 | 0.750 | 0.750 |
| graph | 1.000 | 1.000 | **0.375** | 0.500 |
| hybrid | 1.000 | 1.000 | 1.000 | 0.500 |
| hybrid_weighted | 1.000 | 1.000 | 0.875 | 0.750 |
| hybrid + rerank | 1.000 | 1.000 | 1.000 | 1.000 |

可以回答“哪类问题该走哪路”：

- **正文事实/单跳**：三路都行；keyword 最便宜，hybrid 排序最稳；
- **术语/表格字段**（`q-017`、`q-018`）：graph 最弱（0.375），因为表格块不参与实体抽取；
  keyword/hybrid 更可靠；hybrid 在这些题上把关键词与向量互补，top-1 全中；
- **图片**（`q-012`、`q-021`）：vector/keyword 都能召回描述块，但排序不稳；重排后 top-1 全中；
- **多跳/实体关系**：graph 单独 top-1 1.000，说明实体链接 + 关系边在跨句问题上有效；
- **不可回答**：四路都会返回 top-10（检索层没有相似度阈值），拒答发生在回答层
  （`no_evidence=true`）。因此 0/4 不代表缺陷，但说明**不能用“检索为空”判断不可回答**。

## 4. 失败样例

| 查询 | 现象 | 原因 |
|---|---|---|
| `q-017` PDF 是否支持内嵌图片 | graph 未命中 | 答案在 docx 表格块；图抽取只处理 `kind=text` 的子块 |
| `q-018` DOCX 引用定位 | graph 未命中 | 同上，表格内容没有实体/关系 |
| `q-021` 产品手册图片块 | graph 未命中、hybrid 排第 3 | 图片 caption 不参与图抽取；向量/关键词可召回但排序一般 |
| `q-012` 员工手册图片 | vector 排第 4、keyword 第 2 | 合成图片的 caption 语义弱，描述文本与问题词面重合少 |

这些不是“实现 bug”，而是当前图谱只覆盖文本实体的边界；如需覆盖表格/图片，应扩展抽取
输入或为图片/表格建立独立索引（见 `design-future.md` 的版面感知方案）。

## 5. 延迟

| 通道 | 平均延迟 | 说明 |
|---|---|---|
| keyword | 10ms | PG tsvector，无外部调用 |
| graph | 17ms | 实体链接 + BFS（语料小） |
| vector | 373ms | 主要耗时是 embedding API |
| hybrid | 376ms | 向量为主，融合开销可忽略 |
| hybrid + rerank | 6054ms | 多一次 LLM listwise（30 候选） |

延迟随语料与网络变化；这里的绝对值只用于横向比较。

## 6. 局限

1. 语料只有 2 个文档、25 条问题，指标波动大，结论应视为方向性而非统计显著；
2. 夹具是合成文档，图片为占位噪点图，图片类问题仅验证链路；
3. 金标区间来自当前解析器输出，换解析器需要重新运行 `make_qa.py`；
4. 重排只测了 hybrid；其他模式的 on/off 未采样；
5. 未评测回答质量（忠实度、引用正确率），本次只评测检索层。
