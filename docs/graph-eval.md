# 图谱抽取评测结果

> 日期：2026-11-19 · 抽取模型 `deepseek-flash` · chunker `v1`
> 夹具：[`tests/fixtures/graph/`](../tests/fixtures/graph/) · 金标：[`gold.json`](../tests/fixtures/graph/gold.json)
> 原始数据：[`scripts/bench/results/graph-eval.json`](../scripts/bench/results/graph-eval.json)
> 复现：`uv run python scripts/make_graph_fixtures.py && uv run python scripts/graph_eval.py`

## 1. 测试语料

| 文件 | 内容 | 目的 |
|---|---|---|
| `gold_doc.pdf` | 2 页、22 个事实句：5 跳实体链、8 类实体、别名（XR 组件）、白名单谓词与 2 个“负责”（预期降级 `OTHER`） | 测关系召回、类型与 span 有效性 |
| `adversarial_doc.docx` | 否定句、代词、错误事实、表格、图片、页眉 | 测误抽；表格/图片本就不该进图 |

金标 21 个实体、22 条关系；比较前按 `normalize_name` 去空格/大小写/括号说明，实体要求
名称与类型都一致，关系要求 `(head, type, tail)` 完全一致。

## 2. 结果

- 知识库：`graph-eval-1795089080`（评完自动删除）
- 实抽：26 个实体、27 条关系边（去重前）

| 指标 | Precision | Recall | F1 | TP/FP/FN |
|---|---|---|---|---|
| 实体（名称+类型） | 0.731 | 0.905 | 0.808 | 19/7/2 |
| 关系（head+type+tail） | 0.778 | 0.955 | 0.857 | 21/6/1 |

- 实体类型准确率（名称命中时）：**0.950**（20 个名称命中）
- `evidence_span` 有效率：**27/27 = 1.000**（每条边都能回到原文区间）
- 丢弃计数：抽取阶段的 `dropped_spans/dropped_relations` 未落库，本次未统计（见 §5）

## 3. 漏抽与多抽

**漏抽（1 条）**

- `2026 年 3 月迁移 -[关联]-> 知识库服务`：模型把事件拆成了「2026 年 3 月 → 迁移 →
  关联知识库服务」两个实体，事件没有被识别为单一实体。这是事件类实体的典型边界。

**多抽（6 条）**

| 三元组 | 来源 | 判断 |
|---|---|---|
| `蓝海数据 -[关联]-> 星云科技` | 对抗句“是同一家公司” | 模型把断言当关系；金标不认可 |
| `星云科技 -[OTHER]-> 新版本` | “可能在 2027 年推出新版本” | 推测性内容不应成边 |
| `知识库服务 -[OTHER]-> 旧版检索组件` | 否定句“不依赖” | 谓词被降级为 `OTHER`，但边本身不该出现 |
| `知识库服务 -[OTHER]-> 上线审批` | 代词“它负责所有上线审批” | 指代未消解，模型补了主语 |
| `2026 年 3 月 -[关联]-> 迁移` / `迁移 -[关联]-> 关联知识库服务` | 事件拆分 | 同一条漏抽的副产物 |

**未出现的误抽**：表格（审计日志/报表服务）与图片 caption 都没有产生实体或边，符合
“只处理 child text chunk”的预期。

## 4. 结论

- 金标关系的 **Recall 0.955 / span 有效率 1.0**：说明在“事实写在同一句、实体用全名”的
  文档上，抽取链路是可用的，证据区间没有错位；
- **Precision 偏低主要来自对抗语料**：否定句、代词、推测句、错误事实会被模型“补全”成
  关系；如果文档里这类表达多，需要加一层断言过滤或人工复核；
- **类型准确率 0.95**：唯一明显问题是事件类（“2026 年 3 月迁移”）被拆散；时间/事件类
  实体建议在 prompt 里补充示例，或接受为已知限制；
- 同句关系是当前抽取的强项，**跨句/跨块关系不会自动合并**（设计如此，多跳靠多个边 +
  查询期 BFS）。测试文档应把每条关系写在同一句里，否则 Recall 低不是实现缺陷。

## 5. 已知观测缺口

`ExtractionResult` 里的 `dropped_spans` / `dropped_relations` / `failed_chunks` 目前只在
内存中，未写入 `ingest_job.timings`。因此本报告只能统计“最终留下的图”的 P/R，无法区分
“模型没抽到”和“抽到但被 span 校验丢弃”。补法：在 `application/indexing.py` 索引成功后把
三个计数写进 timings，再在评测里一并输出。

## 6. 如何复跑与扩展

```bash
# 生成/更新夹具（事实改动时同步改 tests/fixtures/graph/gold.json）
uv run python scripts/make_graph_fixtures.py

# 新建 KB、装载、评测、自动清理；也可以 --kb <id> 评已有库
uv run python scripts/graph_eval.py \
  --out-md /tmp/graph-eval.md --out-json /tmp/graph-eval.json

# 只跑指标单测
uv run pytest tests/unit/test_graph_eval.py -q
```

扩展建议：

1. 增加 5–10 条多跳问题，用 `scripts/bench/run_bench.py` 对比 `graph` 与 `vector/keyword`
   的 Recall/MRR（金标已能定位到句子）；
2. 加入跨句关系、英文实体、数字单位等对抗样本，观察 Precision 变化；
3. 把丢弃计数落库后，增加“抽取成功率/校验丢弃率”两项指标。
