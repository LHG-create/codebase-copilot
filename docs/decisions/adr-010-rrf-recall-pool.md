# ADR-010：RRF 融合的候选池宽度（兼记「库扩容 → 跨领域干扰」）

- 状态：已采纳
- 日期：2026-10-15

## 背景

ADR-009 把 AsciiDoc 表格修好后（表格不再被切成孤儿行），实测同一个 Spring 问题，
**回答反而变差了**：

> Spring MVC 里 `@PathVariable` 和 `@RequestParam` 有什么区别？路由 `/users/{id}` 的参数该用哪个？

修复前：模型能给出 3 条要点，引用到了 `@PathVariable` 和 `@RequestParam` 的定义。
修复后：模型说「资料中**没有出现** `@RequestParam` 的任何说明」——直接拒答。

## 诊断过程

第一步，确认「片段本身是否还在库里、能否被召回」：

```
纯向量检索（top 30）：
  "@RequestParam"                          → arguments.adoc 片段排 #7
  "@RequestParam Servlet request parameters" → 排 #3   (score 0.549)
```

结论：**片段能被召回，排名还很高**。所以问题不在索引侧。

第二步，检查 `_retrieve()` 的融合逻辑，发现真正的根因：

```python
docs = self.retriever.invoke(question)          # 单路只取 top_k = 8 条
english_docs = self.vectorstore.similarity_search(english, k=self.top_k)  # 同样 8 条
# RRF 融合后取 top_k
```

**两条通道各自只取 `top_k` 条，然后融合。** 这是 RRF 的误用。

RRF 的分数是 `Σ 1/(60 + rank)`。池子太小时会出现反直觉的结果：

| 文档 | 中文路排名 | 英文路排名 | RRF 总分 |
|---|---|---|---|
| 真答案片段 | #12（超出池，**不计分**） | — | 0 |
| 噪声文档 | #7 | #7 | 0.0294 |
| 真答案片段（若池够大） | #12 | #3 | 0.0288 |

即：**单通道只取 top_k，会让「排名稍靠后但内容正确」的片段直接出局**，
而「两路都中等」的文档反而胜出。表格切分把 `arguments.adoc` 的大表切成 10 块后，
每块语义更窄、排名自然下滑，这个既有缺陷就被放大暴露了出来。

## 决策

把单路召回池放大到 `max(top_k × 4, 32)`，融合后再截断到 `top_k`
（`app/chain.py::RagAssistant.__init__` 的 `self.recall_k`）。

## 实测：候选池宽度扫描

同一个问题，只改候选池大小：

| 候选池 | `@RequestParam` 片段 | 融合后 top 8 的构成 |
|---|---|---|
| **8（原逻辑）** | ❌ 无 | FastAPI 的 `query-params.md` **占 4 席** |
| 16 | ✅ 有 | `arguments.adoc` 升到 #2 |
| **32（采用）** | ✅ 有 | Spring 文档占满 8 席，**专门的 `requestparam.adoc` 也被召回**，FastAPI 干扰清零 |
| 48 | ✅ 有 | 开始混入噪声（`rsocket.adoc` 等） |

采用 32（即 `top_k × 4`）：既能把正确片段拉进池，又不至于引入无关文档。

## 效果

同一问题的端到端回答：

- **修复前**：3 条要点；结尾「资料中未给出 `/users/{id}` 的原文示例」。
- **修复后**：4 条要点（用途 / 取值来源 / 类型支持 / 路由结论）；
  补充了 `required = false`、`java.util.Optional`、`MultiValueMap`、
  `content-type: application/x-www-form-urlencoded` 等细节；
  引用到专门的 `requestparam.adoc`。

## 附带发现：库扩容导致评测指标变化（与本次改动无关）

跑评测时发现指标相比早期记录下降了：

| 指标 | 早期（4385 片段） | 现在（9956 片段） |
|---|---|---|
| 文件级 HitRate | 97.5% | 92.5% |
| 符号级 HitRate | 87.5% | 77.5% |
| 文件级 MRR | 0.902 | 0.833 |

**首先排除本次改动的嫌疑**：`eval_retrieval.py` 用的是
`store.as_retriever(k=settings.top_k)`，走**纯向量 top-k**，
既不经过查询改写，也不经过 `_retrieve()` 的 RRF 融合。
因此改写 `recall_k` 对评测结果**没有任何影响**。

真实原因是**知识库扩容**：从 4 个框架（397 文档 / 4385 片段）扩到
6 个框架（981 文档 / 9956 片段），新增的 Spring 626 篇文档稀释了候选池。
评测集 40 条全部指向本项目源码，在近万片段的池子里自然更难排进 top-8。

这暴露了一个真实的产品问题：**跨领域干扰**。同一个现象在两个地方都出现了：

1. 评测集（问本项目源码）被框架文档挤占；
2. Spring 问题被 FastAPI 的 `query-params.md` 挤占（上表中「原逻辑」那一行）。

**应对方向**（尚未实施）：按领域分库 / 检索前先做文档路由，
或检索后按 `source` 前缀做过滤与加权。这是「检索质量」的下一个改进点，
也是简历上可讲的一个技术演进。

## 验证

- 候选池宽度扫描（上表）：32 为最优。
- 端到端回答对比：见上文「效果」。
- 离线测试：15 项全通过（`tests/test_smoke.py`）。
