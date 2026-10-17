# ADR-006：中文查询的英文改写（Query Translation）

- 状态：已采纳
- 日期：2026-10-15

## 背景

知识库扩充了 FastAPI / Vue 等**英文官方文档**后，出现大量「中文问题检索不到英文文档」的失败案例。

典型问题：「FastAPI 怎么定义路径参数」→ 模型回答「无法回答」，尽管
`data/docs/fastapi/tutorial/path-params.md` 里就有标准答案
（"You can declare path parameters with the same syntax as Python format strings"）。

## 根因（实测定位，非猜测）

对同一批问题分别用中文、英文、中英拼接三种 query 检索，结论是：

1. **短中文查询能命中**：「路径参数」→ `path-params.md` 第 1 名。
2. **长中文查询被口语词稀释**：「FastAPI 怎么定义路径参数」→ `path-params.md` 排不进 top_k，
   被 `query-params*.md` 等大量同样含 "parameter" 的碎片淹没。
3. **纯英文查询稳定命中**：「path parameters」/「path parameter define」→ `path-params.md` 第 1 名。
4. **中英拼接反而更差**：「FastAPI 怎么定义路径参数 path parameter define」→ 仍命中不到，
   因为 embedding 把两种语言的语义「平均」后两边都像又都不像。
5. **框架名是噪声**：翻译若保留 "FastAPI"，会被这个几乎每页都出现的高频词主导，
   检索出一堆通用页面（features / async / behind-a-proxy），反而淹没真正有区分度的 "path parameter"。

## 决策

实现**查询改写（Query Translation）**，核心三点：

1. **中文字符检测**：仅当 query 含 CJK 字符时才触发改写，英文/代码符号查询零开销。
2. **LLM 翻译成英文检索关键词**：prompt 明确要求「去掉框架名等宽泛词，只输出官方文档标题风格的
   核心技术术语」（如 "path parameters"、"lifecycle hooks"）。实测这一步是关键——
   去掉框架名后 "path parameters" 才能精准命中。
3. **双路独立检索 + RRF 融合**：中文 query 和英文关键词**各自**检索一次（绝不拼成一条），
   再用 RRF（k=60）按排名融合去重。让英文路的高排名结果公平进入 top_k。

## 权衡

- **成本**：每次中文提问多一次轻量 LLM 调用（翻译）。但英文查询、代码符号查询不触发，且翻译失败
  优雅降级为原 query 检索，不影响主流程。
- **收益**：跨语言召回从「大量失败」提升到「核心文档稳定进入 top_k」，端到端问答从「无法回答」
  变为带引用的正确回答。

## 备选方案（已否决）

- **中英拼接单路检索**：实测更差（见根因第 4 条）。
- **纯英文替换（丢弃中文路）**：中文文档会召回变差，且翻译可能丢语义。
- **文档侧翻译入库**：把英文文档全部翻译成中文再入库，工程量巨大且译文质量不可控，检索时效性差。
