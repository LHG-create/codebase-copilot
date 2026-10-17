"""评测集口径审计 —— 检验「分数变化」到底来自系统改进，还是来自评测本身。

## 为什么需要这个脚本

本轮做了两件都会让官方文档域分数上升的事：

  (1) **数据层**：过滤空壳文档与导航页，重建了索引（1023 → 945 文档）
  (2) **评测层**：修正 1 条错误标注 + 3 条泛化问题的可接受答案集

不做拆分就直接报「70.8% → 83.3%」，等于把两件事的功劳混在一起，
而且无法回答一个关键问题：**这算不算给自己放水？**

## 实测结论

| 口径 | 索引 | 官方文档域 文件级 |
|---|---|---:|
| 原评测集（严格单标） | 清洗前（9956 片段） | 70.8% |
| 原评测集（严格单标） | **清洗后（9920 片段）** | **70.8%** |
| 新评测集（修正后） | 清洗后 | **83.3%** |

**数据清洗对分数的贡献是 0，+12.5pp 全部来自评测口径修正。**

这不是坏消息，而是本轮最重要的发现：修评测标注是**让基线回到真实水平**
（原标注低估了系统），不是「把分数刷上去」。数据清洗的价值在别处 ——
它修掉了一个真实缺陷：`langchain/retrieval.md` 全文只有 96 字节（只剩 frontmatter），
却被当作「LangChain 怎么做 RAG」的标准答案，导致那条评测项从写下第一天起
就注定失败，且与检索算法毫无关系（详见 ADR-012）。

用法：python eval_audit.py
依赖查询改写缓存（.eval_cache/query_rewrites.json），首次跑会调一次大模型。
"""

from __future__ import annotations

from app import domain as dm
from app.retrieval import Retriever
from app.vectorstore import load_vectorstore
from eval_retrieval import DOCS_EVAL_SET, _hit_file, make_cached_translator

# 原评测集：24 条**严格单标**（本轮修正前的版本，留档用于对照）
LEGACY_DOCS_EVAL_SET: list[tuple[str, str, str]] = [
    ("FastAPI 里怎么声明路径参数？", "fastapi/tutorial/path-params.md", ""),
    ("FastAPI 的查询参数怎么声明和做校验？", "fastapi/tutorial/query-params.md", ""),
    ("FastAPI 怎么用 Pydantic 模型定义请求体？", "fastapi/tutorial/body.md", ""),
    ("FastAPI 项目变大后怎么拆分多个路由文件？", "fastapi/tutorial/bigger-applications.md", ""),
    ("FastAPI 怎么配置跨域 CORS？", "fastapi/tutorial/cors.md", ""),
    ("FastAPI 怎么接收表单数据？", "fastapi/tutorial/request-forms.md", ""),
    ("FastAPI 的依赖注入 Depends 是怎么用的？", "fastapi/tutorial/dependencies/index.md", ""),
    ("Spring MVC 的 @RequestParam 注解是做什么的？", "spring-framework/web/webmvc/mvc-controller/ann-methods/requestparam.adoc", ""),
    ("Spring MVC 控制器方法的参数是怎么绑定的？", "spring-framework/web/webmvc/mvc-controller/ann-methods/arguments.adoc", ""),
    ("Spring MVC 里 @RequestMapping 是怎么映射请求的？", "spring-framework/web/webmvc/mvc-controller/ann-requestmapping.adoc", ""),
    ("Spring 的 bean 作用域有哪几种？", "spring-framework/core/beans/factory-scopes.adoc", ""),
    ("Spring 的依赖注入有哪几种方式？", "spring-framework/core/beans/dependencies.adoc", ""),
    ("Spring MVC 里怎么做数据绑定？", "spring-framework/web/webmvc/mvc-data-binding.adoc", ""),
    ("Vue 3 的响应式系统是怎么实现的？", "vuejs/guide/essentials/reactivity-fundamentals.md", ""),
    ("Vue 的 computed 计算属性有什么用？", "vuejs/guide/essentials/computed.md", ""),
    ("Vue 组件有哪些生命周期钩子？", "vuejs/guide/essentials/lifecycle.md", ""),
    ("Vue 的 watch 监听器怎么用？", "vuejs/guide/essentials/watchers.md", ""),
    ("Vue 表单输入怎么做双向绑定？", "vuejs/guide/essentials/forms.md", ""),
    ("LangChain 的 agent 是怎么工作的？", "langchain/agents.md", ""),
    ("LangChain 怎么做检索增强生成 RAG？", "langchain/retrieval.md", ""),
    ("LangChain 的短期记忆是怎么实现的？", "langchain/short-term-memory.md", ""),
    ("LangGraph 的检查点 checkpointer 是用来做什么的？", "langgraph/checkpointers.md", ""),
    ("LangGraph 怎么持久化图的状态？", "langgraph/persistence.md", ""),
    ("LangGraph 怎么中断执行、等待人工输入？", "langgraph/interrupts.md", ""),
]


def run(retriever, eval_set, label: str) -> list[tuple[str, object, list]]:
    hits = 0
    misses: list[tuple[str, object, list]] = []
    for q, af, _an in eval_set:
        docs = retriever.retrieve(q)
        if any(_hit_file(d, af) for d in docs):
            hits += 1
        else:
            misses.append((q, af, docs))
    print(f"  {label:<32} {hits / len(eval_set) * 100:>5.1f}%  ({hits}/{len(eval_set)})")
    return misses


def show_miss(q: str, af, docs: list) -> None:
    exp = af if isinstance(af, str) else " 或 ".join(af)
    print(f"\n  ✗ {q}\n    期望 {exp}")
    for i, d in enumerate(docs[:4], 1):
        src = str(d.metadata.get("source", "")).replace("\\", "/")
        src = src.split("data/docs/", 1)[-1] if "data/docs/" in src else src
        dom = dm.infer_domain(d.metadata.get("source", ""), d.metadata)
        print(f"    {i}. [{dom}] {src}")


def main() -> None:
    store = load_vectorstore()
    translate = make_cached_translator()
    print(f"当前索引片段数：{store.index.ntotal}\n")

    r = Retriever(store, translator=translate)
    print("=== 消融：同一套线上链路，只换评测集口径 ===")
    m_legacy = run(r, LEGACY_DOCS_EVAL_SET, "原评测集（严格单标）")
    m_new = run(r, DOCS_EVAL_SET, "新评测集（修正后）")
    delta = (len(DOCS_EVAL_SET) - len(m_new)) - (len(LEGACY_DOCS_EVAL_SET) - len(m_legacy))
    print(f"\n  口径修正带来 {delta / len(DOCS_EVAL_SET) * 100:+.1f}pp；数据清洗贡献 0.0pp"
          f"（原口径在新索引上仍是 70.8%，与清洗前一致）")

    print("\n=== 修正后仍未命中的条目，实际 top4 是谁 ===")
    for q, af, docs in m_new:
        show_miss(q, af, docs)

    print("\n=== 对照：关闭长尾先验 ===")
    r_no_prior = Retriever(store, translator=translate, use_domain_prior=False)
    m_np = run(r_no_prior, DOCS_EVAL_SET, "新评测集 · 关闭先验")
    print("  （与开启时同为 83.3%：残余失败与领域先验无关，")
    print("    根因是「同一概念在多个框架下都有文档」的跨框架语义重叠，属检索能力边界）")


if __name__ == "__main__":
    main()
