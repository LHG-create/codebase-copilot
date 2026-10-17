"""代码知识库问答的评测集与基线评测脚本。

评测逻辑：
  每条评测项 = 一个问题 + 一个「正确答案所在的文件」(+ 可选的符号名)。
  判定标准：检索 top-k 个片段里，是否命中「正确答案所在的文件 / 符号」。

指标：
  - Hit Rate@k：正确答案出现在前 k 个检索结果里的比例（k 默认 4）
  - MRR：正确答案排名倒数的平均（越接近 1 越好，反映排得有多靠前）

评测集分两个域，**任何检索优化都必须两个域同时不退化**：
  - 项目代码域（40 条）：本项目自己的源码，答案精确到函数/类/变量
  - 官方文档域（24 条）：FastAPI / Spring / Vue / LangChain / LangGraph 官方文档

之所以要双域：知识库里 99.2% 的片段是官方文档，只有 0.8% 是本项目代码。
如果只在一个域上调参，很容易把一个域的指标做上去、另一个域做塌。

用法：
  python eval_retrieval.py
只依赖已建好的向量库（storage/），不调用 LLM，纯离线、可复现。
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings
from app.llm import get_embeddings
from app.vectorstore import load_vectorstore

PROJECT_ROOT = Path(__file__).resolve().parent
REWRITE_CACHE = PROJECT_ROOT / ".eval_cache" / "query_rewrites.json"


def make_cached_translator():
    """带磁盘缓存的查询改写函数（评测用）。

    线上链路含「中文问题翻译成英文关键词」这一步，它要调一次大模型。
    直接放进评测会让结果不可复现、也拖慢每次跑；缓存到磁盘后，
    同一问题只翻译一次，评测依然可离线复现。
    """
    from app.query_rewriter import translate_to_english

    REWRITE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, str] = {}
    if REWRITE_CACHE.exists():
        try:
            cache = json.loads(REWRITE_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}

    def translate(question: str) -> str:
        if question in cache:
            return cache[question]
        english = translate_to_english(question)
        cache[question] = english
        REWRITE_CACHE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return english

    return translate

# 评测集：问题 -> (答案文件, 答案函数/类名)
# 答案函数名用 "" 表示「答案在整个文件级别」（比如模块头、整体说明）
EVAL_SET: list[tuple[str, str, str]] = [
    # —— 链的组装 chain.py ——
    ("RAG 问答链是怎么把检索和生成串起来的？", "chain.py", "RagAssistant.__init__"),
    ("回答时怎么把检索到的片段标上编号？", "chain.py", "format_docs"),
    ("assistant 的 answer 方法返回什么结构？", "chain.py", "RagAssistant.answer"),
    ("系统提示词在哪里定义，要求助手遵守什么规则？", "chain.py", "SYSTEM_PROMPT"),
    ("构建 assistant 的入口函数是什么？", "chain.py", "build_assistant"),
    ("RagAssistant 类的整体职责是什么？", "chain.py", "RagAssistant"),
    ("提示词模板 PROMPT 是怎么定义的？", "chain.py", "PROMPT"),
    # —— 模型工厂 llm.py ——
    ("对话模型是怎么根据配置选择供应商的？", "llm.py", "get_chat_model"),
    ("向量模型和对话模型为什么要分开配置？", "config.py", "Settings"),
    ("get_embeddings 返回什么类型的对象？", "llm.py", "get_embeddings"),
    ("API Key 是怎么和 provider 对应起来的？", "llm.py", "_KEY_BY_PROVIDER"),
    ("ensure_configured 函数是干嘛的？", "llm.py", "ensure_configured"),
    # —— 向量库 vectorstore.py ——
    ("向量库是怎么保存的，用的什么后端？", "vectorstore.py", "build_vectorstore"),
    ("加载向量库时如果没建过会怎样？", "vectorstore.py", "load_vectorstore"),
    ("向量库的索引文件名是什么？", "vectorstore.py", "INDEX_NAME"),
    # —— 配置 config.py ——
    ("项目的全局配置从哪里读取，默认对话模型是什么？", "config.py", "Settings"),
    ("切分片段大小和检索 top_k 的默认值是多少？", "config.py", "Settings"),
    ("数据目录 DATA_DIR 是怎么定义的？", "config.py", "DATA_DIR"),
    ("向量库存储目录 STORAGE_DIR 指向哪里？", "config.py", "STORAGE_DIR"),
    ("settings 这个全局对象是怎么创建的？", "config.py", "settings"),
    # —— 代码切分器 code_splitter.py ——
    ("代码文件是怎么按函数粒度切分的？", "code_splitter.py", "split_code_file"),
    ("为什么不用通用切分器切代码？", "code_splitter.py", "split_code_file"),
    ("函数签名是怎么从 AST 还原出来的？", "code_splitter.py", "_signature_of"),
    ("docstring 是怎么从函数里提取的？", "code_splitter.py", "_get_docstring"),
    ("代码单元 _CodeUnit 包含哪些字段？", "code_splitter.py", "_CodeUnit"),
    ("_collect_units 是怎么遍历 AST 收集单元的？", "code_splitter.py", "_collect_units"),
    # —— 文档加载 loader.py ——
    ("markdown 里的代码块是怎么被保护不被切碎的？", "loader.py", "_split_text_protecting_code"),
    ("哪些文件类型是支持加载的？", "loader.py", "SUPPORTED_SUFFIXES"),
    ("load_documents 是怎么读取文档的？", "loader.py", "load_documents"),
    ("split_documents 是怎么区分代码和文档的？", "loader.py", "split_documents"),
    ("_FENCE_RE 这个正则表达式是干嘛的？", "loader.py", "_FENCE_RE"),
    # —— 混合检索 hybrid_retriever.py ——
    ("混合检索是怎么融合向量和 BM25 的？", "hybrid_retriever.py", "HybridRetriever"),
    ("RRF 融合算法是怎么算的？", "hybrid_retriever.py", "_rrf_fuse"),
    ("BM25 的分词函数是怎么处理中文的？", "hybrid_retriever.py", "HybridRetriever._bm25_preprocess"),
    ("代码片段给 BM25 的文本是怎么构造的？", "hybrid_retriever.py", "_code_tokens"),
    ("build_hybrid_retriever 是干嘛的？", "hybrid_retriever.py", "build_hybrid_retriever"),
    # —— 入口 ingest.py / chat_cli.py / app_ui.py ——
    ("建库的入口脚本做了什么？", "ingest.py", "main"),
    ("命令行问答怎么循环接收用户输入？", "chat_cli.py", "main"),
    ("Streamlit 界面的 load_assistant 是干嘛的？", "app_ui.py", "load_assistant"),
    ("网页界面是怎么渲染出处的？", "app_ui.py", "render_sources"),
]


def _load_eval_set() -> list[tuple[str, str, str]]:
    return EVAL_SET


# 官方文档域评测集：(问题, 答案文档路径后缀, 符号名[文档无符号，固定为空])
# 路径用「后缀」形式，避免不同领域下的同名文件（如多个 index.md）互相误判。
DOCS_EVAL_SET: list[tuple[str, str | tuple[str, ...], str]] = [
    # —— FastAPI ——
    ("FastAPI 里怎么声明路径参数？", "fastapi/tutorial/path-params.md", ""),
    # 问题同时问「声明」和「校验」：query-params.md 只覆盖前者，
    # query-params-str-validations.md 两者都讲（实测它排第一），两页都算对。
    (
        "FastAPI 的查询参数怎么声明和做校验？",
        ("fastapi/tutorial/query-params.md", "fastapi/tutorial/query-params-str-validations.md"),
        "",
    ),
    ("FastAPI 怎么用 Pydantic 模型定义请求体？", "fastapi/tutorial/body.md", ""),
    ("FastAPI 项目变大后怎么拆分多个路由文件？", "fastapi/tutorial/bigger-applications.md", ""),
    ("FastAPI 怎么配置跨域 CORS？", "fastapi/tutorial/cors.md", ""),
    ("FastAPI 怎么接收表单数据？", "fastapi/tutorial/request-forms.md", ""),
    ("FastAPI 的依赖注入 Depends 是怎么用的？", "fastapi/tutorial/dependencies/index.md", ""),
    # —— Spring Framework ——
    ("Spring MVC 的 @RequestParam 注解是做什么的？", "spring-framework/web/webmvc/mvc-controller/ann-methods/requestparam.adoc", ""),
    # arguments.adoc 是「控制器方法支持哪些参数类型」的清单（what），
    # 而 mvc-data-binding.adoc 讲的才是「参数怎么绑定到对象」（how）——问题问的是机制。
    (
        "Spring MVC 控制器方法的参数是怎么绑定的？",
        (
            "spring-framework/web/webmvc/mvc-controller/ann-methods/arguments.adoc",
            "spring-framework/web/webmvc/mvc-data-binding.adoc",
        ),
        "",
    ),
    ("Spring MVC 里 @RequestMapping 是怎么映射请求的？", "spring-framework/web/webmvc/mvc-controller/ann-requestmapping.adoc", ""),
    ("Spring 的 bean 作用域有哪几种？", "spring-framework/core/beans/factory-scopes.adoc", ""),
    # 原答案指向 dependencies.adoc，但它只是 474 字节的章节导语页
    # （正文是「接下来这一节将介绍…」），不含任何 DI 方式的说明。
    # 真正讲「构造器注入 / Setter 注入」的是 factory-collaborators.adoc，
    # 它的标题就是 "= Dependency Injection"。这是修正的第二例
    # 「评测标注指向导语页」，与 ADR-012 的 retrieval.md 空壳同族。
    ("Spring 的依赖注入有哪几种方式？", "spring-framework/core/beans/dependencies/factory-collaborators.adoc", ""),
    ("Spring MVC 里怎么做数据绑定？", "spring-framework/web/webmvc/mvc-data-binding.adoc", ""),
    # —— Vue.js ——
    # reactivity-fundamentals 讲 ref/reactive 的**用法**，
    # reactivity-in-depth 讲 Proxy 与依赖收集的**实现原理**——问题问的是「怎么实现的」。
    (
        "Vue 3 的响应式系统是怎么实现的？",
        (
            "vuejs/guide/essentials/reactivity-fundamentals.md",
            "vuejs/guide/extras/reactivity-in-depth.md",
        ),
        "",
    ),
    ("Vue 的 computed 计算属性有什么用？", "vuejs/guide/essentials/computed.md", ""),
    ("Vue 组件有哪些生命周期钩子？", "vuejs/guide/essentials/lifecycle.md", ""),
    ("Vue 的 watch 监听器怎么用？", "vuejs/guide/essentials/watchers.md", ""),
    ("Vue 表单输入怎么做双向绑定？", "vuejs/guide/essentials/forms.md", ""),
    # —— LangChain ——
    ("LangChain 的 agent 是怎么工作的？", "langchain/agents.md", ""),
    # 原答案指向 langchain/retrieval.md，但它全文只有 96 字节 ——
    # 只剩一段 frontmatter，正文一个字都没有（官网自己标了 noindex: true）。
    # 这条评测项从写下第一天起就注定失败，且与检索算法无关，是数据源的坑（ADR-012）。
    # LangChain 真正的 RAG 内容在 knowledge-base.md。
    ("LangChain 怎么做检索增强生成 RAG？", "langchain/knowledge-base.md", ""),
    ("LangChain 的短期记忆是怎么实现的？", "langchain/short-term-memory.md", ""),
    # —— LangGraph ——
    ("LangGraph 的检查点 checkpointer 是用来做什么的？", "langgraph/checkpointers.md", ""),
    ("LangGraph 怎么持久化图的状态？", "langgraph/persistence.md", ""),
    ("LangGraph 怎么中断执行、等待人工输入？", "langgraph/interrupts.md", ""),
]


def _match_one_source(doc: Document, answer_file: str) -> bool:
    """单个答案的路径匹配。

    answer_file 支持两种写法：
      - 文件名（"chain.py"）：用于项目代码域
      - 路径后缀（"fastapi/tutorial/path-params.md"）：用于官方文档域，
        这样多个领域下的同名文件（如各种 index.md）不会互相误判。
    """
    if not answer_file:
        return False
    if "/" not in answer_file:
        return doc.metadata.get("file_name", "") == answer_file
    source = str(doc.metadata.get("source", "")).replace("\\", "/")
    return source.endswith("/" + answer_file)


def _hit_file(doc: Document, answer_file) -> bool:
    """文件级命中：检索片段来自正确答案所在文件。

    **answer_file 允许是「可接受答案集合」（list / tuple）。**

    为什么需要这个？因为单一标准答案对「泛化问题」是不成立的。
    实测发现三条评测项的原答案都只覆盖了问题的一部分：
      - 「查询参数怎么声明**和**做校验」只标了只讲"声明"的那页
      - 「参数是怎么**绑定**的」只标了讲"支持哪些参数类型"的那页（问机制答清单）
      - 「响应式系统是怎么**实现**的」只标了讲用法的页（问实现答用法）
    单一标注会把这些判为「检索失败」，但检索其实挑出了更完整的文档 ——
    这是评测在说谎，不是检索出错（详见 ADR-012）。
    """
    if isinstance(answer_file, (list, tuple, set)):
        return any(_match_one_source(doc, a) for a in answer_file)
    return _match_one_source(doc, answer_file)


def _hit_symbol(doc: Document, answer_file: str, answer_name: str) -> bool:
    """符号级命中：检索片段精确命中正确答案所在的函数/类/变量。"""
    if not _hit_file(doc, answer_file):
        return False
    if answer_name == "":
        return True
    return doc.metadata.get("name", "") == answer_name


def _evaluate_with(retriever, eval_set: list[tuple[str, str, str]]) -> dict:
    """用给定 retriever 跑评测，返回指标。"""
    file_hits = 0
    sym_hits = 0
    file_rrs: list[float] = []
    sym_rrs: list[float] = []
    details: list[dict] = []

    for question, answer_file, answer_name in eval_set:
        docs = retriever.invoke(question)

        file_rank = None
        sym_rank = None
        for i, doc in enumerate(docs, start=1):
            if file_rank is None and _hit_file(doc, answer_file):
                file_rank = i
            if sym_rank is None and _hit_symbol(doc, answer_file, answer_name):
                sym_rank = i
            if file_rank is not None and sym_rank is not None:
                break

        file_rrs.append(1.0 / file_rank if file_rank else 0.0)
        sym_rrs.append(1.0 / sym_rank if sym_rank else 0.0)
        if file_rank:
            file_hits += 1
        if sym_rank:
            sym_hits += 1

        details.append(
            {
                "question": question,
                "answer": f"{answer_file}:{answer_name or '(file)'}",
                "file_rank": file_rank,
                "symbol_rank": sym_rank,
            }
        )

    total = len(eval_set)
    return {
        "total": total,
        "file_hit_rate": file_hits / total,
        "symbol_hit_rate": sym_hits / total,
        "file_mrr": sum(file_rrs) / total,
        "symbol_mrr": sum(sym_rrs) / total,
        "file_hits": file_hits,
        "symbol_hits": sym_hits,
        "details": details,
    }


def evaluate(k: int = 4, eval_set: list[tuple[str, str, str]] | None = None) -> dict:
    """跑完整评测，返回「文件级」与「符号级」两套指标。

    文件级命中（宽松）：前 k 个结果里有正确答案所在文件。
    符号级命中（严格）：前 k 个结果里有精确的那个函数/类/变量。

    代码 RAG 的合理评估应两级都看：文件级反映「能不能定位到对的文件」，
    符号级反映「能不能精确定位到对的位置」。
    """
    store = load_vectorstore()
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return _evaluate_with(retriever, eval_set or _load_eval_set())


def evaluate_hybrid(k: int = 4, eval_set: list[tuple[str, str, str]] | None = None) -> dict:
    """用混合检索（向量 + BM25 + RRF）跑同一评测集，用于对比。"""
    from app.hybrid_retriever import build_hybrid_retriever

    store = load_vectorstore()
    retriever = build_hybrid_retriever(store, k=k)
    return _evaluate_with(retriever, eval_set or _load_eval_set())


def _fmt(ratio: float) -> str:
    return f"{ratio * 100:.1f}%"


def _row(name: str, v: dict) -> str:
    return (
        f"  {name:<16s} 文件级 {_fmt(v['file_hit_rate']):>7s} ({v['file_hits']}/{v['total']})"
        f"  |  符号级 {_fmt(v['symbol_hit_rate']):>7s} ({v['symbol_hits']}/{v['total']})"
        f"  |  MRR {v['file_mrr']:.3f}/{v['symbol_mrr']:.3f}"
    )


def main() -> None:
    from app.hybrid_retriever import build_hybrid_retriever
    from app.retrieval import Retriever

    k = settings.top_k
    store = load_vectorstore()

    # 旧口径：纯向量 top-k（不经改写、不经 RRF、不经重排）
    vec_retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # 线上真实链路：领域路由 + 查询改写 + 双路 RRF + 领域先验重排
    #
    # `use_domain_route=True` 必须显式传：Retriever 自身的默认值是 False，
    # 那是为了让诊断脚本默认走「不带路由的纯检索」以便做对照。
    # 漏传过一次，结果这里测出的分数比线上真实表现低一大截（87.5% vs 95.8%）
    # —— 一个「评测跑的不是线上链路」的口径错误，比模型答错更隐蔽。
    online_retriever = Retriever(
        store,
        top_k=k,
        translator=make_cached_translator(),
        use_domain_route=True,
    )
    # 对照：混合检索（向量 + BM25 + RRF），评测显示持续负优化，默认关闭
    hybrid_retriever = build_hybrid_retriever(store, k=k)

    print(f"评测集：项目代码 {len(EVAL_SET)} 条 + 官方文档 {len(DOCS_EVAL_SET)} 条"
          f"  |  top_k={k}  recall_k={online_retriever.recall_k}"
          f"  alpha={settings.domain_prior_alpha}")
    print("=" * 84)

    for label, eval_set in (("项目代码域", EVAL_SET), ("官方文档域", DOCS_EVAL_SET)):
        rows = [
            ("纯向量（旧口径）", _evaluate_with(vec_retriever, eval_set)),
            ("混合检索（对照）", _evaluate_with(hybrid_retriever, eval_set)),
            ("线上链路（最终）", _evaluate_with(online_retriever, eval_set)),
        ]
        print(f"【{label}】{len(eval_set)} 条")
        base = rows[0][1]
        for name, metrics in rows:
            d_file = (metrics["file_hit_rate"] - base["file_hit_rate"]) * 100
            d_sym = (metrics["symbol_hit_rate"] - base["symbol_hit_rate"]) * 100
            suffix = "" if name.startswith("纯向量") else f"   ({d_file:+.1f}pp / {d_sym:+.1f}pp)"
            print(_row(name, metrics) + suffix)
        misses = [d for d in rows[-1][1]["details"] if d["file_rank"] is None]
        if misses:
            print(f"    线上链路下文件级未命中 {len(misses)} 条：")
            for m in misses:
                print(f"      ✗ {m['question']}  （期望 {m['answer']}）")
        print()

    print("提示：官方文档片段的符号级命中等同于文件级（文档没有函数符号）。")
    print("      项目代码域的「符号级」才是真正严格的指标（精确到函数/类/变量）。")


if __name__ == "__main__":
    main()
