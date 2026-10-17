"""领域路由评测：显式线索路由 vs 基线（vs 已证伪的语义路由）。

背景 —— 一个「不是排序能解决」的短板：

知识库有 7 个互不相干的知识域。问「Spring 的依赖注入有哪几种方式」时，
检索要在全库找「依赖注入」，而 FastAPI 的 dependencies 文档、本项目代码的
中文注释里都在讲依赖注入；同理「怎么配置 CORS」在 Spring WebFlux 和
FastAPI 下各有一份文档。

本脚本对比四条路线：

  A 基线（无路由）        —— 全局检索，即改造前的线上链路
  B 语义路由（对照）      —— 用领域质心向量判问题属于哪个域（**已证伪**：
                            本项目代码本身就是用 LangChain 写的，术语与
                            langchain 域高度重叠，质心分不开）
  C 线索路由              —— 认问题里的显式线索（框架名 / 本项目自称 / 代码符号），
                            唯一命中则限定在该域内检索
  D 线索路由 + 加权 RRF   —— 同 C，但把改写路权重降到 0.5（改写是有损派生）

同时统计线索判定的分布：唯一命中 / 歧义（应触发澄清追问）/ 无线索（走全局）。

用法：
    python eval_routing.py
除查询改写（带磁盘缓存）与语义路由对照外不调用 LLM。
"""

from __future__ import annotations

from app import domain as dm
from app.config import STORAGE_DIR, settings
from app.llm import get_embeddings
from app.retrieval import Retriever
from app.vectorstore import load_vectorstore
from eval_retrieval import (
    DOCS_EVAL_SET,
    EVAL_SET,
    _hit_file,
    _hit_symbol,
    make_cached_translator,
)


def _bucket() -> dict:
    return {"file": [], "sym": [], "frank": [], "srank": []}


def _record(bucket: dict, docs, answer_file, answer_name) -> None:
    file_rank = sym_rank = None
    for i, doc in enumerate(docs, start=1):
        if file_rank is None and _hit_file(doc, answer_file):
            file_rank = i
        if sym_rank is None and _hit_symbol(doc, answer_file, answer_name):
            sym_rank = i
        if file_rank is not None and sym_rank is not None:
            break
    bucket["file"].append(file_rank is not None)
    bucket["sym"].append(sym_rank is not None)
    bucket["frank"].append(1.0 / file_rank if file_rank else 0.0)
    bucket["srank"].append(1.0 / sym_rank if sym_rank else 0.0)


def _metrics(bucket: dict) -> dict:
    total = len(bucket["file"])
    return {
        "file": sum(bucket["file"]) / total,
        "sym": sum(bucket["sym"]) / total,
        "fmrr": sum(bucket["frank"]) / total,
        "smrr": sum(bucket["srank"]) / total,
        "fh": sum(bucket["file"]),
        "sh": sum(bucket["sym"]),
        "n": total,
    }


def _fmt(m: dict) -> str:
    return (
        f"{m['file'] * 100:>6.1f}% ({m['fh']:>2d}/{m['n']})"
        f"{m['sym'] * 100:>9.1f}% ({m['sh']:>2d}/{m['n']})"
        f"{m['fmrr']:>9.3f}{m['smrr']:>9.3f}"
    )


def target_domain(answer) -> str:
    """从标准答案路径推出它属于哪个域。"""
    first = answer[0] if isinstance(answer, (list, tuple)) else answer
    return first.split("/")[0] if "/" in first else "project"


def main() -> None:
    top_k = settings.top_k
    store = load_vectorstore()
    embeddings = get_embeddings()
    translate = make_cached_translator()

    print(f"知识库片段数：{store.index.ntotal}    top_k={top_k}")
    print("=" * 96)

    strategies = {
        "A 基线（无路由）": Retriever(
            store,
            top_k=top_k,
            use_query_rewrite=True,
            use_domain_prior=True,
            translator=translate,
        ),
        "C 线索路由": Retriever(
            store,
            top_k=top_k,
            use_query_rewrite=True,
            use_domain_prior=True,
            use_domain_route=True,
            translator=translate,
        ),
        "D 线索路由+加权RRF": Retriever(
            store,
            top_k=top_k,
            use_query_rewrite=True,
            use_domain_prior=True,
            use_domain_route=True,
            rewrite_weight=0.5,
            translator=translate,
        ),
    }

    # ---- 语义路由对照（已证伪，留作证据）----
    centroids = dm.load_centroids(STORAGE_DIR / dm.CENTROID_FILENAME)
    if centroids is None:
        vectors, docs = dm.collect_vectors(store)
        centroids = dm.compute_centroids(vectors, docs)
        dm.save_centroids(centroids, STORAGE_DIR / dm.CENTROID_FILENAME)
    semantic_stores = dm.build_domain_stores(store)

    for label, eval_set in (("项目代码域", EVAL_SET), ("官方文档域", DOCS_EVAL_SET)):
        buckets = {name: _bucket() for name in strategies}
        buckets["B 语义路由（对照）"] = _bucket()

        unique = ambiguous = no_hint = correct = 0
        ambiguous_questions: list[str] = []
        misses: dict[str, list[str]] = {name: [] for name in strategies}
        router = strategies["C 线索路由"]

        for question, answer, answer_name in eval_set:
            for name, retriever in strategies.items():
                docs = retriever.retrieve(question)
                _record(buckets[name], docs, answer, answer_name)
                if not any(_hit_file(d, answer) for d in docs):
                    misses[name].append(question)

            # B 语义路由：用问题向量判域，再去子库检索
            query_vector = embeddings.embed_query(question)
            route, _scores = dm.route_query(query_vector, centroids)
            sub = semantic_stores.get(route) if route else None
            if sub is not None:
                routed = sub.similarity_search_by_vector(query_vector, k=top_k)
            else:
                routed = store.similarity_search(question, k=top_k)
            _record(buckets["B 语义路由（对照）"], routed, answer, answer_name)

            # 线索判定分布
            decision = router.route(question)
            if decision.is_ambiguous:
                ambiguous += 1
                ambiguous_questions.append(question)
            elif decision.primary:
                unique += 1
                if decision.primary == target_domain(answer):
                    correct += 1
            else:
                no_hint += 1

        print(f"【{label}】{len(eval_set)} 条")
        summary = f"  线索判定：唯一命中 {unique} / 歧义 {ambiguous} / 无线索 {no_hint}"
        if unique:
            summary += f"      唯一命中中方向正确 {correct}/{unique}"
        print(summary)
        print(f"  {'策略':<20s}{'文件级':>14s}{'符号级':>16s}{'文件MRR':>9s}{'符号MRR':>9s}")
        for name in list(strategies) + ["B 语义路由（对照）"]:
            print(f"  {name:<20s}{_fmt(_metrics(buckets[name]))}")
        if misses["C 线索路由"]:
            print("  C 线索路由未命中：")
            for item in misses["C 线索路由"]:
                print(f"    - {item}")
        if ambiguous_questions:
            print("  会触发澄清追问的问题：")
            for item in ambiguous_questions:
                print(f"    - {item}")
        print()

    # ---- 澄清追问行为验证 ----
    # 评测集里的问题只带单一框架名，不会触发歧义，所以单独用探针验证。
    print("【澄清追问行为验证】")
    router = strategies["C 线索路由"]
    for probe in (
        "Spring 和 FastAPI 的依赖注入有什么区别？",
        "LangChain 和 LangGraph 有什么区别？",
        "Vue 和 Spring 各自怎么处理请求？",
        "怎么配置跨域 CORS？",
        "本项目的检索链路是什么？",
    ):
        decision = router.route(probe)
        if decision.is_ambiguous:
            state = "触发澄清"
        elif decision.primary:
            state = f"路由到 {decision.primary}"
        else:
            state = "无线索 -> 全局检索"
        print(f"  [{state:22s}] {probe}")
    print()

    print("说明：符号级对官方文档域恒为 0（文档没有函数级答案），只看文件级。")


if __name__ == "__main__":
    main()
