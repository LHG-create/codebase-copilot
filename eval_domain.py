"""跨领域干扰的诊断与检索策略对比脚本。

背景 —— 一个被评测数据暴露出来的真实问题：

知识库从「只有本项目代码」扩到「再加 6 个官方框架文档」后，片段数涨到 9956，
其中**本项目代码只有 82 个片段（0.8%）**。结果是问本项目代码的问题时，
检索结果被官方文档挤占，指标从 文件级 97.5% / 符号级 87.5% 掉到 92.5% / 77.5%。

本脚本做两件事：

一、诊断：干扰到底是不是唯一原因？
   - 打印知识库的领域分布（0.8% vs 99.2%）
   - 「仅在项目子库里检索」得到**零干扰上限**，与全局检索对比

二、策略对比（两个域都要跑，避免只在一个域上调参过拟合）：
   A 纯向量            —— 也是老评测的口径
   B 改写 + 双路 RRF    —— 线上真实链路
   C B + 领域先验重排    —— 本次新增，纠正「大领域天然占优」
   D 质心语义路由        —— 曾经以为的解法，用数据证伪
   E 仅项目子库          —— 诊断用上限，不是可用策略

用法：
    python eval_domain.py
除查询改写（带磁盘缓存，只调一次模型）外不调用 LLM。
"""

from __future__ import annotations

from app import domain as dm
from app.config import STORAGE_DIR, settings
from app.llm import get_embeddings
from app.retrieval import Retriever, domain_prior_map
from app.vectorstore import load_vectorstore
from eval_retrieval import (
    DOCS_EVAL_SET,
    EVAL_SET,
    _hit_file,
    _hit_symbol,
    make_cached_translator,
)

# 待对比的领域先验强度（0 = 关闭）
ALPHA_GRID = [0.0, 0.02, 0.04, 0.06, 0.09]


# ------------------------------------------------------------------ 评测工具


def _new_bucket() -> dict:
    return {"file_hits": [], "sym_hits": [], "file_ranks": [], "sym_ranks": []}


def _record(bucket: dict, docs, answer_file: str, answer_name: str) -> None:
    file_rank = sym_rank = None
    for i, doc in enumerate(docs, start=1):
        if file_rank is None and _hit_file(doc, answer_file):
            file_rank = i
        if sym_rank is None and _hit_symbol(doc, answer_file, answer_name):
            sym_rank = i
        if file_rank is not None and sym_rank is not None:
            break
    bucket["file_hits"].append(file_rank is not None)
    bucket["sym_hits"].append(sym_rank is not None)
    bucket["file_ranks"].append(1.0 / file_rank if file_rank else 0.0)
    bucket["sym_ranks"].append(1.0 / sym_rank if sym_rank else 0.0)


def _metrics(bucket: dict) -> dict:
    total = len(bucket["file_hits"])
    return {
        "file": sum(bucket["file_hits"]) / total,
        "sym": sum(bucket["sym_hits"]) / total,
        "file_mrr": sum(bucket["file_ranks"]) / total,
        "sym_mrr": sum(bucket["sym_ranks"]) / total,
        "file_hits": sum(bucket["file_hits"]),
        "sym_hits": sum(bucket["sym_hits"]),
        "total": total,
    }


def _fmt(m: dict) -> str:
    return (
        f"{m['file'] * 100:>6.1f}% ({m['file_hits']:>2d}/{m['total']})"
        f"{m['sym'] * 100:>9.1f}% ({m['sym_hits']:>2d}/{m['total']})"
        f"{m['file_mrr']:>9.3f}{m['sym_mrr']:>9.3f}"
    )


# ------------------------------------------------------------------ 主流程


def main() -> None:
    top_k = settings.top_k
    store = load_vectorstore()
    embeddings = get_embeddings()

    print(f"知识库片段数：{store.index.ntotal}")
    print("=" * 92)

    # ---------------- 诊断 1：领域分布 ----------------
    vectors, docs = dm.collect_vectors(store)
    groups = dm.group_indices_by_domain(docs)
    sizes = {name: len(idx) for name, idx in groups.items()}
    print("【知识库领域分布】")
    for name, count in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"  {dm.DOMAIN_LABELS.get(name, name):22s} {count:>6d} 片段  ({count / len(docs) * 100:5.1f}%)")
    print()

    project_store = dm.build_domain_stores(store).get("project")

    # ---------------- 检索器 ----------------
    translate = make_cached_translator()
    retrievers = {
        "A 纯向量": Retriever(store, top_k=top_k, use_query_rewrite=False, use_domain_prior=False),
        "B 改写+双路RRF": Retriever(store, top_k=top_k, use_query_rewrite=True, use_domain_prior=False, translator=translate),
        "C B+领域先验": Retriever(store, top_k=top_k, use_query_rewrite=True, use_domain_prior=True, translator=translate),
    }

    # ---------------- 诊断 2：语义路由（对照，已证伪）----------------
    centroids = dm.compute_centroids(vectors, docs)
    dm.save_centroids(centroids, STORAGE_DIR / dm.CENTROID_FILENAME)
    dm_stores = dm.build_domain_stores(store)

    print("【对照：质心语义路由】判「问题属于哪个领域」这条路走不通 ——")
    print("  本项目代码本身就是用 LangChain 写的，术语高度重叠，质心分不开两者。")
    print()

    # ---------------- 双域评测 ----------------
    for label, eval_set in (("项目代码域", EVAL_SET), ("官方文档域", DOCS_EVAL_SET)):
        buckets = {name: _new_bucket() for name in retrievers}
        buckets["D 语义路由（对照）"] = _new_bucket()
        buckets["E 仅项目子库（上限）"] = _new_bucket()

        route_correct = 0

        for question, answer_file, answer_name in eval_set:
            for name, retriever in retrievers.items():
                _record(buckets[name], retriever.retrieve(question), answer_file, answer_name)

            # D 语义路由：用问题向量判领域，再去子库检索
            qvec = embeddings.embed_query(question)
            route, _scores = dm.route_query(qvec, centroids)
            if route == "project":
                route_correct += 1
            sub = dm_stores.get(route) if route else None
            routed = (
                sub.similarity_search_by_vector(qvec, k=top_k)
                if sub is not None
                else store.similarity_search(question, k=top_k)
            )
            _record(buckets["D 语义路由（对照）"], routed, answer_file, answer_name)

            # E 诊断上限：只在项目子库里检索
            if project_store is not None:
                proj = project_store.similarity_search(question, k=top_k)
            else:
                proj = [d for d in store.similarity_search(question, k=top_k * 8)
                        if dm.infer_domain(d.metadata.get("source", ""), d.metadata) == "project"][:top_k]
            _record(buckets["E 仅项目子库（上限）"], proj, answer_file, answer_name)

        print(f"【{label}】{len(eval_set)} 条"
              + (f"   （语义路由判对 {route_correct}/{len(eval_set)}）" if label == "项目代码域" else ""))
        print(f"  {'策略':<18s}{'文件级':>14s}{'符号级':>16s}{'文件MRR':>9s}{'符号MRR':>9s}")
        for name in list(retrievers) + ["D 语义路由（对照）", "E 仅项目子库（上限）"]:
            print(f"  {name:<18s}{_fmt(_metrics(buckets[name]))}")
        print()

    # ---------------- 领域先验强度扫描 ----------------
    print("【领域先验强度 alpha 扫描】（项目域文件级 / 官方文档域文件级）")
    for alpha in ALPHA_GRID:
        row = [f"  alpha={alpha:<7.3f}"]
        for eval_set in (EVAL_SET, DOCS_EVAL_SET):
            retriever = Retriever(
                store, top_k=top_k, use_query_rewrite=True,
                use_domain_prior=alpha > 0, domain_prior_alpha=alpha, translator=translate,
            )
            bucket = _new_bucket()
            for question, answer_file, answer_name in eval_set:
                _record(bucket, retriever.retrieve(question), answer_file, answer_name)
            m = _metrics(bucket)
            row.append(f"  {m['file'] * 100:5.1f}% / {m['sym'] * 100:5.1f}%")
        print("".join(row))
    print()

    prior = domain_prior_map(sizes, settings.domain_prior_alpha)
    print(f"【领域先验权重】alpha={settings.domain_prior_alpha}")
    for name, value in sorted(prior.items(), key=lambda kv: -kv[1]):
        print(f"  {dm.DOMAIN_LABELS.get(name, name):22s} +{value:.4f}")
    print("  说明：权重 = alpha * log(总片段数/该领域片段数) / log(总片段数)，"
          "片段越少的领域加成越高。")


if __name__ == "__main__":
    main()
