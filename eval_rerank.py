"""重排序（rerank）对照实验 —— 一个被数据证伪的方案。

## 实验要回答的问题

`_diag3` 式诊断发现：官方文档域 7 条未命中里，6 条的正确答案**就在 96 条召回池里**，
只是排在第 14~56 名。粗排把正确答案召回了，精排应该能把它挑上来 ——
这正是 rerank（cross-encoder 精排）的教科书场景，于是做了这个实验。

## 实测结论：**rerank 不值得引入**（两域都没有正收益）

| 域 | 策略 | 文件级 | 符号级 | MRR |
|---|---|---:|---:|---:|
| 项目代码域 | A 现状（无精排） | 97.5% | 80.0% | 0.823 |
| | B 精排（中文 query） | **100.0%** | **85.0%** | **0.869** |
| | C 精排（英文改写 query） | 95.0% | 82.5% | 0.794 |
| 官方文档域 | A 现状 | **83.3%** | — | **0.649** |
| | B 精排（中文 query） | 79.2% | — | 0.618 |
| | C 精排（英文改写 query） | 75.0% | — | 0.538 |

三点发现：

1. **精排对「项目代码」是正优化**：文件级 +2.5pp、符号级 +5.0pp（97.5%/80.0% → 100%/85%）。
   合理 —— 本项目代码的注释和 docstring 是**中文**，`gte-rerank-v2` 也是中文强项，
   两边语言对齐，精排判断得准。
2. **精排对「跨语言文档检索」是负优化**：83.3% → 79.2%。
   中文问题配英文文档，精排模型的跨语言匹配能力不如多语言向量模型
   （`text-embedding-v3`），把正确答案压了下去。
3. **把英文改写当精排 query 反而更差**（两个域都降）。这和它在**检索**里的作用正好相反：
   检索只需要「关键词」（短语命中率高），精排需要「完整语义」（改写丢掉的信息正是它要用的）。

> **注意：这套数字与首次实验（同一个脚本）不同。** 首次跑时精排在两个域都是负优化，
> 当时知识库里的项目代码副本还是**过期且不完整**的。
> 副本修对后，项目域的精排收益就显现出来了。
> **同一个实验、同一个模型，数据一变结论就变** —— 这也是 ADR-012 教训 5 的又一个例证。

**结论：仍然不引入。** 它是一次「此消彼长」（项目域 +5pp 换文档域 -4.1pp），
而项目「双域都不牺牲」的原则不允许这种交换；
更何况要单独在项目域启用它，需要先准确判断问题属于哪个领域 ——
而 ADR-011 已经证实**语义路由只有 65% 准确率**，判错就是全盘皆输。
本脚本保留下来作为可复现证据。

用法：python eval_rerank.py（需要 DASHSCOPE_API_KEY；结果会缓存到 .eval_cache/rerank.json）
"""

from __future__ import annotations

import json
from pathlib import Path

from dashscope import TextReRank

from app import domain as dm
from app.config import settings
from app.retrieval import Retriever, rrf_fuse
from app.vectorstore import load_vectorstore
from eval_retrieval import (
    DOCS_EVAL_SET,
    EVAL_SET,
    _hit_file,
    _hit_symbol,
    make_cached_translator,
)

CACHE_PATH = Path(__file__).resolve().parent / ".eval_cache" / "rerank.json"
CANDIDATE_K = 48  # 送去精排的候选数（精排按量计费，不宜把整个召回池都送进去）
DOC_CHARS = 600  # 每条候选送给精排的截断长度

_cache: dict[str, dict[str, float]] | None = None


def _doc_key(doc) -> str:
    return doc.metadata.get("source", "") + "::" + doc.page_content[:80]


def _load_cache() -> dict[str, dict[str, float]]:
    global _cache
    if _cache is None:
        if CACHE_PATH.exists():
            try:
                _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(_load_cache(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def rerank_scores(query: str, docs: list) -> list[float]:
    """对 docs 逐条打相关性分。带磁盘缓存，让实验可离线复现、不重复计费。"""
    cache = _load_cache()
    entry = cache.setdefault(query, {})

    missing = [d for d in docs if _doc_key(d) not in entry]
    if missing:
        try:
            resp = TextReRank.call(
                model="gte-rerank-v2",
                query=query,
                documents=[d.page_content[:DOC_CHARS] for d in missing],
                top_n=len(missing),
                return_documents=False,
                api_key=settings.dashscope_api_key,
            )
            if resp.status_code != 200:
                print(f"  [warn] rerank 失败 {resp.status_code}，降级为原序")
                return [entry.get(_doc_key(d), 0.0) for d in docs]
            for item in resp.output.results:
                entry[_doc_key(missing[item.index])] = item.relevance_score
            _save_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] rerank 异常，降级为原序: {exc}")
            return [entry.get(_doc_key(d), 0.0) for d in docs]

    return [entry.get(_doc_key(d), 0.0) for d in docs]


def rerank_top(docs: list, scores: list[float], top_k: int) -> list:
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in order[:top_k]]


def dedup_by_file(docs: list, top_k: int, max_per_file: int = 2) -> list:
    """同一个文件最多占 max_per_file 席，避免一个文件把 top_k 刷屏。"""
    picked: list = []
    seen: dict[str, int] = {}
    for doc in docs:
        key = doc.metadata.get("source", "")
        if seen.get(key, 0) >= max_per_file:
            continue
        seen[key] = seen.get(key, 0) + 1
        picked.append(doc)
        if len(picked) == top_k:
            break
    return picked


def evaluate(builder, eval_set, with_symbol: bool) -> dict:
    file_hits = sym_hits = 0
    file_rr = 0.0
    for q, af, an in eval_set:
        docs = builder(q)
        fr = next((i for i, d in enumerate(docs, 1) if _hit_file(d, af)), None)
        if fr:
            file_hits += 1
            file_rr += 1 / fr
        if with_symbol:
            sr = next((i for i, d in enumerate(docs, 1) if _hit_symbol(d, af, an)), None)
            if sr:
                sym_hits += 1
    n = len(eval_set)
    return {"file": file_hits / n, "sym": sym_hits / n, "mrr": file_rr / n}


def main() -> None:
    translate = make_cached_translator()
    store = load_vectorstore()
    r = Retriever(store, translator=translate)
    prior = r._ensure_prior()
    k = r.top_k

    # 先把每条问题的「融合结果池」算好，四种策略共用，避免重复检索
    pools: dict[str, tuple[list, str]] = {}
    for eval_set in (EVAL_SET, DOCS_EVAL_SET):
        for q, _af, _an in eval_set:
            zh = store.similarity_search(q, k=r.recall_k)
            en_q = translate(q) if translate else ""
            en = store.similarity_search(en_q, k=r.recall_k) if en_q else []
            fused = rrf_fuse([zh, en]) if en else [(d, 0.0) for d in zh]
            fused = [
                (
                    d,
                    s + prior.get(dm.infer_domain(d.metadata.get("source", ""), d.metadata), 0.0),
                )
                for d, s in fused
            ]
            fused.sort(key=lambda p: p[1], reverse=True)
            pools[q] = ([d for d, _ in fused[:CANDIDATE_K]], (en_q or q))

    def make_builder(strategy: str):
        def build(q: str) -> list:
            docs, en_q = pools[q]
            if strategy == "A":
                return docs[:k]
            if strategy == "B":
                return rerank_top(docs, rerank_scores(q, docs), k)
            if strategy == "C":
                return rerank_top(docs, rerank_scores(en_q, docs), k)
            if strategy == "D":
                return dedup_by_file(rerank_top(docs, rerank_scores(en_q, docs), k * 3), k, 2)
            raise ValueError(strategy)

        return build

    labels = {
        "A": "A 现状（无精排）",
        "B": "B 精排（中文 query）",
        "C": "C 精排（英文改写 query）",
        "D": "D C + 文件去冗余",
    }

    print(f"候选池 {CANDIDATE_K} 条 → 精排 → top_k={k}")
    for eval_set, with_symbol, name in (
        (EVAL_SET, True, "项目代码域"),
        (DOCS_EVAL_SET, False, "官方文档域"),
    ):
        print(f"\n【{name}】{len(eval_set)} 条")
        print(f"  {'策略':<24}{'文件级':>9}{'符号级':>9}{'MRR':>9}")
        for s in ("A", "B", "C", "D"):
            m = evaluate(make_builder(s), eval_set, with_symbol)
            sym = f"{m['sym'] * 100:>8.1f}%" if with_symbol else f"{'—':>9}"
            print(f"  {labels[s]:<24}{m['file'] * 100:>8.1f}%{sym}{m['mrr']:>9.3f}")
    print(
        "\n结论：精排对项目代码域是正优化（符号级 +5.0pp），"
        "对跨语言文档检索是负优化（-4.1pp）；此消彼长且无法可靠地只在一个域启用，"
        "故不引入。详见 ADR-012。"
    )


if __name__ == "__main__":
    main()
