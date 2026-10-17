"""检索链路：查询改写 -> 双路 RRF -> 领域先验重排。

线上真正跑的检索是这样的：

    中文问题 ─┬─> 向量检索（原问题）
              │                                    ┌─> RRF 融合 ─> 领域先验重排 ─> top_k
              └─> 向量检索（英文改写后的关键词）──┘

**为什么要把这条链路从 chain.py 里抽出来？**

因为之前评测跑的是「纯向量 top-k」，而线上跑的是「中文改写 + 双路 RRF」，
两者根本不是一条链路。后果很严重：

  - 评测看不出查询改写的价值（它压根没走改写）
  - 线上真实效果没有数字支撑
  - 想优化时不知道该优化哪一段

这是「评测与实现脱节」的典型症状。抽成独立模块后，评测脚本 import 的就是
线上同一段代码，指标才真正代表用户体验。

三段各自的来历：
  1. 查询改写（ADR-006）：中文问题要命中英文文档，必须先翻译成英文关键词。
  2. 双路 RRF（ADR-010）：中英两路各自独立检索再融合，比拼成一条 query 更准；
     且单路召回池必须显著大于最终 top_k，否则「排名稍后但正确」的片段进不了融合池。
  3. 领域先验重排（ADR-011）：知识库里 99.2% 的片段是官方文档，本项目代码只占 0.8%。
     按领域规模做逆频率加权（规模越小、加权越高，类似 IDF 的思想），
     纠正「大领域天然占优」的检索偏差。
"""

from __future__ import annotations

import math
from collections import Counter

from langchain_core.documents import Document

from app import domain as dm
from app import domain_router as dr
from app.config import settings
from app.query_rewriter import contains_chinese, translate_to_english


def rrf_fuse(
    ranked_lists: list[list[Document]],
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[Document, float]]:
    """Reciprocal Rank Fusion：把多路检索结果按排名融合。

    score(doc) = Σ w_i / (rrf_k + rank_i)，rank 从 1 开始。
    只看排名、不看原始分值，因此天然免疫「两路分数量纲不同」的问题。

    weights: 每路的权重，默认等权。**为什么需要不等权？**

    原问题路表达的是用户的真实意图，改写路是它的有损派生 —— 可能丢词，
    也可能引入原文没有的概念。实测「LangChain 的 agent 是怎么工作的？」
    被改写成 "agent workflow reasoning action observation"（关键词堆砌）后，
    正确答案在改写路掉到第 95 位，而在原问题路排第 3 位；等权 RRF 求和后
    它仍然被挤出 top-8。降低改写路权重，等于告诉融合器：
    「原问题说相关，比改写说相关更可信」。
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}

    def _key(doc: Document) -> str:
        return doc.metadata.get("source", "") + "::" + doc.page_content[:80]

    for weight, docs in zip(weights, ranked_lists):
        for rank, doc in enumerate(docs, start=1):
            key = _key(doc)
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [(by_key[key], scores[key]) for key in ordered]


def domain_prior_map(
    sizes: dict[str, int],
    alpha: float,
    rare_ratio: float = 0.05,
) -> dict[str, float]:
    """按领域规模算「逆频率」加权表（只对长尾领域生效）。

    prior(领域) = alpha * log(总片段数 / 该领域片段数) / log(总片段数)

    直觉：本项目代码只有 82 个片段（0.8%），官方文档有 9800+ 个。
    在同一个向量空间里做 top-k，大领域天然占优 —— 会有大量语义相近的
    官方文档片段把少数但正确的代码片段挤出候选。给**长尾领域**一点加成，
    相当于告诉排序器：「能在一堆大领域里挤进候选的小领域片段，本身就更值得看」。

    **为什么限定 `rare_ratio`（默认只给占比 ≤5% 的领域加权）？**

    这是被一次端到端真机验证揪出来的坑：不加限制时，FastAPI（1071 片段 / 10.8%）
    拿到的先验比 Spring Framework（3591 片段 / 36.1%）更高，
    结果问「Spring MVC 的 @PathVariable 和 @RequestParam」时，
    top-4 被 FastAPI 的 `query-params.md` 占满，模型直接拒答。

    也就是说：**先验本该只用于纠正「0.8% vs 99.2%」这种极端失衡，
    一旦让它在中等领域之间重新分配权重，就会用一个新的偏差替换掉旧的。**
    限定 rare_ratio 后，只有真正的长尾（本项目代码、用户沉淀案例）获得加成，
    其余领域一律 0，先验不再干预它们之间的相对排序。
    """
    total = sum(sizes.values())
    if not sizes or total <= 1 or alpha <= 0:
        return {}
    denominator = math.log(total)
    prior: dict[str, float] = {}
    for name, count in sizes.items():
        if count / total > rare_ratio:
            continue
        prior[name] = alpha * math.log(total / max(count, 1)) / denominator
    return prior


class Retriever:
    """知识库检索器：查询改写 + 双路 RRF + 领域先验重排。"""

    def __init__(
        self,
        vectorstore,
        top_k: int | None = None,
        recall_k: int | None = None,
        use_query_rewrite: bool = True,
        use_domain_prior: bool = True,
        use_domain_route: bool = False,
        domain_prior_alpha: float | None = None,
        use_bm25: bool = False,
        translator=None,
        domain_prior_rare_ratio: float | None = None,
        rewrite_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        self.vectorstore = vectorstore
        self.top_k = top_k if top_k is not None else settings.top_k
        # 单路召回池必须显著大于最终 top_k：融合是多通道的，
        # 每路只取 top_k 会让「排名稍后但内容正确」的片段直接出局（ADR-010）。
        # 96 是双域评测扫出来的值（ADR-011）——池子小的时候，
        # 「领域先验」没有足够的候选可挑。
        self.recall_k = (
            recall_k if recall_k is not None else max(settings.recall_k, self.top_k * 4)
        )
        self.use_query_rewrite = use_query_rewrite
        # 允许注入改写函数：评测脚本借此挂上缓存，避免每次都真的调一次大模型，
        # 让「准链路评测」依然可离线复现。
        self._translator = translator or translate_to_english
        self.use_domain_prior = use_domain_prior
        self.domain_prior_alpha = (
            settings.domain_prior_alpha if domain_prior_alpha is None else domain_prior_alpha
        )
        self.domain_prior_rare_ratio = (
            settings.domain_prior_rare_ratio
            if domain_prior_rare_ratio is None
            else domain_prior_rare_ratio
        )
        self.use_bm25 = use_bm25
        self.rrf_k = rrf_k
        # 改写路相对原问题路的权重（1.0 = 等权）。改写是有损派生，
        # 实测把权重降到 0.5 能救回被改写噪声挤掉的正确答案，见 rrf_fuse 注释。
        self.rewrite_weight = rewrite_weight
        self.use_domain_route = use_domain_route
        self._prior: dict[str, float] | None = None
        self._bm25 = None
        # 领域路由的懒加载缓存：子库按需构建，符号表只算一次
        self._domain_stores: dict[str, object] = {}
        self._symbols: set[str] | None = None
        # 最近一次检索真实落到哪个领域（None = 全局检索），供前端展示路由结果
        self.last_route: str | None = None

    # ---------------------------------------------------------- 领域先验

    def _ensure_prior(self) -> dict[str, float]:
        """算一次领域规模分布，缓存住（近万片段的统计只需做一次）。"""
        if self._prior is None:
            sizes = Counter(
                dm.infer_domain(doc.metadata.get("source", ""), doc.metadata)
                for doc in dm.collect_documents(self.vectorstore)
            )
            self._prior = domain_prior_map(
                dict(sizes), self.domain_prior_alpha, self.domain_prior_rare_ratio
            )
        return self._prior

    # ---------------------------------------------------------- 领域路由

    def _project_symbols(self) -> set[str]:
        """本项目代码的符号表（懒加载，只算一次）。"""
        if self._symbols is None:
            self._symbols = dr.collect_project_symbols(self.vectorstore)
        return self._symbols

    def route(self, question: str) -> dr.RouteDecision:
        """判定问题属于哪个领域（只判定，不检索）。

        线索唯一命中 -> 返回该领域；线索指向多个领域 -> 歧义（调用方应当
        澄清而不是硬答）；没有线索 -> 空决策（老实走全局检索）。
        """
        if not self.use_domain_route:
            return dr.RouteDecision()
        return dr.decide_route(question, self._project_symbols())

    def _store_for(self, domain: str | None):
        """按需构建并缓存领域子库。

        懒加载的原因：给 7 个域各建一遍子库要读一次全量向量再建 7 个索引，
        而绝大多数会话只会碰到其中一两个域。
        """
        if not domain:
            return None
        if domain not in self._domain_stores:
            self._domain_stores.update(
                dm.build_domain_stores(self.vectorstore, only={domain})
            )
        return self._domain_stores.get(domain)

    # ---------------------------------------------------------- 检索

    def _bm25_retriever(self):
        """懒加载 BM25 路（对近万片段建索引较慢，只在显式开启时才构建）。"""
        if self._bm25 is None:
            from app.hybrid_retriever import build_hybrid_retriever

            self._bm25 = build_hybrid_retriever(self.vectorstore, k=self.recall_k)
        return self._bm25

    def _search_in(self, store, question: str, k: int) -> list[Document]:
        """在**一个**向量库上跑完整检索（改写 + 双路 RRF + 先验），返回前 k 条。

        抽出来是为了让「单域检索」和「跨域均衡检索」共用同一套打分逻辑 ——
        两处各写一份，迟早会在参数上漂移。
        """
        base_docs = store.similarity_search(question, k=self.recall_k)

        ranked_lists = [base_docs]
        weights = [1.0]

        # 中文问题额外走一遍「英文关键词」路：中文长句会被口语词稀释，
        # 难以命中英文文档；翻译成文档标题风格的英文术语后命中率高得多。
        if self.use_query_rewrite and contains_chinese(question):
            english = self._translator(question)
            if english:
                ranked_lists.append(store.similarity_search(english, k=self.recall_k))
                weights.append(self.rewrite_weight)

        # 可选第三路：BM25 关键词检索。
        # 双域评测显示它在「中文问题 + 英文文档」场景持续负优化（ADR-005），
        # 因此默认关闭，只作为可对照的开关保留。
        # 另：BM25 索引建在全局库上，域内检索时不参与，否则又把跨域干扰带回来了。
        if self.use_bm25 and store is self.vectorstore:
            ranked_lists.append(self._bm25_retriever().invoke(question))
            weights.append(1.0)

        if len(ranked_lists) == 1:
            fused = [(doc, 0.0) for doc in base_docs]
        else:
            fused = rrf_fuse(ranked_lists, rrf_k=self.rrf_k, weights=weights)

        if self.use_domain_prior:
            prior = self._ensure_prior()
            if prior:
                fused = [
                    (
                        doc,
                        score
                        + prior.get(
                            dm.infer_domain(doc.metadata.get("source", ""), doc.metadata),
                            0.0,
                        ),
                    )
                    for doc, score in fused
                ]
                fused.sort(key=lambda pair: pair[1], reverse=True)

        return [doc for doc, _ in fused[:k]]

    def _balanced_retrieve(self, question: str, domains: list[str]) -> list[Document]:
        """比较类问题的检索：每个涉事领域各取若干条，再合并。

        为什么不能只靠全局检索：「Spring 和 FastAPI 的依赖注入有什么区别？」
        走全局 top-4 时，四条**全部**来自 FastAPI —— 因为「依赖注入」这个词在
        FastAPI 文档里匹配度最高。模型手里只有一边的资料，只能回答「无法比较」。
        实测过，这不是模型的问题，是检索给错了料。

        所以比较类问题必须**按域配额**取：每个域至少 2 条，总数仍是 top_k。
        这样两边的技术栈都会进入上下文，模型才有得比。
        """
        subset = domains[: max(2, min(3, len(domains)))]
        per_domain = max(2, self.top_k // len(subset))

        picked: list[Document] = []
        for domain in subset:
            store = self._store_for(domain)
            if store is None:
                continue
            picked.extend(self._search_in(store, question, k=per_domain))

        if not picked:
            # 一个域都没建成子库（片段太少）→ 退回全局，别让比较类问题反而没结果
            return self._search_in(self.vectorstore, question, k=self.top_k)

        # 跨域合并后可能有重复（同一片段被两个域的子库都包含，理论上不会，但防御一下）
        seen: set[tuple[str, str]] = set()
        unique: list[Document] = []
        for doc in picked:
            key = (doc.metadata.get("source", ""), doc.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            unique.append(doc)

        return unique[: self.top_k]

    def retrieve(
        self,
        question: str,
        forced_domain: str | None = None,
    ) -> list[Document]:
        """检索 top_k 个片段。

        forced_domain: 强制在指定领域内检索。用户在澄清追问里点选了某个领域后，
        前端会带着这个约束把问题重放一遍，因此这里要能显式接受它。
        为 None 时按问题里的线索自动路由。

        三种走法：
        - 用户指定 / 线索唯一 → 限定在单域内检索
        - **比较类且有多个涉事领域 → 按域配额各取若干条再合并**（否则只能拿到一边的资料）
        - 其余（无线索）→ 全局检索
        """
        decision = None
        domain = forced_domain
        if domain is None and self.use_domain_route:
            decision = self.route(question)
            # 注意：比较类问题的 primary 也是 None，但它的处理方式和「无线索」不同，
            # 所以必须在下面单独判一次 comparison，不能只看 domain。
            domain = decision.primary

        if decision is not None and decision.comparison and len(decision.domains) >= 2:
            self.last_route = None  # 没有收窄到单域，界面不显示「已路由到某域」
            return self._balanced_retrieve(question, decision.domains)

        # 拿到目标领域的子库；路由无结果、或该域片段太少没单独建库时，
        # 退回全局检索 —— 这是「软路由」的兜底，保证最坏不比不做路由更差。
        store = self._store_for(domain) or self.vectorstore
        self.last_route = domain if store is not self.vectorstore else None
        return self._search_in(store, question, k=self.top_k)

    # 与 LangChain Retriever 的接口对齐，方便直接丢进评测脚本 / chain 里用
    def invoke(
        self,
        question: str,
        forced_domain: str | None = None,
        **kwargs,
    ) -> list[Document]:
        return self.retrieve(question, forced_domain=forced_domain)


def build_retriever(vectorstore, **kwargs) -> Retriever:
    return Retriever(vectorstore, **kwargs)
