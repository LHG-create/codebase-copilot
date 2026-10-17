"""领域感知检索（Domain-Aware Retrieval）。

为什么要做这件事 —— 一个被评测数据暴露出来的真实问题：

知识库一开始只有「本项目自己的代码」（约 100 个片段），后来陆续加入了
FastAPI / LangChain / LangGraph / Spring Boot / Spring Framework / Vue.js 六个
官方框架文档，片段数从 4 千涨到近万，其中 **98% 是官方文档**。

结果是：问本项目代码的问题（如「RagAssistant.answer 返回什么结构」）时，
检索结果被 FastAPI / Spring 的文档片段挤占，评测指标从
文件级 97.5% / 符号级 87.5% 掉到 92.5% / 77.5%。

这不是检索算法退化，而是「一次检索被迫横跨 6 个互不相干的知识域」：
「怎么处理参数」这个问题，在 FastAPI 的 path-params 文档和本项目的
code_splitter 里语义都近，但它们属于完全不同的技术栈，谁也替不了谁。

解决思路分三层：

1. **打标**：`infer_domain()` 从文档路径推断每个片段所属领域（建库/检索时执行）。
   这是本模块的**主职** —— 领域先验、子库切分、路由全都建立在它之上。
2. **子库切分**：`build_domain_stores()` 从主索引无损切分出各领域的子索引，
   供领域路由把检索限定在单一技术栈内使用（见 `domain_router.py` / ADR-013）。
3. **质心与语义路由**：`route_query()` 用「领域质心向量」判断问题在问哪个领域。
   质心 = 该领域全部片段向量的平均方向，用 faiss 的 `reconstruct_n` 直接从
   已建好的索引里现算，**零额外 embedding 调用**。

**关于语义路由：已被数据证伪，不要接线上。**

  ADR-011 用它做过初判，ADR-013 做了完整评测：项目代码域 97.5% → 65.0%、
  官方文档域 87.5% → 83.3%。原因是本项目代码本身就是用 LangChain 写的，
  术语与 langchain 域高度重叠，质心分不开两者。

  真正可用的路由方案是 `domain_router.py` 里的**显式线索路由** ——
  从问题里认框架名（用户往往已经说清了），而不是从内容里猜。
  本模块的质心函数保留作对照证据。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

# 官方文档领域 = fetch_docs.py 下载到 data/docs/<name>/ 的一级目录名
DOC_DOMAINS: tuple[str, ...] = (
    "fastapi",
    "langchain",
    "langgraph",
    "spring-boot",
    "spring-framework",
    "vuejs",
)

# 领域 -> 展示名（前端出处卡片 / 评测报告里用）
DOMAIN_LABELS: dict[str, str] = {
    "project": "本项目代码",
    "fastapi": "FastAPI",
    "langchain": "LangChain",
    "langgraph": "LangGraph",
    "spring-boot": "Spring Boot",
    "spring-framework": "Spring Framework",
    "vuejs": "Vue.js",
    "case": "沉淀案例",
    "other": "其他",
}

CENTROID_FILENAME = "domain_centroids.npz"


# ---------------------------------------------------------------- 1. 打标


def infer_domain(source: str = "", metadata: dict | None = None) -> str:
    """从片段来源推断所属领域。

    规则按优先级：用户案例 > 官方文档 > 本项目代码 > 其他。
    之所以靠路径而不是靠内容，是因为路径是建库时写死的、100% 可靠，
    而内容分类（哪怕是 LLM 判）都有出错空间，且会给每次检索加延迟。
    """
    meta = metadata or {}
    if meta.get("origin") == "user-case" or meta.get("kind") == "case":
        return "case"

    path = str(source or "").replace("\\", "/").lower()
    # 统一成以 / 开头，保证 "/data/docs/" 这种带边界的匹配不会漏掉相对路径
    normalized = path if path.startswith("/") else "/" + path

    marker = "/data/docs/"
    if marker in normalized:
        name = normalized.split(marker, 1)[1].split("/", 1)[0]
        if name in DOC_DOMAINS:
            return name

    if "/data/cases/" in normalized:
        return "case"

    if "/data/codebase/" in normalized or "/app/" in normalized or normalized.endswith(".py"):
        return "project"

    # data/ 根目录下的笔记（本项目自己的学习笔记/说明文档）也算本项目，
    # 否则它们会落进 "other" 这个杂项类，反而吃到「小领域先验」的红利，
    # 在检索本项目代码的问题时挤掉真正的源码片段。
    if "/data/" in normalized:
        return "project"

    return "other"


# ---------------------------------------------------------------- 2. 质心与路由


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化，零向量保持为零（不产生 NaN）。"""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return matrix / norms


def collect_vectors(store) -> tuple[np.ndarray, list[Document | None]]:
    """从已建好的 FAISS 索引里取出全部向量与对应文档。

    关键点：走 `index.reconstruct_n` 直接读原始向量，
    **不重新调用 embedding 接口**。近万个片段的库切分成本约几十秒、0 次 API 调用。
    """
    index = store.index
    total = index.ntotal
    if total == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    vectors = np.asarray(index.reconstruct_n(0, total), dtype=np.float32)

    docs: list[Document | None] = []
    for i in range(total):
        doc_id = store.index_to_docstore_id.get(i)
        docs.append(store.docstore.search(doc_id) if doc_id else None)
    return vectors, docs


def collect_documents(store) -> list[Document]:
    """取出索引里的全部文档（只要文本与元数据，不要向量）。

    用于统计领域分布、构建代码符号索引 —— 比 `collect_vectors` 省内存。
    优先走 docstore 的内部字典（一次取全），比按 id 逐个 search 快得多。
    """
    docstore = getattr(store, "docstore", None)
    if docstore is not None and hasattr(docstore, "_dict"):
        return list(docstore._dict.values())

    docs: list[Document] = []
    for i in range(store.index.ntotal):
        doc_id = store.index_to_docstore_id.get(i)
        if not doc_id:
            continue
        doc = store.docstore.search(doc_id)
        if doc is not None:
            docs.append(doc)
    return docs


def group_indices_by_domain(docs: list[Document | None]) -> dict[str, list[int]]:
    """把片段下标的按领域分组。"""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, doc in enumerate(docs):
        if doc is None:
            continue
        buckets[infer_domain(doc.metadata.get("source", ""), doc.metadata)].append(i)
    return dict(buckets)


def compute_centroids(
    vectors: np.ndarray,
    docs: list[Document | None],
) -> dict[str, np.ndarray]:
    """算出每个领域的质心向量（单位向量）。

    质心是该领域所有片段向量的平均方向，可以理解为「这个技术栈的语义中心」。
    用它和问题向量做余弦相似度，就能判断问题落在哪个领域。
    """
    groups = group_indices_by_domain(docs)
    unit = _normalize(vectors)

    centroids: dict[str, np.ndarray] = {}
    for domain, indices in groups.items():
        if not indices:
            continue
        centroid = unit[indices].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-8:
            centroids[domain] = (centroid / norm).astype(np.float32)
    return centroids


def save_centroids(centroids: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **{k: v for k, v in centroids.items()})


def load_centroids(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(str(path)) as data:
        return {key: data[key] for key in data.files}


def route_query(
    query_vector,
    centroids: dict[str, np.ndarray],
    min_score: float = 0.0,
) -> tuple[str | None, dict[str, float]]:
    """判断问题属于哪个领域。

    返回 (命中的领域 or None, 各领域的相似度分数)。
    相似度全都低于 min_score 时返回 None —— 说明这个问题不属于任何一个
    明确的领域（或是个泛泛的问题），此时应当走全局检索而不是硬塞进某个领域。
    """
    if not centroids:
        return None, {}

    vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None, {}
    vector = vector / norm

    scores = {domain: float(np.dot(vector, centroid)) for domain, centroid in centroids.items()}
    best = max(scores, key=scores.get)
    if scores[best] < min_score:
        return None, scores
    return best, scores


# ---------------------------------------------------------------- 3. 领域子库


def build_domain_stores(
    store,
    min_docs: int = 8,
    only: set[str] | None = None,
) -> dict[str, object]:
    """把主索引无损切分成「每领域一个子索引」，零 embedding 调用。

    利用 `FAISS.from_embeddings()` —— 它接收 (文本, 向量) 对，
    **不会再去调 embedding 接口**，所以我们能把主索引里的向量原样搬进子库。
    片段数少于 min_docs 的领域不单独建库（样本太少，检索价值不大）。

    `only` 用于按需建库：线上路由到哪个领域就只建哪个领域，
    避免为了问一个 Spring 的问题而先把 7 个域全建一遍。

    这样检索「本项目代码」的问题时就只在约 100 个片段里搜，
    官方文档的 9800 个片段根本不会出现在候选池里 —— 跨领域干扰被物理消除。
    """
    from langchain_community.vectorstores import FAISS

    vectors, docs = collect_vectors(store)
    if not len(docs):
        return {}

    groups = group_indices_by_domain(docs)

    stores: dict[str, object] = {}
    for domain, indices in groups.items():
        if only is not None and domain not in only:
            continue
        if len(indices) < min_docs:
            continue
        pairs = [(docs[i].page_content, vectors[i].tolist()) for i in indices]
        metadatas = [dict(docs[i].metadata, domain=domain) for i in indices]
        stores[domain] = FAISS.from_embeddings(
            pairs,
            embedding=store.embeddings,
            metadatas=metadatas,
        )
    return stores


def prefer_domain(
    docs: list[Document],
    domain: str | None,
    k: int,
    min_keep: int = 2,
) -> list[Document]:
    """领域内候选优先，不足再用其他领域补位（软过滤，不是硬切）。

    之所以要求 `min_keep` 条领域内候选才生效：如果候选池里该领域只有 0~1 条，
    说明这次路由判定很可能是错的，此时应该老老实实退回全局结果，而不是
    为了「贯彻路由」把整池好结果都扔了。
    """
    if not domain or not docs:
        return list(docs)[:k]

    inside: list[Document] = []
    outside: list[Document] = []
    for doc in docs:
        target = inside if infer_domain(doc.metadata.get("source", ""), doc.metadata) == domain else outside
        target.append(doc)

    if len(inside) < min_keep:
        return list(docs)[:k]
    return (inside + outside)[:k]


# ---------------------------------------------------------------- 4. 一键刷新


def refresh_domain_assets(store, storage_dir: Path) -> dict[str, int]:
    """算好质心并落盘，返回「领域 -> 片段数」。

    建库（ingest.py）后调用一次即可；之后检索端直接读 npz，无需重算。
    """
    vectors, docs = collect_vectors(store)
    centroids = compute_centroids(vectors, docs)
    save_centroids(centroids, storage_dir / CENTROID_FILENAME)
    return {domain: len(indices) for domain, indices in group_indices_by_domain(docs).items()}
