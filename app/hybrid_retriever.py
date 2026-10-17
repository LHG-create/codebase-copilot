"""混合检索器：向量检索 + BM25 关键词检索 + RRF 融合。

为什么需要混合检索？

纯向量检索擅长「语义相似」，但对「精确符号」不敏感。
代码库里大量内容是函数名、类名、变量名（如 build_vectorstore、RagAssistant），
用户问「向量库怎么保存」时，向量检索可能因为语义泛化而漏掉
「build_vectorstore」这个精确符号。

BM25（基于词频的关键词检索）恰好擅长精确匹配：
  - 代码符号（英文）作为关键词索引
  - 中文问题经 jieba 分词后也能做关键词匹配

RRF（Reciprocal Rank Fusion）把两路结果按排名融合：
  score = Σ 1/(k + rank_i)，k 默认 60。
它不依赖两路的分值量纲，是混合检索的业界标准做法。
"""

from __future__ import annotations

import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


def _code_tokens(doc: Document) -> str:
    """为 BM25 生成每个文档的「关键词文本」。

    关键设计：BM25 对代码片段的文本要「突出符号、弱化中文」。
    因为代码片段里的中文（docstring、注释）和普通笔记里的中文会互相干扰，
    而用户问代码问题时，真正有区分度的是函数名、类名这些英文符号。

    所以这里把符号重复三遍，让它成为 BM25 里的强关键词，
    同时保留代码体供英文关键词匹配。
    """
    kind = doc.metadata.get("kind")
    if kind in {"function", "method", "class", "variable", "module"}:
        name = doc.metadata.get("name", "")
        signature = doc.metadata.get("signature", "")
        # 符号重复 3 次 → 提升 BM25 词频权重
        symbol_block = f"{name} {signature} {name} {name}"
        return f"{symbol_block}\n{doc.page_content}"
    return doc.page_content


class HybridRetriever(BaseRetriever):
    """向量 + BM25 双路检索，RRF 融合。"""

    vector_store: VectorStore
    k: int = 4
    rrf_k: int = 60
    _bm25: BM25Retriever | None = None

    def _build_bm25(self) -> BM25Retriever:
        docs = _all_documents(self.vector_store)
        # 为代码片段构造更利于关键词匹配的文本
        bm25_docs = []
        for d in docs:
            bm25_docs.append(
                Document(page_content=_code_tokens(d), metadata=d.metadata)
            )
        retriever = BM25Retriever.from_documents(
            bm25_docs, preprocess_func=self._bm25_preprocess
        )
        retriever.k = self.k
        return retriever

    @property
    def _bm25_retriever(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = self._build_bm25()
        return self._bm25

    @staticmethod
    def _bm25_preprocess(text: str) -> list[str]:
        """BM25 的分词函数：中文用 jieba，英文符号保留原样。"""
        tokens = list(jieba.cut(text))
        return [t.strip() for t in tokens if t.strip()]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # 1) 向量路
        vec_docs = self.vector_store.similarity_search(query, k=self.k)
        # 2) 关键词路
        bm25_docs = self._bm25_retriever.invoke(query)

        # 3) RRF 融合
        return _rrf_fuse([vec_docs, bm25_docs], self.k, self.rrf_k)


def _all_documents(vector_store: VectorStore) -> list[Document]:
    """从向量库取出所有已入库的文档（用于重建 BM25 索引）。

    依赖 FAISS 的 docstore；若 docstore 为空则降级返回空列表。
    """
    try:
        # langchain FAISS 的 docstore 存储了全部原始文档
        docstore = vector_store.docstore
        return list(docstore._dict.values()) if hasattr(docstore, "_dict") else []
    except Exception:
        return []


def _rrf_fuse(ranked_lists: list[list[Document]], k: int, rrf_k: int = 60) -> list[Document]:
    """把多路排序结果用 RRF 融合，返回 top-k。

    score(doc) = Σ 1/(rrf_k + rank_in_that_list)，rank 从 1 开始。
    """
    scores: dict[str, float] = {}
    doc_by_id: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            # 用 page_content 的哈希做去重键（代码片段内容通常唯一）
            key = doc.metadata.get("source", "") + "::" + doc.page_content[:80]
            doc_by_id[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_by_id[key] for key in sorted_keys[:k]]


def build_hybrid_retriever(vector_store: VectorStore, k: int = 4) -> HybridRetriever:
    return HybridRetriever(vector_store=vector_store, k=k)
