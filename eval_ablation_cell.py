"""补测消融实验缺的那一格：**无查询改写 + 长尾先验**。

eval_domain.py 跑了 A（无改写无先验）、B（改写无先验）、C（改写+先验）三格，
2x2 消融表缺「无改写 + 先验」这一格。旧版 README 里这一格是 97.5%/87.5%，
但那是在旧索引快照（代码片段 45 个）上测的 —— 索引重建后必须复测，否则表格里
会混着两个不同知识库快照的数字。

用法：python eval_ablation_cell.py
不调 LLM（无改写路径），只做向量检索 + 先验重排。
"""

from __future__ import annotations

from app.config import settings
from app.retrieval import Retriever
from app.vectorstore import load_vectorstore
from eval_domain import _fmt, _metrics, _new_bucket, _record
from eval_retrieval import DOCS_EVAL_SET, EVAL_SET


def main() -> None:
    store = load_vectorstore()
    print(f"知识库片段数：{store.index.ntotal}  top_k={settings.top_k}  alpha={settings.domain_prior_alpha}")

    # 无改写（单路原问题检索）+ 领域先验重排
    retriever = Retriever(
        store,
        top_k=settings.top_k,
        use_query_rewrite=False,
        use_domain_prior=True,
    )

    for label, eval_set in (("项目代码域", EVAL_SET), ("官方文档域", DOCS_EVAL_SET)):
        bucket = _new_bucket()
        for question, answer_file, answer_name in eval_set:
            _record(bucket, retriever.retrieve(question), answer_file, answer_name)
        print(f"【{label}】{len(eval_set)} 条")
        print(f"  {'无改写 + 先验':<18s}{_fmt(_metrics(bucket))}")
        print()


if __name__ == "__main__":
    main()
