"""性能基准：一次问答的时间到底花在哪。

用法：
    python bench_latency.py               # 8 条问题，每条 1 次
    python bench_latency.py -n 12 -r 2    # 12 条，每条 2 次

为什么值得单独做这个脚本：

调优时最容易犯的错是「凭感觉优化」。我原本以为瓶颈肯定在生成（等模型吐字），
实测打脸：**检索占了端到端 40%**（均值 1.5s / 3.7s）。

原因不是向量检索慢 —— 实测 FAISS 在 10147 个片段上 k=4 与 k=96 只差 8ms（197ms vs 190ms），
说明搜索本身开销很小，时间几乎都花在**查询向量化的远程调用**上；
而「检索」这一步还包含**一次查询改写的 LLM 调用**（中文问题要先翻成英文关键词才能跨语言召回）。
所以降延迟的第一优先级是缓存改写结果，而不是换向量库或调 top_k。

这个数字也说明：只看「端到端 3.6 秒」会误判优化方向，必须分段看。

这也是简历上唯一能拿来说「性能」的实证数字：P50 / P95 + 分段占比。

注意：脚本里的「生成」= 端到端 - 检索，包含了拼上下文、模板渲染等少量开销，
不是纯模型耗时。这个近似够用，但对外说的时候别写成「模型耗时」。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

# 覆盖不同领域与问题形态：单域、跨域比较、需要澄清、代码符号级
QUESTIONS = [
    "FastAPI 的依赖注入是怎么实现的？",
    "Vue 3 的响应式原理是什么？",
    "Spring Boot 自动配置的原理？",
    "LangGraph 的条件边怎么用？",
    "RAG 检索里 RRF 是怎么融合两路结果的？",
    "这个项目的领域路由是怎么判断问题属于哪个框架的？",
    "比较一下 Spring 和 FastAPI 的依赖注入",
    "PR 审查时怎么防止 Agent 写坏文件？",
]


def _pct(values: list[float], p: float) -> float:
    """取分位数。样本量小的时候 interpolate 没意义，用最近位次即可。"""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=len(QUESTIONS), help="问题条数")
    parser.add_argument("-r", "--repeat", type=int, default=1, help="每条问题跑几次取均值")
    parser.add_argument("-o", "--out", default="", help="把结果写成 markdown 文件")
    parser.add_argument("--json", default="", help="把原始样本写成 json")
    args = parser.parse_args()

    # 延迟加载：让 --help 不用付索引加载的代价
    from app.chain import build_assistant

    chain = build_assistant()
    questions = QUESTIONS[: args.num]

    print("[i] 预热（加载索引、建链）...")
    t0 = time.perf_counter()
    chain.retriever.retrieve(questions[0])
    print(f"[i] 预热完成，首次检索 {time.perf_counter() - t0:.2f}s\n")

    rows: list[dict] = []
    for question in questions:
        retrieve_ms: list[float] = []
        total_ms: list[float] = []
        for _ in range(args.repeat):
            t = time.perf_counter()
            chain.retriever.retrieve(question)
            retrieve_ms.append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            result = chain.answer(question)
            total_ms.append((time.perf_counter() - t) * 1000)

        if result.get("clarification"):
            note = "（触发澄清，未生成）"
        elif not result.get("sources"):
            note = "（未检索到内容）"
        else:
            note = ""

        rows.append(
            {
                "question": question,
                "retrieve_ms": statistics.mean(retrieve_ms),
                "total_ms": statistics.mean(total_ms),
                "note": note,
            }
        )
        print(
            f"  检索 {rows[-1]['retrieve_ms']:7.0f}ms | "
            f"端到端 {rows[-1]['total_ms']:7.0f}ms {note}"
        )

    totals = [r["total_ms"] for r in rows]
    retrieves = [r["retrieve_ms"] for r in rows]
    generates = [max(0.0, r["total_ms"] - r["retrieve_ms"]) for r in rows]

    def line(name: str, values: list[float]) -> str:
        return (
            f"| {name} | {statistics.mean(values):.0f} | "
            f"{_pct(values, 0.5):.0f} | {_pct(values, 0.95):.0f} | "
            f"{min(values):.0f} | {max(values):.0f} |"
        )

    report = [
        "",
        "| 阶段 | 均值(ms) | P50 | P95 | 最小 | 最大 |",
        "|---|---|---|---|---|---|",
        line("检索（含改写/RRF/重排）", retrieves),
        line("生成（端到端-检索）", generates),
        line("端到端", totals),
        "",
        f"样本：{len(rows)} 条问题 × {args.repeat} 次；"
        f"检索占端到端 {statistics.mean(retrieves) / max(1e-9, statistics.mean(totals)) * 100:.1f}%。",
    ]
    text = "\n".join(report)
    print(text)

    if args.json:
        Path(args.json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[i] 原始样本已写入 {args.json}")

    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[i] markdown 已写入 {args.out}")


if __name__ == "__main__":
    main()
