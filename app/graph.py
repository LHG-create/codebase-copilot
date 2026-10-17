"""把 RAG 链路建成一张 LangGraph 状态图（可观测、可中断）。

=== 为什么要有这个文件 ===

`app/chain.py` 里那条链路是「顺序调用」写死的：路由 → 检索 → 生成。
能跑，但有两个说不清的地方：

1. **每一步花了多久、做了什么判断，事后无从查证。** 用户反馈「答得不对」时，
   是路由判错了？检索没召回？还是模型没用好资料？顺序调用只能靠加 print 猜。
2. **「停下来问用户」这种控制流塞不进线性管道。** 歧义时要中断并等待用户选择，
   线性写法只能靠返回值里塞个特殊字段，调用方自己判断（原来就是这么干的）。

状态图把这两件事都变成一等公民：

    route ──┬─(歧义)─> clarify ─> END
            └─(明确)─> retrieve ─┬─(无结果)─> END
                                 └─(有结果)─> generate ─> END

=== 图里没有什么（有意为之）===

- **没有 checkpointer（对话记忆）**：多轮历史仍由调用方通过 `chat_history`
  显式传入。原因是不想出现两套「历史真源」——会话状态同时存在于 LangGraph
  的 checkpoint 和 Streamlit 的 session_state 里，迟早会对不上。
- **没有循环边**：澄清后不是在图里等，而是返回给调用方，用户选择后**重新发起**
  一次带 `forced_domain` 的图执行。这样图是无环的，调试时不会陷进重试循环。

=== 输出协议与 chain.stream 完全一致 ===

目的是让 CLI / Web / API 三端不用区分自己接的是哪条链路：
    {"clarification": {...}, "sources": []}     # 需要澄清，未生成回答
    {"sources": [...], "stream_started": True, "route": "fastapi"}
    {"delta": "..."}                            # 流式回答片段
    {"answer": "...", "sources": []}            # 检索为空
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from app.chain import PROMPT, PROMPT_WITH_HISTORY, RagAssistant, format_docs
from app.domain import DOMAIN_LABELS
from app.domain_router import build_clarification
from app.llm import get_chat_model


def _append_trace(existing: list[dict], new: list[dict]) -> list[dict]:
    """trace 字段的 reducer：各节点追加自己的记录，而不是互相覆盖。"""
    return (existing or []) + (new or [])


class RagState(TypedDict, total=False):
    """图的状态。

    total=False 是因为不同分支走完后，字段本来就填不满（例如澄清分支没有 context），
    强行要求全字段会让每个节点都写一堆无意义的空值。
    """

    question: str
    chat_history: list[dict]
    forced_domain: str | None
    decision: dict[str, Any]
    route: str | None
    clarification: dict | None
    sources: list[Document]
    context: str
    answer: str
    trace: Annotated[list[dict], _append_trace]


def _emit(payload: dict) -> None:
    """把增量结果推给流式消费方；非流式执行时静默忽略。

    `get_stream_writer()` 只在 graph.stream(..., stream_mode="custom") 下有效，
    在 graph.invoke() 下调用会抛错 —— 所以必须吞掉异常，否则同一条链路
    在「非流式」和「流式」两种用法下行为不一致。
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return
    if writer is not None:
        writer(payload)


class RagGraph:
    """RAG 状态图。对外暴露与 RagAssistant 相同的 answer / stream 签名。"""

    def __init__(self, assistant: RagAssistant | None = None) -> None:
        self.assistant = assistant or RagAssistant()
        self.retriever = self.assistant.retriever
        self.vectorstore = self.assistant.vectorstore
        self.top_k = self.assistant.top_k
        self._chain = PROMPT | get_chat_model() | StrOutputParser()
        self._chain_with_history = PROMPT_WITH_HISTORY | get_chat_model() | StrOutputParser()
        self.graph = self._build()
        # 最近一次执行的逐步耗时，供界面/接口展示（便于回答「为什么答错了」）
        self.last_trace: list[dict] = []

    # ------------------------------------------------------------ 建图

    def _build(self):
        builder = StateGraph(RagState)
        builder.add_node("route", self._node_route)
        builder.add_node("clarify", self._node_clarify)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("generate", self._node_generate)

        builder.set_entry_point("route")
        builder.add_conditional_edges(
            "route",
            self._edge_after_route,
            {"clarify": "clarify", "retrieve": "retrieve"},
        )
        builder.add_conditional_edges(
            "retrieve",
            self._edge_after_retrieve,
            {"generate": "generate", "end": END},
        )
        builder.add_edge("clarify", END)
        builder.add_edge("generate", END)
        return builder.compile()

    # ------------------------------------------------------------ 节点

    def _node_route(self, state: RagState) -> dict:
        started = time.perf_counter()
        question = state["question"]
        forced = state.get("forced_domain")

        if forced:
            # 用户已在澄清里选定领域，不再重复判定 —— 判一遍反而可能翻掉用户的选择
            decision = {"domains": [forced], "primary": forced, "ambiguous": False}
            route = forced
            note = "用户已指定领域，跳过判定"
        else:
            decided = self.retriever.route(question)
            decision = {
                "domains": decided.domains,
                "primary": decided.primary,
                "ambiguous": decided.is_ambiguous,
                "comparison": decided.comparison,
            }
            route = decided.primary
            if decided.is_ambiguous:
                # 用中文域名额外的括号列表，避免把 Python 的
                # `['fastapi', 'spring-framework']` 原样带进界面和 CLI
                note = f"歧义：{'、'.join(DOMAIN_LABELS.get(d, d) for d in decided.domains)}"
            elif decided.comparison and len(decided.domains) > 1:
                note = f"比较类问题，跨域检索（{'/'.join(DOMAIN_LABELS.get(d, d) for d in decided.domains)}）"
            elif route:
                note = f"限定到 {DOMAIN_LABELS.get(route, route)}"
            else:
                note = "无线索，走全库"

        return {
            "decision": decision,
            "route": route,
            "trace": [
                {
                    "node": "route",
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                    "note": note,
                }
            ],
        }

    def _node_clarify(self, state: RagState) -> dict:
        started = time.perf_counter()
        decision = state.get("decision") or {}
        chosen = list(decision.get("domains") or [])
        clarification = build_clarification(state["question"], _DecisionView(chosen))
        _emit({"clarification": clarification, "sources": []})
        return {
            "clarification": clarification,
            "answer": clarification["prompt"],
            "sources": [],
            "trace": [
                {
                    "node": "clarify",
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                    "note": f"等待用户选择：{'、'.join(DOMAIN_LABELS.get(d, d) for d in chosen)}",
                }
            ],
        }

    def _node_retrieve(self, state: RagState) -> dict:
        started = time.perf_counter()
        docs = self.retriever.retrieve(
            state["question"], forced_domain=state.get("forced_domain")
        )
        route = self.retriever.last_route
        if docs:
            _emit({"sources": docs, "stream_started": True, "route": route})

        note = f"召回 {len(docs)} 条"
        if docs:
            top = docs[0].metadata.get("file_name") or "?"
            note += f"，首条 {top}"
        return {
            "sources": docs,
            "route": route,
            "context": format_docs(docs) if docs else "",
            "trace": [
                {
                    "node": "retrieve",
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                    "note": note,
                }
            ],
        }

    def _node_generate(self, state: RagState) -> dict:
        started = time.perf_counter()
        context = state.get("context") or ""
        question = state["question"]
        history = state.get("chat_history") or []

        if history:
            chain = self._chain_with_history
            inputs = {
                "chat_history": RagAssistant._format_history(history),
                "context": context,
                "question": question,
            }
        else:
            chain = self._chain
            inputs = {"context": context, "question": question}

        chunks: list[str] = []
        for piece in chain.stream(inputs):
            chunks.append(piece)
            _emit({"delta": piece})

        answer = "".join(chunks)
        return {
            "answer": answer,
            "trace": [
                {
                    "node": "generate",
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                    "note": f"生成 {len(answer)} 字",
                }
            ],
        }

    # ------------------------------------------------------------ 条件边

    @staticmethod
    def _edge_after_route(state: RagState) -> str:
        decision = state.get("decision") or {}
        return "clarify" if decision.get("ambiguous") else "retrieve"

    @staticmethod
    def _edge_after_retrieve(state: RagState) -> str:
        return "generate" if state.get("sources") else "end"

    # ------------------------------------------------------------ 对外接口

    def _initial_state(
        self, question: str, chat_history: list[dict] | None, forced_domain: str | None
    ) -> RagState:
        return {
            "question": question,
            "chat_history": chat_history or [],
            "forced_domain": forced_domain,
            "trace": [],
        }

    def answer(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        forced_domain: str | None = None,
    ) -> dict:
        """非流式执行一次完整图，返回与 RagAssistant.answer 相同的结构。"""
        state = self.graph.invoke(
            self._initial_state(question, chat_history, forced_domain)
        )
        self.last_trace = state.get("trace") or []

        docs = state.get("sources") or []
        if state.get("clarification"):
            return {
                "answer": state["answer"],
                "sources": [],
                "route": None,
                "clarification": state["clarification"],
                "trace": self.last_trace,
            }
        if not docs:
            return {
                "answer": "没有检索到相关内容，请确认知识库里是否包含该主题。",
                "sources": [],
                "route": state.get("route"),
                "clarification": None,
                "trace": self.last_trace,
            }
        return {
            "answer": state.get("answer", ""),
            "sources": docs,
            "route": state.get("route"),
            "clarification": None,
            "trace": self.last_trace,
        }

    def stream(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        forced_domain: str | None = None,
    ):
        """流式执行：产出与 RagAssistant.stream 完全一致的 chunk 序列。

        用 stream_mode=["custom", "values"]：
          - custom 是各节点通过 _emit 推出来的增量（sources / delta / clarification）
          - values 用来拿最终状态（无结果时补一条兜底文案，并记录 trace）
        """
        final: dict = {}
        for mode, payload in self.graph.stream(
            self._initial_state(question, chat_history, forced_domain),
            stream_mode=["custom", "values"],
        ):
            if mode == "custom":
                yield payload
            else:
                final = payload

        self.last_trace = final.get("trace") or []

        # 没召回任何片段：图直接走到 END，没有任何节点推过文本，这里补兜底
        if not final.get("sources") and not final.get("clarification"):
            yield {
                "answer": "没有检索到相关内容，请确认知识库里是否包含该主题。",
                "sources": [],
            }


class _DecisionView:
    """把路由决策的最小视图喂给 build_clarification（它只需要领域列表）。

    直接复用 dataclass 会让 graph 依赖 RouteDecision 的全部字段，
    而澄清只用到 domains 一项 —— 这里做个窄接口，避免把实现细节焊进图里。
    """

    def __init__(self, domains: list[str]) -> None:
        self.domains = domains


def build_graph(assistant: RagAssistant | None = None, **kwargs) -> RagGraph:
    """构造状态图。kwargs 透传给 RagAssistant（top_k / 各类开关）。"""
    if assistant is None and kwargs:
        assistant = RagAssistant(**kwargs)
    return RagGraph(assistant=assistant)
