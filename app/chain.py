"""RAG 问答链：检索 -> 拼上下文 -> 生成 -> 带出处返回。

刻意没用 create_retrieval_chain，直接用 LCEL 组装。
好处是每一步都摆在明面上，出问题一眼能看出是检索错了还是模型答歪了。

检索本身（查询改写 / 双路 RRF / 领域先验重排）在 app/retrieval.py 里，
本模块只负责「拿到片段 -> 拼上下文 -> 调模型 -> 带回出处」。
这样拆的好处是评测脚本可以 import 同一个 Retriever，
**评测跑的就是线上真正跑的那条链路**。
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.domain_router import build_clarification
from app.llm import get_chat_model
from app.retrieval import Retriever
from app.vectorstore import load_vectorstore

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。\n"
    "规则：\n"
    "1. 只能依据【参考资料】回答，不要使用你自己的先验知识。\n"
    "2. 如果参考资料里没有答案，直接回答「根据已有资料无法回答该问题」，不要编造。\n"
    "3. 用中文回答，条理清晰；引用到某条资料时在其后标注编号，例如 [1]。"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}"),
    ]
)

# 带历史的模板：{chat_history} 是纯文本形式的历史对话，由调用方拼接
PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "以下是之前几轮对话的摘要，供你理解当前问题（如果为「无」则忽略）：\n"
            "{chat_history}\n\n"
            "【参考资料】\n{context}\n\n【问题】\n{question}",
        ),
    ]
)


def format_docs(docs: list[Document]) -> str:
    """把检索到的片段拼成带编号的上下文，编号和引用要对得上。

    代码片段会额外带上「文件:行号-行号 + 符号名」，让模型能引用到精确位置。
    """
    blocks = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        name = meta.get("file_name") or meta.get("source") or "未知来源"

        # 代码片段：附上符号名和精确行号
        kind = meta.get("kind")
        loc = ""
        if kind in {"function", "method", "class", "variable", "module", "file"}:
            symbol = meta.get("name", "")
            start = meta.get("start_line")
            end = meta.get("end_line")
            if start and end:
                loc = f"（{name}:{start}-{end} 行"
                if symbol and symbol != "(module)":
                    loc += f"，{symbol}"
                loc += "）"
            elif symbol and symbol != "(module)":
                loc = f"（{name}，{symbol}）"

        header = f"[{index}] 来源：{name}{loc}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


class RagAssistant:
    def __init__(
        self,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_query_rewrite: bool = True,
        use_domain_prior: bool = True,
        use_domain_route: bool = True,
        rewrite_weight: float | None = None,
    ) -> None:
        self.top_k = top_k if top_k is not None else settings.top_k
        self.vectorstore = load_vectorstore()
        # 检索链路统一放在 app/retrieval.py，评测脚本 import 的是同一个类，
        # 保证「评测的数字」就是「线上真实表现」。
        self.retriever = Retriever(
            self.vectorstore,
            top_k=self.top_k,
            use_query_rewrite=use_query_rewrite,
            use_domain_prior=use_domain_prior,
            use_domain_route=use_domain_route,
            rewrite_weight=(
                settings.rewrite_weight if rewrite_weight is None else rewrite_weight
            ),
            use_bm25=use_hybrid,
        )
        self.chain = PROMPT | get_chat_model() | StrOutputParser()

    def _retrieve(self, question: str, forced_domain: str | None = None) -> list[Document]:
        """检索召回（实现见 app/retrieval.py）。

        链路：原问题向量路 + 中文改写英文路 -> RRF 融合 -> 领域先验重排 -> top_k。

        为什么用 RRF 而不是简单拼接？
        实测发现：中文长查询（如「FastAPI 怎么定义路径参数」）会被口语词稀释，
        命中不到英文文档 path-params.md；而翻译后的英文关键词
        （如「path parameter define」）能稳稳命中。但若把中英文拼成一条 query
        再检索，embedding 会互相干扰、两边都像又都不像，反而更差。
        所以正确做法是「各自独立检索，再 RRF 融合」，让英文路的高排名结果
        也能公平进入最终 top_k。

        注意两路的召回池用的是 recall_k（远大于 top_k），而不是 top_k 本身。
        单通道只取 top_k 会让「排名稍后但内容正确」的片段彻底进不了融合池
        ——实测 Spring 的 @RequestParam 片段在纯向量检索中排 #3，
        却因为两路各只取 8 条而在融合后消失（详见 ADR-010）。
        """
        return self.retriever.retrieve(question, forced_domain=forced_domain)

    def add_to_knowledge(self, docs: list[Document]) -> int:
        """把新片段增量追加进知识库（用于用户案例沉淀），返回追加的片段数。

        检索端（self.vectorstore / self.retriever）持有的是同一个 FAISS 对象，
        追加后立即可召回，不需要重启或重新加载。
        """
        from app.vectorstore import append_documents

        return append_documents(self.vectorstore, docs)

    def answer(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        forced_domain: str | None = None,
    ) -> dict:
        """返回 {"answer": str, "sources": list[Document], "route": str|None,
        "clarification": dict|None}。

        chat_history: 形如 [{"role": "user"/"assistant", "content": "..."}] 的历史，
        用于支持多轮追问。传 None 或空列表则退化为单轮问答。
        forced_domain: 用户在澄清追问里选定的领域。带着它重放问题时，
        检索直接限定在该领域内，不再重新判定。

        **歧义时这里不回答，而是返回澄清请求**（clarification 非空）——
        详见 app/domain_router.py 里关于「认 vs 猜」的取舍说明。
        """
        if forced_domain is None:
            decision = self.retriever.route(question)
            if decision.is_ambiguous:
                clarification = build_clarification(question, decision)
                return {
                    "answer": clarification["prompt"],
                    "sources": [],
                    "route": None,
                    "clarification": clarification,
                }

        docs = self._retrieve(question, forced_domain)
        if not docs:
            return {
                "answer": "没有检索到相关内容，请确认知识库里是否包含该主题。",
                "sources": [],
                "route": self.retriever.last_route,
                "clarification": None,
            }

        context = format_docs(docs)
        if chat_history:
            history_text = self._format_history(chat_history)
            text = self._chain_with_history.invoke(
                {"chat_history": history_text, "context": context, "question": question}
            )
        else:
            text = self.chain.invoke({"context": context, "question": question})
        return {
            "answer": text,
            "sources": docs,
            "route": self.retriever.last_route,
            "clarification": None,
        }

    def stream(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        forced_domain: str | None = None,
    ):
        """流式回答：先检索返回 sources，再逐步产出 answer 文本。

        用法：
            result = assistant.stream(question, chat_history)
            sources = result["sources"]
            for piece in result["answer_stream"]:
                ...

        判定为歧义时只 yield 一个澄清请求（clarification 非空），不生成回答。
        """
        if forced_domain is None:
            decision = self.retriever.route(question)
            if decision.is_ambiguous:
                yield {
                    "clarification": build_clarification(question, decision),
                    "sources": [],
                }
                return

        docs = self._retrieve(question, forced_domain)
        if not docs:
            yield {"answer": "没有检索到相关内容，请确认知识库里是否包含该主题。", "sources": []}
            return

        context = format_docs(docs)
        if chat_history:
            history_text = self._format_history(chat_history)
            chain = self._chain_with_history
            inputs = {"chat_history": history_text, "context": context, "question": question}
        else:
            chain = self.chain
            inputs = {"context": context, "question": question}

        yield {"sources": docs, "stream_started": True, "route": self.retriever.last_route}
        for chunk in chain.stream(inputs):
            yield {"delta": chunk}

    @staticmethod
    def _format_history(chat_history: list[dict]) -> str:
        """把历史对话拼成纯文本，控制长度避免撑爆上下文。"""
        lines = []
        # 只保留最近几轮，避免过长
        for msg in chat_history[-6:]:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = (msg.get("content") or "").strip()
            if content:
                lines.append(f"{role}：{content[:200]}")
        return "\n".join(lines) if lines else "无"

    @property
    def _chain_with_history(self):
        # 懒加载，避免每次 answer 都重新构建
        if not hasattr(self, "_history_chain"):
            self._history_chain = PROMPT_WITH_HISTORY | get_chat_model() | StrOutputParser()
        return self._history_chain


def build_assistant(
    top_k: int | None = None,
    use_hybrid: bool = False,
    use_query_rewrite: bool = True,
    use_domain_prior: bool = True,
    use_domain_route: bool = True,
    rewrite_weight: float | None = None,
) -> RagAssistant:
    return RagAssistant(
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_query_rewrite=use_query_rewrite,
        use_domain_prior=use_domain_prior,
        use_domain_route=use_domain_route,
        rewrite_weight=rewrite_weight,
    )
