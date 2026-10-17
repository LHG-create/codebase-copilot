"""第 3 步：Web 界面。

用法（注意：不能用 python app_ui.py 跑）：streamlit run app_ui.py

说明：本界面刻意只用 Streamlit 原生组件（st.markdown / st.code / st.expander / st.chat_message），
不注入任何 unsafe_allow_html 的自定义 HTML，以避免 Streamlit 前端 React 的 removeChild 渲染崩溃。
"""

import sys
from pathlib import Path

import streamlit as st

# 裸模式防呆：用 python app_ui.py 直接跑时给出正确命令
if not st.runtime.exists():
    print("=" * 62)
    print("启动方式不对。Streamlit 应用必须用 streamlit 命令启动：")
    print()
    print("    streamlit run app_ui.py")
    print()
    print("在 PyCharm 里请把 Run Configuration 的 module 设为 streamlit，")
    print("Parameters 填：run app_ui.py")
    print("=" * 62)
    sys.exit(0)

from app.case_store import count_cases, extract_code, save_case, split_case_file  # noqa: E402
from app.chain import build_assistant  # noqa: E402
from app.config import settings  # noqa: E402
from app.domain import DOMAIN_LABELS, infer_domain  # noqa: E402

st.set_page_config(
    page_title="代码知识库问答",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="正在加载向量库与模型 ...")
def load_assistant(
    use_hybrid: bool = False,
    use_domain_prior: bool = True,
    use_domain_route: bool = True,
):
    return build_assistant(
        use_hybrid=use_hybrid,
        use_domain_prior=use_domain_prior,
        use_domain_route=use_domain_route,
    )


@st.cache_resource(show_spinner="正在构建 LangGraph 状态图 ...")
def load_graph(
    use_hybrid: bool = False,
    use_domain_prior: bool = True,
    use_domain_route: bool = True,
):
    """同一套检索链路，换成状态图执行（多一个「执行轨迹」可看）。"""
    from app.graph import build_graph

    return build_graph(
        use_hybrid=use_hybrid,
        use_domain_prior=use_domain_prior,
        use_domain_route=use_domain_route,
    )


# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown("## ⚙️ 设置")
    use_domain_prior = st.toggle(
        "长尾领域先验重排",
        value=True,
        help=(
            "全库近万片段里本项目代码只占 0.8%，会被海量官方文档淹没。"
            "开启后给「占比 ≤5% 的长尾领域」排序加成（类似 IDF 的思想），"
            "避免真正正确的少数片段被挤出候选。"
        ),
    )
    use_hybrid = st.toggle(
        "混合检索（向量 + BM25 + RRF）",
        value=False,
        help="实验性功能。双域评测显示在中文问题场景下混合检索持续负优化，默认关闭。",
    )
    use_domain_route = st.toggle(
        "领域路由 + 澄清追问",
        value=True,
        help=(
            "从问题里认框架名（FastAPI / Spring / Vue …），把检索限定在该技术栈内，"
            "避免「依赖注入」「CORS」这类跨框架概念互相干扰。"
            "线索指向多个技术栈时不猜，而是反问你要问哪一个。"
        ),
    )
    use_graph = st.toggle(
        "使用 LangGraph 状态图",
        value=False,
        help=(
            "把「路由 → 澄清 → 检索 → 生成」建成显式状态图，回答后可以展开看"
            "每一步的耗时与判断，便于定位「到底是路由错了还是检索没召回」。"
        ),
    )
    st.caption(
        f"对话：{settings.llm_provider}/{settings.llm_model}  ·  "
        f"向量：{settings.embedding_provider}/{settings.embedding_model}  ·  top_k={settings.top_k}"
    )
    if st.button("🧹 清空对话"):
        st.session_state.history = []
        st.session_state.chat_history_msgs = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        **📚 知识库内容**
        - 本项目源码（AST 结构化切分，可精确到函数/类/行号）
        - FastAPI 官方文档
        - LangChain / LangGraph 官方文档
        - Vue.js 官方文档
        - Spring Framework / Spring Boot 官方文档

        全库约 1 万片段，**本项目代码只占 0.8%**。
        检索时会对「片段数少的领域」做先验加权，避免被海量官方文档淹没。
        """
    )
    st.caption(f"📌 已沉淀用户案例：{count_cases()} 条（你在问答里点「记住」积累的）")
    st.markdown("---")
    st.markdown(
        """
        **💡 试试这样问**
        - 「`build_vectorstore` 做了什么？」
        - 「FastAPI 怎么定义路径参数？」
        - 「LangGraph 的 StateGraph 怎么用？」
        - 「Vue 的生命周期钩子有哪些？」
        - 「Spring Boot 怎么配置数据源？」

        中文问题会自动翻译成英文关键词检索，跨语言命中英文文档。
        回答下方的出处平铺展示，精确到**文件与行号**。

        **📌 沉淀你的案例**
        贴一段代码提问，觉得回答有用，就点回答下方的「记住这个案例」——
        它会把这条问答写进知识库，**下次能被检索到**（越用越聪明）。
        """
    )


# ---------- 标题 ----------
st.title("💬 代码知识库问答")
st.caption("基于 AST 结构化切分的 RAG · 答案精确溯源到文件与行号")

# 沉淀结果提示（rerun 后显示一次，随即清除，避免重复弹出）
_flash = st.session_state.pop("_case_flash", None)
if _flash:
    if _flash[0] == "success":
        st.success(_flash[1])
    else:
        st.error(_flash[1])


# ---------- 加载助手 ----------
try:
    assistant = load_assistant(
        use_hybrid=use_hybrid,
        use_domain_prior=use_domain_prior,
        use_domain_route=use_domain_route,
    )
    engine = (
        load_graph(
            use_hybrid=use_hybrid,
            use_domain_prior=use_domain_prior,
            use_domain_route=use_domain_route,
        )
        if use_graph
        else assistant
    )
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error("初始化失败，请检查 .env 里的 API Key 和 provider 配置。")
    st.exception(exc)
    st.stop()


# ---------- 会话状态 ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history_msgs" not in st.session_state:
    st.session_state.chat_history_msgs = []


def _read_source_range(source: str, start: int, end: int, context: int = 0) -> str:
    """从源码字符串里截取 start-end 行（1-based），可选带上下文行。"""
    lines = source.splitlines()
    s = max(1, start - context)
    e = min(len(lines), end + context)
    if s > len(lines):
        return ""
    return "\n".join(lines[s - 1 : e])


def _locate_real_source(meta: dict) -> Path | None:
    """优先定位到项目里「真实」的源码文件，而非 data/codebase 里的副本。"""
    fname = meta.get("file_name", "")
    if not fname.endswith(".py"):
        return None
    project_root = Path(__file__).resolve().parent
    for c in (project_root / "app" / fname, project_root / fname):
        if c.exists():
            return c
    fallback = meta.get("source")
    if fallback and Path(fallback).exists():
        return Path(fallback)
    return None


def render_sources(sources) -> None:
    """渲染出处：代码片段显示「文件:行号 · 符号」，文档片段显示文件名。

    注意：这里刻意不用 st.expander（可交互的嵌套容器）。
    st.expander 在 st.chat_message 里配合流式输出时，会触发 Streamlit 前端 React 的
    removeChild 崩溃（"被移除的节点不是该节点的子节点"）。改用纯展示组件
    （st.markdown / st.code）平铺，彻底避开这个坑。
    """
    if not sources:
        return

    st.markdown("---")
    st.markdown(f"**📎 出处（{len(sources)} 条）**")

    for index, doc in enumerate(sources, start=1):
        meta = doc.metadata
        fname = meta.get("file_name", "未知来源")
        kind = meta.get("kind")
        start = meta.get("start_line")
        end = meta.get("end_line")
        symbol = meta.get("name", "")

        is_code = kind in {"function", "method", "class", "variable", "module", "file"} and start and end

        # 领域标签：让用户一眼看出这条出处来自哪个技术栈
        # （全库近万片段里，本项目代码只占 0.8%，标注能显著提升可读性）
        domain_label = DOMAIN_LABELS.get(infer_domain(meta.get("source", ""), meta), "")
        tag = f"  ·  {domain_label}" if domain_label else ""

        if is_code:
            sym = symbol if symbol not in ("(module)", "(file)") else ""
            loc = f"{fname}:{start}-{end}"
            if sym:
                loc += f" · {sym}"
            st.markdown(f"**[{index}]** `{loc}`{tag}")
            # 代码片段：直接平铺展示，不再嵌套可交互容器
            st.code(doc.page_content[:600], language="python")
        else:
            st.markdown(f"**[{index}]** {fname}{tag}")
            st.caption(doc.page_content[:300] + (" ..." if len(doc.page_content) > 300 else ""))


def save_current_case(item: dict) -> None:
    """把一轮问答沉淀进知识库（写案例文件 + 增量入库），结果写入 flash 待显示。

    这是一次「RAG 自我进化」：用户确认有用的问答，会变成知识库的一部分，
    下次同类问题即可被检索到。沉淀内容与官方文档物理隔离（data/cases/），
    并带 origin=user-case 标记，便于在检索结果里区分与排查。
    """
    try:
        raw_code = extract_code(item["question"])
        path = save_case(item["question"], item["answer"], raw_code=raw_code)
        added = assistant.add_to_knowledge(split_case_file(path))
        item["saved"] = True
        st.session_state["_case_flash"] = (
            "success",
            f"已沉淀到知识库：{path.name}（新增 {added} 个片段，立即生效）",
        )
    except Exception as exc:  # noqa: BLE001
        st.session_state["_case_flash"] = ("error", f"沉淀失败：{exc}")


# ---------- 渲染历史 ----------
for idx, item in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        st.markdown(item["answer"])
        render_sources(item["sources"])
        if item.get("saved"):
            st.caption("✅ 已沉淀到知识库")
        elif st.button(
            "📌 记住这个案例",
            key=f"save_case_{idx}",
            help="把这条问答写进知识库，下次同类问题可被检索到",
        ):
            save_current_case(item)
            st.rerun()


# ---------- 输入 ----------
typed = st.chat_input("针对你的代码或文档提问 ...")

# 用户在澄清追问里点选了某个技术栈 —— 带着这个约束把原问题重放一遍
pick = st.session_state.pop("_clarify_pick", None)
question = pick["question"] if pick else typed
forced_domain = pick["domain"] if pick else None

if question:
    with st.chat_message("user"):
        st.markdown(question)

    clarification = None
    with st.chat_message("assistant"):
        # 一次性完整输出（非流式）。
        # 原因：st.write_stream 流式渲染含代码块的 Markdown 时，React 会在
        # 「文本节点 ↔ 代码块节点」之间反复 diff，触发 removeChild 崩溃
        # （"被移除的节点不是该节点的子节点"）。改为一次性渲染，彻底避开。
        try:
            result = engine.answer(
                question,
                chat_history=st.session_state.chat_history_msgs,
                forced_domain=forced_domain,
            )
            answer = result["answer"]
            sources = result["sources"]
            clarification = result.get("clarification")
        except Exception as exc:
            st.error("调用模型失败，请检查 API Key 余额与网络。")
            st.exception(exc)
            st.stop()

        if clarification:
            # 歧义：不猜，把候选项摆给用户；点选后带领域约束重放这个问题
            st.markdown(f"🤔 {answer}")
            options = clarification["options"]
            cols = st.columns(len(options))
            for i, opt in enumerate(options):
                if cols[i].button(opt["label"], key=f"clarify_{i}", use_container_width=True):
                    st.session_state["_clarify_pick"] = {
                        "question": clarification["question"],
                        "domain": opt["domain"],
                    }
                    st.rerun()
        else:
            st.markdown(answer)
            if result.get("route"):
                label = DOMAIN_LABELS.get(result["route"], result["route"])
                st.caption(f"🎯 已路由到「{label}」知识域")
            trace = result.get("trace") or []
            if trace:
                with st.expander(
                    "🔍 执行轨迹（"
                    + " → ".join(step["node"] for step in trace)
                    + "）",
                    expanded=False,
                ):
                    for step in trace:
                        st.markdown(f"- **{step['node']}**　{step['ms']} ms　{step['note']}")
            render_sources(sources)

    # 等待澄清的这一轮不算完成，不写入历史（避免历史里塞满没有答案的提问）
    if clarification:
        st.stop()

    st.session_state.history.append(
        {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
    )
    st.session_state.chat_history_msgs.append({"role": "user", "content": question})
    st.session_state.chat_history_msgs.append({"role": "assistant", "content": answer})

    # rerun 让刚产生的这轮回答作为「历史」重新渲染，从而显示「📌 记住这个案例」按钮
    st.rerun()
