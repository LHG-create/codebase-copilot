"""把知识库问答暴露成 HTTP 接口（FastAPI）。

启动：
    uvicorn app.api:app --reload --port 8000
    或 python -m app.api

打开 http://127.0.0.1:8000/docs 就是自动生成的交互式文档，不用手写说明书。

=== 为什么接口里的「澄清」不是错误 ===

歧义时返回 **200 + clarification 字段**，而不是 4xx。因为「问题需要补充信息」
是正常的业务流程，不是客户端错误 —— 前端拿到选项后让用户点一下，再带
`forced_domain` 重新 POST 同一个问题即可。用 4xx 会逼调用方区分「参数写错了」
和「我需要再问一句」，这两种情况的处理方式完全不同。

=== 为什么 sources 要裁剪 ===

检索出来的 Document 里带着完整正文（单个片段可达上千字）和一堆内部元数据
（chunk_id、领域推断用的中间字段）。全量返回会让响应体膨胀十几倍，
而调用方真正需要的是「这段答案出自哪个文件的哪几行」。
所以只返回定位信息 + 正文摘要，正文要全文就得回原始文档看。
"""

from __future__ import annotations

# --- 防呆：让人拿「脚本方式」跑这个文件也能起来 -------------------------------
# 正常入口是 `python -m app.api`（PyCharm 配置 10 已改成模块模式）。
# 但谁都可能右键点一次 Run，此时 sys.path[0] 是 app/ 目录，
# 下面的 `from app import domain` 会直接 ModuleNotFoundError、窗口一闪就关。
# 这里把项目根补进去，两种方式都能跑。以包方式导入时 __package__ 非空，不进这个分支。
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import domain as dm
from app.config import BASE_DIR, settings
from app.domain_router import FRAMEWORK_ALIASES
from app.graph import build_graph

# 前端单页应用的入口文件。和 API 同源托管，省掉一个静态服务器 /
# 跨域配置，也避免用户为了看界面再装一套 node 构建链。
WEB_INDEX = BASE_DIR / "web" / "index.html"

# 单个来源片段的正文摘要长度上限（字符）
_SNIPPET_CHARS = 400

VALID_DOMAINS = sorted({domain for _, domain in FRAMEWORK_ALIASES} | {"project"})


# ================================================================ 数据模型


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        description="最近几轮对话，最多取最后 6 条（3 轮）参与生成",
        max_length=20,
    )
    forced_domain: str | None = Field(
        default=None,
        description=(
            "用户在澄清里选定的领域。填了它就不再重新判定路由。"
            f"可选值：{VALID_DOMAINS}"
        ),
    )


class SourceOut(BaseModel):
    index: int
    file_name: str | None = None
    # 相对路径（如 fastapi/tutorial/cors.md）：前端要展示「在哪个目录下」，
    # 只给 basename 会让多个 index.md / config.py 看起来一模一样。
    path: str | None = None
    domain: str | None = None
    domain_label: str | None = None
    kind: str | None = None
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    snippet: str
    score_note: str | None = None


class ClarificationOption(BaseModel):
    domain: str
    label: str


class ClarificationOut(BaseModel):
    type: str = "clarify"
    question: str
    prompt: str
    options: list[ClarificationOption]


class TraceStep(BaseModel):
    node: str
    ms: float
    note: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    route: str | None = None
    clarification: ClarificationOut | None = None
    trace: list[TraceStep] = Field(default_factory=list)


class StatsResponse(BaseModel):
    chunks: int
    documents: int
    top_k: int
    recall_k: int
    domains: dict[str, int]
    domain_labels: dict[str, str]


class CaseRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    answer: str = Field(..., min_length=1, max_length=8000)


class CaseResponse(BaseModel):
    saved: bool
    path: str
    chunks_added: int


# ================================================================ 应用


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时构建一次状态图，后续请求复用。

    放在 lifespan 而不是模块顶层：`uvicorn --reload` 会反复导入模块，
    顶层加载会让每次热重载都重新读一遍索引（近万片段，好几秒）。
    """
    app.state.graph = build_graph()
    yield
    app.state.graph = None


app = FastAPI(
    title="代码知识库问答 API",
    description=(
        "基于 LangChain + FAISS 的代码/文档问答服务。"
        "检索链路：领域路由 → 查询改写 → 双路 RRF → 领域先验重排。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _graph():
    graph = getattr(app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪，请稍后重试")
    return graph


def _display_path(source: str) -> str:
    """把绝对路径裁剪成「从 data/ 或项目根算起」的相对路径。

    前端用它区分同名文件；同时也避免把用户的绝对路径（含用户名）
    暴露在界面上。
    """
    if not source:
        return ""
    normalized = str(source).replace("\\", "/")
    for marker in ("/data/docs/", "/data/codebase/", "/data/cases/"):
        if marker in normalized:
            return normalized.split(marker, 1)[1]
    root = str(BASE_DIR).replace("\\", "/")
    if normalized.startswith(root + "/"):
        return normalized[len(root) + 1:]
    return normalized


def _to_source(index: int, doc) -> SourceOut:
    meta = doc.metadata or {}
    source = meta.get("source", "")
    domain = meta.get("domain") or (dm.infer_domain(source, meta) if source else None)
    body = doc.page_content or ""
    snippet = body if len(body) <= _SNIPPET_CHARS else body[:_SNIPPET_CHARS] + " ..."
    return SourceOut(
        index=index,
        file_name=meta.get("file_name"),
        path=_display_path(source),
        domain=domain,
        domain_label=dm.DOMAIN_LABELS.get(domain or "", domain),
        kind=meta.get("kind"),
        symbol=meta.get("name"),
        start_line=meta.get("start_line"),
        end_line=meta.get("end_line"),
        snippet=snippet,
    )


def _validate_forced_domain(forced: str | None) -> None:
    if forced is not None and forced not in VALID_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"未知领域 {forced!r}，可选值：{VALID_DOMAINS}",
        )


def _history(req: AskRequest) -> list[dict] | None:
    return [turn.model_dump() for turn in req.chat_history] or None


# ---------------------------------------------------------------- 端点


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """前端单页应用。和 API 同源，因此页面里直接 fetch('/ask/stream') 即可。"""
    if not WEB_INDEX.exists():
        raise HTTPException(
            status_code=500,
            detail=f"前端文件缺失：{WEB_INDEX}（应位于项目根的 web/index.html）",
        )
    return FileResponse(WEB_INDEX, media_type="text/html")


@app.get("/health", summary="健康检查")
def health() -> dict[str, Any]:
    """探活 + 索引是否就绪。不触发任何模型调用。"""
    graph = getattr(app.state, "graph", None)
    ready = graph is not None
    return {
        "status": "ok" if ready else "loading",
        "index_ready": ready,
        "chunks": len(graph.vectorstore.docstore._dict) if ready else 0,
    }


@app.get("/stats", response_model=StatsResponse, summary="知识库概况")
def stats() -> StatsResponse:
    """片段数、文档数、各领域片段分布。用于确认「知识库里到底有什么」。"""
    graph = _graph()
    store = graph.vectorstore
    per_domain: dict[str, int] = {}
    sources: set[str] = set()
    for doc in store.docstore._dict.values():
        src = (doc.metadata or {}).get("source", "")
        if src:
            sources.add(src)
        domain = (doc.metadata or {}).get("domain") or dm.infer_domain(src, doc.metadata)
        per_domain[domain] = per_domain.get(domain, 0) + 1

    return StatsResponse(
        chunks=len(store.docstore._dict),
        documents=len(sources),
        top_k=graph.top_k,
        recall_k=graph.retriever.recall_k,
        domains=dict(sorted(per_domain.items(), key=lambda kv: -kv[1])),
        domain_labels=dm.DOMAIN_LABELS,
    )


@app.post("/ask", response_model=AskResponse, summary="提问（一次性返回）")
def ask(req: AskRequest) -> AskResponse:
    """同步回答。

    - 问题涉及多个技术栈且无法判断时，返回 `clarification`（answer 里是一句追问）
    - 拿到选项后，用同一个 question + `forced_domain` 再请求一次即可
    """
    _validate_forced_domain(req.forced_domain)
    graph = _graph()
    result = graph.answer(
        req.question,
        chat_history=_history(req),
        forced_domain=req.forced_domain,
    )
    return AskResponse(
        answer=result["answer"],
        sources=[_to_source(i, d) for i, d in enumerate(result["sources"], start=1)],
        route=result.get("route"),
        clarification=result.get("clarification"),
        trace=result.get("trace") or [],
    )


@app.post("/ask/stream", summary="提问（SSE 流式）")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """流式回答，text/event-stream 协议。

    事件格式（每条都是 `data: <json>\\n\\n`）：

        {"type": "clarification", "data": {...}}
        {"type": "sources", "data": [...]}
        {"type": "delta", "data": "文本片段"}
        {"type": "done", "data": {"route": "fastapi", "trace": [...]}}

    用 SSE 而不是 WebSocket：问答是单向推送，SSE 在浏览器里用
    `EventSource` 一行就能接，还自带断线重连，没有必要上 WebSocket。
    """
    _validate_forced_domain(req.forced_domain)
    graph = _graph()
    history = _history(req)

    def event_stream():
        try:
            for chunk in graph.stream(
                req.question, chat_history=history, forced_domain=req.forced_domain
            ):
                if chunk.get("clarification"):
                    yield _sse("clarification", chunk["clarification"])
                elif chunk.get("stream_started"):
                    sources = [
                        _to_source(i, d).model_dump()
                        for i, d in enumerate(chunk.get("sources") or [], start=1)
                    ]
                    yield _sse("sources", sources)
                elif chunk.get("delta"):
                    yield _sse("delta", chunk["delta"])
                elif chunk.get("answer"):
                    # 检索为空时的兜底文案
                    yield _sse("delta", chunk["answer"])
            yield _sse("done", {"route": None, "trace": graph.last_trace})
        except Exception as exc:  # 生成阶段的异常也要让客户端知道，不能静默断开
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/cases", response_model=CaseResponse, summary="沉淀一条问答为案例")
def save_case_endpoint(req: CaseRequest) -> CaseResponse:
    """把「确认有用」的问答写进知识库，增量索引后立刻可被检索到。

    这是项目的自我进化能力：用得越多，能答的越多。只追加这一条案例，
    不重建整个索引 —— 近万片段重建要几分钟，而单条案例是毫秒级。
    """
    from app.case_store import extract_code, save_case, split_case_file

    graph = _graph()
    try:
        path = save_case(req.question, req.answer, raw_code=extract_code(req.question))
        added = graph.assistant.add_to_knowledge(split_case_file(path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"案例沉淀失败：{exc}") from exc

    return CaseResponse(saved=True, path=str(path), chunks_added=added)


def _sse(event_type: str, data: Any) -> str:
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _open_when_ready(url: str, host: str, port: int, timeout: float = 60.0) -> None:
    """等端口真的通了再开浏览器。

    直接延迟固定秒数不可靠：索引要加载近万片段，机器慢的时候 2 秒还没 listen，
    浏览器会先甩一个「无法访问此网站」，用户以为服务挂了。这里改成轮询端口。
    """

    def _worker() -> None:
        import socket
        import time
        import webbrowser

        probe = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.3)
                if sock.connect_ex((probe, port)) == 0:
                    try:
                        webbrowser.open(url)
                    except Exception:
                        pass
                    return
            time.sleep(0.3)

    import threading

    threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    import os

    import uvicorn

    host, port = settings.api_host, settings.api_port
    # 监听 0.0.0.0 时浏览器不能直接开 0.0.0.0，换成回环地址
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"

    print("=" * 66)
    print("  代码知识库 · Codebase Copilot")
    print(f"  网页界面   {url}")
    print(f"  接口文档   {url}docs")
    print("  停止服务   Ctrl + C（或点 PyCharm 的红色方块）")
    print("=" * 66, flush=True)

    # 服务器 / 无头环境跑的时候不该弹浏览器，给个开关关掉。
    if os.getenv("WBRAG_NO_BROWSER", "").strip().lower() not in ("1", "true", "yes", "on"):
        _open_when_ready(url, host, port)

    try:
        # 传对象而不是 "app.api:app"：脚本方式运行时模块已以 __main__ 加载过一次，
        # 再用字符串导入会让整个模块被导入第二遍（索引与配置各存两份）。
        # reload=False 时传对象完全够用，且不受 sys.path 影响。
        uvicorn.run(app, host=host, port=port, reload=False)
    except OSError as exc:
        print(f"\n[启动失败] 端口 {port} 可能已被占用：{exc}")
        print("           多半是上一次的服务还开着。任务管理器里结束 python.exe 再试。")
        raise SystemExit(1) from exc
