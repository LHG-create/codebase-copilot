"""第 2 步（可选）：命令行里直接问答，最快验证链路通不通。

用法：python chat_cli.py

支持三件事，和网页端保持能力一致：
- **澄清追问**：问题涉及多个技术栈时列出编号选项，输入序号后带着约束重放
- **多轮对话**：自动维护最近几轮历史，`/clear` 清空
- **流式输出**：逐字打印，首字时间比整段返回快得多

命令：/clear 清空历史、/history 查看历史、q 退出。
"""

from __future__ import annotations

from app.chain import build_assistant


def _print_options(clarification: dict) -> list[dict]:
    """打印澄清选项，返回选项列表。"""
    print(f"\n⚠️  {clarification['prompt']}")
    options = clarification.get("options") or []
    for index, option in enumerate(options, start=1):
        print(f"  {index}) {option['label']}")
    print("  0) 都不限定，按全库检索")
    return options


def _choose_domain(clarification: dict):
    """让用户选一个领域。

    返回 领域名（带约束重放）、None（不限定的全库检索）、
    或 False（用户直接回车放弃这个问题）。
    """
    options = _print_options(clarification)
    try:
        raw = input("请输入序号（直接回车 = 放弃本次提问）：").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    if not raw:
        return False
    if not raw.isdigit():
        print("  序号无效，按全库检索处理。")
        return None
    index = int(raw)
    if index == 0:
        return None
    if 1 <= index <= len(options):
        chosen = options[index - 1]
        print(f"  → 已限定到「{chosen['label']}」")
        return chosen["domain"]
    print("  序号超出范围，按全库检索处理。")
    return None


def _ask_once(assistant, question: str, history: list[dict]) -> dict | None:
    """问一轮，内部消化澄清追问；返回 {answer, sources, route}，放弃时返回 None。"""
    forced_domain: str | None = None

    while True:
        buffer: list[str] = []
        sources: list = []
        route: str | None = None
        clarification: dict | None = None
        started = False

        for chunk in assistant.stream(
            question,
            chat_history=history or None,
            forced_domain=forced_domain,
        ):
            if chunk.get("clarification"):
                clarification = chunk["clarification"]
                break
            if "sources" in chunk:
                sources = chunk["sources"]
                route = chunk.get("route")
                if route:
                    print(f"\n（已路由到 {route} 域）")
                continue
            delta = chunk.get("delta") or chunk.get("answer")
            if delta:
                if not started:
                    print("\n助手：", end="", flush=True)
                    started = True
                print(delta, end="", flush=True)
                buffer.append(delta)

        if clarification:
            choice = _choose_domain(clarification)
            if choice is False:
                return None
            forced_domain = choice  # None 表示"都不限定"
            continue

        answer = "".join(buffer)
        if started:
            print()
        return {"answer": answer, "sources": sources, "route": route}


def _print_sources(sources: list) -> None:
    if not sources:
        return
    print("  出处：")
    for index, doc in enumerate(sources, start=1):
        meta = doc.metadata
        name = meta.get("file_name") or meta.get("source") or "?"
        symbol = meta.get("name")
        start, end = meta.get("start_line"), meta.get("end_line")
        loc = ""
        if start and end and symbol and symbol != "(module)":
            loc = f":{start}-{end} {symbol}"
        elif start and end:
            loc = f":{start}-{end}"
        snippet = doc.page_content[:50].replace("\n", " ")
        print(f"    [{index}] {name}{loc}  {snippet} ...")


def main() -> None:
    try:
        assistant = build_assistant()
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return

    print(f"知识库问答已就绪（top_k={assistant.top_k}）。输入 q 退出。")
    print("命令：/clear 清空历史、/history 查看历史\n")

    history: list[dict] = []

    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question == "/clear":
            history.clear()
            print("  历史已清空。\n")
            continue
        if question == "/history":
            print(f"  当前 {len(history) // 2} 轮历史。\n")
            continue

        result = _ask_once(assistant, question, history)
        if result is None:
            print()
            continue

        if result["route"]:
            print(f"  （已路由到 {result['route']} 域）")
        _print_sources(result["sources"])
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()
