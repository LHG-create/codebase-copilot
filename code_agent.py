"""代码修改 Agent 的命令行入口。

用法：
    python code_agent.py                        # 交互式，逐条下需求
    python code_agent.py "给 config 加一个超时参数"   # 一次性执行
    python code_agent.py --yes "..."            # 跳过确认，直接写盘（慎用）

默认行为是**只出 diff、不写盘**：Agent 跑完后把 unified diff 打出来，
你输入 y 才真正落盘。理由见 app/code_agent.py 头部说明 ——
模型给错答案的代价很低，改错你代码的代价很高，两者不该共用同一个开关。

命令：/revert 还原本次会话已写入的修改、q 退出。
"""

from __future__ import annotations

import sys

from app.code_agent import AgentRun, CodeAgent


def _print_run(run: AgentRun) -> None:
    if run.steps:
        print("\n── 执行过程 ──")
        for step in run.steps:
            arg_hint = ""
            if step.args:
                first_key = next(iter(step.args))
                value = str(step.args[first_key]).replace("\n", " ")[:70]
                arg_hint = f"{first_key}={value}"
            print(f"  [{step.index}] {step.tool}({arg_hint})")
            preview = step.output.strip().splitlines()
            if step.tool != "propose_edit" and preview:
                print(f"        {preview[0][:100]}")

    pending = run.pending()
    if pending:
        print("\n── 待应用的修改 ──")
        for change in pending:
            print(f"  {change.path}  (+{change.added_lines} / -{change.removed_lines})")

    if run.summary:
        print(f"\n── 结论 ──\n{run.summary}")


def _handle(agent: CodeAgent, task: str, auto_yes: bool) -> None:
    run = agent.plan(task)
    _print_run(run)

    pending = run.pending()
    if not pending:
        print("\n（没有产生任何文件修改）\n")
        return

    if not auto_yes:
        print("\n" + "=" * 70)
        for change in pending:
            print(change.diff)
        print("=" * 70)
        try:
            answer = input("确认写入这些修改？(y/N) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            return
        if answer not in {"y", "yes"}:
            print("已取消，磁盘没有任何改动。\n")
            return

    written = agent.apply(run)
    print(f"\n已写入 {len(written)} 个文件：{', '.join(c.path for c in written)}")
    print("如需还原，输入 /revert\n")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    auto_yes = "--yes" in sys.argv

    try:
        agent = CodeAgent()
    except Exception as exc:
        print(f"初始化失败：{exc}")
        return

    print(f"代码修改 Agent 已就绪，工作目录：{agent.root}")
    print("默认只出 diff 不写盘。输入 q 退出。\n")

    # 一次性模式
    if args:
        _handle(agent, " ".join(args), auto_yes)
        return

    while True:
        try:
            task = input("任务：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task.lower() in {"q", "quit", "exit"}:
            break
        if task == "/revert":
            restored = agent.revert()
            print(f"已还原：{', '.join(restored) if restored else '（没有需要还原的修改）'}\n")
            continue

        _handle(agent, task, auto_yes)


if __name__ == "__main__":
    main()
