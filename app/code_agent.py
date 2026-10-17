"""代码修改 Agent：按自然语言需求改这个仓库的代码（基于 tool calling）。

=== 它和「问答」的区别 ===

问答是「读」：检索片段 → 组织答案。
这个 Agent 是「写」：定位文件 → 读实现 → 提出修改 → 跑测试验证。

工具集刻意做小（5 个），因为**给模型的权限越大，出事的半径越大**：

    list_files   看有哪些文件
    grep         按正则找代码位置
    read_file    读文件（带行号，读完全文而不是猜）
    propose_edit 提交整文件替换（只生成 diff，**不落盘**）
    run_tests    跑 pytest（命令固定，不接受自定义命令字符串）

=== 三条安全边界（这是本模块最该被审视的部分）===

1. **写盘是两阶段的。** `plan()` 只产出 `FileChange`（含 unified diff），
   要真正写入必须再调 `apply()`。默认永远不写盘 —— 因为 LLM 会犯错，
   而「错误答案」成本低，「错误地改了你的代码」成本高，两者不该同一个开关。

2. **路径锁在 root 内。** 所有路径先 `resolve()` 再校验 `is_relative_to(root)`，
   挡掉 `../../.ssh/id_rsa` 这类逃逸；同时硬性拒绝 `.env`、`.git/`、
   `storage/`、`data/` —— 这些目录要么含密钥，要么是生成物，让模型去改它们
   没有任何正当理由。

3. **不提供任意命令执行。** `run_tests` 内部拼的是固定的 `pytest` 命令，
   模型的入参只能是一个测试路径。若开放「执行任意 shell」，这个模块就等于
   把本机控制权交给了模型 —— 那已经不是 Agent，是后门。

=== 已知限制 ===

不做多文件事务：改 A 成功、改 B 失败时会留下半成品（有 diff 可回溯，但没有回滚）。
真要上生产，应该先把改动写进临时分支或影子目录再整体替换。
"""

from __future__ import annotations

import difflib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app.config import BASE_DIR
from app.llm import get_chat_model

MAX_STEPS = 12
MAX_READ_LINES = 600
MAX_WRITE_BYTES = 200_000
MAX_TOOL_OUTPUT = 4000
TEST_TIMEOUT_SECONDS = 300

# 硬性禁写目录 / 文件（相对于 root）
_FORBIDDEN_PATTERNS = (
    r"(^|/)\.env(\.|$)",
    r"(^|/)\.git(/|$)",
    r"(^|/)storage(/|$)",
    r"(^|/)data(/|$)",
    r"(^|/)\.venv(/|$)",
    r"(^|/)__pycache__(/|$)",
    r"\.pyc$",
)

SYSTEM_PROMPT = """你是一个严谨的代码修改助手，工作在一个 Python RAG 项目里。

工作流程（必须遵守）：
1. 先用 grep / list_files 定位相关文件，不要凭印象猜文件路径。
2. 用 read_file 读懂现有实现和代码风格，再动手。
3. 用 propose_edit 提交**整文件替换**内容（它只生成 diff，不会立即写盘）。
4. 改完调用 run_tests 验证；测试失败就根据报错继续改。
5. 最后用两三句话说明：改了哪个文件、为什么这么改、测试结果如何。

硬性约束：
- 只改必要的文件，不要顺手重写无关模块。
- 保持项目现有风格：中文注释、解释「为什么」而不是「是什么」。
- 不要修改 .env / storage/ / data/ 下的任何东西。
- 不要引入新的第三方依赖（除非用户明确要求）。
- 如果需求本身有歧义，先用一句话说明你缺什么信息，不要猜着改。

写代码时必须保证：
- import 齐全，没有未定义的名字。
- 边界条件（空值、空列表）有处理。
- 不硬编码密钥。
"""


@dataclass
class FileChange:
    """一次文件修改（尚未落盘）。"""

    path: str
    old_content: str
    new_content: str

    @property
    def diff(self) -> str:
        return "".join(
            difflib.unified_diff(
                self.old_content.splitlines(keepends=True),
                self.new_content.splitlines(keepends=True),
                fromfile=f"a/{self.path}",
                tofile=f"b/{self.path}",
                n=3,
            )
        )

    @property
    def added_lines(self) -> int:
        return sum(
            1
            for line in self.diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

    @property
    def removed_lines(self) -> int:
        return sum(
            1
            for line in self.diff.splitlines()
            if line.startswith("-") and not line.startswith("---")
        )


@dataclass
class Step:
    """Agent 的一步：调了哪个工具、入参是什么、拿到什么结果。"""

    index: int
    tool: str
    args: dict
    output: str


@dataclass
class AgentRun:
    task: str
    steps: list[Step] = field(default_factory=list)
    changes: list[FileChange] = field(default_factory=list)
    summary: str = ""
    applied: bool = False

    def pending(self) -> list[FileChange]:
        return [c for c in self.changes if c.old_content != c.new_content]


class CodeAgent:
    """代码修改 Agent。默认 dry-run：只产出 diff，不碰磁盘。"""

    def __init__(self, root: Path | str = BASE_DIR, model=None) -> None:
        self.root = Path(root).resolve()
        # 模型延迟到 plan() 才构造：工具集与路径校验是纯本地逻辑，
        # 没有 Key 也应该能构造出 agent 来（并让单元测试覆盖安全边界）。
        self._model = model
        self._pending: dict[str, FileChange] = {}
        self._applied: list[FileChange] = []

    def _get_llm(self):
        if getattr(self, "llm", None) is None:
            self.llm = (self._model or get_chat_model()).bind_tools(self._build_tools())
        return self.llm

    # ------------------------------------------------------------ 路径安全

    def _resolve(self, relative: str) -> Path:
        """把相对路径解析成绝对路径，并确保它落在 root 内。"""
        if not relative or not str(relative).strip():
            raise ValueError("路径不能为空")
        candidate = (self.root / str(relative).strip()).resolve()
        if candidate != self.root and not candidate.is_relative_to(self.root):
            raise ValueError(f"路径越界（只能访问项目目录内的文件）：{relative}")
        return candidate

    def _check_writable(self, relative: str) -> None:
        # 注意不能用 lstrip("./")：它会把 ".env" 的首字符 "." 一起吃掉，
        # 变成 "env" 从而绕过下面的禁写正则 —— 这个坑真的踩过一次。
        normalized = str(relative).replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = normalized.lstrip("/")
        for pattern in _FORBIDDEN_PATTERNS:
            if re.search(pattern, normalized):
                raise ValueError(f"出于安全考虑，不允许修改：{relative}")

    # ------------------------------------------------------------ 工具实现

    def _iter_sources(self, glob: str):
        """遍历匹配的文件。不含目录的 glob（如 '*.py'）自动递归到全项目。

        Path.glob('*.py') 只匹配根目录 —— 直接用它会让 Agent 在
        「搜不到代码」的假象里瞎猜路径。所以这里补一次 `**/` 前缀。
        """
        pattern = glob if "/" in glob or "\\" in glob else f"**/{glob}"
        for path in sorted(self.root.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            if any(
                part in {
                    ".venv",
                    "__pycache__",
                    ".git",
                    ".pytest_cache",
                    "storage",
                    "node_modules",
                    "data",
                }
                for part in rel.split("/")
            ):
                continue
            yield path, rel

    def _tool_list_files(self, pattern: str = "*.py") -> str:
        """按 glob 列出项目内文件（忽略隐藏目录与缓存）。"""
        try:
            matches = []
            for _, rel in self._iter_sources(pattern):
                matches.append(rel)
                if len(matches) >= 200:
                    break
            return "\n".join(matches) if matches else "（没有匹配的文件）"
        except Exception as exc:
            return f"列出文件失败：{exc}"

    def _tool_grep(self, pattern: str, glob: str = "*.py") -> str:
        """按正则搜索代码，返回「文件:行号: 内容」。"""
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"正则表达式无效：{exc}"

        hits: list[str] = []
        for path, rel in self._iter_sources(glob):
            try:
                for number, line in enumerate(
                    path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
                ):
                    if regex.search(line):
                        hits.append(f"{rel}:{number}: {line.strip()[:160]}")
                        if len(hits) >= 120:
                            return "\n".join(hits) + "\n（结果过多，已截断）"
            except OSError:
                continue
        return "\n".join(hits) if hits else "（没有匹配）"

    def _tool_read_file(self, path: str) -> str:
        """读取项目内的一个文本文件（返回带行号的内容）。"""
        try:
            target = self._resolve(path)
        except ValueError as exc:
            return str(exc)
        if not target.is_file():
            return f"文件不存在：{path}"
        if target.stat().st_size > MAX_WRITE_BYTES * 2:
            return f"文件太大（>{MAX_WRITE_BYTES * 2} 字节），不适合整体读取：{path}"
        try:
            lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            return f"读取失败：{exc}"
        truncated = len(lines) > MAX_READ_LINES
        body = "\n".join(f"{i:>4}| {line}" for i, line in enumerate(lines[:MAX_READ_LINES], start=1))
        if truncated:
            body += f"\n...（共 {len(lines)} 行，已截断到前 {MAX_READ_LINES} 行）"
        return body

    def _tool_propose_edit(self, path: str, new_content: str) -> str:
        """提交一个文件的**完整新内容**（只生成 diff，不写盘）。"""
        try:
            self._check_writable(path)
            target = self._resolve(path)
        except ValueError as exc:
            return str(exc)

        if not target.is_file():
            return (
                f"文件不存在：{path}。本工具只做「修改已有文件」，"
                "新建文件请先说明必要性，让用户确认。"
            )
        if len(new_content.encode("utf-8")) > MAX_WRITE_BYTES:
            return f"内容过大（>{MAX_WRITE_BYTES} 字节），拒绝写入"

        old = target.read_text(encoding="utf-8", errors="ignore")
        change = FileChange(path=target.relative_to(self.root).as_posix(), old_content=old, new_content=new_content)
        if old == new_content:
            return "新旧内容完全相同，没有产生修改。"

        self._pending[change.path] = change
        diff = change.diff
        head = diff if len(diff) <= MAX_TOOL_OUTPUT else diff[:MAX_TOOL_OUTPUT] + "\n...（diff 已截断）"
        return (
            f"已记录修改 {change.path}（+{change.added_lines} / -{change.removed_lines} 行），"
            f"**尚未写盘**。\n\n{head}"
        )

    def _tool_run_tests(self, target: str = "tests/") -> str:
        """运行 pytest（命令固定，只接受一个测试路径）。"""
        try:
            resolved = self._resolve(target)
        except ValueError as exc:
            return str(exc)

        cmd = [sys.executable, "-m", "pytest", str(resolved), "-q", "--no-header"]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_SECONDS,
                encoding="utf-8",
                errors="ignore",
            )
        except subprocess.TimeoutExpired:
            return f"测试超时（>{TEST_TIMEOUT_SECONDS}s）"
        except OSError as exc:
            return f"无法运行测试：{exc}"

        output = (completed.stdout or "") + (completed.stderr or "")
        if len(output) > MAX_TOOL_OUTPUT:
            output = output[-MAX_TOOL_OUTPUT:]
        verdict = "通过" if completed.returncode == 0 else f"失败（exit={completed.returncode}）"
        return f"pytest {verdict}\n{output.strip()}"

    def _build_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self._tool_list_files,
                name="list_files",
                description="按 glob 列出项目内的文件路径，例如 'app/*.py' 或 'tests/*.py'。",
            ),
            StructuredTool.from_function(
                func=self._tool_grep,
                name="grep",
                description="用正则表达式搜索代码，返回『文件:行号: 内容』。用来定位函数、类、字符串出现在哪里。",
            ),
            StructuredTool.from_function(
                func=self._tool_read_file,
                name="read_file",
                description="读取一个文件的内容（带行号）。修改前必须先读。",
            ),
            StructuredTool.from_function(
                func=self._tool_propose_edit,
                name="propose_edit",
                description=(
                    "提交某个文件的完整新内容。只生成 diff 并暂存，不会立刻写盘。"
                    "path 必须是已存在的文件；new_content 必须是完整文件内容，不是片段。"
                ),
            ),
            StructuredTool.from_function(
                func=self._tool_run_tests,
                name="run_tests",
                description="运行 pytest。target 是测试路径，例如 'tests/' 或 'tests/test_smoke.py'。",
            ),
        ]

    # ------------------------------------------------------------ 主循环

    def plan(self, task: str, use_rag: bool = True) -> AgentRun:
        """执行一次 Agent 会话，产出待应用的修改（不写盘）。

        use_rag=True 时先把知识库检索结果塞进上下文 —— 这个仓库里的代码
        问答链路本身就是它的知识库，让 Agent 带着「项目自己的约定」动手，
        比让它从零读文件更省步数、也更少跑偏。
        """
        run = AgentRun(task=task)
        messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

        context_block = ""
        if use_rag:
            context_block = self._rag_context(task)
        messages.append(
            HumanMessage(content=(f"【项目内相关资料】\n{context_block}\n\n" if context_block else "")
            + f"【任务】\n{task}")
        )

        for step_index in range(1, MAX_STEPS + 1):
            ai: AIMessage = self._get_llm().invoke(messages)
            messages.append(ai)

            if not getattr(ai, "tool_calls", None):
                run.summary = (ai.content or "").strip()
                break

            for call in ai.tool_calls:
                name = call.get("name", "?")
                args = call.get("args") or {}
                output = self._dispatch(name, args)
                run.steps.append(
                    Step(index=step_index, tool=name, args=args, output=output)
                )
                messages.append(ToolMessage(content=output, tool_call_id=call.get("id") or name))
        else:
            run.summary = f"达到最大步数（{MAX_STEPS}），已停止。请检查任务是否过于宽泛。"

        run.changes = list(self._pending.values())
        return run

    def _dispatch(self, name: str, args: dict) -> str:
        mapping = {
            "list_files": self._tool_list_files,
            "grep": self._tool_grep,
            "read_file": self._tool_read_file,
            "propose_edit": self._tool_propose_edit,
            "run_tests": self._tool_run_tests,
        }
        func = mapping.get(name)
        if func is None:
            return f"未知工具：{name}"
        try:
            return func(**args)
        except TypeError as exc:
            return f"工具 {name} 的参数不匹配：{exc}"
        except Exception as exc:
            return f"工具 {name} 执行出错：{exc}"

    @staticmethod
    def _rag_context(task: str, top_k: int = 3) -> str:
        """用项目自己的检索链路找相关资料；失败时静默降级（Agent 仍可工作）。"""
        try:
            from app.chain import build_assistant

            result = build_assistant().answer(task)
            sources = result.get("sources") or []
            if not sources:
                return ""
            return result.get("answer", "")[:1500]
        except Exception:
            return ""

    def apply(self, run: AgentRun) -> list[FileChange]:
        """把 plan 产出的修改真正写入磁盘。

        单独一步、显式调用 —— 这是本模块的安全阀：**模型不能自己决定写盘**，
        必须由调用方（人或上层服务）确认后才执行。
        """
        written: list[FileChange] = []
        for change in run.changes:
            self._check_writable(change.path)
            target = self._resolve(change.path)
            target.write_text(change.new_content, encoding="utf-8")
            written.append(change)
            self._applied.append(change)
        run.applied = bool(written)
        self._pending.clear()
        return written

    def revert(self) -> list[str]:
        """把本次会话已写入的修改还原（内存里存着旧内容）。"""
        restored: list[str] = []
        for change in reversed(self._applied):
            target = self._resolve(change.path)
            target.write_text(change.old_content, encoding="utf-8")
            restored.append(change.path)
        self._applied.clear()
        return restored
