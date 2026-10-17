"""代码结构感知切分器。

为什么不用通用的 RecursiveCharacterTextSplitter 切代码？

普通的按字数切分会把一个函数拦腰截断，导致：
  1. 函数签名和函数体被拆成两段，检索时召回一半
  2. import 语句、装饰器、类定义和它们的用法失去关联

本模块用 Python 标准库 ast（抽象语法树）做「结构化切分」：
  - 以函数 / 方法 / 类为最小切分单元，保证每个片段语义完整
  - 每个片段保留签名、文件路径、起止行号到 metadata
  - 类内部的方法会被单独拆出，同时附带「所属类」信息

这样检索到的结果是「完整的、可定位到具体行号的代码块」，
而不是「被切碎的文本」。这是本项目区别于通用 RAG 的核心设计。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document


@dataclass
class _CodeUnit:
    """一次切分得到的代码单元。"""

    kind: str          # function / method / class / module
    name: str          # 函数或类名
    signature: str     # 签名（含 def 关键字和参数）
    start_line: int    # 1-based 起始行
    end_line: int      # 1-based 结束行
    docstring: str = ""  # 文档字符串（若有）


def _get_docstring(node: ast.AST) -> str:
    """提取函数/类的 docstring，用于增强检索语义。"""
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        val = body[0].value
        if isinstance(val, str):
            return val.strip()
    return ""


def _signature_of(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """还原函数签名（不含函数体），例如：def foo(a, b=1):"""
    args = func.args
    # 只取形参名与默认值的最简表达，避免复杂注解导致字符串过长
    arg_names = [a.arg for a in args.posonlyargs + args.args]
    defaults = args.defaults
    # 有默认值的参数从后往前对齐
    parts: list[str] = []
    n_no_default = len(arg_names) - len(defaults)
    for i, name in enumerate(arg_names):
        if i >= n_no_default:
            # 默认值用 ast.unparse 还原源码
            default_src = ast.unparse(defaults[i - n_no_default])
            parts.append(f"{name}={default_src}")
        else:
            parts.append(name)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    if args.kwonlyargs:
        parts.append("*")
        for a in args.kwonlyargs:
            parts.append(a.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    prefix = "async def" if isinstance(func, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {func.name}({', '.join(parts)})"


def _collect_units(tree: ast.Module, source: str) -> list[_CodeUnit]:
    """遍历 AST，收集模块级函数、类、以及类内方法。"""
    units: list[_CodeUnit] = []
    lines = source.splitlines()

    for node in tree.body:
        # 模块级变量（如 SYSTEM_PROMPT = "..."）也拆成独立单元，
        # 否则「这个常量在哪定义」这类问题检索不到。
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            name = ""
            if isinstance(node, ast.Assign) and node.targets:
                t = node.targets[0]
                if isinstance(t, ast.Name):
                    name = t.id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
            if name and (name.isupper() or not name.startswith("_")):
                units.append(
                    _CodeUnit(
                        kind="variable",
                        name=name,
                        signature=f"{name} = ...",
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring="",
                    )
                )
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append(
                _CodeUnit(
                    kind="function",
                    name=node.name,
                    signature=_signature_of(node),
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    docstring=_get_docstring(node),
                )
            )
        elif isinstance(node, ast.ClassDef):
            units.append(
                _CodeUnit(
                    kind="class",
                    name=node.name,
                    signature=f"class {node.name}",
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    docstring=_get_docstring(node),
                )
            )
            # 类内方法单独拆出，附上所属类
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    units.append(
                        _CodeUnit(
                            kind="method",
                            name=f"{node.name}.{child.name}",
                            signature=_signature_of(child),
                            start_line=child.lineno,
                            end_line=child.end_lineno or child.lineno,
                            docstring=_get_docstring(child),
                        )
                    )
                # 类内字段/属性（如 pydantic Settings 的字段）单独拆出，
                # 否则「chunk_size 默认值多少」这类问题检索不到具体字段。
                elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                    field_name = ""
                    if isinstance(child, ast.Assign) and child.targets:
                        t = child.targets[0]
                        if isinstance(t, ast.Name):
                            field_name = t.id
                    elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                        field_name = child.target.id
                    if field_name and not field_name.startswith("_"):
                        units.append(
                            _CodeUnit(
                                kind="field",
                                name=f"{node.name}.{field_name}",
                                signature=f"{field_name} = ...",
                                start_line=child.lineno,
                                end_line=child.end_lineno or child.lineno,
                                docstring="",
                            )
                        )

    # 模块级说明（文件头部注释/模块 docstring 之前的 import 区）作为一个独立单元，
    # 便于回答「这个文件整体是干嘛的、依赖了什么」这类问题。
    module_doc = ast.get_docstring(tree) or ""
    if module_doc or lines:
        # 取第一个代码单元之前的部分作为「模块头」
        first_line = min((u.start_line for u in units), default=len(lines) + 1)
        header_lines = lines[: first_line - 1]
        header_text = "\n".join(header_lines).strip()
        if header_text or module_doc:
            units.insert(
                0,
                _CodeUnit(
                    kind="module",
                    name="(module)",
                    signature="",
                    start_line=1,
                    end_line=first_line - 1,
                    docstring=module_doc or header_text,
                ),
            )

    return units


def split_code_file(path: Path) -> list[Document]:
    """把单个 .py 文件切成结构完整的 Document 列表。"""
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        # 非 UTF-8 或读不了的文件，退回按原样整文件入一个 chunk，不抛异常中断建库
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # 语法不完整的代码文件（如片段），退回按整文件切
        return [
            Document(
                page_content=source,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "kind": "file",
                    "name": path.stem,
                    "start_line": 1,
                    "end_line": source.count("\n") + 1,
                },
            )
        ]

    docs: list[Document] = []
    for unit in _collect_units(tree, source):
        # 提取该单元对应的源码文本（含原缩进，保证可读）
        raw_lines = source.splitlines()[unit.start_line - 1 : unit.end_line]
        code_text = "\n".join(raw_lines)

        # page_content = 签名 + docstring + 代码，签名和 docstring 前置增强检索
        parts = []
        if unit.signature:
            parts.append(unit.signature)
        if unit.docstring:
            parts.append(unit.docstring)
        parts.append(code_text)
        content = "\n\n".join(parts)

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "kind": unit.kind,
                    "name": unit.name,
                    "signature": unit.signature,
                    "start_line": unit.start_line,
                    "end_line": unit.end_line,
                },
            )
        )

    return docs
