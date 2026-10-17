"""用户问答案例沉淀：把「用户确认有用」的问答写回知识库，实现 RAG 的自我进化。

为什么要做这个：
  普通 RAG 的知识库是静态的（只能靠重新抓文档来更新）。把用户问过、且确认有用的
  问答沉淀下来，系统就能随使用过程逐渐积累私有知识（你踩过的坑、验证过的写法），
  下次遇到同类问题可直接命中 —— 这就是「越用越聪明」。

安全设计（防止幻觉污染知识库）：
  1. 只沉淀用户手动确认的内容（前端「📌 记住这个案例」按钮触发），绝不自动写入。
  2. 案例独立存放在 data/cases/，与官方文档物理隔离；元数据标注 origin=user-case，
     检索结果可区分「官方文档」与「历史问答」，便于排查与按需过滤。
  3. 文件名带问答内容的 hash，同一问答重复沉淀会自动去重，不会堆积重复条目。
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from app.config import DATA_DIR

CASES_DIR = DATA_DIR / "cases"

# 匹配 ```lang ... ``` 代码块（lang 可省略）
_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """从用户输入里提取第一段代码块；没有代码块则返回空串。"""
    match = _CODE_FENCE_RE.search(text or "")
    return match.group(1).rstrip() if match else ""


def _digest(question: str, answer: str) -> str:
    raw = f"{question or ''}\n---\n{answer or ''}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def build_case_markdown(
    question: str,
    answer: str,
    raw_code: str = "",
    language: str = "python",
) -> str:
    """把一轮问答渲染成带 frontmatter 的 markdown，便于检索也便于人工查看。"""
    question = (question or "").strip()
    answer = (answer or "").strip()
    # 标题只取问题里的自然语言部分，剔除代码块，避免 title 里塞进一堆 ``` 和代码
    title_source = _CODE_FENCE_RE.sub(" ", question)
    title = " ".join(title_source.split())[:60] or "用户问答案例"

    parts = [
        "---",
        f"title: {title}",
        "type: case",
        "origin: user-qa",
        f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"language: {language}",
        "tags: [用户问答, 代码案例]",
        "---",
        "",
        "# 用户问题",
        "",
        question,
        "",
    ]
    if raw_code:
        parts += [
            "# 原始代码（用户提供）",
            "",
            f"```{language}",
            raw_code.strip(),
            "```",
            "",
        ]
    parts += [
        "# 参考解答",
        "",
        answer,
        "",
    ]
    return "\n".join(parts)


def save_case(
    question: str,
    answer: str,
    raw_code: str = "",
    language: str = "python",
    cases_dir: Path | None = None,
) -> Path:
    """把一轮问答写成一个案例文件，返回文件路径。

    文件名含内容 hash：同一问答重复保存会复用同一文件（幂等）。
    cases_dir 可注入（测试用），默认写入 data/cases/。
    """
    target = cases_dir or CASES_DIR
    target.mkdir(parents=True, exist_ok=True)
    digest = _digest(question, answer)

    # 幂等：同内容已存在（任意时间戳）就直接复用，避免重复沉淀
    for existing in target.glob(f"case-*-{digest}.md"):
        return existing

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = target / f"case-{stamp}-{digest}.md"
    path.write_text(
        build_case_markdown(question, answer, raw_code=raw_code, language=language),
        encoding="utf-8",
    )
    return path


def split_case_file(path: Path) -> list[Document]:
    """把案例文件切分为片段，并打上 user-case 元数据标记。

    复用 loader 的「保护代码块」切分逻辑，保证案例里的代码不被拦腰截断。
    """
    from app.loader import _split_text_protecting_code  # 局部导入，避免循环依赖

    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = {
        "source": str(path),
        "file_name": path.name,
        "origin": "user-case",
        "kind": "case",
    }
    return _split_text_protecting_code(text, meta)


def count_cases() -> int:
    """已沉淀的案例数量（供前端展示）。"""
    if not CASES_DIR.exists():
        return 0
    return len(list(CASES_DIR.glob("case-*.md")))


def list_case_documents() -> list[Document]:
    """读取并切分全部案例（供全量建库或排查用）。"""
    docs: list[Document] = []
    if not CASES_DIR.exists():
        return docs
    for path in sorted(CASES_DIR.glob("case-*.md")):
        docs.extend(split_case_file(path))
    return docs
