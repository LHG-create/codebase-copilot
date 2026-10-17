"""文档加载与切分。

支持三类内容，用不同的切分策略：
  - .py 代码文件：走 code_splitter 的 ast 结构化切分（按函数/类粒度）
  - .txt / .md / .mdx / .pdf 文档：走 RecursiveCharacterTextSplitter 按语义切分，
    其中 .md 里的 ``` 代码块会被保护，不被拦腰截断。
  - .adoc 文档（Spring 官方文档）：先经 app.adoc 转成纯文本，再按语义切分。
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.code_splitter import split_code_file
from app.config import DATA_DIR, settings

SUPPORTED_SUFFIXES = {".txt", ".md", ".mdx", ".adoc", ".pdf", ".py"}

# 低信息密度的元信息文件：不进知识库。
# 这类文件（changelog / release notes / 贡献指南等）体量大但检索价值低，
# 切分后会产生大量碎片，淹没真正有价值的教程内容，严重拉低检索质量。
_EXCLUDED_FILENAMES = {
    "release-notes.md",
    "changelog.md",
    "changelog-js.md",
    "changelog-py.md",
    "changes.md",
    "contributing.md",
    "help-fastapi.md",
    "fastapi-people.md",
    "history-design-future.md",
    "external-links.md",
    "management.md",
    "editor-support.md",
    "benchmarks.md",
    "alternatives.md",
    # Spring 文档里的纯导航 / 索引文件，无检索价值
    "nav.adoc",
    "index.adoc",
    "page-layout.adoc",
}

# 去掉 YAML frontmatter 后的最小正文字数，低于它的文档视为「空壳」，不入库。
#
# 为什么只判断「内容非空」不够：
#   不少官方文档的页面只剩一段 frontmatter、正文一个字都没有 ——
#   例如 langchain/retrieval.md 全文仅 96 字节，官网自己都标了 `noindex: true`。
#   这类文件会以「标题向量」的形式混进候选池：既白占检索位，
#   又让评测出现「永远命中不了」的假失败 —— 你无论怎么调检索算法都救不了它，
#   因为答案文档本身是空的（详见 ADR-012）。
#   实测阈值 200：被挡掉的全是导航页 / 索引页 / 团队页，
#   而 254 字的 langchain/studio.md、348 字的 Spring CLI 安装页都完整保留。
MIN_BODY_CHARS = 200

# ---------------------------------------------------------------- 导语页判据
#
# MIN_BODY_CHARS 只能挡住「一个字都没有」的空壳，挡不住「有正文、但整页只是
# 预告下一节」的导语页。典型例子是 Spring 的 core/beans/dependencies.adoc：
# 421 字符，正文只有一句「This next section explains how you go from ... to ...」，
# 内容是空的、语义却和「依赖注入 / bean 依赖」非常接近，因此它在向量空间里
# 会稳定地排在真内容前面 —— 曾经就是「Spring 依赖注入」这条评测的标准答案，
# 而正确答案其实是 dependencies/factory-collaborators.adoc（21KB，标题就叫
# "= Dependency Injection"）。
#
# 判据的难点是**不能误伤**「末尾附带一句『详见下节』的真实内容页」。
# 所以不是「文里出现了 next section 就算」，而是：
#   1. 正文足够短（<= 430 字）—— 长的必然是正常页面
#   2. 没有代码块、没有表格、没有成规模的列表（>=2 项）—— 有这些就是有实质内容
#   3. 段落数不超过 3 —— 段落多说明有多个论述
#   4. 命中导语句式的那一段**占了正文 45% 以上** —— 整页就是这句过渡语
# 实测在 10000 个片段上命中 7 条，逐条人工核对全部为真导语页；
# 曾命中的 security/index.md（OAuth2 flows 列表）与 mvc-freemarker.adoc（489 字）
# 分别被规则 2 和规则 1 排除，未发生误伤。
LEAD_IN_MAX_CHARS = 430
_LEAD_IN_MIN_CHARS = 60

_LEAD_IN_RE = re.compile(
    r"this next section"
    r"|the next sections?"
    r"|in the next sections?"
    r"|next chapters?"
    r"|the following sections?"
    r"|here (?:you will see|you can find)",
    re.IGNORECASE,
)

# markdown / asciidoc 列表项（`- x` / `* x` / `** x`）
_LIST_ITEM_RE = re.compile(r"(?m)^\s*[-*]{1,2}\s+\S")

# 匹配文件开头的 YAML frontmatter（--- ... ---）
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """去掉文件开头的 YAML frontmatter。"""
    match = _FRONTMATTER_RE.match(text or "")
    return text[match.end():] if match else (text or "")


def _body_length(text: str) -> int:
    """去掉 YAML frontmatter 后的正文长度（字符数）。"""
    return len(_strip_frontmatter(text).strip())


def looks_like_lead_in(text: str) -> bool:
    """判断整页是否只是「导语 / 过渡页」（只预告下一节，自身无实质内容）。

    判据细节与标定过程见上方 LEAD_IN_MAX_CHARS 处的注释。
    """
    body = _strip_frontmatter(text).strip()
    if not (_LEAD_IN_MIN_CHARS <= len(body) <= LEAD_IN_MAX_CHARS):
        return False
    if "```" in body or "\n|" in body:
        return False
    if len(_LIST_ITEM_RE.findall(body)) >= 2:
        return False
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if len(paragraphs) > 3:
        return False
    matched = [p for p in paragraphs if _LEAD_IN_RE.search(p)]
    if not matched:
        return False
    return sum(len(p) for p in matched) / len(body) >= 0.45


# 匹配 markdown 代码块 ```...```（含 ```lang 形式）
_FENCE_RE = re.compile(r"```.*?\n.*?```", re.DOTALL)

# 匹配连续的 markdown 表格行（每行以 | 开头、以 | 结尾）
# 注意 |=== 这类 AsciiDoc 残留标记行尾不是 |，不会误匹配
_RE_TABLE_BLOCK = re.compile(r"(?:^[ \t]*\|.*\|[ \t]*\n?)+", re.MULTILINE)


def _is_separator_row(line: str) -> bool:
    """判断是否 markdown 表格的分隔行（如 `| --- | --- |`）。"""
    core = line.replace("|", "").replace(" ", "").strip()
    return bool(core) and set(core) <= {"-", ":"}


def _split_table_block(block: str, chunk_size: int) -> list[str]:
    """把 markdown 表格按行切成若干块，每块都带上表头。

    表格是「键值对」结构，表头（如 `| Controller method argument | Description |`）
    是理解每一行的关键语境。若直接按 chunk_size 硬切，除首块外的所有行都会
    失去表头——实测 Spring 的 Method Arguments 表就是这样退化成了一堆
    「| `@PathVariable` | ...」的孤立行，并与相邻的 @MatrixVariable 行混淆。
    因此这里按行分组、逐块补表头。
    """
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if not lines:
        return []
    if len(lines) < 2:
        return [block.strip()]

    if _is_separator_row(lines[1]):
        header, body = lines[:2], lines[2:]
    else:
        header, body = lines[:1], lines[1:]
    if not body:
        return [block.strip()]

    header_text = "\n".join(header)
    base = len(header_text) + 1

    pieces: list[str] = []
    cur: list[str] = []
    cur_len = base
    for line in body:
        if cur and cur_len + len(line) + 1 > chunk_size:
            pieces.append(header_text + "\n" + "\n".join(cur))
            cur, cur_len = [], base
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        pieces.append(header_text + "\n" + "\n".join(cur))
    return pieces


def _is_case_path(path: Path, data_dir: Path = DATA_DIR) -> bool:
    """判断文件是否位于 <data_dir>/cases/ 用户案例目录下。

    用户案例与官方文档物理隔离，元数据要单独标记为 user-case，
    以便把「官方文档」和「历史问答」在检索结果里区分开。
    """
    try:
        return path.resolve().is_relative_to((data_dir / "cases").resolve())
    except Exception:
        return False


def _is_excluded(path: Path) -> bool:
    """判断文件是否属于低价值元信息，应被排除。"""
    name = path.name.lower()
    if name in _EXCLUDED_FILENAMES:
        return True
    # `nav-*.adoc` 是 Spring / Spring Boot 文档里的纯导航片段（只有链接列表，
    # 正文常常不到 300 字），同样没有任何可检索内容。
    return name.startswith("nav-")


def iter_source_texts(data_dir: Path = DATA_DIR):
    """遍历 data/ 下所有**文本类**文档，产出 (path, 归一化文本)。

    只做读取与格式归一化（AsciiDoc -> 纯文本），不做任何过滤。
    抽出来是为了让「建库」和「索引体检（prune_index.py）」用**同一份判据**：
    两边都拿整页原文去判断，就不会出现「建库时放过的页面被体检脚本误杀」。
    .py 由 code_splitter 处理、.pdf 需要额外依赖，都不在这里产出。
    """
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if _is_excluded(path):
            continue
        suffix = path.suffix.lower()
        if suffix in {".py", ".pdf"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".adoc":
            from app.adoc import adoc_to_text

            text = adoc_to_text(text)
        yield path, text


def exclusion_reason(path: Path, text: str, data_dir: Path = DATA_DIR) -> str | None:
    """判断一整个文档是否应当被排除，返回原因（None 表示保留）。

    这是文档卫生规则的**唯一入口**：建库过滤与索引体检都调它。
    用户案例（data/cases/）一律不过滤 —— 案例本身可能很短，但它是
    用户亲手确认过的答案，不该被文档卫生规则误伤。
    """
    if _is_case_path(path, data_dir):
        return None
    if _body_length(text) < MIN_BODY_CHARS:
        return "空壳页"
    if looks_like_lead_in(text):
        return "导语页"
    return None


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """把 data/ 目录下所有支持的文档读成 Document 列表（含目录递归）。

    注意：.py 文件不在这里读成整文件，而是在 split_documents 阶段按 ast 切分，
    因此这里对 .py 直接跳过原始加载，交给 split_documents 统一处理。
    """
    documents: list[Document] = []

    # 1) 文本类文档：过一遍文档卫生规则（空壳页 / 导语页）
    for path, text in iter_source_texts(data_dir):
        if exclusion_reason(path, text, data_dir):
            continue
        documents.append(Document(page_content=text, metadata={"source": str(path), "file_name": path.name}))

    # 2) PDF：走专用加载器，按加载出来的每一页各判一次
    for path in sorted(data_dir.rglob("*.pdf")):
        if not _is_excluded(path):
            from langchain_community.document_loaders import PyPDFLoader

            for doc in PyPDFLoader(str(path)).load():
                if exclusion_reason(path, doc.page_content, data_dir):
                    continue
                doc.metadata["source"] = str(path)
                doc.metadata["file_name"] = path.name
                documents.append(doc)

    for doc in documents:
        if _is_case_path(Path(doc.metadata["source"]), data_dir):
            doc.metadata["origin"] = "user-case"
            doc.metadata["kind"] = "case"

    return documents
    return documents


def _split_text_protecting_code(text: str, metadata: dict) -> list[Document]:
    """切分文本，但保护 ``` 代码块与 markdown 表格的完整性。

    做法：先用占位符把代码块整体替换掉，切完再还原。占位符本身不含语义，
    不会干扰向量化；切分边界也永远不会落在代码块内部。
    表格则单独抽出，按行切成「每块都带表头」的小片段（见 _split_table_block）。
    """
    fences = _FENCE_RE.findall(text)
    placeholders = [f"\n\n__CODEBLOCK_{i}__\n\n" for i in range(len(fences))]

    masked = _FENCE_RE.sub(lambda m: placeholders.pop(0), text)

    # 抽出表格块：切成带表头的小块单独入库，原位置留空行
    table_pieces: list[str] = []

    def _hide_table(m: re.Match) -> str:
        table_pieces.extend(_split_table_block(m.group(0), settings.chunk_size))
        return "\n\n"

    masked = _RE_TABLE_BLOCK.sub(_hide_table, masked)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_text(masked)

    docs: list[Document] = []
    for chunk in chunks:
        # 还原代码块
        restored = chunk
        for i, fence in enumerate(fences):
            restored = restored.replace(f"__CODEBLOCK_{i}__", fence)
        if not restored.strip():
            continue
        docs.append(Document(page_content=restored, metadata=dict(metadata)))

    # 表格片段追加在最后（内容独立，不与正文混切）
    for piece in table_pieces:
        if piece.strip():
            docs.append(Document(page_content=piece, metadata=dict(metadata)))

    return docs


def split_documents(documents: list[Document], data_dir: Path = DATA_DIR) -> list[Document]:
    """把文档切分成片段。代码文件与普通文档分开处理。"""
    chunks: list[Document] = []

    # 1) 普通文档（.txt / .md / .pdf）：按语义切分，保护代码块
    for doc in documents:
        if doc.metadata.get("file_name", "").endswith(".py"):
            continue
        chunks.extend(_split_text_protecting_code(doc.page_content, doc.metadata))

    # 2) 代码文件：按 ast 结构化切分
    for path in sorted(data_dir.rglob("*.py")):
        chunks.extend(split_code_file(path))

    # 3) 统一编号
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks
