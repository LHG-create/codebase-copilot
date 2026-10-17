"""AsciiDoc 转纯文本：把 Spring 官方文档的 .adoc 源码清洗成可入库的文本。

Spring Framework / Spring Boot 的官方文档用 AsciiDoc（.adoc）编写，
包含大量 AsciiDoc 专用标记。这里用「正则清洗」而非完整解析，
只做入库检索需要的最小转换——目标是得到干净、可读、可被 embedding 的文本，
不需要还原成完美排版。

处理清单（按出现频率）：
  - [[anchor]] 锚点 → 删除
  - :attr: value 属性指令 / ifdef / endif 等 → 删除
  - = / == / === 标题 → 转为 markdown 风格的 # / ## / ###（保留标题语义）
  - [source,java,...] 等属性行 → 删除
  - [tabs] / ====== 分块 / Java:: / Kotlin:: 标签 → 删除
  - |=== 表格块 → 转成 Markdown 表格（保留表头，行间不留空行）
  - ---- 代码块分隔 → 转为 ``` （复用 markdown 代码块保护逻辑）
  - `code` 行内代码 → 保留反引号
  - _italic_ / *bold* → 去掉标记符，保留文字
  - include::xxx[] 指令 → 删除（被 include 的子文件会作为独立文档入库）
  - xref:xxx[] / link:xxx[] 交叉引用 → 尽量提取显示文本
  - NOTE: / TIP: / WARNING: 等提示 → 保留文字
"""

from __future__ import annotations

import re

# 锚点 [[anchor]] 或 [[anchor,text]]
_RE_ANCHOR = re.compile(r"\[\[[^\]]+\]\]")
# 属性指令 :name: value（行首）
_RE_ATTR = re.compile(r"^:[a-zA-Z0-9_-]+:\s*.*$", re.MULTILINE)
# 条件指令 ifdef/endif/ifeval 等（行首）
_RE_COND = re.compile(r"^ifn?def::.*$|^endif::.*$|^ifeval::.*$", re.MULTILINE)
# include 指令 include::file.adoc[opts] 以及 include-code::xxx[]
_RE_INCLUDE = re.compile(r"^include(?:-code)?::.*$", re.MULTILINE)
# 属性引用 {attr-name}（Spring 文档大量用 {url-xxx} / {code-xxx-src} 等占位符）
_RE_ATTR_REF = re.compile(r"\{[a-zA-Z0-9_-]+\}")
# 块属性行 [source,java] / [tabs] / [NOTE] 等（行首方括号，但不是锚点）
_RE_BLOCK_ATTR = re.compile(r"^\[(source|tabs|listing|NOTE|TIP|IMPORTANT|WARNING|CAUTION|role|cols|options)\b.*\]$", re.MULTILINE)
# 标题 = / == / ===（行首，最多 6 个 =）
_RE_HEADING = re.compile(r"^(={1,6})\s+(.+)$", re.MULTILINE)
# 代码块分隔 ----
_RE_CODE_DELIM = re.compile(r"^----\s*$", re.MULTILINE)
# 行内：*bold* / _italic_（保守处理，只处理单词边界内联标记）
# AsciiDoc 里 *bold* 单星为粗体；但要避免误伤乘法（2 * 3）和列表星号（行首）。
# 用「非空格、非换行、非星号」限定星号包裹的内容，且星号两侧不能是字母数字（避免 2*3）。
_RE_BOLD = re.compile(r"(?<![\w*])\*([^*\s][^*\n]*[^*\s])\*(?![\w*])")
_RE_ITALIC = re.compile(r"(?<![\w*])_([^_]+)_(?![\w_])")
# tab 块标签 Java:: / Kotlin::（行首，后面跟着 :）
_RE_TAB_LABEL = re.compile(r"^[A-Za-z]+\:\:\s*$", re.MULTILINE)
# 交叉引用 xref:id[text] / xref:id / link:url[text]
_RE_XREF = re.compile(r"xref:[^\[\s]+(?:\[([^\]]*)\])?")
_RE_LINK = re.compile(r"link:[^\[\s]+(?:\[([^\]]*)\])?")
# 图片 image::xxx[]
_RE_IMAGE = re.compile(r"image::[^\[\s]+(?:\[[^\]]*\])?")
# 表格块：可选的 [cols="1,2", options="header"] 属性行 + |=== ... |===
# 捕获组 1 = 属性行内容（用于取列数），组 2 = 表格体
_RE_TABLE = re.compile(
    r"^(?:\[([^\]]*)\]\s*\n)?\|===\s*\n(.*?)^\|===\s*$",
    re.DOTALL | re.MULTILINE,
)
# 从 [cols="1,2"] 里取列数
_RE_COLS = re.compile(r"cols\s*=\s*[\"']([0-9,\s]+)[\"']")


def _table_to_markdown(attr: str | None, body: str) -> str:
    """把 AsciiDoc 表格体转成 Markdown 表格。

    AsciiDoc 表格用 |=== 包裹，单元格以 | 开头、内容可跨行书写；列数由
    [cols="1,2"] 或表头行的 | 数量决定。

    转成 Markdown 表格（行与行之间没有空行）很关键：否则切分器会把每行
    当成独立段落，`| @PathVariable | ...` 会变成没有表头语境的「孤儿行」，
    并与相邻的 `@MatrixVariable` / `@RequestParam` 行互相混淆，检索时
    随机命中邻居行（Spring Method Arguments 表格实测就是这个症状）。
    """
    cells = [" ".join(c.split()) for c in body.split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    while cells and cells[-1] == "":
        cells.pop()
    if not cells:
        return ""

    # 列数：优先从 [cols="1,2"] 推断，否则用表头行的 | 数量
    ncol = 0
    if attr:
        m = _RE_COLS.search(attr)
        if m:
            ncol = len([x for x in m.group(1).split(",") if x.strip()])
    if ncol <= 0:
        head = body.split("\n\n", 1)[0]
        ncol = head.count("|")
    if ncol <= 0:
        ncol = 2

    lines: list[str] = []
    for i in range(0, len(cells), ncol):
        row = cells[i : i + ncol]
        row += [""] * (ncol - len(row))
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:  # 表头下补 Markdown 分隔行
            lines.append("| " + " | ".join(["---"] * ncol) + " |")
    return "\n" + "\n".join(lines) + "\n\n"


def adoc_to_text(text: str) -> str:
    """把 AsciiDoc 源码清洗成纯文本。

    返回的文本里，代码块用 ``` 包裹（与 markdown 切分的保护逻辑兼容），
    标题转成 # 前缀，其余标记尽量删除、保留可读文字。
    """
    if not text:
        return ""

    # 0) 表格：|=== 块 → Markdown 表格（行连续，避免切分时被拆成孤儿行）
    text = _RE_TABLE.sub(lambda m: _table_to_markdown(m.group(1), m.group(2)), text)

    # 1) 删除锚点、属性指令、条件指令、include 指令
    text = _RE_ANCHOR.sub("", text)
    text = _RE_ATTR.sub("", text)
    text = _RE_COND.sub("", text)
    text = _RE_INCLUDE.sub("", text)
    # 属性引用 {xxx} → 删除（保留不了真实值，删掉比残留占位符干净）
    text = _RE_ATTR_REF.sub("", text)

    # 2) 交叉引用 / 链接 / 图片：优先保留显示文本
    def _xref_repl(m: re.Match) -> str:
        return m.group(1) if m.group(1) else ""

    text = _RE_XREF.sub(_xref_repl, text)
    text = _RE_LINK.sub(_xref_repl, text)
    text = _RE_IMAGE.sub("", text)

    # 3) 块属性行、tab 标签删除
    text = _RE_BLOCK_ATTR.sub("", text)
    text = _RE_TAB_LABEL.sub("", text)

    # 4) 代码块 ---- 转成 ``` （成对转换，简化：直接全替换）
    text = _RE_CODE_DELIM.sub("```", text)

    # 5) 标题 = / == / === 转 markdown 标题
    def _heading_repl(m: re.Match) -> str:
        level = len(m.group(1))
        return "#" * level + " " + m.group(2).strip()

    text = _RE_HEADING.sub(_heading_repl, text)

    # 6) 行内标记清理
    text = _RE_BOLD.sub(r"\1", text)
    text = _RE_ITALIC.sub(r"\1", text)

    # 7) 压缩多余空行（3 个以上换行 → 2 个）
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
