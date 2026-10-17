# ADR-007：Spring 官方文档的 AsciiDoc 支持

- 状态：已采纳
- 日期：2026-10-15

## 背景

用户希望知识库覆盖 Spring Boot / Spring Framework。但 Spring 官方文档与已有的
FastAPI / Vue / LangChain（均为 Markdown）不同，使用的是 **AsciiDoc（`.adoc`）** 格式，
包含大量专用标记（锚点、属性指令、include、交叉引用、代码块等），不能直接按 Markdown 处理。

## 决策

采用「**正则清洗**」而非「完整 AsciiDoc 解析器」，新增 `app/adoc.py` 的 `adoc_to_text()`：

1. **删除结构性指令**：`[[anchor]]`、`:attr: value`、`ifdef/endif`、`include::xxx[]`。
   include 指令直接删除而非展开——因为抓取时已把被 include 的子文件也作为独立文档下载，
   内容不会丢。
2. **标题转换**：`= / == / ===` → `# / ## / ###`，保留层级语义。
3. **代码块转换**：`----` → ` ``` `，复用 loader 里已有的「代码块保护」切分逻辑，一举两得。
4. **交叉引用/链接**：`xref:id[text]` / `link:url[text]` 提取显示文本 `text`，丢弃目标地址。
5. **行内标记**：`*bold*` / `_italic_` 去标记符保留文字。正则用「星号两侧不能是字母数字」
   的负向断言，避免误伤乘法（`2 * 3`）和列表星号（行首 `* item`）。
6. **低价值文件排除**：`nav.adoc`、`index.adoc`、`page-layout.adoc` 等纯导航文件不进库；
   转换后为空文本的文档也跳过，避免产生空片段。

## 权衡

- **为什么不用完整解析器（asciidoctor / asciidoc-py）**：RAG 入库只需要「干净可读的文本」，
  不需要还原成完美排版。完整解析器依赖重、安装可能在本机（中文路径/网络）出问题，
  而正则清洗无第三方依赖、完全可控。实测转换结果满足检索需求。
- **正则的局限**：极少数复杂嵌套语法（如带条件的 include、复杂 tab 块）可能处理不完美，
  但对 embedding 检索影响可忽略——正文关键词和标题都在。

## 结果

抓取 Spring Framework（464 个 `.adoc`）+ Spring Boot（162 个 `.adoc`），
连同原有 4 个框架，知识库扩充到 6 个框架、1000+ 文档。
「Spring Boot 怎么配置数据源」等中文问题可正确跨语言检索命中。
