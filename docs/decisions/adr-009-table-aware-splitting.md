# ADR-009：表格感知切分（表格不再被切碎成「孤儿行」）

- 状态：已采纳
- 日期：2026-10-15

## 背景

用户问了一个 Spring MVC 的问题：

> `@PathVariable` 和 `@RequestParam` 有什么区别？路由 `/users/{id}` 的参数该用哪个？

模型回答得**诚实但单薄**——它引用了 `@PathVariable`「用于访问 URI 模板变量」这一定义，
却明确说明「资料中未给出 `/users/{id}` 这种路由的原文示例」。

排查发现两个事实：

1. **知识库里有更好的资料，但没被召回。**
   `spring-framework/web/webmvc/mvc-controller/ann-requestmapping.adoc:111` 里有：

   ```java
   @GetMapping("/owners/{ownerId}/pets/{petId}")
   public Pet findPet(@PathVariable Long ownerId, @PathVariable Long petId) { ... }
   ```

   这正是用户所问场景的原文示例。

2. **召回的第 1 名是一个「表格碎片」。**
   实际召回结果：

   ```
   #1 arguments.adoc | | `@MatrixVariable` | For access to name-value pairs in ...
   ```

   这行是 Method Arguments 表里 `@PathVariable` 的**邻居行**。

## 根因

AsciiDoc 表格被两层处理破坏：

**第一层：`adoc_to_text()` 完全没处理 `|===` 表格块。**
表格标记原样留存，行间保留原有空行：

```
|===                     ← 起始标记残留
| Controller method argument | Description
| `WebRequest`, `NativeWebRequest` | ...

| `@PathVariable`        ← 与邻居之间有空行
| For access to URI template variables. See URI patterns.

| `@MatrixVariable`
| For access to name-value pairs in URI path segments.
```

由于 `RecursiveCharacterTextSplitter` 的首选分隔符是 `\n\n`，
这些行被切成**彼此独立的段落**。`@PathVariable` 那一行于是变成
没有表头语境的「孤儿行」。

**第二层：即便表格行连续，还会被 chunk_size 硬切。**
`chunk_size=800` 而 Method Arguments 表转成文本后约 5600 字，
必然被切成多块，**表头只留在第一块**。

两层叠加的后果：`| @PathVariable | ...` 与 `| @MatrixVariable | ...`
在 embedding 空间里几乎无法区分（都是「注解名 + URI 相关说明」），
检索于是**随机命中邻居行**——这正是 #1 召回 `@MatrixVariable` 的原因。

## 决策

分两处修复，职责分离：转换负责「结构」，切分负责「边界」。

### 1. `app/adoc.py`：`|===` 表格块 → 标准 Markdown 表格

新增 `_table_to_markdown()`：

- 列数从 `[cols="1,2"]` 取；无该属性时用表头行的 `|` 数量推断。
- 输出标准的表头 + 分隔行 + 数据行结构，**行与行之间不插空行**。

```
| Controller method argument | Description |
| --- | --- |
| `@PathVariable` | For access to URI template variables. See URI patterns. |
| `@RequestParam` | For access to the Servlet request parameters, including multipart files. ... |
```

### 2. `app/loader.py`：表格感知切分，逐块补表头

新增 `_split_table_block()` 与 `_is_separator_row()`，并在
`_split_text_protecting_code()` 里像保护代码块一样把表格**整体抽出**：

- 表格不进 `RecursiveCharacterTextSplitter`，而是按行分组。
- 分组时**每块都注入表头（含分隔行）**，块内行间不留空行。
- 表格作为独立片段入库，与正文互不混切。

关键点：**表头就是语境**。`| Controller method argument | Description |`
让模型知道「这是一张 controller 方法参数说明表」，而不是一堆零散的注解名。

## 影响

- **收益面超出 Spring**：FastAPI / Vue / LangChain 的 Markdown 表格
  此前同样会被切成孤儿行，本次一并修复。
- **片段数增加**：一张大表会变成多个带表头的片段（Method Arguments 表 → 10 块）。
  这是**预期**的：用更多片段换取每一块的语境完整性，检索精度优先于片段数量。
- **顺带清理**：`|===` 标记不再残留到入库文本里。

## 已知局限

- AsciiDoc 的 `xref:目标[显示文本]` 被转成纯显示文本（`See URI patterns`），
  **丢失了跳转目标**（`#mvc-ann-requestmapping-uri-templates`）。
  模型因此无法顺着链接找到 `/owners/{ownerId}` 示例——本例中该示例
  只能靠 `ann-requestmapping.adoc` 自身被独立召回（实测排名 2~4）。
  后续可考虑把目标锚点也保留为文本线索。
- 若单元格内含 `|` 字面量（未转义），按 `|` 切分会错位；Spring 文档中未出现此情况。
- 表格内嵌套代码块（`[source,java]` + `----`）尚未特殊处理。

## 验证

- `arguments.adoc` 转换后，Method Arguments 表被切成 10 块，**每块均带表头与分隔行**；
  `@PathVariable` 落在带表头的片段内，并与 `@MatrixVariable`、`@RequestParam` 同行可见
  （恰好覆盖用户所问的对比场景）。
- 离线测试（`tests/test_smoke.py`）新增 3 项：AsciiDoc 表格转 Markdown、
  表格切分逐块保留表头且不丢行、普通文档切分时表格独立成片段。全部通过（共 15 项）。
