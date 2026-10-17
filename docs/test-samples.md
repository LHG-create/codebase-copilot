# 测试样例

一组**可直接复制粘贴**的问题，覆盖系统全部能力，并写明「预期看到什么」。
文中「实测」列均在 **2026-10-17、`top_k=4`、开启领域路由** 的条件下跑出，
不是凭印象写的。

> 前置条件：`storage/` 里已有建好的索引（945 个文档 / 约 1 万片段）。
> 没建过就先跑 `ingest.py`，否则所有入口都会提示「向量库不存在」。

## 怎么用

| 入口 | 启动方式 |
|---|---|
| **网页（推荐，效果完整）** | 运行 `10. 网页界面 web（含 API）`，浏览器打开 `127.0.0.1:8000` |
| 网页（极简备用） | 运行 `3. 网页界面 app_ui`，或 `streamlit run app_ui.py` |
| 命令行 | 运行 `2. 命令行问答 chat_cli`，命令：`/clear` 清历史、`/history` 看历史、`q` 退出 |
| HTTP | 运行 `10. 网页界面 web（含 API）`，`127.0.0.1:8000/docs` 在线试（示例见 README「HTTP 接口怎么用」） |

先按下面的顺序试，能最快看出系统各层是否都还活着：
**冒烟 → 项目代码域 → 官方文档域 → 澄清 → 比较类 → 边界 → Agent**。

> 新版网页支持深链接：`http://127.0.0.1:8000/?q=你的问题`，直接跳过输入框发问，方便分享与复现。

---

## 一、冒烟（3 条，先确认链路通）

| # | 问题 | 预期 | 实测 |
|---|---|---|---|
| 1 | `这个项目用的是什么向量库？` | 答 FAISS，并给出出处；不应说「不知道」 | 第 2 位 `vectorstore.py::load_vectorstore` |
| 2 | `RAG 问答链是怎么把检索和生成串起来的？` | 提到 `RagAssistant`、`PROMPT \| get_chat_model() \| StrOutputParser()`，答案下方列出处 | 文件级第 1 |
| 3 | `Spring 的 bean 作用域有哪几种？` | 答 singleton / prototype 等，验证**官方文档域**也通 | 文档级第 1 |

> 三条分别验证「能答本项目」「能答官方文档」「都能给出处」。
> 想看引用编号长什么样，直接看第 2 条回答下方的 `[1] [2]`。

---

## 二、项目代码域（考察符号级定位）

这些问题的答案都在**本项目源码**里，用来验证「引用能精确到函数名 + 行号」。
每条都标注了预期命中的符号。

| # | 问题 | 预期符号 / 要点 | 实测符号排名 |
|---|---|---|---|
| 4 | `系统提示词在哪里定义，要求助手遵守什么规则？` | `chain.py::SYSTEM_PROMPT`，规则含「只依据资料」「标注引用编号」「不知道就说不知道」 | **1** |
| 5 | `函数签名是怎么从 AST 还原出来的？` | `code_splitter.py::_signature_of`：从 `args.posonlyargs + args.args` 重建形参，默认值用 `ast.unparse` 还原 | **1** |
| 6 | `markdown 里的代码块是怎么被保护不被切碎的？` | `loader.py::_split_text_protecting_code`：先把 ``` 代码块换成占位符，切完再还原，切分边界不会落进代码块 | **1** |
| 7 | `哪些文件类型是支持加载的？` | `loader.py::SUPPORTED_SUFFIXES` = `.txt .md .mdx .adoc .pdf .py` | **1** |
| 8 | `项目的全局配置从哪里读取，默认对话模型是什么？` | `config.py::Settings`，pydantic-settings 读 `.env`，默认 `deepseek` / `deepseek-chat` | **1** |
| 9 | `加载向量库时如果没建过会怎样？` | `vectorstore.py::load_vectorstore` 抛 `FileNotFoundError` 并提示先跑 `ingest.py` | **1** |
| 10 | `对话模型是怎么根据配置选择供应商的？` | `llm.py::get_chat_model`，按 `settings.llm_provider` 分支 | **1** |
| 11 | `Streamlit 界面的 load_assistant 是干嘛的？` | `app_ui.py::load_assistant`，用 `st.cache_resource` 缓存，避免每次 rerun 重建 | **1** |

**看点**：展开出处，行号是精确的（如 `chain.py:123-140`），不是「整文件」；
代码片段还带 `kind`（function / class / module）。

---

## 三、官方文档域（考察领域路由）

这些问题只提到**一个**框架。预期：系统认出框架名 → 检索限定到该域 → 不会混进别家的文档。

| # | 问题 | 预期命中文档 | 实测 |
|---|---|---|---|
| 12 | `FastAPI 怎么配置跨域 CORS？` | `fastapi/tutorial/cors.md` | **第 1** |
| 13 | `FastAPI 的依赖注入 Depends 是怎么用的？` | `fastapi/tutorial/dependencies/index.md` | **第 1** |
| 14 | `Spring 的依赖注入有哪几种方式？` | `spring-framework/core/beans/dependencies/factory-collaborators.adoc`（构造器 / Setter 两种） | **第 1** |
| 15 | `Spring 的 bean 作用域有哪几种？` | `spring-framework/core/beans/factory-scopes.adoc` | **第 1** |
| 16 | `Spring MVC 控制器方法的参数是怎么绑定的？` | `spring-framework/web/webmvc/mvc-data-binding.adoc` | **第 1** |
| 17 | `Vue 组件有哪些生命周期钩子？` | `vuejs/guide/essentials/lifecycle.md` | **第 1** |
| 18 | `Vue 表单输入怎么做双向绑定？` | `vuejs/guide/essentials/forms.md` | **第 1** |
| 19 | `LangGraph 的检查点 checkpointer 是用来做什么的？` | `langgraph/checkpointers.md` | **第 1** |

第 14 条值得单独提：这份文档是**评测标注被修正后的正确答案**。
原来的标注指向 `dependencies.adoc`，但它只是 474 字节的章节导语页
（正文是「接下来这一节将介绍…」），真正的 DI 内容在 `factory-collaborators.adoc`。
这条评测项此前「从写下第一天就不可能通过」，与检索算法无关（见 ADR-012 / ADR-014）。

---

## 四、澄清追问（多框架、非比较）

问题里同时提到两个框架，但**不是在比较**。预期：不猜，返回选项让你点。

| # | 问题 | 预期 | 实测 |
|---|---|---|---|
| 20 | `Spring 和 FastAPI 的依赖注入各怎么写？` | 不直接回答，返回两个选项：FastAPI / Spring Framework | 触发澄清 |
| 21 | `Vue 和 Spring 各自怎么处理请求？` | 返回 Vue.js / Spring Framework 两个选项 | 触发澄清 |

**怎么验证「点选真的生效」**：

1. 在网页端点其中一个选项 → 问题会自动带上该领域重放，答案只来自选定技术栈；
2. 命令行里输入序号（如 `1`）→ 打印「→ 已限定到「FastAPI」」后再回答；
3. HTTP 里拿 `clarification.options[].domain`，带着 `forced_domain` 重发一次：

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"Spring 和 FastAPI 的依赖注入各怎么写？","forced_domain":"fastapi"}'
```

> 选 `0`（都不限定）走全库检索，用来对比「不澄清会怎样」——见下一节。

---

## 五、比较类问题（不追问、两侧取数）

这是**刻意与第四节相反**的设计：问题是在比较，答案天然需要两边资料，
所以既不该追问，也不该收窄，而要**按域配额各取几条**。

| # | 问题 | 预期 | 实测 |
|---|---|---|---|
| 22 | `Spring 和 FastAPI 的依赖注入有什么区别？` | 不触发澄清；出处里**两个技术栈都有**（约各 2 条）；回答分两边讲 | FastAPI 2 + Spring 2 |
| 23 | `Vue 和 React 的响应式有什么区别？` | 只认到 `vuejs`（React 不在知识库里），因此只答 Vue 一侧，**并说明缺少 React 资料** | 限定到 vuejs |

第 22 条是这个系统一个很典型的坑：**如果只改「不追问」不改取数**，
走全局检索时 top-4 会**全被 FastAPI 占满**（「依赖注入」在 FastAPI 文档里匹配度更高），
模型只能如实回答「资料不足，无法比较」——把一道能答的题变成答不了。

第 23 条是**越界样例**：知识库里没有 React，正确答案是「只答 Vue + 明确说明另一侧缺失」，
而不是编一段 React 的内容。

---

## 六、无线索与边界

| # | 问题 | 预期 | 实测 |
|---|---|---|---|
| 24 | `依赖注入怎么用？` | 没有任何框架线索 → 走全局，**不追问**；答案通常会集中讲 FastAPI，Spring 可能不出现 | 全局；命中 FastAPI 3 条 + Vue 1 条 |
| 25 | `怎么定义一个异步接口？` | 同上，走全局；结果会横跨 Spring / LangGraph / LangChain | 全局，4 条来自 3 个域 |
| 26 | `今天天气怎么样？` | 应明确说「根据资料无法回答」，不编造 | 拒答成功；但仍列出 4 条无关出处（LangGraph 文档里的 `get_weather` 示例）—— 检索层不知道「库里没有」，只能靠提示词兜 |
| 27 | 空问题（API 传 `{"question":""}`） | HTTP **422**，不进入检索 | 422 |

第 24 条是**已知短板的最好例证**：系统只能「认」问题里写出来的框架名，
写不出来时它不会主动问。这条不是 bug，是 ADR-015 明确记录的下一步。

---

## 七、代码修改 Agent（默认不写盘）

| # | 任务 | 预期 | 实测 |
|---|---|---|---|
| 28 | `在 app/config.py 里给 api_port 补一句注释，说明为什么默认端口是 8000` | 走「读文件 → 提交修改 → 跑测试」，最后打印 unified diff，**磁盘无改动**；输入 `y` 才落盘 | 9 步完成，自动跑出 37 passed |
| 29 | `给 app/domain.py 的 DOMAIN_LABELS 加一行中文注释` | 同上，只出 diff | 待手动确认 |
| 30 | `把 .env 里的模型换成 gpt-4o` | **直接拒绝**（`.env` 在禁写名单里） | 拒绝 |
| 31 | `读取 ../../Windows/win.ini` | **直接拒绝**（路径越界） | 拒绝 |

跑完想撤销：在 Agent 里输入 `/revert`（只还原本次会话写过的文件）。

---

## 八、已知会失败的样例（故意保留，别当 bug）

这三条建议**亲手跑一遍**——它们比成功的样例更能说明系统的边界在哪。

| # | 问题 | 现象 | 原因 |
|---|---|---|---|
| 32 | `命令行问答怎么循环接收用户输入？` | 检索不到 `chat_cli.py::main` | 索引是 `chat_cli.py` 重构**之前**建的，旧片段对不上；重跑 `ingest.py` 即可 |
| 33 | `LangChain 的 agent 是怎么工作的？` | 有时第 1、有时未命中 | **改写质量**：同一问题会被改写成 `agent workflow reasoning action observation` 或 `... tool calling`，正确片段在英文路可能掉到第 95 位 |
| 34 | `代码 Agent 是怎么防止写坏文件的？` | 答不出实现细节 | `app/code_agent.py`（连同 `graph.py` / `api.py` / `domain_router.py`）是后加的，**尚未入索引**（重建索引要重新 embedding 近万片段，故暂缓） |

第 33 条已写进 ADR-013 的遗留问题：正确片段在**原问题路**排第 3，
在**改写路**掉到第 95 —— 这不是排序算法的问题，是改写 prompt 的问题。

---

## 九、一键自动化核对（不用手点）

```bash
.venv\Scripts\python -m pytest tests -q      # 38 项离线测试，约 4 秒，不烧 API Key
.venv\Scripts\python eval_retrieval.py       # 双域命中率与 MRR（会调 embedding，约 1–2 分钟）
.venv\Scripts\python eval_routing.py         # 四种路由策略对比 + 澄清触发统计（较慢，建议后台跑）
```

对应的 PyCharm 运行配置：`4. 测试 pytest`、`5. 评测 eval_retrieval`、`9. 领域路由 eval_routing`。

## 十、前端界面自检（10 项，2 分钟）

跑 `10. 网页界面 web（含 API）` 打开 `127.0.0.1:8000` 后，按顺序点一遍：

| # | 操作 | 预期 |
|---|---|---|
| 1 | 打开首页 | 左「知识域」7 个、片段数约 9,999；顶部「索引就绪」绿灯 |
| 2 | 点第一条示例卡 | 问题自动发出，先出现「正在判定领域…」，再流式出字 |
| 3 | 看回答下方 | 出处卡片带领域色条、相对路径、符号名、行号；悬停可展开 |
| 4 | 点答案里的 `[2]` | 自动滚动到第 2 张卡片并高亮闪一下 |
| 5 | 看右栏 | 「状态图轨迹」有 3–4 个节点与各自耗时；「本次路由」显示技术栈 |
| 6 | 点右上角月亮/太阳 | 深浅主题切换，刷新后仍保持 |
| 7 | 刷新页面 | 对话与出处都还在（`sessionStorage`） |
| 8 | 往上滚 | 出现「回到底部」浮层，点击回到底 |
| 9 | 点右上角下载图标 | 下载 `codebase-chat-YYYY-MM-DD.md`，含问题、回答与出处 |
| 10 | 输入 `Spring 和 FastAPI 的依赖注入各怎么写？` | 出现两张选项卡；点 FastAPI 后**不会重复插入同一条提问**，答案只来自 FastAPI |

窄屏可用浏览器设备模拟（iPhone 13 / 414px）再看一遍：侧栏与轨迹面板变成抽屉，首屏不会被输入框顶飞。

## 关于排名抖动（为什么第 2–4 位会变）

检索链里有一步**中文问题 → 英文关键词**的改写，它调的是对话模型，
`LLM_TEMPERATURE=0.2`，所以同一问题两次跑出来的关键词可能不同：
例如 `Vue 3 的响应式系统是怎么实现的？` 会得到
`reactivity system implementation` 或 `... proxy ref reactive effect`。

因此：

- **路由判定、是否澄清**是纯字符串匹配，**完全确定**，可以当断言用；
- **第 1 位**通常稳定（差 3–5 个排名位不会翻转它），可以当预期；
- **第 2–4 位**会抖动，别拿它判断「系统坏了」。

`eval_retrieval.py` 为此把改写结果**缓存到 `.eval_cache/query_rewrites.json`**，
保证评测数字可复现——代价是那份缓存是「某一次改写」的快照，
与即时运行的结果可能有细微差异。这一点在 ADR-015 里也记了一笔。
