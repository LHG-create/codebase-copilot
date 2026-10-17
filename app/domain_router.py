"""领域路由与澄清追问（Domain Routing & Clarification）。

为什么要做这件事 —— 被评测数据反复暴露的一个真实短板：

知识库里现在有 7 个互不相干的知识域：本项目代码 + FastAPI / LangChain /
LangGraph / Spring Boot / Spring Framework / Vue.js 六套官方文档。问
「Spring 的依赖注入有哪几种方式」时，检索会在整个库里找「依赖注入」，
而 FastAPI 的 dependencies 文档、本项目代码的中文注释里都在讲依赖注入；
同理「怎么配置 CORS」在 Spring WebFlux 和 FastAPI 下各有一份文档。

**这不是排序算法不够好，而是问题本身就缺了一个关键信息：问的是哪个技术栈。**

两次对照实验说明了这一点：

1. 用「领域质心向量」做语义路由 —— **已证伪**。本项目代码本身就是用 LangChain
   写的，术语与 langchain 域高度重叠，质心分不开两者（见 eval_domain.py 的 D 组）。
2. 用「问题里的显式线索」做路由 —— **有效**。实测「FastAPI 怎么配置跨域 CORS？」
   的正确答案在英文路候选池里排第 1，却在跨域 RRF 融合后被 Spring 的
   webflux-cors 挤出 top-4；而只要认出问题里的「FastAPI」并把检索限定进
   fastapi 域内，它立刻回到第 1 位。

所以本模块做两件事：

- **认（route）**：从问题里抽取显式线索 —— 框架名（FastAPI / Spring / Vue …）、
  本项目自称（"本项目"、"我的代码"）、本项目代码符号（RagAssistant、format_docs）。
  线索唯一命中时，把检索限定到该领域。
- **问（clarify）**：线索指向多个领域时，不猜，而是把选项摆给用户。

为什么是「认 + 问」而不是「猜」：

  猜错的代价是不对称的。猜对只是略快一点，猜错却会给出错误技术栈的答案，
  而且用户很难察觉 —— Spring 的 CORS 配置和 FastAPI 的完全是两回事，
  但两段答案「看起来都对」。反过来，用户问「依赖注入怎么用」时，
  本来就该被告知还需要补充信息。这是交互层的责任，不是排序层的。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain import DOMAIN_LABELS

# ================================================================ 1. 线索表

# 框架别名 -> 领域。**顺序敏感**：长别名必须排在短别名前面，
# 否则 "Spring Boot 的问题" 会先被 "spring" 匹配成 spring-framework 域。
# 抽取时命中的别名会从文本中"消费"掉，因此一个词只会贡献一次。
FRAMEWORK_ALIASES: tuple[tuple[str, str], ...] = (
    ("fastapi", "fastapi"),
    # Spring Boot 必须先于 Spring Framework 判定
    ("spring boot", "spring-boot"),
    ("springboot", "spring-boot"),
    ("spring mvc", "spring-framework"),
    ("spring webflux", "spring-framework"),
    ("spring framework", "spring-framework"),
    ("spring core", "spring-framework"),
    ("spring", "spring-framework"),
    ("vue.js", "vuejs"),
    ("vuejs", "vuejs"),
    ("vue 3", "vuejs"),
    ("vue3", "vuejs"),
    ("vue", "vuejs"),
    ("langchain", "langchain"),
    ("langgraph", "langgraph"),
)

# 指代「本项目自己」的说法 —— 命中即路由到 project 域
PROJECT_SELF_REFERENCE: tuple[str, ...] = (
    "本项目",
    "这个项目",
    "该项目",
    "项目里",
    "项目内",
    "项目代码",
    "我的代码",
    "自己的代码",
    "本仓库",
    "这个仓库",
    "本工程",
)

# ================================================================ 2. 线索抽取


def _normalize(text: str) -> str:
    return (text or "").lower().replace("　", " ")


def extract_framework_hints(question: str) -> list[str]:
    """抽出问题里提到的框架领域（去重、保序）。

    做的是「全词消费」而不是「正则扫描」：命中的别名会从剩余文本里抹掉，
    于是 "Spring Boot 怎么配置" 只会得到 ["spring-boot"]，不会再被 "spring"
    补一个 spring-framework —— 那会让本来明确的问题无端变成歧义。
    """
    remaining = _normalize(question)
    found: list[str] = []
    for alias, domain in FRAMEWORK_ALIASES:
        if alias in remaining:
            if domain not in found:
                found.append(domain)
            remaining = remaining.replace(alias, " ")
    return found


def looks_like_code_token(token: str) -> bool:
    """判断一个词是否"像代码标识符"而非普通英文单词。

    这是本项目符号线索的准入条件。必要性在于：符号表里的词要拿去和用户的
    中文问题做子串匹配，如果放进了 `config`、`loader` 这种普通英文单词，
    那么「Spring 怎么配置 config」会被误判成在问本项目代码。

    条件（满足其一）：含下划线 / 以 .py 结尾 / 含大写字母（驼峰）。
    普通小写英文单词一律不算。
    """
    if not token or len(token) < 4:
        return False
    if "_" in token or token.endswith(".py"):
        return True
    return any(ch.isupper() for ch in token)


def extract_project_hint(question: str, project_symbols: set[str]) -> bool:
    """问题里是否出现了「本项目」的自称，或本项目代码的符号 / 文件名。"""
    text = _normalize(question)
    if any(ref in text for ref in PROJECT_SELF_REFERENCE):
        return True
    return any(symbol in text for symbol in project_symbols)


# ================================================================ 3. 路由决策


@dataclass
class RouteDecision:
    """一次路由的判定结果。

    domains     命中的领域列表（去重、保序）
    comparison  问题是否在要求「比较两个技术栈」
    """

    domains: list[str] = field(default_factory=list)
    comparison: bool = False

    @property
    def is_ambiguous(self) -> bool:
        """线索指向两个及以上领域 —— 此时应当澄清而不是猜。

        **例外：比较类问题不澄清。** 「Spring 和 FastAPI 的依赖注入有什么区别」
        这种问法同时提到两个框架，但它的答案**本来就需要两边的内容**，
        把它限定到单一领域只会让模型回答「资料里没有 Spring」。实测过：
        澄清选 FastAPI 后，模型如实回答「参考资料仅介绍了 FastAPI……
        完全没有提及 Spring，因此无法进行比较」—— 澄清反而把能答的问题
        变成了答不了。所以比较意图下走全局检索，让两边的文档都进候选池。
        """
        return len(self.domains) >= 2 and not self.comparison

    @property
    def primary(self) -> str | None:
        return self.domains[0] if len(self.domains) == 1 else None

    def __bool__(self) -> bool:
        return bool(self.domains)


# 比较意图 —— 命中则不应把问题收窄到单一领域
_COMPARISON_RE = re.compile(
    r"区别|差异|不一样|不同点|对比|比较|相比|谁更|哪个更好|哪个好"
    r"|\bvs\.?\b|\bversus\b|\bdifference\b|\bcompare\b",
    re.IGNORECASE,
)


def is_comparison(question: str) -> bool:
    """问题是否在要求比较（需要同时用到多个技术栈的资料）。"""
    return bool(_COMPARISON_RE.search(question or ""))


def decide_route(question: str, project_symbols: set[str] | None = None) -> RouteDecision:
    """判定问题应当路由到哪个领域。

    注意：**返回空 RouteDecision 是常态，不是失败**。用户问
    「怎么用 Pydantic 定义请求体」这种不带任何线索的问题时，就应当
    老老实实走全局检索，而不是硬塞进某个域。
    """
    domains = extract_framework_hints(question)
    if extract_project_hint(question, project_symbols or set()) and "project" not in domains:
        domains.append("project")
    return RouteDecision(domains=domains, comparison=is_comparison(question))


def build_clarification(question: str, decision: RouteDecision) -> dict:
    """把歧义判定转成前端可渲染的澄清请求。

    返回结构里带上 `question` 是为了让前端在用户点选后能原样重放问题，
    只补一个领域约束 —— 用户不需要重打一遍。
    """
    return {
        "type": "clarify",
        "question": question,
        "prompt": "这个问题在多个知识域下都有内容，你想问的是哪一个？",
        "options": [
            {"domain": domain, "label": DOMAIN_LABELS.get(domain, domain)}
            for domain in decision.domains
        ],
    }


# ================================================================ 4. 符号表


def collect_project_symbols(store) -> set[str]:
    """从索引里收集本项目代码的符号名 / 文件名，供线索匹配用。

    只收 project 域的片段，且只保留「像代码」的词（见 looks_like_code_token）。
    近万片段里筛一遍约几十毫秒，调用方（Retriever）只需做一次并缓存。
    """
    from app import domain as dm

    symbols: set[str] = set()
    for doc in dm.collect_documents(store):
        if dm.infer_domain(doc.metadata.get("source", ""), doc.metadata) != "project":
            continue
        for key in ("name", "file_name"):
            raw = str(doc.metadata.get(key) or "").strip()
            if not raw or raw == "(module)":
                continue
            if looks_like_code_token(raw):
                symbols.add(raw.lower())
    return symbols
