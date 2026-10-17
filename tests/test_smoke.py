"""离线冒烟测试：不调用任何大模型 API，验证「加载 -> 切分 -> 入库 -> 检索」链路。

运行：pytest -v
"""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# FAISS 的 C++ 层用 fopen 写文件，不接受中文路径。
# 系统临时目录（D:\Temp\pytest-of-林海亘）带中文，所以这里固定用项目内的英文路径。
TEST_STORAGE = PROJECT_ROOT / ".test_storage"


class HashEmbeddings(Embeddings):
    """伪向量：把文本哈希成固定长度向量，只为跑通链路，不代表语义。"""

    def __init__(self, size: int = 128) -> None:
        self.size = size

    def _to_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (digest * (self.size // len(digest) + 1))[: self.size]
        return [byte / 255.0 for byte in repeated]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._to_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._to_vector(text)


def test_config_loads():
    from app.config import settings

    assert settings.chunk_size > 0
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.top_k > 0


def test_split_documents(tmp_path):
    from app.loader import split_documents

    doc = Document(page_content="苹果是一种常见水果，富含维生素。" * 200, metadata={"file_name": "fruit.md"})
    # 传入空目录，避免 split_documents 去扫描 data/ 里的 .py 文件干扰断言
    chunks = split_documents([doc], data_dir=tmp_path)

    assert len(chunks) > 1
    assert all(chunk.metadata["file_name"] == "fruit.md" for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))


def test_load_documents_filters_suffix(tmp_path):
    from app.loader import load_documents

    # 内容要超过 MIN_BODY_CHARS，否则会被「空壳文档」过滤规则挡掉（见 ADR-012）
    (tmp_path / "a.md").write_text("LangChain 是编排框架。" * 30, encoding="utf-8")
    (tmp_path / "b.txt").write_text("FAISS 是向量检索库。" * 30, encoding="utf-8")
    (tmp_path / "c.csv").write_text("ignored", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert len(docs) == 2
    assert {doc.metadata["file_name"] for doc in docs} == {"a.md", "b.txt"}
    assert all(doc.metadata["source"] for doc in docs)


def test_vectorstore_roundtrip(tmp_path):
    from app.loader import split_documents
    from app.vectorstore import build_vectorstore, load_vectorstore

    embeddings = HashEmbeddings()
    docs = [
        Document(page_content="LangChain 用于构建大模型应用。", metadata={"file_name": "a.md"}),
        Document(page_content="今天天气不错，适合出门散步。", metadata={"file_name": "b.md"}),
    ]
    chunks = split_documents(docs, data_dir=tmp_path)

    build_vectorstore(chunks, TEST_STORAGE, embeddings=embeddings)
    store = load_vectorstore(TEST_STORAGE, embeddings=embeddings)

    assert store.index.ntotal == len(chunks)
    assert len(store.similarity_search("LangChain", k=1)) == 1
    assert (TEST_STORAGE / "index.faiss").exists()


def test_load_vectorstore_missing_raises():
    import pytest

    from app.vectorstore import load_vectorstore

    with pytest.raises(FileNotFoundError):
        load_vectorstore(PROJECT_ROOT / ".test_storage_not_built")


def test_contains_chinese():
    """查询改写的中文字符检测是纯逻辑，可离线验证。"""
    from app.query_rewriter import contains_chinese

    assert contains_chinese("FastAPI 怎么定义路径参数") is True
    assert contains_chinese("路径参数") is True
    assert contains_chinese("path parameters declare") is False
    assert contains_chinese("build_vectorstore 做了什么") is True
    assert contains_chinese("RagAssistant") is False
    assert contains_chinese("") is False


def test_adoc_to_text():
    """AsciiDoc 转纯文本：锚点/属性/include 删除，标题/代码块转换，行内标记清理。"""
    from app.adoc import adoc_to_text

    src = """[[beans]]
= The IoC Container
:page-section-summary-toc: 1

This chapter covers IoC.

[source,java]
----
\tpublic interface Foo {}
----

Implement the `Foo` interface. See xref:types[Advice Types].

include::other.adoc[]

include-code::config/MyConfig[]

See the {code-spring-boot-src}/autoconfigure/ dir.

*This is bold* and _italic_.
"""
    out = adoc_to_text(src)

    assert "[[beans]]" not in out
    assert ":page-section-summary-toc:" not in out
    assert "include::" not in out
    assert "include-code::" not in out
    assert "{code-spring-boot-src}" not in out
    assert "[source" not in out
    assert "= The IoC Container" not in out
    assert "# The IoC Container" in out
    assert "```" in out  # 代码块分隔
    assert "`Foo`" in out  # 行内代码保留
    assert "Advice Types" in out  # xref 提取显示文本
    assert "This is bold" in out
    assert "italic" in out
    assert "*This is bold*" not in out


def test_adoc_preserves_multiply_and_list():
    """乘法号与列表星号不能被误删。"""
    from app.adoc import adoc_to_text

    out = adoc_to_text("* item one\n* item two\n\nresult = 2 * 3\n")
    assert "* item one" in out
    assert "2 * 3" in out


def test_extract_code():
    """从用户输入里提取代码块（案例沉淀用）。"""
    from app.case_store import extract_code

    assert extract_code("```python\nx = 1\n```") == "x = 1"
    assert extract_code("```\ny = 2\n```") == "y = 2"
    assert extract_code("只有文字，没有代码块") == ""
    assert extract_code("") == ""


def test_case_store_save_is_idempotent_and_tagged(tmp_path):
    """案例沉淀：内容 hash 幂等复用同一文件；切分后带 origin=user-case 标记。"""
    from app.case_store import extract_code, save_case, split_case_file

    question = "这段 FastAPI 代码哪里错了？\n```python\ndef get_user(user_id):\n    pass\n```"
    answer = "路径参数缺少类型注解。"

    p1 = save_case(question, answer, raw_code=extract_code(question), cases_dir=tmp_path)
    p2 = save_case(question, answer, raw_code=extract_code(question), cases_dir=tmp_path)

    assert p1 == p2, "同一问答应复用同一文件（幂等）"
    assert p1.exists()

    text = p1.read_text(encoding="utf-8")
    assert "# 用户问题" in text
    assert "# 参考解答" in text
    assert "origin: user-qa" in text

    docs = split_case_file(p1)
    assert docs
    assert all(d.metadata.get("origin") == "user-case" for d in docs)


def test_append_documents_grows_index(tmp_path):
    """增量追加：不重建索引即可让新片段进入库（用伪向量离线验证）。"""
    from app.loader import split_documents
    from app.vectorstore import append_documents, build_vectorstore, load_vectorstore

    embeddings = HashEmbeddings()
    base = split_documents(
        [Document(page_content="LangChain 是编排框架。", metadata={"file_name": "a.md"})],
        data_dir=tmp_path,
    )
    build_vectorstore(base, TEST_STORAGE, embeddings=embeddings)

    store = load_vectorstore(TEST_STORAGE, embeddings=embeddings)
    before = store.index.ntotal

    new_docs = [
        Document(
            page_content="用户沉淀的案例内容。",
            metadata={"file_name": "case.md", "origin": "user-case"},
        )
    ]
    added = append_documents(store, new_docs, TEST_STORAGE)

    assert added == 1
    assert store.index.ntotal == before + 1


def test_loader_tags_case_directory(tmp_path):
    """data/cases/ 下的文档应被标记 origin=user-case，与官方文档区分。"""
    from app.loader import load_documents

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    # 同样要超过 MIN_BODY_CHARS，避免被空壳文档过滤规则挡掉（见 ADR-012）
    (cases_dir / "case-x.md").write_text(
        "# 用户问题\n某问题\n" + "案例正文内容。" * 40, encoding="utf-8"
    )
    (tmp_path / "doc.md").write_text("普通文档内容。" * 40, encoding="utf-8")

    by_name = {d.metadata["file_name"]: d for d in load_documents(tmp_path)}

    assert by_name["case-x.md"].metadata.get("origin") == "user-case"
    assert by_name["doc.md"].metadata.get("origin") is None


def test_adoc_table_to_markdown():
    """AsciiDoc 表格应转成带表头的 Markdown 表格，且行间不插空行。"""
    from app.adoc import adoc_to_text

    src = """[cols="1,2",options="header"]
|===
| Argument | Description

| `@PathVariable`
| For access to URI template variables.

| `@RequestParam`
| For access to the Servlet request parameters.
|===
"""
    out = adoc_to_text(src)

    assert "| Argument | Description |" in out
    assert "| --- | --- |" in out
    # 每行应为自包含的「参数 | 说明」，而不是被空行拆散的孤儿行
    assert "| `@PathVariable` | For access to URI template variables. |" in out
    assert "| `@RequestParam` | For access to the Servlet request parameters. |" in out
    assert "|===" not in out


def test_table_rows_keep_header():
    """表格被切成多块时，每一块都必须带上表头，且不能丢行。"""
    from app.loader import _split_table_block

    rows = ["| arg | desc |", "| --- | --- |"]
    rows += [
        f"| `@Arg{i}` | Description number {i}, long enough to force splitting. |"
        for i in range(1, 25)
    ]
    block = "\n".join(rows)

    pieces = _split_table_block(block, chunk_size=300)

    assert len(pieces) > 1, "应被切成多块（否则这个用例验证不到表头注入）"
    for piece in pieces:
        assert piece.startswith("| arg | desc |"), "每块都必须以表头开头"
        assert "| --- | --- |" in piece

    joined = "\n".join(pieces)
    for i in range(1, 25):
        assert f"`@Arg{i}`" in joined, f"@Arg{i} 不能在切分中丢失"


def test_split_keeps_table_intact():
    """普通文档切分时，表格应作为独立片段，且与正文互不混切。"""
    from app.loader import _split_text_protecting_code

    text = "引言段落。\n\n| key | value |\n| --- | --- |\n| `@PathVariable` | URI template variables. |\n\n结尾段落。\n"
    docs = _split_text_protecting_code(text, {"file_name": "t.md"})

    table_docs = [d for d in docs if d.page_content.lstrip().startswith("|")]
    assert len(table_docs) == 1
    assert "@PathVariable" in table_docs[0].page_content
    assert table_docs[0].page_content.startswith("| key | value |")


def test_infer_domain():
    """领域识别：从文档路径推断所属技术栈（跨领域干扰治理的基础，见 ADR-011）。"""
    from app.domain import infer_domain

    # 官方文档
    assert infer_domain(r"D:\p\rag\data\docs\fastapi\tutorial\path-params.md") == "fastapi"
    assert infer_domain("/p/rag/data/docs/spring-framework/web/webmvc.adoc") == "spring-framework"
    assert infer_domain("/p/rag/data/docs/spring-boot/ROOT/index.adoc") == "spring-boot"
    assert infer_domain("/p/rag/data/docs/vuejs/guide/essentials/computed.md") == "vuejs"

    # 本项目代码：data/codebase 副本、项目里的 app/、以及 data/ 下的笔记
    assert infer_domain("/p/rag/data/codebase/chain.py") == "project"
    assert infer_domain("/p/rag/app/chain.py") == "project"
    assert infer_domain("/p/rag/data/rag-实践笔记.md") == "project"

    # 用户沉淀案例（元数据标记优先于路径）
    assert infer_domain("/p/rag/data/cases/x.md") == "case"
    assert infer_domain("/p/rag/data/docs/fastapi/a.md", {"origin": "user-case"}) == "case"

    # 完全无法归类的才进 other
    assert infer_domain("/somewhere/else/readme.md") == "other"


def test_domain_prior_prefers_small_domains():
    """领域规模先验：只给长尾领域加权，片段越少加成越高。"""
    from app.retrieval import domain_prior_map

    sizes = {"project": 82, "langgraph": 779, "fastapi": 1071, "spring-framework": 3591}
    prior = domain_prior_map(sizes, alpha=0.06)

    # 只有占比 ≤5% 的长尾领域（本项目代码 82 / 5523 ≈ 1.5%）拿到加成
    assert set(prior) == {"project"}
    # 加成幅度要落在 RRF 分数的量级上（RRF 单路第一约 1/(60+1) ≈ 0.016）：
    # 太小则不起作用，太大则会压过相关性本身
    assert 0.01 < prior["project"] < 0.05

    # 关键约束：先验不得在中等领域之间重新分配权重。
    # 否则 FastAPI(10.8%) 会压过 Spring Framework(36.1%)，
    # 问 Spring 注解时 top-4 被 FastAPI 文档占满（真实踩过的坑，见 ADR-011）。
    assert "fastapi" not in prior
    assert "spring-framework" not in prior
    assert "langgraph" not in prior

    # alpha=0 表示关闭先验；空输入不应崩
    assert domain_prior_map(sizes, alpha=0.0) == {}
    assert domain_prior_map({}, alpha=0.06) == {}


def test_rrf_fuse_dedups_and_ranks_by_both_channels():
    """RRF 融合：两路都出现的文档排最前，且结果去重。"""
    from langchain_core.documents import Document

    from app.retrieval import rrf_fuse

    def make(name: str):
        return Document(page_content=f"内容-{name}", metadata={"source": f"{name}.md"})

    a, b, c = make("a"), make("b"), make("c")
    fused = rrf_fuse([[a, b], [b, c]])

    keys = [doc.metadata["source"] for doc, _ in fused]
    assert len(fused) == 3, "同一片段在两条通道里出现，融合后只应保留一份"
    assert set(keys) == {"a.md", "b.md", "c.md"}
    assert keys[0] == "b.md", "两路都出现的片段应排最前"
    # 分数应单调递减
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_retriever_respects_domain_prior_switch():
    """Retriever 的开关应真实生效：关闭先验时不计算、不排序。"""
    from langchain_core.embeddings import Embeddings

    from app.retrieval import Retriever, domain_prior_map

    class TinyEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[float(len(t)), 1.0, 0.0] for t in texts]

        def embed_query(self, text):
            return [float(len(text)), 1.0, 0.0]

    # 只验证「先验表」这条链路，不依赖真实向量库
    domprior = domain_prior_map({"project": 5, "big": 5000}, alpha=0.06)
    assert domprior["project"] > 0
    assert "big" not in domprior, "非长尾领域不应拿到先验加成"

    # 关闭先验的 Retriever 不应再去统计领域
    from langchain_community.vectorstores import FAISS

    store = FAISS.from_texts(["hello world"], TinyEmbeddings())
    retriever = Retriever(store, top_k=1, use_query_rewrite=False, use_domain_prior=False)
    assert retriever.retrieve("hello") == list(store.similarity_search("hello", k=1))
    assert retriever._prior is None


def test_body_length_ignores_frontmatter():
    """正文长度必须忽略 YAML frontmatter —— 只剩 frontmatter 的页面就是空壳。

    这是 ADR-012 修的那个坑：langchain/retrieval.md 全文 96 字节、
    正文一个字都没有，却因为「内容非空」的旧判据混进了知识库，
    还被当成「LangChain 怎么做 RAG」的标准答案。
    """
    from app.loader import _body_length

    shell = "---\ntitle: Retrieval\nsidebarTitle: Retrieval\nurl: /oss/x\nnoindex: true\n---\n"
    assert _body_length(shell) == 0
    assert _body_length("") == 0

    with_fm = "---\ntitle: X\n---\n\n# 正文\n\n" + "content " * 40
    assert _body_length(with_fm) > 200

    # 没有 frontmatter 的文档（如 .adoc 转换结果）直接按全文算
    assert _body_length("# Title\n\n" + "abc " * 80) > 200


def test_excluded_nav_files():
    """nav-*.adoc 这类纯导航片段、以及 changelog 类元信息文件都应被排除。"""
    from pathlib import Path

    from app.loader import _is_excluded

    assert _is_excluded(Path("data/docs/spring-boot/api/partials/nav-java-api.adoc"))
    assert _is_excluded(Path("data/docs/spring-framework/release-notes.md"))
    assert not _is_excluded(Path("data/docs/fastapi/tutorial/query-params.md"))


def test_load_documents_drops_empty_shell(tmp_path):
    """只剩 frontmatter 的空壳页、纯导航页不入库；有实质内容的文档照常保留。"""
    from app.loader import MIN_BODY_CHARS, load_documents

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "shell.md").write_text(
        "---\ntitle: Retrieval\nnoindex: true\n---\n", encoding="utf-8"
    )
    (docs_dir / "nav-x.adoc").write_text("= Nav\n\n* link\n", encoding="utf-8")
    (docs_dir / "real.md").write_text(
        "---\ntitle: Real\n---\n\n" + "这是一段有实质内容的正文。" * 30, encoding="utf-8"
    )

    names = {doc.metadata["file_name"] for doc in load_documents(docs_dir)}
    assert "shell.md" not in names, "只剩 frontmatter 的空壳页不该入库"
    assert "nav-x.adoc" not in names, "纯导航页不该入库"
    assert "real.md" in names, f"正文超过 {MIN_BODY_CHARS} 字的文档应保留"


def test_hit_file_accepts_multiple_answers():
    """文件级命中要支持「可接受答案集合」—— 泛化问题本来就该有多解（ADR-012）。"""
    from langchain_core.documents import Document

    from eval_retrieval import _hit_file

    doc = Document(
        page_content="x",
        metadata={"source": "/a/b/fastapi/tutorial/query-params.md"},
    )
    assert _hit_file(doc, "fastapi/tutorial/query-params.md")
    # 集合里任一对上就算命中
    assert _hit_file(
        doc,
        (
            "fastapi/tutorial/query-params-str-validations.md",
            "fastapi/tutorial/query-params.md",
        ),
    )
    assert not _hit_file(doc, ("a/b.md", "c/d.md"))


# ---------------------------------------------------------------- 领域路由
# ADR-013：从问题里「认」框架名并限定检索范围；线索指向多个技术栈时不猜，而是问。


def test_extract_framework_hints():
    from app.domain_router import extract_framework_hints

    assert extract_framework_hints("FastAPI 怎么配置跨域 CORS？") == ["fastapi"]
    assert extract_framework_hints("Spring 的依赖注入有哪几种方式？") == ["spring-framework"]
    assert extract_framework_hints("Vue 3 的响应式是怎么实现的？") == ["vuejs"]
    assert extract_framework_hints("LangGraph 的 checkpointer 是什么？") == ["langgraph"]
    assert extract_framework_hints("怎么用 Pydantic 定义请求体？") == []


def test_spring_boot_takes_priority_over_spring():
    """长别名必须先匹配并被「消费」掉，否则 Spring Boot 的问题会被误判成 spring-framework。"""
    from app.domain_router import extract_framework_hints

    assert extract_framework_hints("Spring Boot 怎么配置数据源？") == ["spring-boot"]
    # 同时提到两个，才应得到两个域（进而触发澄清）
    assert extract_framework_hints("Spring Boot 里怎么用 Spring MVC？") == [
        "spring-boot",
        "spring-framework",
    ]


def test_route_ambiguous_when_multiple_domains():
    from app.domain_router import decide_route

    decision = decide_route("Spring 和 FastAPI 的依赖注入各怎么写？")
    assert decision.is_ambiguous
    assert decision.primary is None
    assert set(decision.domains) == {"spring-framework", "fastapi"}


def test_comparison_question_is_not_ambiguous():
    """「A 和 B 有什么区别」不该触发澄清。

    这条是实测踩出来的：原先它对多域问题一律澄清，用户选了 FastAPI 之后，
    模型只能看到 FastAPI 的资料，于是回答「参考资料未提及 Spring，无法比较」——
    澄清把一个本来能答的问题变成了答不了。比较类问题的答案天然需要两边的内容，
    所以命中比较词时不收窄、也不追问，直接全库检索。
    """
    from app.domain_router import decide_route, is_comparison

    decision = decide_route("Spring 和 FastAPI 的依赖注入有什么区别？")
    assert decision.comparison is True
    assert decision.is_ambiguous is False
    assert set(decision.domains) == {"spring-framework", "fastapi"}
    # 仍然不该收窄到单一领域
    assert decision.primary is None

    assert is_comparison("Vue 和 React 相比哪个更好？")
    assert is_comparison("FastAPI vs Flask")
    assert not is_comparison("FastAPI 怎么配置 CORS？")


def test_route_no_hint_falls_back_to_global():
    from app.domain_router import decide_route

    decision = decide_route("怎么用 Pydantic 定义请求体？")
    assert not decision.domains
    assert decision.primary is None
    assert not decision.is_ambiguous
    assert not decision, "空决策应为假值，方便调用方直接回退全局检索"


def test_route_project_by_self_reference_and_symbol():
    from app.domain_router import decide_route

    assert decide_route("本项目的检索链路是什么？").primary == "project"
    # 本项目代码的符号名同样是线索（符号表由 collect_project_symbols 从库里取）
    assert decide_route("RagAssistant.answer 返回什么？", {"ragassistant"}).primary == "project"


def test_looks_like_code_token_rejects_plain_words():
    """普通小写英文单词不能进符号表，否则「Spring 怎么配置 config」会被误判成本项目代码。"""
    from app.domain_router import looks_like_code_token

    assert looks_like_code_token("format_docs")
    assert looks_like_code_token("RagAssistant")
    assert looks_like_code_token("chain.py")
    assert not looks_like_code_token("config")
    assert not looks_like_code_token("loader")
    assert not looks_like_code_token("ab")


def test_build_clarification_shape():
    from app.domain_router import build_clarification, decide_route

    question = "Vue 和 Spring 各自怎么处理请求？"
    clarification = build_clarification(question, decide_route(question))

    assert clarification["type"] == "clarify"
    assert clarification["question"] == question, "要带回原问题，用户点选后才能重放"
    assert len(clarification["options"]) == 2
    assert {opt["domain"] for opt in clarification["options"]} == {"vuejs", "spring-framework"}
    assert all(opt["label"] for opt in clarification["options"])


# ============================================================ 文档卫生：导语页


def test_lead_in_page_detected():
    """只预告下一节的整页文档应被判为导语页；有实质内容的短页不能误伤。"""
    from app.loader import looks_like_lead_in

    lead_in = (
        "A typical enterprise application does not consist of a single object. "
        "This next section explains how you go from defining a number of bean "
        "definitions that stand alone to a fully realized application where "
        "objects collaborate to achieve a goal."
    )
    assert looks_like_lead_in(lead_in)

    # 有真实分节内容的短页：不判为导语页
    real = (
        "# Dependency Injection\n\nThere are two major variants of DI: constructor-based "
        "and setter-based. Constructor-based DI is accomplished by the container calling "
        "a constructor with a number of arguments, each representing a dependency."
    )
    assert not looks_like_lead_in(real)

    # 有代码块 / 列表的不算导语页（说明有实质内容）
    assert not looks_like_lead_in("The next section shows this:/n/n```python\nx = 1\n```")
    assert not looks_like_lead_in(
        "The next sections cover these:/n/n- one\n- two\n"
    )

    # 太长的页面必然不是导语页
    long_body = "This next section explains everything. " + "细节。" * 200
    assert not looks_like_lead_in(long_body)


def test_exclusion_reason_skips_user_cases(tmp_path):
    """用户案例不参与文档卫生过滤 —— 案例可能很短，但它是用户确认过的答案。"""
    from app.loader import exclusion_reason

    case = tmp_path / "cases" / "case-x.md"
    case.parent.mkdir(parents=True)
    case.write_text("短", encoding="utf-8")
    assert exclusion_reason(case, "短", tmp_path) is None

    doc = tmp_path / "docs" / "empty.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("太短", encoding="utf-8")
    assert exclusion_reason(doc, "太短", tmp_path) == "空壳页"


# ============================================================ 状态图分支


def test_graph_edges_pick_the_right_branch():
    """不建图、不调模型，只验证条件边的判据（这是澄清/生成两条分支的开关）。"""
    from app.graph import RagGraph

    assert RagGraph._edge_after_route({"decision": {"ambiguous": True}}) == "clarify"
    assert RagGraph._edge_after_route({"decision": {"ambiguous": False}}) == "retrieve"
    assert RagGraph._edge_after_route({}) == "retrieve", "没有决策时应当继续检索而不是澄清"

    assert RagGraph._edge_after_retrieve({"sources": [object()]}) == "generate"
    assert RagGraph._edge_after_retrieve({"sources": []}) == "end"
    assert RagGraph._edge_after_retrieve({}) == "end"


# ============================================================ 代码 Agent 安全边界


def test_code_agent_rejects_path_escape():
    from app.code_agent import CodeAgent

    agent = CodeAgent(root=PROJECT_ROOT)
    for bad in ("../../etc/passwd", r"..\..\windows\system32\drivers", "/etc/shadow"):
        try:
            agent._resolve(bad)
        except ValueError as exc:
            assert "越界" in str(exc) or "只能访问" in str(exc)
        else:
            raise AssertionError(f"应当拒绝越界路径：{bad}")

    # 正常路径应当放行
    assert agent._resolve("app/config.py").name == "config.py"


def test_code_agent_rejects_sensitive_writes():
    """禁写名单里最容易漏的是 '.env' —— 用 lstrip('./') 归一化会把它变成 'env' 而绕过检查。"""
    from app.code_agent import CodeAgent

    agent = CodeAgent(root=PROJECT_ROOT)
    for bad in (".env", ".env.local", "./.env", "storage/index.faiss", "data/docs/x.md", ".git/config"):
        try:
            agent._check_writable(bad)
        except ValueError:
            continue
        raise AssertionError(f"应当拒绝写入：{bad}")

    # 源码文件可写
    agent._check_writable("app/config.py")


def test_code_agent_tools_are_local_only():
    """工具集必须只有固定的五个，多一个都意味着权限扩张。"""
    from app.code_agent import CodeAgent

    names = {tool.name for tool in CodeAgent(root=PROJECT_ROOT)._build_tools()}
    assert names == {"list_files", "grep", "read_file", "propose_edit", "run_tests"}


def test_comparison_retrieval_takes_each_side(monkeypatch):
    """比较类问题必须按域配额取数，否则 top-4 会被单一技术栈占满。

    这条来自实测：问「Spring 和 FastAPI 的依赖注入有什么区别」走全局检索时，
    四条来源**全部**来自 FastAPI，「依赖注入」在 FastAPI 文档里匹配度最高。
    模型手里只有一边的资料，只能回答「无法比较」—— 不是模型的问题，是给药给少了。
    """
    from app.retrieval import Retriever

    class FakeStore:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def similarity_search(self, query: str, k: int = 4):
            return [
                Document(page_content=f"{self.tag}-{i}", metadata={"source": f"{self.tag}.md"})
                for i in range(k)
            ]

    retriever = Retriever(
        FakeStore("global"),
        top_k=4,
        use_query_rewrite=False,   # 关掉改写，避免测试里调模型
        use_domain_prior=False,
        use_domain_route=True,
    )
    stores = {"fastapi": FakeStore("fastapi"), "spring-framework": FakeStore("spring")}
    monkeypatch.setattr(retriever, "_store_for", lambda domain: stores.get(domain) if domain else None)
    monkeypatch.setattr(retriever, "route", lambda q: _fake_decision())

    docs = retriever.retrieve("Spring 和 FastAPI 的依赖注入有什么区别？")
    tags = [doc.page_content.split("-")[0] for doc in docs]
    assert len(docs) == 4
    assert "fastapi" in tags and "spring" in tags, f"两侧资料必须都进候选，实际：{tags}"


def _fake_decision():
    """构造一个「比较 + 两个领域」的决策，避免测试依赖真实路由规则。"""
    from app.domain_router import RouteDecision

    return RouteDecision(domains=["fastapi", "spring-framework"], comparison=True)
