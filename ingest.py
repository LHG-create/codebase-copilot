"""第 1 步：把 data/ 里的文档灌进向量库。

用法：python ingest.py
每改一次文档、或者改过项目源码，就跑一次。
"""

import shutil

from app.config import BASE_DIR, DATA_DIR, STORAGE_DIR
from app.loader import load_documents, split_documents
from app.vectorstore import build_vectorstore


# 需要同步进知识库的入口脚本（其余 .py 是评测 / 抓取等工具脚本，刻意不入库）
_ENTRY_SCRIPTS = ("ingest.py", "chat_cli.py", "app_ui.py", "code_agent.py", "prune_index.py")


def sync_codebase() -> tuple[int, int]:
    """把项目源码同步到 data/codebase/，保证知识库里是本项目的**当前**实现。

    为什么需要这一步：`data/codebase/` 是源码副本，靠手工维护必然过期。
    实测发现它落后真实源码 5 个文件（chain.py / app_ui.py / config.py /
    loader.py / vectorstore.py），并且缺少后来新增的模块
    （domain.py / retrieval.py / query_rewriter.py …）。

    这个缺陷特别隐蔽：**检索依然能命中「对的文件」，但给出的内容是旧实现**，
    用户从答案里根本看不出来（详见 ADR-012）。
    放进 ingest 里自动执行，就不存在「忘记同步」这种可能。
    """
    target = DATA_DIR / "codebase"
    target.mkdir(parents=True, exist_ok=True)

    # 只同步「用户真正会问的项目代码」：app/ 下的全部模块 + 三个入口脚本。
    #
    # 刻意**不**同步 eval_*.py —— 它们里面写着评测集的原文（问题 + 标准答案路径），
    # 一旦入库就会和评测问题高度相似、抢占召回位，造成教科书级的「评测集污染」：
    # 实测官方文档域命中率被从 83.3% 打到 66.7%，而且某个问题的第一名出处
    # 居然是 `eval_retrieval.py` 自己（详见 ADR-012）。
    sources = sorted((BASE_DIR / "app").glob("*.py"))
    sources += [BASE_DIR / n for n in _ENTRY_SCRIPTS if (BASE_DIR / n).exists()]

    updated = 0
    for path in sources:
        dest = target / path.name
        if not dest.exists() or dest.read_bytes() != path.read_bytes():
            shutil.copy2(path, dest)
            updated += 1

    # 源码里已删除的文件，副本也要同步删除（该目录由本函数全权维护）
    keep = {p.name for p in sources}
    for stale in target.glob("*.py"):
        if stale.name not in keep:
            stale.unlink()

    return updated, len(sources)


def main() -> None:
    updated, total = sync_codebase()
    print(f"[i] 项目源码已同步到 data/codebase/：{updated}/{total} 个文件有更新")

    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents, DATA_DIR)
    if not chunks:
        print(f"[!] {DATA_DIR} 里没找到任何可入库的内容")
        print("    支持的格式：.txt / .md / .pdf / .py")
        return

    py_count = sum(1 for c in chunks if c.metadata.get("kind") in {"function", "method", "class", "module", "file"})
    print(f"[i] 读取 {len(documents)} 个文档，切分为 {len(chunks)} 个片段（其中代码片段 {py_count} 个）")
    print("[i] 正在调用向量模型建库，首次可能稍慢 ...")

    try:
        build_vectorstore(chunks, STORAGE_DIR)
    except RuntimeError as exc:
        print(f"[!] {exc}")
        return

    print(f"[OK] 向量库已写入 {STORAGE_DIR}")
    print("     下一步：python chat_cli.py  或  streamlit run app_ui.py")


if __name__ == "__main__":
    main()
