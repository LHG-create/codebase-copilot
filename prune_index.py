"""从**已建好的索引**里摘除脏文档片段，不需要重新 embedding。

为什么单独有这个脚本：
  数据卫生问题（空壳页、导语页）修进 `app/loader.py` 后，只对「下次建库」生效；
  当前这份索引里仍然躺着那些脏片段。全量重建要重新调用近万次 embedding
  （既费钱又费时），而这些文档本身是可以**精确识别**的，所以直接从 FAISS
  里按 source 删掉即可 —— 零 API 调用，几秒钟完成。

判据不是另写一套：调用 `app.loader.exclusion_reason()`，与建库时严格同一个函数、
同一份整页原文。所以不会出现「建库放过的页面被体检脚本误杀」这种偏差
（初版脚本按**片段**长度判断，把 AST 切出来的短代码片段全判成了空壳页 —— 已修）。

用法：
    python prune_index.py            # 只报告会删什么（dry-run，不改文件）
    python prune_index.py --apply    # 真正删除并落盘

删除是**不可逆**的（片段会从 index.faiss / index.pkl 里消失），所以默认 dry-run。
注意它只动官方文档：`data/cases/` 下的用户案例不受任何数据卫生规则影响。
"""

from __future__ import annotations

import sys
from collections import defaultdict

from app.config import DATA_DIR, STORAGE_DIR
from app.loader import exclusion_reason, iter_source_texts
from app.vectorstore import INDEX_NAME, load_vectorstore


def find_dirty_sources() -> dict[str, str]:
    """返回 {绝对路径: 排除原因}，只包含应当被清理的文档。"""
    dirty: dict[str, str] = {}
    for path, text in iter_source_texts(DATA_DIR):
        reason = exclusion_reason(path, text)
        if reason:
            dirty[str(path)] = reason
    return dirty


def main() -> None:
    apply = "--apply" in sys.argv

    dirty = find_dirty_sources()
    store = load_vectorstore()
    total = len(store.docstore._dict)

    # 统计每个脏文档在索引里占了多少片段
    per_source: dict[str, int] = defaultdict(int)
    for doc in store.docstore._dict.values():
        src = doc.metadata.get("source", "")
        if src in dirty:
            per_source[src] += 1

    print(f"索引片段总数：{total}")
    print(f"命中脏文档：{len(dirty)} 个，占用片段 {sum(per_source.values())} 条\n")

    for src, reason in sorted(dirty.items(), key=lambda kv: -per_source.get(kv[0], 0)):
        if not per_source.get(src):
            continue
        short = src.replace("\\", "/").split("data/")[-1]
        print(f"  [{reason}] {short}  ({per_source[src]} 片段)")

    if not per_source:
        print("\n没有需要清理的片段，索引是干净的。")
        return

    if not apply:
        print("\n这是 dry-run，没有修改任何文件。确认无误后执行：")
        print("    python prune_index.py --apply")
        return

    ids = [
        doc_id
        for doc_id, doc in store.docstore._dict.items()
        if doc.metadata.get("source", "") in dirty
    ]
    store.delete(ids)
    store.save_local(str(STORAGE_DIR), index_name=INDEX_NAME)
    print(f"\n已删除 {len(ids)} 条片段，索引剩余 {len(store.docstore._dict)} 条，已落盘。")
    print("建议随后跑一次评测，确认指标没有退化：python eval_retrieval.py")


if __name__ == "__main__":
    main()
