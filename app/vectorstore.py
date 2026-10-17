"""向量库的建库与加载（FAISS 本地持久化）。

选 FAISS 而不是 Chroma：FAISS 索引读写稳定、无额外原生运行时依赖。
产物是 storage/ 下的 index.faiss + index.pkl 两个文件。
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import STORAGE_DIR
from app.llm import get_embeddings

INDEX_NAME = "index"


def build_vectorstore(
    chunks: list[Document],
    persist_dir: Path = STORAGE_DIR,
    embeddings: Embeddings | None = None,
) -> FAISS:
    """重建向量库。同名索引文件直接覆盖，不需要先删目录。"""
    if not chunks:
        raise ValueError("chunks 为空，没有可入库的内容")

    persist_dir.mkdir(parents=True, exist_ok=True)

    store = FAISS.from_documents(chunks, embeddings or get_embeddings())
    store.save_local(str(persist_dir), index_name=INDEX_NAME)
    return store


def load_vectorstore(
    persist_dir: Path = STORAGE_DIR,
    embeddings: Embeddings | None = None,
) -> FAISS:
    """加载已建好的向量库；没建过就报错提示先跑 ingest.py。"""
    if not (persist_dir / f"{INDEX_NAME}.faiss").exists():
        raise FileNotFoundError(f"向量库不存在：{persist_dir}\n请先执行：python ingest.py")

    return FAISS.load_local(
        str(persist_dir),
        embeddings or get_embeddings(),
        index_name=INDEX_NAME,
        # 索引由本机 ingest.py 生成，内容是文档文本，可安全反序列化
        allow_dangerous_deserialization=True,
    )


def append_documents(
    store: FAISS,
    chunks: list[Document],
    persist_dir: Path = STORAGE_DIR,
) -> int:
    """把新片段增量追加进已加载的向量库，并落盘，返回追加的片段数。

    用于「用户问答案例沉淀」：不必全量重建索引（近万个片段重建要几分钟），
    只把新案例加进去即可。就地修改传入的 store —— 检索端持有的是同一对象，
    追加后立即生效，无需重新加载。
    """
    if not chunks:
        return 0
    store.add_documents(chunks)
    store.save_local(str(persist_dir), index_name=INDEX_NAME)
    return len(chunks)
