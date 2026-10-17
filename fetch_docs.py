"""下载官方框架文档到 data/docs/，供知识库使用。

用 GitHub 的 codeload tarball 下载仓库快照（比 git clone 更稳定，尤其在本机网络环境），
只解压文档目录，避免下载整个仓库体积。

用法：
    python fetch_docs.py fastapi          # 只下载 FastAPI 文档
    python fetch_docs.py langgraph        # 只下载 LangGraph 文档
    python fetch_docs.py all              # 下载全部

下载后的文件在 data/docs/<name>/ 下，然后跑 ingest.py 建库即可。
"""

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# 每个框架的仓库、分支、以及要提取的文档目录
SOURCES = {
    "fastapi": {
        "repo": "fastapi/fastapi",
        "branch": "master",
        "dirs": ["docs/en/docs"],
        "desc": "FastAPI 官方文档（英文）",
    },
    "langgraph": {
        "repo": "langchain-ai/docs",
        "branch": "main",
        "dirs": ["src/oss/langgraph"],
        "desc": "LangGraph 官方文档（英文，mdx）",
    },
    "langchain": {
        "repo": "langchain-ai/docs",
        "branch": "main",
        "dirs": ["src/oss/langchain"],
        "desc": "LangChain 官方文档（英文，mdx）",
    },
    "vuejs": {
        "repo": "vuejs/docs",
        "branch": "main",
        "dirs": ["src"],
        "desc": "Vue.js 官方文档",
    },
    "spring-boot": {
        "repo": "spring-projects/spring-boot",
        "branch": "main",
        "dirs": ["documentation/spring-boot-docs/src/docs/antora/modules"],
        "desc": "Spring Boot 官方文档（英文，AsciiDoc）",
    },
    "spring-framework": {
        "repo": "spring-projects/spring-framework",
        "branch": "main",
        "dirs": ["framework-docs/modules/ROOT/pages"],
        "desc": "Spring Framework 官方文档（英文，AsciiDoc）",
    },
}

DATA_DIR = Path(__file__).resolve().parent / "data"
DOCS_DIR = DATA_DIR / "docs"


def _download_tarball(repo: str, branch: str, dest: Path) -> bool:
    """从 codeload 下载 tarball 快照。"""
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
    print(f"    下载: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fetch-docs"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        return True
    except Exception as e:
        print(f"[!] 下载失败: {e}")
        return False


def _extract_docs(tarball: Path, repo: str, dirs: list[str], target: Path) -> int:
    """解压 tarball 里指定目录下的文档文件（markdown / mdx / asciidoc）。"""
    count = 0
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            # 路径形如: <repo名>-<hash>/docs/en/docs/xxx.md
            parts = member.name.split("/", 1)
            if len(parts) < 2:
                continue
            rel_to_root = parts[1]  # 去掉第一层目录

            # 判断是否属于目标 docs 目录
            matched_dir = None
            for d in dirs:
                if rel_to_root == d or rel_to_root.startswith(d + "/"):
                    matched_dir = d
                    break
            if matched_dir is None:
                continue

            # 只取 markdown / mdx / asciidoc
            name = member.name.lower()
            if not (name.endswith(".md") or name.endswith(".mdx") or name.endswith(".adoc")):
                continue

            rel = Path(rel_to_root).relative_to(matched_dir)
            if str(rel).endswith(".mdx"):
                rel = rel.with_suffix(".md")

            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def _count_files(target: Path) -> int:
    """统计 target 目录下已下载的文档文件数量。"""
    suffixes = {".md", ".mdx", ".adoc"}
    return sum(1 for p in target.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def fetch(name: str) -> bool:
    cfg = SOURCES[name]
    target = DOCS_DIR / name

    print(f"[i] 下载 {name} —— {cfg['desc']}")
    tarball = DOCS_DIR / f".tmp_{name}.tar.gz"

    if not _download_tarball(cfg["repo"], cfg["branch"], tarball):
        tarball.unlink(missing_ok=True)
        return False

    target.mkdir(parents=True, exist_ok=True)
    try:
        count = _extract_docs(tarball, cfg["repo"], cfg["dirs"], target)
    finally:
        tarball.unlink(missing_ok=True)

    print(f"[OK] {name} 下载完成，共 {count} 个文档文件 → {target}")
    return count > 0


def main() -> None:
    names = sys.argv[1:] or ["all"]
    if "all" in names:
        names = list(SOURCES.keys())

    for name in names:
        if name not in SOURCES:
            print(f"[!] 未知来源: {name}，可选：{', '.join(SOURCES)}")
            continue
        fetch(name)
        print()

    print("[i] 完成。现在运行 ingest.py 重新建库即可。")


if __name__ == "__main__":
    main()
