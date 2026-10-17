# ADR-001：向量库从 Chroma 换到 FAISS

- 状态：已采纳
- 日期：2026-10-15

## 背景

项目最初按主流教程选用 Chroma 作为向量库，理由是「轻量、纯本地、零配置」。

## 问题

首次跑通建库时，`collection.add()` 直接**段错误**（进程退出码 139，access violation）。
定位过程：

1. 隔离测试原生 `chromadb`，在 `add()` 处崩溃 → 排除了 LangChain 封装层的因素
2. 检查 Chroma 的依赖 `onnxruntime`，发现 `import onnxruntime` 在本机即崩溃
3. 结论：Chroma 的 Rust 绑定（`_upsert`）在本机环境无法运行，且其依赖的 onnxruntime 同样崩溃

## 决策

改用 **FAISS**（`faiss-cpu 1.15.1`）。验证结果：索引创建、写入、搜索全部正常。

## 后果

- 正向：FAISS 无额外原生运行时依赖，检索快，稳定。
- 代价：FAISS 不提供内置持久化，需要自己管理 `index.faiss` + `index.pkl` 两个文件（已封装在 `vectorstore.py`）。
- 教训：**「主流选择」不等于「本机可用」**。遇到原生扩展崩溃，先隔离测试定位到根因，再换方案，而不是反复重试。
