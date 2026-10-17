# ADR-002：项目必须放在纯英文路径

- 状态：已采纳
- 日期：2026-10-15

## 背景

本机 Windows 用户名是中文（`林海亘`），PyCharm 默认项目目录 `C:\Users\林海亘\PycharmProjects\` 含中文。

## 问题

用 FAISS 建库时，`write_index` 报错：

```
could not open D:/.../中文目录/idx.faiss for writing: No such file or directory
```

## 调查

FAISS 的 `write_index` 底层调用 C 的 `fopen`，而 `fopen` 在本机的中文路径下无法正确打开文件。
这是 C 文件 API 的编码问题，Python 层无法规避。

实测确认：
- 英文路径 `D:/chroma-probe/ascii/idx.faiss` → 写入成功
- 中文路径 `D:/chroma-probe/中文目录/idx.faiss` → 写入失败

## 决策

项目整体迁移到 `D:\projects\langchain-rag`（纯英文路径）。

## 后果

- 正向：彻底规避了所有走 C/C++ 文件 API 的库（FAISS、部分深度学习库）在中文路径下的崩溃。
- 约束：**以后本机所有 Python 项目都放纯英文路径**，不要用 PyCharm 默认的中文路径。
- 衍生坑：系统临时目录 `D:\Temp\pytest-of-林海亘` 也含中文，导致 pytest 的 `tmp_path` 不能用于 FAISS 落盘测试，测试改用项目内英文目录。
