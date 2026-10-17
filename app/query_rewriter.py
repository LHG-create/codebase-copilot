"""查询改写：解决「中文问题检索英文文档」的跨语言语义鸿沟。

问题背景（实测验证）：
  知识库里大量内容是英文文档（FastAPI / Vue 等官方文档），
  用户用中文问「FastAPI 怎么定义路径参数」时，向量检索会被
  「怎么定义」这类口语词稀释，导致 path-params.md 这种最相关的
  片段排不进 top_k，最终模型回答「无法回答」。

  但纯英文 query「path parameters declare」能稳稳命中。
  → 根因不是入库侧，而是查询侧的语言鸿沟。

方案（工业界标准做法 Query Translation）：
  检测到中文查询时，先用 LLM 把问题翻译成「英文检索关键词」，
  再用「中文 query + 英文 query」双路召回，合并去重。
  这样既保留中文语义命中（中文文档），又补上英文关键词命中（英文文档）。

设计取舍：
  - 翻译用一次轻量 LLM 调用，但只在「含中文」时触发，英文/代码符号查询零开销。
  - 翻译结果要求「只输出英文关键词短语」，不写完整句子，避免引入噪声。
  - 翻译失败时优雅降级：直接用原 query 检索，不影响主流程。
"""

from __future__ import annotations

import re

from app.llm import get_chat_model

# 检测是否含中文字符（\u4e00-\u9fff 覆盖 CJK 基本区）
_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")

# 翻译提示词：要求输出干净的英文检索关键词
_TRANSLATE_PROMPT = (
    "You are a search query translator for a technical documentation knowledge base. "
    "Translate the Chinese question into concise English search keywords.\n"
    "Rules:\n"
    "1. Output ONLY English keywords separated by spaces.\n"
    "2. DO NOT include broad framework names (e.g. FastAPI, Vue, React, Django), "
    "because they appear on every page and dilute the search. "
    "Focus on the specific technical concept being asked about.\n"
    "3. Use the exact terminology used in official docs titles "
    "(e.g. 'path parameters', 'lifecycle hooks', 'query parameters').\n"
    "4. No full sentences, no punctuation, no explanations.\n\n"
    "Chinese question: {question}\n"
    "English keywords:"
)


def contains_chinese(text: str) -> bool:
    """判断文本是否含中文字符。"""
    return bool(_HAS_CJK.search(text or ""))


def translate_to_english(question: str) -> str:
    """把中文问题翻译成英文检索关键词。失败时返回空串。

    返回空串表示「无需改写或改写失败」，调用方应直接用原 query 检索。
    """
    if not contains_chinese(question):
        return ""

    try:
        model = get_chat_model()
        result = model.invoke(_TRANSLATE_PROMPT.format(question=question))
        keywords = (result.content or "").strip() if hasattr(result, "content") else str(result).strip()
        # 去掉可能残留的标点和换行，只留关键词
        keywords = re.sub(r"[，。！？；：,.;:!?\n]+", " ", keywords).strip()
        # 翻译结果如果跟原问题一模一样（模型没翻），视为失败
        if not keywords or keywords == question:
            return ""
        return keywords
    except Exception:
        # 任何异常都降级，绝不让改写拖垮主流程
        return ""
