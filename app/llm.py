"""模型工厂：改 .env 里的 provider 就能换模型，业务代码一行都不用动。"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings

_KEY_BY_PROVIDER = {
    "openai": ("OPENAI_API_KEY", "openai_api_key"),
    "deepseek": ("DEEPSEEK_API_KEY", "deepseek_api_key"),
    "dashscope": ("DASHSCOPE_API_KEY", "dashscope_api_key"),
}


def ensure_configured() -> None:
    """提前校验 Key，避免跑到一半才甩出一大段英文堆栈。"""
    problems = []

    for role, provider in (
        ("对话模型", settings.llm_provider),
        ("向量模型", settings.embedding_provider),
    ):
        provider = provider.strip().lower()
        if provider == "ollama":
            continue  # 本地模型不需要 Key
        entry = _KEY_BY_PROVIDER.get(provider)
        if entry and not getattr(settings, entry[1], "").strip():
            problems.append(f"  - {role} 用的是 {provider}，需要 {entry[0]}，但 .env 里是空的")

    if problems:
        raise RuntimeError(
            "配置不完整，请先在 .env 里补齐：\n"
            + "\n".join(problems)
            + "\n\n不想申请 Key 的话，可以把 LLM_PROVIDER 和 EMBEDDING_PROVIDER 都改成 ollama，"
            "完全本地跑（需要本机已启动 Ollama）。"
        )


def get_chat_model() -> BaseChatModel:
    ensure_configured()
    provider = settings.llm_provider.strip().lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model or "gpt-4o-mini",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
        )

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model or "deepseek-chat",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.llm_temperature,
        )

    if provider == "dashscope":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model or "qwen-plus",
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            temperature=settings.llm_temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model or "qwen2.5:7b",
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    raise ValueError(f"不支持的 LLM_PROVIDER: {provider}（可选 deepseek/dashscope/openai/ollama）")


def get_embeddings() -> Embeddings:
    ensure_configured()
    provider = settings.embedding_provider.strip().lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model or "text-embedding-3-small",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    if provider == "dashscope":
        from langchain_community.embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(
            model=settings.embedding_model or "text-embedding-v3",
            dashscope_api_key=settings.dashscope_api_key,
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embedding_model or "nomic-embed-text",
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"不支持的 EMBEDDING_PROVIDER: {provider}（可选 dashscope/openai/ollama）")
