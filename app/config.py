"""全局配置：统一从 .env 读取，代码里不要再出现硬编码的 Key。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 对话模型
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2

    # 向量模型（和对话模型分开配，因为 DeepSeek 不提供 embedding）
    embedding_provider: str = "dashscope"
    embedding_model: str = "text-embedding-v3"

    # 各家 Key
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ollama_base_url: str = "http://localhost:11434"

    # 检索参数
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 8  # 知识库扩充到 3000+ 片段后，top_k 需相应调大，否则召回不足

    # 单路召回池宽度：RRF 是多通道融合，池子必须显著大于 top_k，
    # 否则「排名稍后但内容正确」的片段进不了融合池（ADR-010 / ADR-011）。
    recall_k: int = 96

    # 领域先验加权强度：给「片段数极少」的领域一点排序加成，纠正
    # 「大领域天然占优」的偏差（本项目代码只占全库 ~1%，详见 ADR-011）。
    #
    # 0.02 是 ADR-012 复扫出来的。ADR-011 定的是 0.06，但那个值是在
    # 「codebase 副本只有 11 个旧文件」的前提下扫出来的；副本改为自动同步后
    # 变成 16 个文件（含 query_rewriter.py / retrieval.py 这些中文注释极多的模块），
    # 它们的注释与「中文提问官方文档」的语义高度重合 —— 实测问
    # 「FastAPI 的查询参数怎么声明和做校验」时 top-3 全是本项目代码的注释片段。
    # 复扫结果：alpha ≤ 0.02 时官方文档域满血（83.3%），0.04 起开始退化。
    domain_prior_alpha: float = 0.02

    # 只给「占比 ≤ 该比例」的长尾领域加权。
    # 这是被一次端到端真机验证揪出来的限制：不加限制时，FastAPI（10.8%）拿到的
    # 先验高于 Spring Framework（36.1%），导致问 Spring 注解时 top-4 被 FastAPI
    # 文档占满、模型直接拒答 —— 先验本该纠正「0.8% vs 99.2%」的极端失衡，
    # 而不是在中等领域之间重新分配权重。
    domain_prior_rare_ratio: float = 0.05

    # 改写路相对原问题路的 RRF 权重（1.0 = 等权）。
    #
    # **这一项是为一个「看起来很有道理」的想法留的墓碑。**
    # 起因：实测「LangChain 的 agent 是怎么工作的？」被改写成
    # "agent workflow reasoning action observation" 这种关键词堆砌后，
    # 正确答案在改写路掉到第 95 位（原问题路是第 3 位），等权融合后被挤出候选。
    # 于是推论「降低改写路权重应当更好」，降到 0.5 后在 top_k=8 的快速试验里
    # 确实两域都没退化。
    #
    # 但完整评测（top_k=4、两个域）显示它是**净负优化**：
    # 官方文档域 95.8% → 91.7%、项目代码域 97.5% → 95.0%，
    # 等于用「救回 1 条」换了「丢掉 3 条」。原因是 top_k 越小，越依赖两路共识
    # 来拉开差距，压低其中一路反而让排序变糊。
    # 结论：保持等权。此项仅作为可对照的开关保留（详见 ADR-013）。
    rewrite_weight: float = 1.0

    # HTTP 接口（app/api.py）监听地址。默认只绑本机 ——
    # 改成 0.0.0.0 就等于把这个（能读你代码的）服务暴露到局域网，需要自己确认过。
    api_host: str = "127.0.0.1"
    api_port: int = 8000


settings = Settings()
