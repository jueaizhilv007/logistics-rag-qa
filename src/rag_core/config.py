"""配置模型与加载。

设计要点：所有 provider 字段使用 str 而不是 Literal，
端点参数用 dict[str, EndpointCfg] 而不是固定字段名。

原因是注册表已具备 UnknownImplError 报错能力（会附带可用实现清单），
若再用 Literal 约束合法取值，等于把「新增一个实现」的改点数从一处放大到两处，
并且堵死了宿主项目在外部注册自有实现的可能。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EndpointCfg(BaseModel):
    """一个 OpenAI 兼容端点，本地与云端通用。"""
    base_url: str
    model: str
    batch_size: int = 32
    temperature: float = 0.0
    max_tokens: int = 1024


class AppCfg(BaseModel):
    name: str = "rag-qa"
    data_dir: str = "./data"


class EmbeddingCfg(BaseModel):
    """向量化服务配置。endpoints 支持任意数量的服务商。"""
    provider: str = "agicto"
    endpoints: dict[str, EndpointCfg] = Field(default_factory=dict)

    def endpoint(self, provider: str | None = None) -> EndpointCfg:
        """取出指定 provider 的端点配置，未配置时报错并列出已有项。"""
        name = provider or self.provider
        if name not in self.endpoints:
            raise KeyError(
                f"未配置 embedding provider `{name}`，"
                f"已在 endpoints 中配置的项：{list(self.endpoints)}"
            )
        return self.endpoints[name]


class LLMCfg(BaseModel):
    """LLM 服务配置。"""
    provider: str = "agicto"
    endpoints: dict[str, EndpointCfg] = Field(default_factory=dict)

    def endpoint(self, provider: str | None = None) -> EndpointCfg:
        name = provider or self.provider
        if name not in self.endpoints:
            raise KeyError(
                f"未配置 llm provider `{name}`，"
                f"已在 endpoints 中配置的项：{list(self.endpoints)}"
            )
        return self.endpoints[name]


class VectorStoreCfg(BaseModel):
    """向量存储配置。options 用于向具体实现透传专有参数，避免每加一个实现就改这里。"""
    provider: str = "lancedb"
    path: str = "./data/index"
    collection: str = "docs"
    metric: str = "cosine"
    options: dict[str, Any] = Field(default_factory=dict)


class ChunkingCfg(BaseModel):
    """切片配置。"""
    strategy: str = "fixed"
    size: int = 500
    overlap: int = 50
    inject_heading: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class HybridCfg(BaseModel):
    """混合检索配置。"""
    bm25_k: int = 10
    vector_k: int = 10
    weights: list[float] = Field(default_factory=lambda: [0.5, 0.5])
    rrf_k: int = 60


class RerankCfg(BaseModel):
    """重排配置。"""
    enabled: bool = False
    provider: str = "bge_local"
    top_n: int = 20
    options: dict[str, Any] = Field(default_factory=dict)


class RetrievalCfg(BaseModel):
    """检索配置。

    mode 默认取 vector：对照实验 E1 的 baseline 口径是「固定切片加纯向量」，
    若默认开启 hybrid，baseline 就不是纯向量，E2 以后的归因全部失效。
    """
    mode: str = "vector"
    top_k: int = 4
    hybrid: HybridCfg = Field(default_factory=HybridCfg)
    rerank: RerankCfg = Field(default_factory=RerankCfg)


class GenerateCfg(BaseModel):
    """生成层配置。"""
    max_context_tokens: int = 6000
    empty_answer: str = "知识库未覆盖该问题，建议转人工"


class LoggingCfg(BaseModel):
    level: str = "INFO"
    dir: str = "./logs"


class Config(BaseModel):
    """完整配置。config_dir 由加载阶段注入，不来自 YAML。"""
    app: AppCfg = Field(default_factory=AppCfg)
    embedding: EmbeddingCfg
    llm: LLMCfg
    vector_store: VectorStoreCfg
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    generate: GenerateCfg = Field(default_factory=GenerateCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)

    config_dir: Path = Path(".")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def resolve_path(self, raw: str) -> Path:
        """把配置中的相对路径挂到 config.yaml 所在目录，避免依赖调用方的工作目录。"""
        p = Path(raw)
        if p.is_absolute():
            return p
        return (self.config_dir / p).resolve()


class Secrets(BaseSettings):
    """只从 .env 与环境变量读取密钥与 provider 覆盖开关。"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    embedding_api_key: str = ""
    llm_api_key: str = ""
    embedding_provider: Optional[str] = None
    llm_provider: Optional[str] = None


def load_config(config_path: str | Path | None = None) -> tuple[Config, Secrets]:
    """加载配置文件，返回配置与密钥。

    .env 显式锚定到 config.yaml 所在目录，不依赖进程工作目录，
    这样从任何目录执行命令行都能拿到同一份密钥。
    """
    if config_path is None:
        config_path = find_config_file()
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    secrets = Secrets(_env_file=config_dir / ".env")

    # 环境变量优先级高于 YAML，便于临时切换 provider 做对照实验
    if secrets.embedding_provider:
        raw["embedding"]["provider"] = secrets.embedding_provider
    if secrets.llm_provider:
        raw["llm"]["provider"] = secrets.llm_provider

    cfg = Config(**raw)
    cfg.config_dir = config_dir

    # 把相对路径统一解析为绝对路径，后续模块不再关心工作目录
    cfg.vector_store.path = str(cfg.resolve_path(cfg.vector_store.path))
    cfg.logging.dir = str(cfg.resolve_path(cfg.logging.dir))

    return cfg, secrets


def find_config_file(start: Path | None = None) -> Path:
    """从给定目录向上逐级查找 config.yaml。

    这是给命令行等入口使用的便利函数，内核不自动调用，
    避免库被 import 时产生文件系统副作用。
    """
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        hit = candidate / "config.yaml"
        if hit.exists():
            return hit
    raise FileNotFoundError(f"未找到 config.yaml（起始目录：{current}）")


def resolve_level(level: str) -> int:
    """把配置中的日志级别字符串转为 logging 常量。"""
    value = getattr(logging, level.upper(), None)
    if not isinstance(value, int):
        raise ValueError(f"不支持的日志级别：{level}")
    return value
