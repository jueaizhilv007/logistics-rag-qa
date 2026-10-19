"""rag_core：可嵌入的 RAG 问答内核。

本模块被 import 时不产生任何副作用：
不读取配置文件、不发起网络请求、不创建全局组件实例。
所有装配都通过 RagCore 显式完成。

    >>> from rag_core import RagCore
    >>> core = RagCore(embedding=emb, store=store, llm=llm)  # 显式注入
    >>> core, cfg, secrets = RagCore.from_config("config.yaml")  # 配置驱动
"""

from rag_core.config import (
    Config,
    EndpointCfg,
    Secrets,
    find_config_file,
    load_config,
    resolve_level,
)
from rag_core.container import RagCore
from rag_core.core.contracts import (
    Chunk,
    IngestResult,
    LoadedDoc,
    LoadedPage,
    QAResult,
    RetrievedChunk,
    TextRecord,
    VectorHit,
)
from rag_core.core.protocols import (
    BaseEmbedding,
    BaseLLM,
    BaseLoader,
    BaseReranker,
    BaseRetriever,
    BaseSplitter,
    BaseVectorStore,
)

__version__ = "0.1.0"

__all__ = [
    "RagCore",
    # 配置
    "Config",
    "EndpointCfg",
    "Secrets",
    "load_config",
    "find_config_file",
    "resolve_level",
    # 数据契约
    "Chunk",
    "TextRecord",
    "VectorHit",
    "RetrievedChunk",
    "QAResult",
    "IngestResult",
    "LoadedDoc",
    "LoadedPage",
    # 行为契约
    "BaseEmbedding",
    "BaseLLM",
    "BaseLoader",
    "BaseReranker",
    "BaseRetriever",
    "BaseSplitter",
    "BaseVectorStore",
    "__version__",
]
