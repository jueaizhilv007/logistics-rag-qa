"""装配根：所有组件只在本文件组装。

两种构造方式对应两种使用场景：

1. 显式注入
    >>> core = RagCore(embedding=emb, store=store, llm=llm)
    宿主项目把 rag_core 当库嵌入时使用。这条路径不读取配置文件、
    不触碰注册表、不依赖工作目录，可以安全地在任意进程中构造。

2. 配置驱动
    >>> core, cfg = RagCore.from_config("config.yaml")
    独立应用与命令行使用。要求调用方已 import 具体实现模块以触发注册，
    未注册时会抛出 UnknownKindError 并列出可用项，不会静默降级到别的实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rag_core.config import Config, GenerateCfg, RetrievalCfg, Secrets, load_config
from rag_core.core.logging import get_logger
from rag_core.core.protocols import (
    BaseEmbedding,
    BaseLLM,
    BaseReranker,
    BaseSplitter,
    BaseVectorStore,
)


class RagCore:
    """内核装配实例。持有各层组件本身，不持有任何全局状态。"""

    def __init__(
        self,
        *,
        embedding: BaseEmbedding,
        store: BaseVectorStore,
        llm: BaseLLM,
        splitter: Optional[BaseSplitter] = None,
        reranker: Optional[BaseReranker] = None,
        chunk_strategy: str = "fixed",
        # 以下两个是配置对象，不是组件实例，与其余参数性质不同。
        # 检索器与生成器要到 S3 / S4 才装配，现阶段先暂存配置，
        # 供 describe() 推导策略标签使用。
        # 显式注入路径下可以不传，此时 describe() 退化为默认策略标签。
        retrieval: Optional[RetrievalCfg] = None,
        generate: Optional[GenerateCfg] = None,
    ):
        self.embedding = embedding
        self.store = store
        self.llm = llm
        self.splitter = splitter
        self.reranker = reranker
        self.chunk_strategy = chunk_strategy
        self.retrieval = retrieval
        self.generate = generate
        self.log = get_logger("rag_core")

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
    ) -> tuple["RagCore", Config, Secrets]:
        """按配置文件装配全部组件。

        返回三元组 (内核, 配置, 密钥)，调用方通常会保留配置用于拼接路径与策略标签。
        registry 在这里延迟导入：显式注入路径因此完全不与注册表耦合。
        """
        from rag_core.registry import create

        cfg, secrets = load_config(config_path)

        emb_provider = cfg.embedding.provider
        embedding = create(
            "embedding",
            emb_provider,
            endpoint=cfg.embedding.endpoint(emb_provider),
            api_key=secrets.embedding_api_key,
        )

        llm_provider = cfg.llm.provider
        llm = create(
            "llm",
            llm_provider,
            endpoint=cfg.llm.endpoint(llm_provider),
            api_key=secrets.llm_api_key,
        )

        # 向量数据库
        store = create(
            "vector_store",
            cfg.vector_store.provider,
            path=Path(cfg.vector_store.path),
            collection=cfg.vector_store.collection,
            metric=cfg.vector_store.metric,
            options=cfg.vector_store.options,
        )

        # 切片器
        splitter = create(
            "splitter",
            cfg.chunking.strategy,
            size=cfg.chunking.size,
            overlap=cfg.chunking.overlap,
            inject_heading=cfg.chunking.inject_heading,
            options=cfg.chunking.options,
        )

        # 重排器
        # 重排器在 retrieval 配置中开启时才创建，避免无用实例化
        reranker = None
        if cfg.retrieval.rerank.enabled:
            reranker = create(
                "reranker",
                cfg.retrieval.rerank.provider,
                top_n=cfg.retrieval.rerank.top_n,
                options=cfg.retrieval.rerank.options,
            )

        # 内核实例
        core = cls(
            embedding=embedding,
            store=store,
            llm=llm,
            splitter=splitter,
            reranker=reranker,
            chunk_strategy=cfg.chunking.strategy,
            retrieval=cfg.retrieval,
            generate=cfg.generate,
        )
        return core, cfg, secrets

    def describe(self, mode: str | None = None) -> str:
        """返回当前策略标签，形如 fixed+vector 或 hierarchical+hybrid+rerank。

        该标签会写入评测结果与界面，是区分各组对照实验的唯一依据，
        因此必须与 retrieval.mode、rerank.enabled 保持同步推导，不允许硬编码。

        已知限制：标签当前从配置推导而非从组件推导，两者可能不一致。
        显式注入路径下若未传 retrieval 配置，这里会静默退化为默认策略标签，
        即使实际注入的是另一套检索组件。S3 装配 retriever 后改为从组件自述推导。
        mode 入参允许外部覆盖，与上一句的「不允许硬编码」冲突，S3 一并移除。
        """
        retrieval_mode = mode or (self.retrieval.mode if self.retrieval else "vector")
        bits = [self.chunk_strategy, retrieval_mode]
        if self.retrieval is not None and self.retrieval.rerank.enabled:
            bits.append("rerank")
        return "+".join(bits)
