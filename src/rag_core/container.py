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
from typing import TYPE_CHECKING, Optional

from rag_core.config import Config, RetrievalCfg, Secrets, load_config
from rag_core.core.logging import get_logger
from rag_core.core.protocols import (
    BaseEmbedding,
    BaseLLM,
    BaseReranker,
    BaseRetriever,
    BaseSplitter,
    BaseVectorStore,
)

if TYPE_CHECKING:
    from rag_core.ingest.pipeline import IngestPipeline


class RagCore:
    """内核装配实例。持有各层组件本身，不持有任何全局状态。"""

    def __init__(
        self,
        *,
        embedding: BaseEmbedding,
        store: BaseVectorStore,
        llm: BaseLLM,
        splitter: Optional[BaseSplitter] = None,
        # 检索器是组件实例而非配置对象。策略标签由它自述（BaseRetriever.describe），
        # 内核不再持有 retrieval / generate 配置——
        # 配置与组件并存时，两者不一致会静默产出错误标签。
        retriever: Optional[BaseRetriever] = None,
        chunk_strategy: str = "fixed",
        # 入库流水线：配置驱动路径下由 from_config 装配，显式注入可自行传入
        ingest: Optional["IngestPipeline"] = None,
        # 检索配置只在配置驱动路径下由 from_config 注入，
        # 供命令行按当前配置展示参数用；不参与策略标签推导
        retrieval: Optional[RetrievalCfg] = None,
    ):
        self.embedding = embedding
        self.store = store
        self.llm = llm
        self.splitter = splitter
        self.retriever = retriever
        self.chunk_strategy = chunk_strategy
        self.ingest = ingest
        self.retrieval = retrieval
        self.log = get_logger("rag_core")

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        retrieval_mode: str | None = None,
    ) -> tuple["RagCore", Config, Secrets]:
        """按配置文件装配全部组件。

        返回三元组 (内核, 配置, 密钥)，调用方通常会保留配置用于拼接路径与策略标签。
        registry 在这里延迟导入：显式注入路径因此完全不与注册表耦合。

        retrieval_mode 用于临时覆盖配置里的 retrieval.mode。
        评测脚本要在同一份语料上跑 E1-E4，各组之间只有检索策略不同，
        若靠改配置文件来回切换，跑完忘记改回来是典型的实验事故来源。
        """
        from rag_core.registry import create

        cfg, secrets = load_config(config_path)
        if retrieval_mode is not None:
            cfg.retrieval.mode = retrieval_mode

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

        # 重排器只在混合检索下装配：纯向量路不送重排，
        # 为一个「配了但用不上」的开关去加载 GB 级 cross-encoder 是白白浪费
        reranker = None
        if cfg.retrieval.mode == "hybrid" and cfg.retrieval.rerank.enabled:
            reranker = create(
                "reranker",
                cfg.retrieval.rerank.provider,
                top_n=cfg.retrieval.rerank.top_n,
                options=cfg.retrieval.rerank.options,
            )

        # 检索器。两种策略需要的构造参数不同，按 mode 组装后再透传。
        # 刻意不做「统一 kwargs、各实现忽略多余项」：
        # 那样会让「配了 vector_k 但当前是 vector 模式」这类无效配置被静默吞掉，
        # 而它往往说明调用方以为自己在跑另一套策略。
        retrieval_kwargs: dict[str, object] = {
            "embedding": embedding,
            "store": store,
            "top_k": cfg.retrieval.top_k,
        }
        if cfg.retrieval.mode == "hybrid":
            retrieval_kwargs.update(
                vector_k=cfg.retrieval.hybrid.vector_k,
                bm25_k=cfg.retrieval.hybrid.bm25_k,
                rrf_k=cfg.retrieval.hybrid.rrf_k,
                reranker=reranker,
                rerank_candidates=cfg.retrieval.rerank.top_n,
            )
        retriever = create("retriever", cfg.retrieval.mode, **retrieval_kwargs)

        # 入库流水线
        # 显式注入路径不装配，由调用方自行构造；这里只在配置驱动路径下提供
        from rag_core.ingest.loaders import all_loaders
        from rag_core.ingest.pipeline import IngestPipeline

        ingest = IngestPipeline(
            loaders=all_loaders(),
            splitter=splitter,
            embedding=embedding,
            store=store,
        )

        # 内核实例
        core = cls(
            embedding=embedding,
            store=store,
            llm=llm,
            splitter=splitter,
            retriever=retriever,
            chunk_strategy=cfg.chunking.strategy,
            ingest=ingest,
            retrieval=cfg.retrieval,
        )
        return core, cfg, secrets

    def describe(self) -> str:
        """返回当前策略标签，形如 fixed+vector 或 hierarchical+hybrid+rerank。

        该标签会写入评测结果与界面，是区分各组对照实验的唯一依据。

        前半段是切片策略，后半段由检索器自述。
        之所以改成组件自述而非从配置推导：配置与组件可能不一致
        （改了配置但没重建组件），此时以组件为准才能反映实际跑了什么；
        反过来则会产出「标签写着 hybrid、实际跑的是 vector」这类不可信数据。
        """
        bits = [self.chunk_strategy]
        if self.retriever is not None:
            bits.append(self.retriever.describe())
        return "+".join(bits)
