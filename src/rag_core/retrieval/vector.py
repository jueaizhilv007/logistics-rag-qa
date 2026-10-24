"""纯向量检索：E1 / E2 共用的检索侧口径。

这是 baseline 检索器。对照实验要求一次只改一个变量，
因此 E1→E2 只换切片策略、检索侧必须固定为纯向量，E3 才引入混合。
若 baseline 本身就带 BM25，E3 的增量无从归因。
"""

from __future__ import annotations

from rag_core.core.contracts import RetrievedChunk
from rag_core.core.protocols import BaseEmbedding, BaseVectorStore
from rag_core.registry import register


@register("retriever", "vector")
class VectorRetriever:
    """单路向量检索。"""

    def __init__(
        self,
        embedding: BaseEmbedding,
        store: BaseVectorStore,
        top_k: int = 4,
    ):
        self.embedding = embedding
        self.store = store
        self.top_k = top_k

    def describe(self) -> str:
        return "vector"

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        limit = k or self.top_k
        hits = self.store.search(self.embedding.embed_query(query), limit)
        results: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            # 单路检索下 bm25_* 保持 None。这不是「忘了填」，
            # 而是「这一路没跑」——证据面板据此显示「仅由向量路召回」。
            # 若填 0 或 -1，会和「召回了但排名垫底」混淆。
            results.append(
                RetrievedChunk(
                    chunk=hit.record.to_chunk(),
                    final_rank=rank,
                    fused_score=hit.score,
                    vector_rank=rank,
                    vector_score=hit.score,
                )
            )
        return results
