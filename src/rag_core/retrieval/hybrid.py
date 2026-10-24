"""混合检索：向量路 + BM25 路，RRF 融合。

### 两路必须独立执行

先算向量、再把结果喂给 BM25 的串行写法，会让「一路失败时另一路也拿不到结果」，
也让将来的并行化无从下手。两路各自取回自己的候选，只在融合层汇合。

### 每路取回的数量要多于最终 top_k

RRF 的价值来自「两路都靠前」的那些样本。
若每路只取最终需要的 4 条，交集之外的样本永远进不了候选，
融合就退化成「两路各出一半、交替占位」。
因此默认每路取 10 条，融合后再截到最终条数。

### 重排为什么放在这里而不是单独一层

重排是对「召回结果」的精排，它的输入必须是融合后的完整候选集。
放在检索器内部，调用方拿到手的就已经是最终结果，
不必知道「先融合再重排」这个顺序。
"""

from __future__ import annotations

from rag_core.core.contracts import RetrievedChunk
from rag_core.core.logging import get_logger
from rag_core.core.protocols import (
    BaseEmbedding,
    BaseReranker,
    BaseVectorStore,
)
from rag_core.registry import register
from rag_core.retrieval.bm25 import BM25Index
from rag_core.retrieval.rrf import rrf_fuse


@register("retriever", "hybrid")
class HybridRetriever:
    """向量 + BM25 双路检索，RRF 融合，可选重排。"""

    def __init__(
        self,
        embedding: BaseEmbedding,
        store: BaseVectorStore,
        top_k: int = 4,
        vector_k: int = 10,
        bm25_k: int = 10,
        rrf_k: int = 60,
        reranker: BaseReranker | None = None,
        rerank_candidates: int = 20,
    ):
        self.embedding = embedding
        self.store = store
        self.top_k = top_k
        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        self.reranker = reranker
        # 送进重排器的候选条数。重排的意义在于「从更大候选里挑更好的」，
        # 候选数若等于最终条数，重排无事可做
        self.rerank_candidates = rerank_candidates
        self.log = get_logger("rag_core.retrieval")
        self._bm25: BM25Index | None = None
        self._indexed_rows = 0

    def describe(self) -> str:
        """自述策略。标签由实际装配决定，不看配置——
        配置里写了 rerank 但没注入重排器时，标签不该声称有重排。
        """
        bits = ["hybrid"]
        if self.reranker is not None:
            bits.append("rerank")
        return "+".join(bits)

    def _ensure_bm25(self) -> None:
        """BM25 索引是构建时刻的快照，语料变了必须重建。

        用行数变化作为重建判据而不是每次重建：
        jieba 分词整张表是本次检索最贵的一步，
        每次查询都做会让交互延迟不可接受。
        """
        rows = self.store.count()
        if self._bm25 is not None and self._indexed_rows == rows:
            return
        self._bm25 = BM25Index.from_store(self.store)
        self._indexed_rows = rows
        self.log.info("BM25 索引已构建：%d 条文本", self._bm25.size)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        limit = k or self.top_k
        self._ensure_bm25()

        vec_hits = self.store.search(self.embedding.embed_query(query), self.vector_k)
        bm_hits = self._bm25.search(query, self.bm25_k) if self._bm25 else []

        # 同一 chunk 可能被两路同时召回，用 chunk_id 去重并各记各的排名
        chunks: dict[str, RetrievedChunk] = {}
        vec_rank: dict[str, int] = {}
        vec_score: dict[str, float] = {}
        bm_rank: dict[str, int] = {}
        bm_score: dict[str, float] = {}

        for rank, hit in enumerate(vec_hits, start=1):
            chunk_id = hit.record.chunk_id
            vec_rank[chunk_id] = rank
            vec_score[chunk_id] = hit.score
            chunks.setdefault(
                chunk_id, RetrievedChunk(chunk=hit.record.to_chunk(), final_rank=rank)
            )

        for rank, hit in enumerate(bm_hits, start=1):
            chunk_id = hit.record.chunk_id
            bm_rank[chunk_id] = rank
            bm_score[chunk_id] = hit.score
            chunks.setdefault(
                chunk_id, RetrievedChunk(chunk=hit.record.to_chunk(), final_rank=rank)
            )

        # 重排需要更大的候选池，否则重排只是对已定好的顺序做无意义复核
        candidate_limit = (
            max(limit, self.rerank_candidates) if self.reranker is not None else limit
        )
        fused = rrf_fuse([list(vec_rank), list(bm_rank)], k=self.rrf_k)

        results: list[RetrievedChunk] = []
        for final_rank, (chunk_id, score) in enumerate(fused[:candidate_limit], start=1):
            item = chunks[chunk_id]
            item.final_rank = final_rank
            item.fused_score = score
            item.vector_rank = vec_rank.get(chunk_id)
            item.vector_score = vec_score.get(chunk_id)
            item.bm25_rank = bm_rank.get(chunk_id)
            item.bm25_score = bm_score.get(chunk_id)
            results.append(item)

        if self.reranker is not None:
            results = self.reranker.rerank(query, results, limit)
        return results
