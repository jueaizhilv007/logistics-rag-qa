"""检索层。

对外只需要三条入口：两种检索器与一个索引类。
RRF 是纯函数，不依赖任何组件，单独成模块便于直接验证。
"""

from rag_core.retrieval.bm25 import BM25Hit, BM25Index
from rag_core.retrieval.hybrid import HybridRetriever
from rag_core.retrieval.rrf import rrf_fuse
from rag_core.retrieval.vector import VectorRetriever

__all__ = [
    "BM25Hit",
    "BM25Index",
    "HybridRetriever",
    "VectorRetriever",
    "rrf_fuse",
]
