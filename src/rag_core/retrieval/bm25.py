"""BM25 稀疏检索索引。

### 为什么需要这一路

纯向量检索对「精确字面量」不敏感。用户问「圆通未保价最多赔多少」时，
向量能召回语义相近的赔付条款，但对金额阈值、单号、地名这类 token，
字面匹配往往比语义匹配更准——向量会把「三倍资费」和「五倍资费」
判成高度相似，而用户问的往往正是二者的区别。

### 为什么必须中文分词

BM25 的基础是词频统计，而中文没有空格。
按空格切分会得到「整句作为一个 token」，倒排索引退化成全表扫描，
BM25 完全失去意义。因此走 jieba。

### 索引是快照，不是视图

BM25Index 在构建时把全表文本读进内存，之后不再与 store 同步。
新增语料必须重建，重建由调用方按行数变化触发（见 HybridRetriever）。
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Sequence

import jieba
from rank_bm25 import BM25Okapi

from rag_core.core.contracts import TextRecord
from rag_core.core.protocols import BaseVectorStore

# jieba 首次分词时默认以 DEBUG 级别打印词典加载过程，
# 会混进 CLI 的正常输出与评测日志里
jieba.setLogLevel(logging.WARNING)


class BM25Hit(NamedTuple):
    """BM25 路径的原始命中项。

    与 VectorHit 分开定义而不是复用：VectorHit 的名字是「向量」，
    用来装 BM25 结果会让读代码的人误以为两者可以互换。
    两路只是形状相同，语义与分数量纲都不同。
    """

    record: TextRecord
    score: float


def tokenize(text: str) -> list[str]:
    """中文分词，丢弃空白 token（空白进倒排会拉低命中质量）。"""
    return [tok for tok in jieba.lcut(text) if tok.strip()]


class BM25Index:
    """一次性构建、可反复查询的 BM25 倒排索引。"""

    def __init__(self, records: Sequence[TextRecord]) -> None:
        self._records = list(records)
        self._tokenized = [tokenize(r.text) for r in self._records]
        # 空语料时不建索引：BM25Okapi 收到空语料会在查询时报错，
        # 而「库是空的」是一个合法状态，检索应返回空列表而不是抛异常
        self._index = BM25Okapi(self._tokenized) if self._tokenized else None

    @classmethod
    def from_store(cls, store: BaseVectorStore) -> "BM25Index":
        """从向量库全量回读文本建索引。

        走 iter_text_records 而不是直接读 chunks：
        该方法只拉文本列、不拉向量列，
        否则建一次稀疏索引要把整个向量库读进内存。
        """
        return cls(list(store.iter_text_records()))

    @property
    def size(self) -> int:
        return len(self._records)

    def search(self, query: str, k: int) -> list[BM25Hit]:
        """返回分数最高的前 k 条命中。"""
        if self._index is None:
            return []
        scores = self._index.get_scores(tokenize(query))
        # 并列时按下标升序，保证同一输入的输出稳定，理由同 rrf_fuse
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        hits: list[BM25Hit] = []
        for i in order:
            score = float(scores[i])
            # 零分必须丢弃。BM25 对「一个查询词都没命中」的文档给 0 分，
            # 这类文档通常占绝大多数；若放进候选，
            # 它们会带着并列的最低分挤进 RRF，稀释真正命中的结果的排名。
            # 因为已按分数降序，遇到第一个 0 即可停止。
            if score <= 0.0:
                break
            hits.append(BM25Hit(self._records[i], score))
            if len(hits) >= k:
                break
        return hits
