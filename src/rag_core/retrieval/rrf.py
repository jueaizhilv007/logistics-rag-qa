"""RRF（Reciprocal Rank Fusion）融合：把多路检索的排名合成一路。

### 为什么是 RRF 而不是分数加权

向量检索的分数是余弦相似度，落在 [0, 1]；
BM25 的分数是词频加权和，无上界，且随语料长度漂移。
两者尺度不可比，加权求和的权重只能靠试出来，
而试出来的权重换个语料就失效。

RRF 只吃排名、不吃分数，天然免疫尺度差异，
这也是它成为混合检索事实标准的原因。

### 公式

    score(d) = Σ_i 1 / (k + rank_i(d))

k=60 是原论文的经验值。k 的作用是压低「某一路排第一」的收益：
k 越大，1/(k+rank) 随 rank 衰减越平缓，单路的头部优势被稀释，
「两路都靠前」的样本更容易胜出。
"""

from __future__ import annotations

from typing import Sequence


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """把多路排名融合为一路，返回按融合分降序的 (key, score) 列表。

    rankings 的每个元素是一路检索器的结果，元素为 chunk_id，
    按该路的排名从 1 开始排列。

    某路没召回的 chunk 不做惩罚：RRF 的语义是
    「被越多路、越靠前地召回，分越高」，未召回是不加分，不是扣分。
    若改成扣分，会出现「两路都召回了但都靠后」输给「一路召回且靠前」的反直觉结果。

    并列时按 key 字典序升序。这一条不是打磨细节：
    评测脚本要反复跑同一批 query，排名不稳定会让实验结果无法复现。
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
