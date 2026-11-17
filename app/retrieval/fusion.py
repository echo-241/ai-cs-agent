"""RRF 融合（Reciprocal Rank Fusion）。

公式：score(d) = Σ 1 / (k + rank_i(d))，rank 从 1 开始，k 默认 60。

为什么按排名而不是按分数融合：
稠密路返回余弦相似度（0~1，越大越好），BM25 返回词频加权分（无上界，越大越好），
两者量纲不可比。直接加权求和必须先做归一化，而归一化方式本身又会引入新的超参；
RRF 只用名次，对分数量纲免疫，是工程上最稳的做法。
k 的作用是压平头部差距：k 越大，第 1 名与第 5 名的差距越小，融合越平均。
"""
from __future__ import annotations

from typing import Iterable


def rrf_fuse(rankings: Iterable[list[str]], k: int = 60,
             weights: list[float] | None = None) -> list[tuple[str, float]]:
    """输入若干条已按相关性降序排列的 id 列表，输出融合后的 (id, score) 降序列表。"""
    rankings = [r for r in rankings if r]
    if not rankings:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
