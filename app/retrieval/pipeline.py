"""检索流水线，支持五种配置切换。

配置说明（mode）：
- dense         纯稠密（改造前的朴素基线）
- dense_rerank  稠密 + 重排（改造前系统的真实形态）
- sparse        纯 BM25
- hybrid        BM25 + 稠密 + RRF，不重排
- hybrid_rerank BM25 + 稠密 + RRF + 重排（改造后）

五种配置共用同一批 chunk、同一个 embedding 模型、同一个 rerank 模型，
只改检索组合方式，保证提升可以归因到链路本身。
"""
from __future__ import annotations

from app.config.settings import settings
from app.indexing.bm25_index import BM25Index
from app.indexing.milvus_store import MilvusStore
from app.retrieval.fusion import rrf_fuse
from app.retrieval.rerank import rerank as do_rerank

MODES = ("dense", "dense_rerank", "sparse", "hybrid", "hybrid_rerank")


class Retriever:
    def __init__(self, store: MilvusStore, bm25: BM25Index) -> None:
        self.store = store
        self.bm25 = bm25

    # ------------------------------------------------------------ 单路
    def _dense(self, query: str, k: int, doc_ids: list[str] | None) -> list[dict]:
        return self.store.search_dense(query, top_k=k, doc_ids=doc_ids)

    def _sparse(self, query: str, k: int, doc_ids: list[str] | None) -> list[dict]:
        return self.bm25.search(query, top_k=k, doc_ids=doc_ids)

    # ------------------------------------------------------------ 融合
    def _hybrid(self, query: str, recall_k: int, doc_ids: list[str] | None) -> tuple[list[dict], dict]:
        dense_hits = self._dense(query, recall_k, doc_ids)
        sparse_hits = self._sparse(query, recall_k, doc_ids)

        pool: dict[str, dict] = {}
        for item in dense_hits:
            pool[item["pk"]] = {**item, "dense_rank": None, "sparse_rank": None}
        for item in sparse_hits:
            pool.setdefault(item["pk"], {**item, "dense_rank": None, "sparse_rank": None})

        for rank, item in enumerate(dense_hits, 1):
            pool[item["pk"]]["dense_rank"] = rank
            pool[item["pk"]]["score"] = item["score"]
        for rank, item in enumerate(sparse_hits, 1):
            pool[item["pk"]]["sparse_rank"] = rank

        fused = rrf_fuse(
            [[h["pk"] for h in dense_hits], [h["pk"] for h in sparse_hits]],
            k=settings.rrf_k,
        )
        debug = {
            "dense": [h["pk"] for h in dense_hits],
            "sparse": [h["pk"] for h in sparse_hits],
            "overlap": len({h["pk"] for h in dense_hits} & {h["pk"] for h in sparse_hits}),
        }
        ranked = []
        for pk, score in fused:
            item = dict(pool[pk])
            item["rrf_score"] = score
            ranked.append(item)
        return ranked, debug

    # ------------------------------------------------------------ 对外
    def search(self, query: str, mode: str = "hybrid_rerank",
               recall_k: int | None = None, final_k: int | None = None,
               doc_ids: list[str] | None = None,
               with_debug: bool = False):
        if mode not in MODES:
            raise ValueError(f"未知配置 {mode}，可选 {MODES}")

        recall_k = recall_k or settings.recall_k
        final_k = final_k or settings.final_k
        debug: dict = {"mode": mode}

        if mode == "dense":
            hits = self._dense(query, recall_k, doc_ids)[:final_k]
        elif mode == "sparse":
            hits = self._sparse(query, recall_k, doc_ids)[:final_k]
        elif mode == "dense_rerank":
            candidates = self._dense(query, recall_k, doc_ids)
            hits = do_rerank(query, candidates, top_n=final_k)
        elif mode == "hybrid":
            ranked, dbg = self._hybrid(query, recall_k, doc_ids)
            debug.update(dbg)
            hits = ranked[:final_k]
        else:  # hybrid_rerank
            ranked, dbg = self._hybrid(query, recall_k, doc_ids)
            debug.update(dbg)
            hits = do_rerank(query, ranked[:recall_k], top_n=final_k)

        for item in hits:
            item["mode"] = mode
        return (hits, debug) if with_debug else hits
