"""进程内单例：向量库、BM25 索引、检索器。

FastAPI 启动时调用 init_runtime() 预热；测试可直接调用。
"""
from __future__ import annotations

from app.common import db
from app.indexing.bm25_index import BM25Index
from app.indexing.milvus_store import MilvusStore
from app.indexing.pipeline import IngestionPipeline

_store: MilvusStore | None = None
_bm25: BM25Index | None = None
_pipeline: IngestionPipeline | None = None
_retriever = None


def init_runtime() -> None:
    global _store, _bm25, _pipeline
    db.init_db()
    if _store is None:
        _store = MilvusStore()
    if _bm25 is None:
        _bm25 = BM25Index()
        _bm25.load()
    if _pipeline is None:
        _pipeline = IngestionPipeline(_store, _bm25)


def get_store() -> MilvusStore:
    if _store is None:
        init_runtime()
    assert _store is not None
    return _store


def get_bm25() -> BM25Index:
    if _bm25 is None:
        init_runtime()
    assert _bm25 is not None
    return _bm25


def get_pipeline() -> IngestionPipeline:
    if _pipeline is None:
        init_runtime()
    assert _pipeline is not None
    return _pipeline


def get_retriever():
    global _retriever
    if _retriever is None:
        from app.retrieval.pipeline import Retriever

        _retriever = Retriever(get_store(), get_bm25())
    return _retriever
