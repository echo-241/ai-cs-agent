"""Milvus 向量存储。

相比原实现（Chroma，一文件一 collection）：改为单 collection + doc_id 标量过滤，
这样才能支持「多文档共存」与「限定某几个文档检索」。
"""
from __future__ import annotations

from typing import Iterable

from pymilvus import DataType, MilvusClient

from app.common.clients import embed_documents, embed_one_cached
from app.config.settings import settings

_ID_MAX = 64
_NAME_MAX = 256
_TEXT_MAX = 8192


class MilvusStore:
    def __init__(self, uri: str | None = None, collection: str | None = None) -> None:
        self.uri = uri or settings.milvus_uri
        self.collection = collection or settings.collection
        self.client = MilvusClient(uri=self.uri)
        self._ensure_collection()

    # ------------------------------------------------------------ schema
    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            return
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=_ID_MAX)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=_ID_MAX)
        schema.add_field("doc_name", DataType.VARCHAR, max_length=_NAME_MAX)
        schema.add_field("page", DataType.INT64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=_TEXT_MAX)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            index_type=settings.index_type,
            metric_type=settings.metric_type,
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    # ------------------------------------------------------------ 写入
    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(collection_name=self.collection, filter=f'doc_id == "{doc_id}"')

    def insert_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        vectors = embed_documents([c["text"] for c in chunks])
        rows = [
            {
                "pk": c["pk"],
                "doc_id": c["doc_id"],
                "doc_name": c["doc_name"],
                "page": int(c.get("page", 1)),
                "chunk_index": int(c.get("chunk_index", 0)),
                "text": c["text"][:_TEXT_MAX],
                "dense": vec,
            }
            for c, vec in zip(chunks, vectors)
        ]
        self.client.insert(collection_name=self.collection, data=rows)
        return len(rows)

    # ------------------------------------------------------------ 检索
    def search_dense(self, query: str, top_k: int = 10,
                     doc_ids: Iterable[str] | None = None) -> list[dict]:
        vector = list(embed_one_cached(query))
        expr = ""
        if doc_ids:
            ids = list(doc_ids)
            quoted = ", ".join(f'"{i}"' for i in ids)
            expr = f"doc_id in [{quoted}]"

        hits = self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="dense",
            limit=top_k,
            filter=expr or "",
            output_fields=["pk", "doc_id", "doc_name", "page", "chunk_index", "text"],
            search_params={"metric_type": settings.metric_type, "params": {"ef": 64}},
        )
        results: list[dict] = []
        for hit in (hits[0] if hits else []):
            entity = hit.get("entity", {})
            results.append({
                "pk": entity.get("pk") or hit.get("id") or hit.get("pk"),
                "doc_id": entity.get("doc_id"),
                "doc_name": entity.get("doc_name"),
                "page": entity.get("page"),
                "chunk_index": entity.get("chunk_index"),
                "text": entity.get("text"),
                "score": float(hit.get("distance", 0.0)),
            })
        return results

    # ------------------------------------------------------------ 统计
    def count(self) -> int:
        """row_count 统计有延迟，改用 count(*) 查询，结果与写入一致。"""
        rows = self.client.query(
            self.collection, filter='pk != ""', output_fields=["count(*)"]
        )
        return int(rows[0]["count(*)"]) if rows else 0

    def drop(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self._ensure_collection()
