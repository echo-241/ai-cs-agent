"""离线建库流水线：加载 → 切分 → 向量化入库 → 重建 BM25 → 登记文档。"""
from __future__ import annotations

from pathlib import Path

from app.common import db
from app.config.settings import settings
from app.indexing.bm25_index import BM25Index
from app.indexing.loader import doc_id_of, load_chunks
from app.indexing.milvus_store import MilvusStore


class IngestionPipeline:
    def __init__(self, store: MilvusStore, bm25: BM25Index) -> None:
        self.store = store
        self.bm25 = bm25

    def ingest_file(self, path: Path) -> dict:
        path = Path(path)
        chunks = load_chunks(path)
        if not chunks:
            raise ValueError(f"{path.name} 未切出任何内容，可能是空文件或扫描件")

        doc_id = doc_id_of(path)
        # 同一份文件重复上传时先删后插，保证幂等
        self.store.delete_doc(doc_id)
        inserted = self.store.insert_chunks(chunks)

        self.bm25.remove_doc(doc_id)
        self.bm25.add(chunks)
        self.bm25.save()

        db.upsert_document(doc_id, path.name, inserted)
        return {
            "doc_id": doc_id,
            "doc_name": path.name,
            "chunks": inserted,
            "total_vectors": self.store.count(),
            "bm25_size": len(self.bm25),
        }

    def ingest_dir(self, directory: Path | None = None) -> list[dict]:
        """批量入库。同名文件同时存在 .pdf 与 .md 时只取 PDF，避免重复入库。"""
        directory = Path(directory or settings.docs_dir)
        by_stem: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {".pdf", ".md", ".txt"}:
                continue
            current = by_stem.get(path.stem)
            if current is None or path.suffix.lower() == ".pdf":
                by_stem[path.stem] = path

        results = []
        for path in sorted(by_stem.values()):
            results.append(self.ingest_file(path))
        return results

    def clear(self) -> None:
        self.store.drop()
        self.bm25.build([])
        self.bm25.save()
        db.clear_documents()
