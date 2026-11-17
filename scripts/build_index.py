"""一键重建索引：清空 → 全量入库 → 打印统计。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runtime import get_pipeline, init_runtime


def main() -> None:
    init_runtime()
    pipeline = get_pipeline()

    print("清空已有索引 ...")
    pipeline.clear()

    print("开始入库 ...")
    results = pipeline.ingest_dir()
    total = 0
    for item in results:
        total += item["chunks"]
        print(f"  {item['doc_name']:<30} {item['chunks']:>3} 块")
    print(f"\n文档 {len(results)} 份，chunk {total} 个")
    print(f"Milvus 向量数 {pipeline.store.count()}，BM25 条目 {len(pipeline.bm25)}")


if __name__ == "__main__":
    main()
