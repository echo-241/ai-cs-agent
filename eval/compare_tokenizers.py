"""分词方式对照实验：jieba / 字符级 / 2-gram。

同一批 chunk、同一批题、同一个 RRF 参数，只换 BM25 的分词方式，
分别看「纯 BM25」和「稠密 + BM25 融合」的召回表现。

用法：python eval/compare_tokenizers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.indexing.bm25_index import BM25Index  # noqa: E402
from app.retrieval.fusion import rrf_fuse  # noqa: E402
from app.runtime import get_store, init_runtime  # noqa: E402

MODES = ["jieba", "char", "bigram"]


def hit_at(pks: list[str], gold: set[str], k: int) -> bool:
    return bool(gold & set(pks[:k]))


def main() -> None:
    init_runtime()
    store = get_store()

    base = BM25Index()
    if not base.load():
        raise SystemExit("先运行 scripts/build_index.py")
    chunks = base.chunks

    questions = [
        json.loads(line)
        for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # 稠密结果只算一次，三种分词方式共用
    dense_cache: dict[str, list[str]] = {}
    for q in questions:
        dense_cache[q["qid"]] = [h["pk"] for h in store.search_dense(q["question"], top_k=settings.recall_k)]

    indexes = {m: BM25Index(mode=m) for m in MODES}
    for m in MODES:
        indexes[m].build(chunks)

    print(f"chunk 总数 {len(chunks)}，题目 {len(questions)} 道\n")
    header = (f"{'分词方式':<10}{'词表大小':>10}{'纯BM25@1':>12}{'纯BM25@5':>12}"
              f"{'融合@1':>10}{'融合@5':>10}")
    print(header)
    print("-" * len(header))

    rows = []
    for mode in MODES:
        idx = indexes[mode]
        vocab = len({t for toks in idx._tokens for t in toks})
        s1 = s5 = h1 = h5 = 0
        for q in questions:
            gold = set(q["gold_pks"])
            sparse_hits = idx.search(q["question"], top_k=settings.recall_k)
            sparse_pks = [h["pk"] for h in sparse_hits]
            s1 += hit_at(sparse_pks, gold, 1)
            s5 += hit_at(sparse_pks, gold, 5)

            fused = rrf_fuse([dense_cache[q["qid"]], sparse_pks], k=settings.rrf_k)
            fused_pks = [pk for pk, _ in fused]
            h1 += hit_at(fused_pks, gold, 1)
            h5 += hit_at(fused_pks, gold, 5)

        n = len(questions)
        rows.append((mode, vocab, s1 / n, s5 / n, h1 / n, h5 / n))
        print(f"{mode:<10}{vocab:>10}{s1 / n:>12.1%}{s5 / n:>12.1%}{h1 / n:>10.1%}{h5 / n:>10.1%}")

    print("\n注：融合列 = 稠密 top10 + 该分词方式的 BM25 top10，RRF(k=60) 后取前 5。")


if __name__ == "__main__":
    main()
