"""诊断 Recall@1 偏低的原因。

假设：同一事实在语料里有多处表述（手册 / FAQ / 全表），
gold 只锚定其中一处，另一处同样能回答问题的块排到第一时，会被判为未命中。

本脚本统计：Recall@1 未命中的题里，有多少是「top-1 其实含标准事实」。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.pipeline import Retriever  # noqa: E402
from app.runtime import get_bm25, get_store, init_runtime  # noqa: E402


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def main() -> None:
    init_runtime()
    retriever = Retriever(get_store(), get_bm25())
    questions = [
        json.loads(line)
        for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    hit_gold = 0
    hit_fact_instead = 0
    real_miss = 0
    rows = []

    for q in questions:
        hit = retriever.search(q["question"], mode="hybrid_rerank", recall_k=10, final_k=1)
        if not hit:
            real_miss += 1
            continue
        top = hit[0]
        pk = top["pk"]
        text = norm(top["text"])

        if pk in set(q["gold_pks"]):
            hit_gold += 1
            continue

        facts = [norm(f) for f in q["facts"]]
        if any(f in text for f in facts):
            hit_fact_instead += 1
            rows.append({
                "qid": q["qid"],
                "question": q["question"],
                "top1": top["text"].split("\n")[0][:36],
                "gold": q["gold_pks"],
            })
        else:
            real_miss += 1
            rows.append({
                "qid": q["qid"],
                "question": q["question"],
                "top1": top["text"].split("\n")[0][:36],
                "gold": q["gold_pks"],
                "note": "真未命中",
            })

    n = len(questions)
    print(f"题目总数 {n}")
    print(f"  top-1 命中 gold            : {hit_gold} ({hit_gold / n:.1%})")
    print(f"  top-1 未中 gold 但含标准事实: {hit_fact_instead} ({hit_fact_instead / n:.1%})")
    print(f"  top-1 真的没找到答案        : {real_miss} ({real_miss / n:.1%})")
    print(f"\n「top-1 能回答问题」的实际比例 = {(hit_gold + hit_fact_instead) / n:.1%}")

    print("\n--- 未命中 gold 但含标准事实的样例 ---")
    for r in rows:
        if "note" not in r:
            print(f"  {r['qid']} {r['question'][:26]} → top1: {r['top1']}")

    print("\n--- 真未命中的样例 ---")
    for r in rows:
        if "note" in r:
            print(f"  {r['qid']} {r['question'][:26]} → top1: {r['top1']}")


if __name__ == "__main__":
    main()
