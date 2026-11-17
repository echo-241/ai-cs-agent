"""由 facts 反查 gold chunk，生成评测集文件。

用法：
  python eval/build_eval.py --set clean       # 干净问法（默认）
  python eval/build_eval.py --set realistic   # 真实问法（同一批 gold，只改提问方式）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.indexing.bm25_index import BM25Index  # noqa: E402

SETS = {
    "clean": ("eval.questions_data", "eval/questions.jsonl"),
    "realistic": ("eval.questions_realistic", "eval/questions_realistic.jsonl"),
}


def norm(text: str) -> str:
    """去掉全部空白，消除 PDF 换行造成的断字。"""
    return re.sub(r"\s+", "", text or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=list(SETS), default="clean")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    module_name, default_out = SETS[args.set]
    module = __import__(module_name, fromlist=["QUESTIONS"])
    QUESTIONS = module.QUESTIONS
    OUT = ROOT / (args.out or default_out)

    bm25 = BM25Index()
    if not bm25.load():
        raise SystemExit("未找到 BM25 索引，先运行 scripts/build_index.py")

    chunks = bm25.chunks
    normalized = [(c["pk"], norm(c["text"]), c) for c in chunks]

    records, problems = [], []
    for qid, category, question, facts in QUESTIONS:
        gold: list[str] = []
        matched_facts = 0
        for fact in facts:
            needle = norm(fact)
            hit = [pk for pk, text, _ in normalized if needle and needle in text]
            if hit:
                matched_facts += 1
            gold.extend(hit)

        gold = sorted(set(gold))
        if matched_facts < len(facts) or not gold:
            problems.append((qid, "事实未命中", facts, len(gold)))
        elif len(gold) > 3:
            problems.append((qid, "gold 过多", facts, len(gold)))

        records.append({
            "qid": qid,
            "category": category,
            "question": question,
            "facts": facts,
            "gold_pks": gold,
        })

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

    total = len(records)
    by_cat: dict[str, int] = {}
    gold_sizes = []
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        gold_sizes.append(len(r["gold_pks"]))

    print(f"题目 {total} 道 → {OUT}")
    print("分类分布：", "  ".join(f"{k}={v}" for k, v in by_cat.items()))
    print(f"gold 平均 {sum(gold_sizes) / len(gold_sizes):.2f} 块，最大 {max(gold_sizes)}，最小 {min(gold_sizes)}")
    print(f"语料 chunk 总数 {len(chunks)}，gold 平均占比 {sum(gold_sizes) / len(gold_sizes) / len(chunks):.1%}")

    if problems:
        print(f"\n需要人工确认的 {len(problems)} 条：")
        for qid, reason, facts, n in problems:
            print(f"  {qid} [{reason}] gold={n}  facts={facts}")


if __name__ == "__main__":
    main()
