"""五组检索配置的复测脚本。

指标口径：
- Recall@5：命中任一 gold 块（事实级多相关判定）即算命中；
- Recall@1：第一名即命中；
- P50/P95 延迟：单次检索耗时。

用法：python eval/run_eval.py [--modes dense,sparse,...] [--limit N]
输出：eval/results.json + 控制台表格
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.pipeline import MODES, Retriever  # noqa: E402
from app.runtime import get_bm25, get_store, init_runtime  # noqa: E402

DEFAULT_QUESTIONS = ROOT / "eval" / "questions.jsonl"
TOP_K = 5


def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def evaluate(retriever: Retriever, questions: list[dict], mode: str) -> dict:
    hit1 = hit5 = 0
    latencies: list[float] = []
    per_category: dict[str, dict] = {}
    failures: list[dict] = []

    for q in questions:
        start = time.perf_counter()
        hits = retriever.search(q["question"], mode=mode, recall_k=10, final_k=TOP_K)
        latencies.append((time.perf_counter() - start) * 1000)

        got = [h["pk"] for h in hits]
        gold = set(q["gold_pks"])
        ok1 = bool(gold & set(got[:1]))
        ok5 = bool(gold & set(got))

        hit1 += ok1
        hit5 += ok5

        cat = per_category.setdefault(q["category"], {"n": 0, "hit1": 0, "hit5": 0})
        cat["n"] += 1
        cat["hit1"] += ok1
        cat["hit5"] += ok5

        if not ok5:
            failures.append({
                "qid": q["qid"],
                "category": q["category"],
                "question": q["question"],
                "gold": q["gold_pks"],
                "top3": got[:3],
                "top3_heading": [h["text"].split("\n")[0][:30] for h in hits[:3]],
            })

    n = len(questions)
    lat_sorted = sorted(latencies)
    return {
        "mode": mode,
        "n": n,
        "recall@1": hit1 / n,
        "recall@5": hit5 / n,
        "p50_ms": statistics.median(latencies),
        "p95_ms": lat_sorted[min(int(n * 0.95), n - 1)],
        "by_category": {
            k: {"n": v["n"], "recall@1": v["hit1"] / v["n"], "recall@5": v["hit5"] / v["n"]}
            for k, v in per_category.items()
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    args = parser.parse_args()

    init_runtime()
    retriever = Retriever(get_store(), get_bm25())
    questions = load_questions(Path(args.questions), args.limit)

    results = []
    for mode in args.modes.split(","):
        mode = mode.strip()
        if not mode:
            continue
        print(f"运行 {mode} ...", flush=True)
        results.append(evaluate(retriever, questions, mode))

    payload = {"top_k": TOP_K, "n": len(questions), "tag": args.tag, "results": results}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- 表格
    cats = ["编号", "术语", "口语化", "跨节"]
    header = f"{'配置':<16}{'Recall@1':>10}{'Recall@5':>10}" + "".join(f"{c + '@5':>12}" for c in cats) + f"{'P50':>9}{'P95':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        row = f"{r['mode']:<16}{r['recall@1']:>10.1%}{r['recall@5']:>10.1%}"
        for c in cats:
            val = r["by_category"].get(c)
            row += f"{val['recall@5']:>12.1%}" if val else f"{'-':>12}"
        row += f"{r['p50_ms']:>8.0f}ms{r['p95_ms']:>8.0f}ms"
        print(row)

    print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
