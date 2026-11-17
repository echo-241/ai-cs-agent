"""延迟复测：验证「准确率提升、延迟不升反降」是否成立。

三个要排除的干扰：
1. 测量顺序偏差 —— 先跑的模式承担冷启动与连接建立，后跑的占便宜；
2. 缓存偏差 —— query embedding 被缓存后，后跑的配置少了一次 API 调用；
3. 候选长度偏差 —— 重排耗时与送入文档的总字数正相关，
   混合检索的候选里含更多 FAQ 短块，送入字数更少，重排自然更快。

做法：两种配置逐题交替执行（A,B,A,B），跑两轮；同时记录每次重排的送入字数。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.retrieval.pipeline import Retriever  # noqa: E402
from app.runtime import get_bm25, get_store, init_runtime  # noqa: E402

MODES = ("dense_rerank", "hybrid_rerank")


def one_round(retriever: Retriever, questions: list[dict], order: tuple[str, ...]) -> dict:
    data = {m: {"lat": [], "chars": []} for m in MODES}
    for q in questions:
        for mode in order:  # 按指定顺序执行
            start = time.perf_counter()
            hits = retriever.search(q["question"], mode=mode,
                                    recall_k=settings.recall_k, final_k=settings.final_k)
            elapsed = (time.perf_counter() - start) * 1000
            data[mode]["lat"].append(elapsed)
            # 重排的输入是 recall 阶段的候选，这里用最终返回块的字数近似其规模
            data[mode]["chars"].append(sum(len(h["text"]) for h in hits))
    return data


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    n = len(ordered)
    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[min(int(n * 0.95), n - 1)],
        "min": ordered[0],
        "max": ordered[-1],
        "std": statistics.pstdev(ordered),
    }


def main() -> None:
    init_runtime()
    retriever = Retriever(get_store(), get_bm25())
    questions = [
        json.loads(line)
        for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print("预热中 ...", flush=True)
    for q in questions[:3]:
        for mode in MODES:
            retriever.search(q["question"], mode=mode)

    # 关键：正序与反序各跑一遍。若差异随顺序翻转，说明是顺序偏差而非真实差异。
    print("第 1 轮（正序 A→B）...", flush=True)
    forward = one_round(retriever, questions, MODES)
    print("第 2 轮（反序 B→A）...", flush=True)
    backward = one_round(retriever, questions, tuple(reversed(MODES)))

    print("\n=== 两轮合并后的延迟（毫秒） ===")
    print(f"{'配置':<16}{'均值':>9}{'中位':>9}{'P95':>9}{'最小':>9}{'最大':>9}{'标准差':>9}")
    print("-" * 70)
    for mode in MODES:
        values = forward[mode]["lat"] + backward[mode]["lat"]
        s = summarize(values)
        print(f"{mode:<16}{s['mean']:>9.1f}{s['median']:>9.1f}{s['p95']:>9.1f}"
              f"{s['min']:>9.1f}{s['max']:>9.1f}{s['std']:>9.1f}")

    def paired(round_data: dict) -> tuple[float, float]:
        diffs = [b - a for a, b in zip(round_data["dense_rerank"]["lat"],
                                       round_data["hybrid_rerank"]["lat"])]
        mean = statistics.mean(diffs)
        stderr = statistics.pstdev(diffs) / (len(diffs) ** 0.5)
        return mean, stderr

    fwd_mean, fwd_err = paired(forward)
    bwd_mean, bwd_err = paired(backward)

    print("\n=== 配对差值（hybrid_rerank − dense_rerank） ===")
    print(f"正序 A→B：{fwd_mean:+.1f}ms ± {1.96 * fwd_err:.1f}ms")
    print(f"反序 B→A：{bwd_mean:+.1f}ms ± {1.96 * bwd_err:.1f}ms")

    if fwd_mean * bwd_mean < 0 and abs(fwd_mean - bwd_mean) > abs(fwd_mean) + abs(bwd_mean):
        print("判定：两轮符号相反 → 差异来自执行顺序（连接复用/预热），不是链路真实开销")
    elif fwd_mean * bwd_mean < 0:
        print("判定：两轮符号相反 → 顺序效应占主导，不能宣称任何一方更快")
    else:
        print("判定：两轮符号一致 → 存在系统性差异")

    print("\n=== 重排候选规模（字数，越大重排越慢） ===")
    for mode in MODES:
        chars = forward[mode]["chars"] + backward[mode]["chars"]
        print(f"{mode:<16} 平均 {statistics.mean(chars):.0f} 字  中位 {statistics.median(chars):.0f} 字")


if __name__ == "__main__":
    main()
