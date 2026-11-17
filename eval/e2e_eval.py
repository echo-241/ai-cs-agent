"""端到端准确率复测：检索 → 生成 → 判定。

为什么除了 Recall 还要测端到端：
老师要求的是「验收准确率」，即最终答复对不对，而不只是「检索有没有命中」。
两个指标一起报，才能说清「提升来自检索链路，而不是生成波动」。

判定方式：LLM 判定员，固定 prompt、固定模型，逐条给出 correct/reason，
原始答案全部落盘，便于人工复核与复现。

用法：python eval/e2e_eval.py --modes dense_rerank,hybrid_rerank
输出：eval/e2e_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.common.clients import get_chat_model  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.retrieval.pipeline import Retriever  # noqa: E402
from app.runtime import get_bm25, get_store, init_runtime  # noqa: E402

DEFAULT_QUESTIONS = ROOT / "eval" / "questions.jsonl"
DEFAULT_OUT = ROOT / "eval" / "e2e_results.json"

ANSWER_PROMPT = """
你是「星域智能」的售后客服助手。

回答要求：
1. 只依据【参考资料】作答，不得编造条款、金额、时效或型号参数；
2. 先给结论，再给依据；引用资料时用 [1][2] 标注对应编号；
3. 资料里没有写明的内容，直接说「资料中未写明」，并建议转人工；
4. 涉及金额、时效、保修期、型号参数时，必须给出具体数值；
5. 语气简洁专业，不要寒暄铺垫，不要输出思考过程。
""".strip()

JUDGE_PROMPT = """
你是严格的答案评定员。输入包含【问题】【标准事实】【客服回答】。

判定为「正确」必须同时满足：
1. 回答给出了问题所问的结论，且与标准事实不矛盾；
2. 标准事实中的关键信息（编号、数值、时效、结论）在回答中出现或被正确转述。

以下情况判为「错误」：
1. 回答出现与标准事实矛盾的数值、编号或结论；
2. 回答以「资料中未写明」或「建议转人工」搪塞，而标准事实中已有答案；
3. 回答答非所问。

只输出一行 JSON，不要任何解释性文字：
{"correct": true 或 false, "reason": "不超过 30 字的中文理由"}
""".strip()


def build_messages(question: str, hits: list[dict]) -> list:
    context = "\n\n".join(
        f"[{i}] 来源：{h.get('doc_name', '')} 第{h.get('page', '?')}页\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    return [
        SystemMessage(content=ANSWER_PROMPT),
        SystemMessage(content=f"【参考资料】\n{context or '（未检索到相关资料）'}"),
        HumanMessage(content=question),
    ]


def judge(question: str, facts: list[str], answer: str) -> tuple[bool, str]:
    payload = (
        f"【问题】{question}\n"
        f"【标准事实】{'；'.join(facts)}\n"
        f"【客服回答】{answer}"
    )
    try:
        raw = get_chat_model().invoke(
            [SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=payload)]
        ).content
        text = str(raw).strip()
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1])
        return bool(data.get("correct")), str(data.get("reason", ""))[:60]
    except Exception as exc:  # noqa: BLE001
        return False, f"判定异常：{exc.__class__.__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="dense_rerank,hybrid_rerank")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    init_runtime()
    retriever = Retriever(get_store(), get_bm25())
    questions = [
        json.loads(line)
        for line in Path(args.questions).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if args.limit:
        questions = questions[:args.limit]

    report = {}
    for mode in args.modes.split(","):
        mode = mode.strip()
        print(f"\n=== {mode} ===", flush=True)
        rows, correct = [], 0

        for i, q in enumerate(questions, 1):
            hits = retriever.search(q["question"], mode=mode,
                                    recall_k=settings.recall_k, final_k=settings.final_k)
            try:
                answer = str(get_chat_model().invoke(build_messages(q["question"], hits)).content).strip()
            except Exception as exc:  # noqa: BLE001
                answer = f"（生成失败：{exc.__class__.__name__}）"

            ok, reason = judge(q["question"], q["facts"], answer)
            correct += ok
            rows.append({
                "qid": q["qid"], "category": q["category"], "question": q["question"],
                "gold_facts": q["facts"], "answer": answer, "correct": ok, "reason": reason,
                "top_headings": [h.get("heading") for h in hits],
            })
            if i % 10 == 0:
                print(f"  {i}/{len(questions)} 累计正确 {correct}", flush=True)

        by_cat: dict[str, dict] = {}
        for r in rows:
            c = by_cat.setdefault(r["category"], {"n": 0, "ok": 0})
            c["n"] += 1
            c["ok"] += int(r["correct"])

        report[mode] = {
            "accuracy": correct / len(rows),
            "n": len(rows),
            "by_category": {k: {"n": v["n"], "accuracy": v["ok"] / v["n"]} for k, v in by_cat.items()},
            "rows": rows,
        }
        print(f"  {mode} 端到端准确率 {correct}/{len(rows)} = {correct / len(rows):.1%}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n配置                端到端准确率   编号      术语      口语化    跨节")
    print("-" * 74)
    for mode, data in report.items():
        row = f"{mode:<18}{data['accuracy']:>10.1%}"
        for cat in ("编号", "术语", "口语化", "跨节"):
            v = data["by_category"].get(cat)
            row += f"{v['accuracy']:>10.1%}" if v else f"{'-':>10}"
        print(row)
    print(f"\n明细已写入 {args.out}")


if __name__ == "__main__":
    main()
