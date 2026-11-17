"""Cross-encoder 重排（DashScope gte 系列 qwen3-rerank）。

与改造前一致，仍是「先粗排候选、再逐条精排」的两段式：
交叉编码要把 query 与候选拼在一起过一次模型，开销随候选数线性增长，
所以只能对少量候选做，不能对全库做。
"""
from __future__ import annotations

import os
import time

import dashscope

from app.config.settings import settings

_MAX_DOCS_PER_CALL = 20  # 单次请求上限保护，超出分批


def _ensure_key() -> None:
    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        dashscope.api_key = key


def rerank(query: str, candidates: list[dict], top_n: int = 4,
           retries: int = 2) -> list[dict]:
    """对候选块重新打分排序，返回带 rerank_score 的前 top_n 条。"""
    if not candidates:
        return []
    _ensure_key()

    docs = [c["text"] for c in candidates]
    scored: list[tuple[int, float]] = []

    for start in range(0, len(docs), _MAX_DOCS_PER_CALL):
        batch = docs[start:start + _MAX_DOCS_PER_CALL]
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = dashscope.TextReRank.call(
                    model=settings.rerank_model,
                    query=query,
                    documents=batch,
                    top_n=len(batch),
                    return_documents=False,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"rerank 返回 {resp.status_code}: {resp.message}")
                for item in resp.output["results"]:
                    scored.append((start + int(item["index"]), float(item["relevance_score"])))
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.0 * (attempt + 1))
        else:
            raise RuntimeError(f"rerank 调用失败：{last_err}")

    scored.sort(key=lambda x: x[1], reverse=True)
    result: list[dict] = []
    for idx, score in scored[:top_n]:
        item = dict(candidates[idx])
        item["rerank_score"] = score
        result.append(item)
    return result
