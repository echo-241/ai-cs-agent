"""模型客户端集中管理。

约束：模型名全部取自配置，不在此处硬编码，便于答辩时说明「只改检索、不改模型」。
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Sequence

import dashscope
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from app.config.settings import settings

if os.getenv("DASHSCOPE_API_KEY"):
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]


@lru_cache(maxsize=1)
def get_embeddings() -> DashScopeEmbeddings:
    return DashScopeEmbeddings(model=settings.embedding_model)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatTongyi:
    return ChatTongyi(model=settings.chat_model, streaming=True)


@lru_cache(maxsize=1)
def get_tool_model() -> ChatTongyi:
    """Function Calling 专用客户端：非流式。

    DashScope 的 incremental_output（流式增量）与 tools 不兼容，
    所以工具决策单独用一个 streaming=False 实例，模型名仍取自同一份配置。
    """
    return ChatTongyi(model=settings.chat_model, streaming=False)


def embed_documents(texts: Sequence[str], batch_size: int = 20, retries: int = 3) -> list[list[float]]:
    """批量向量化，带重试。

    DashScope text-embedding-v1 单次最多 25 条，这里取 20 留余量。
    """
    emb = get_embeddings()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                vectors.extend(emb.embed_documents(batch))
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"向量化失败（批次 {start}）：{last_err}")
    return vectors


def embed_query(text: str) -> list[float]:
    return get_embeddings().embed_query(text)


@lru_cache(maxsize=2048)
def embed_one_cached(text: str) -> tuple[float, ...]:
    """带缓存的单条向量化，评测时同一条 query 会被多组配置复用。"""
    return tuple(get_embeddings().embed_query(text))
