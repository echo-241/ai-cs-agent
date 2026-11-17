"""BM25 稀疏索引（jieba 分词 + rank_bm25）。

为什么不用 Milvus 内置 BM25：
1. 内置 BM25 的 k1/b 不可调
2. 需要「纯 BM25」这一组独立配置，自建索引可以直接取 top-k；
3. 分词可控 —— 编号类 token 必须先被完整切出来，否则 BM25 的优势就没了。
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from app.config.settings import settings

# 编号 token：XR-200 / E1024 / RF20261108000123 / LW03 / Q001 / 3.2.1
_CODE_RE = re.compile(r"[A-Za-z]{1,4}-?\d{2,}[A-Za-z]*|\d+\.\d+(?:\.\d+)?|\d{2,}")

# 业务词典：保证「错误码」「无理由退货」等术语不被切碎
_USER_WORDS = [
    "错误码", "无理由退货", "性能故障", "保修期", "保外维修", "集尘基站",
    "激光雷达", "悬崖传感器", "碰撞传感器", "沿墙传感器", "主刷电机", "边刷电机",
    "退款单号", "检测费", "人工费", "红冲", "换开", "包邮", "派送失败", "丢件",
]
for _w in _USER_WORDS:
    jieba.add_word(_w)

_STOP = set(" \t\r\n，。、；：？！“”‘’（）()【】[]{}<>《》-—_/\\|·,.;:?!")


def _split_codes(text: str) -> tuple[list[str], list[str]]:
    """把文本拆成（编号 token 列表, 剩余中文片段列表）。

    编号整体保留是三种分词方式共用的前提：否则 E1024 被拆成字符，
    BM25 就失去了「精确匹配稀有 token」这个唯一优势。
    """
    codes: list[str] = []
    segments: list[str] = []
    pos = 0
    for match in _CODE_RE.finditer(text):
        segments.append(text[pos:match.start()])
        codes.append(match.group().replace("-", "").lower())
        pos = match.end()
    segments.append(text[pos:])
    return codes, segments


def _tokenize_jieba(text: str) -> list[str]:
    codes, segments = _split_codes(text)
    tokens: list[str] = []
    for seg in segments:
        tokens.extend(t for t in jieba.lcut(seg) if t.strip() and t not in _STOP)
    tokens.extend(codes)
    return [t.lower() for t in tokens]


def _tokenize_char(text: str) -> list[str]:
    """字符级：每个汉字一个 token，零依赖。"""
    codes, segments = _split_codes(text)
    tokens: list[str] = [ch for seg in segments for ch in seg
                         if not ch.isspace() and ch not in _STOP]
    tokens.extend(codes)
    return [t.lower() for t in tokens]


def _tokenize_bigram(text: str) -> list[str]:
    """2-gram：两字一组，零依赖，能保留「充电」「电流」这类词。"""
    codes, segments = _split_codes(text)
    tokens: list[str] = []
    for seg in segments:
        chars = [ch for ch in seg if not ch.isspace() and ch not in _STOP]
        tokens.extend(chars[i:i + 2] for i in range(len(chars) - 1))
        if len(chars) == 1:
            tokens.append(chars[0])
    tokens.extend(codes)
    return [t.lower() if isinstance(t, str) else "".join(t).lower() for t in tokens]


MODES = {
    "jieba": _tokenize_jieba,
    "char": _tokenize_char,
    "bigram": _tokenize_bigram,
}


def tokenize(text: str, mode: str = "jieba") -> list[str]:
    """先抽编号，再按指定方式切剩余中文，最后小写归一。

    编号里的连字符会被去掉：XR-200 与 XR200 视为同一 token，
    避免用户输入时漏打连字符导致完全匹配失败。
    """
    fn = MODES.get(mode, _tokenize_jieba)
    return [t for t in fn(text) if t]


class BM25Index:
    """单机 BM25 索引，语料规模在万级 chunk 以内性能足够。

    mode 决定分词方式：jieba（默认）/ char（字符级）/ bigram（2-gram）。
    后两者零依赖，用同一批 chunk 可以随时切换做对照。
    """

    def __init__(self, mode: str = "jieba") -> None:
        self.mode = mode
        self.chunks: list[dict] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    # ------------------------------------------------------------ 构建
    def build(self, chunks: list[dict]) -> None:
        self.chunks = list(chunks)
        self._tokens = [tokenize(c["text"], self.mode) for c in self.chunks]
        corpus = [t if t else ["_empty_"] for t in self._tokens]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def add(self, chunks: list[dict]) -> None:
        """增量：把新 chunk 并入后整体重建（几万条以内重建耗时 < 1s）。"""
        self.build(self.chunks + list(chunks))

    def remove_doc(self, doc_id: str) -> None:
        remain = [c for c in self.chunks if c["doc_id"] != doc_id]
        if len(remain) != len(self.chunks):
            self.build(remain)

    # ------------------------------------------------------------ 检索
    def search(self, query: str, top_k: int = 10,
               doc_ids: list[str] | None = None) -> list[dict]:
        if not self._bm25 or not self.chunks:
            return []
        q_tokens = tokenize(query, self.mode)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        order = scores.argsort()[::-1]
        results: list[dict] = []
        for idx in order:
            chunk = self.chunks[int(idx)]
            if doc_ids and chunk["doc_id"] not in doc_ids:
                continue
            score = float(scores[int(idx)])
            if score <= 0:
                # 全零说明没有词面重合，继续往后也不会有正分（argsort 已降序）
                break
            item = dict(chunk)
            item["score"] = score
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    # ------------------------------------------------------------ 持久化
    def save(self, path: Path | None = None) -> None:
        path = path or (settings.bm25_dir / "index.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"chunks": self.chunks, "mode": self.mode}, fh)

    def load(self, path: Path | None = None) -> bool:
        path = path or (settings.bm25_dir / "index.pkl")
        if not path.exists():
            return False
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("mode"):
            self.mode = payload["mode"]
        self.build(payload.get("chunks", []))
        return True

    def __len__(self) -> int:
        return len(self.chunks)
