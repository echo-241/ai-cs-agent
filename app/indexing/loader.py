"""文档加载与结构化切分。

与改造前（AI-Project02）的差异：
- 原实现直接对整页文本做递归切分，PDF 换行会把标题甩到上一块末尾，
  导致「E1024 电源模块过温」这个标题和它的正文被分到两个 chunk；
- 现在先按标题把文档切成小节，再对超长小节按 200 字/20 重叠细分，
  并把标题回填到每个子块开头 —— 粒度不变，只修正边界。

四个检索配置共用同一批 chunk，保证对比是受控实验。
"""
from __future__ import annotations

import bisect
import hashlib
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings

SUPPORTED_SUFFIX = {".pdf", ".md", ".txt"}

# 形如 "2.2.1 E1024 电源模块过温"、"3.1 七天无理由退货"、"Q001"
_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+){0,3}\s+\S[^\n]{0,38}|Q\d{3}|第[一二三四五六七八九十]+[章节])\s*$"
)
_SENTENCE_SEPARATORS = ["(?<=。)", "(?<=！)", "(?<=？)", "(?<=；)", "\n", " ", ""]


def _read_pdf(path: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return [page.get_text() for page in doc]


def _read_text(path: Path) -> list[str]:
    return [path.read_text(encoding="utf-8", errors="ignore")]


def doc_id_of(path: Path) -> str:
    digest = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    return f"{path.stem.replace(' ', '_')}-{digest}"


def _split_sections(full_text: str) -> list[tuple[str, str, int]]:
    """返回 [(标题, 正文, 起始偏移)]。"""
    sections: list[tuple[str, str, int]] = []
    heading, buf = "", []
    offset = 0
    heading_start = 0

    for line in full_text.split("\n"):
        stripped = line.strip()
        if _HEADING_RE.match(stripped):
            if heading or any(x.strip() for x in buf):
                sections.append((heading, "\n".join(buf).strip(), heading_start))
            heading, buf, heading_start = stripped, [], offset
        else:
            buf.append(line)
        offset += len(line) + 1

    if heading or any(x.strip() for x in buf):
        sections.append((heading, "\n".join(buf).strip(), heading_start))
    return sections


def _page_of(offset: int, page_offsets: list[int]) -> int:
    idx = bisect.bisect_right(page_offsets, offset) - 1
    return max(idx, 0) + 1


def load_chunks(path: Path) -> list[dict]:
    """把一份文档切成带结构信息的块。"""
    if path.suffix.lower() not in SUPPORTED_SUFFIX:
        raise ValueError(f"不支持的格式：{path.suffix}，仅支持 {SUPPORTED_SUFFIX}")

    pages = _read_pdf(path) if path.suffix.lower() == ".pdf" else _read_text(path)

    full_text = ""
    page_offsets: list[int] = []
    for page_text in pages:
        page_offsets.append(len(full_text))
        full_text += page_text + "\n"

    doc_id = doc_id_of(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        is_separator_regex=True,
        separators=_SENTENCE_SEPARATORS,
    )

    chunks: list[dict] = []
    for heading, body, start in _split_sections(full_text):
        prefix = f"{heading}\n" if heading else ""
        room = max(settings.chunk_size - len(prefix), 60)

        if len(body) <= room:
            pieces = [body] if body else [""]
        else:
            pieces = splitter.split_text(body) if body else [""]

        for piece in pieces:
            text = (prefix + piece).strip()
            if len(text) < 15:
                continue
            idx = len(chunks)
            chunks.append({
                "pk": f"{doc_id}-{idx:04d}",
                "doc_id": doc_id,
                "doc_name": path.name,
                "page": _page_of(start, page_offsets),
                "chunk_index": idx,
                "heading": heading,
                "text": text,
            })
    return chunks
