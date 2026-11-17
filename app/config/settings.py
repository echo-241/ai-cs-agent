"""全局配置。

设计约束（对应任务二）：
1. 不更换模型 —— embedding / chat / rerank 三个模型名固定，改造只在检索链路上做；
2. 不改动文档内容 —— 语料原样入库；
3. 切分参数固定，保证四组配置对比是同一次受控实验。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
BM25_DIR = DATA_DIR / "bm25"
DB_DIR = DATA_DIR / "db"
UPLOAD_DIR = DATA_DIR / "uploads"


def _load_dotenv() -> None:
    """轻量 .env 加载，避免额外依赖。"""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    # -------------------------------------------------- 模型（不更换）
    embedding_model: str = "text-embedding-v1"
    embedding_dim: int = 1536
    chat_model: str = "deepseek-v4-flash-0731"
    rerank_model: str = "qwen3-rerank"

    # -------------------------------------------------- 切分（固定，控制变量）
    chunk_size: int = 200
    chunk_overlap: int = 20

    # -------------------------------------------------- Milvus
    milvus_uri: str = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    collection: str = "cs_knowledge"
    metric_type: str = "COSINE"
    index_type: str = "HNSW"

    # -------------------------------------------------- 检索
    recall_k: int = 10          # 每路召回条数，也是 rerank 的输入规模
    final_k: int = 4            # 最终送入模型的片段数
    rrf_k: int = 60             # RRF 平滑常数，取原论文经验值
    # 四组待对比的配置
    retriever_modes: tuple[str, ...] = ("dense", "sparse", "hybrid", "hybrid_rerank")

    # -------------------------------------------------- 商城（MySQL）
    # 用 MySQL 而不是 Postgres：本机已装 8.4、向量检索由 Milvus 承担，不需要 pgvector
    mysql_host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_db: str = os.getenv("MYSQL_DB", "mitao_shop")

    # -------------------------------------------------- 记忆
    history_window: int = 10    # 上下文里保留的最近消息条数
    summary_threshold: int = 10  # 未摘要消息达到多少条触发增量摘要

    # -------------------------------------------------- 语料
    docs_dir: Path = field(default=DOCS_DIR)
    bm25_dir: Path = field(default=BM25_DIR)
    db_dir: Path = field(default=DB_DIR)


settings = Settings()

for _d in (DATA_DIR, DOCS_DIR, BM25_DIR, DB_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)
