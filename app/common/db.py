"""SQLite 持久层。

原 ChatService 用进程内 dict 存会话，多 worker 部署或进程重启即丢失。
这里改为 SQLite，顺带承载文档登记、事实槽、摘要与工单。
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.config.settings import DB_DIR

_DB_PATH = DB_DIR / "app.db"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    doc_name    TEXT NOT NULL,
    chunks      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS facts (
    session_id  TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    PRIMARY KEY (session_id, key)
);
CREATE TABLE IF NOT EXISTS summaries (
    session_id  TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    upto_id     INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    intent      TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    order_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    sku         TEXT NOT NULL,
    item        TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    amount      INTEGER NOT NULL,
    address     TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_session ON orders(session_id, created_at);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect():
    conn = sqlite3.connect(str(_DB_PATH), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        with _lock:
            yield conn
            conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        # 轻量迁移：老库补列，避免升级时丢会话
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(summaries)")}
        if "upto_id" not in cols:
            conn.execute("ALTER TABLE summaries ADD COLUMN upto_id INTEGER NOT NULL DEFAULT 0")


# ---------------------------------------------------------------- 文档
def upsert_document(doc_id: str, doc_name: str, chunks: int) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents(doc_id, doc_name, chunks, created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(doc_id) DO UPDATE SET chunks=excluded.chunks, doc_name=excluded.doc_name",
            (doc_id, doc_name, chunks, _now()),
        )


def list_documents() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def clear_documents() -> None:
    with connect() as conn:
        conn.execute("DELETE FROM documents")


# ---------------------------------------------------------------- 会话
def ensure_session(session_id: str, user_id: str, title: str = "") -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, user_id, title, created_at) VALUES(?,?,?,?)",
            (session_id, user_id, title, _now()),
        )


def list_sessions(user_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    with connect() as conn:
        for table in ("messages", "facts", "summaries", "tickets", "orders"):
            conn.execute(f"DELETE FROM {table} WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))


# ---------------------------------------------------------------- 消息
def add_message(session_id: str, role: str, content: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
            (session_id, role, content, _now()),
        )


def recent_messages(session_id: str, limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def count_messages(session_id: str) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()
    return int(row["c"])


def messages_since(session_id: str, after_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, role, content FROM messages WHERE session_id=? AND id>? ORDER BY id",
            (session_id, after_id),
        ).fetchall()
    return [dict(r) for r in rows]


def last_message_id(session_id: str) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()
    return int(row["m"])


# ---------------------------------------------------------------- 事实槽
def put_facts(session_id: str, slots: dict) -> None:
    with connect() as conn:
        for key, value in slots.items():
            if value in (None, "", [], {}):
                continue
            conn.execute(
                "INSERT INTO facts(session_id, key, value) VALUES(?,?,?) "
                "ON CONFLICT(session_id, key) DO UPDATE SET value=excluded.value",
                (session_id, key, str(value)),
            )


def get_facts(session_id: str) -> dict:
    with connect() as conn:
        rows = conn.execute("SELECT key, value FROM facts WHERE session_id=?", (session_id,)).fetchall()
    return {r["key"]: r["value"] for r in rows}


# ---------------------------------------------------------------- 摘要
def get_summary(session_id: str) -> tuple[str, int]:
    with connect() as conn:
        row = conn.execute(
            "SELECT summary, upto_id FROM summaries WHERE session_id=?", (session_id,)
        ).fetchone()
    return (row["summary"], int(row["upto_id"])) if row else ("", 0)


def save_summary(session_id: str, summary: str, upto_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO summaries(session_id, summary, upto_id, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, "
            "upto_id=excluded.upto_id, updated_at=excluded.updated_at",
            (session_id, summary, upto_id, _now()),
        )


# ---------------------------------------------------------------- 工单
def create_ticket(session_id: str, intent: str, content: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tickets(session_id, intent, content, created_at) VALUES(?,?,?,?)",
            (session_id, intent, content, _now()),
        )
        return int(cur.lastrowid)


# ---------------------------------------------------------------- 订单（Function Calling 写入）
def create_order(order_id: str, session_id: str, sku: str, item: str,
                 quantity: int, amount: int, address: str, status: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO orders(order_id, session_id, sku, item, quantity, amount, address, status, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (order_id, session_id, sku, item, quantity, amount, address, status, _now()),
        )


def get_order(order_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
    return dict(row) if row else None


def find_recent_order(session_id: str, sku: str, quantity: int, address: str,
                      within_seconds: int = 300) -> dict | None:
    """演示用幂等：5 分钟内同会话、同参数的重复触发直接返回已有订单。

    真实项目应改用客户端传来的幂等键，这里按参数去重只是为了挡住模型重复调用。
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE session_id=? AND sku=? AND quantity=? AND address=? "
            "ORDER BY created_at DESC LIMIT 1",
            (session_id, sku, quantity, address),
        ).fetchone()
    if not row:
        return None
    age = (datetime.now() - datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")).total_seconds()
    return dict(row) if 0 <= age < within_seconds else None
