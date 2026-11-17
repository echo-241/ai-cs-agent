"""觅淘商城数据层（MySQL）：商品、库存、订单。

人工购买与 AI 助手代买共用这一套表和同一套下单工具（app/agent/tools.py），
区别只在订单的 source 字段：user / agent。

首次调用自动建库建表并写入演示商品，不需要手动执行 SQL 脚本。
"""
from __future__ import annotations

import re
import threading
from datetime import datetime, timedelta

import pymysql
from pymysql.cursors import DictCursor

from app.config.settings import settings

# 演示商品：型号取自知识库，价格与库存为 mock
SEED_PRODUCTS = [
    {"sku": "XR-200S", "item": "XR-200S 扫地机器人（简配版）", "price": 1299, "stock": 20},
    {"sku": "XR-300 Plus", "item": "XR-300 Plus 扫地机器人（大容量版）", "price": 2599, "stock": 8},
    {"sku": "XR-500 Max", "item": "XR-500 Max 扫地机器人（旗舰版）", "price": 3999, "stock": 5},
    {"sku": "XG-100 Pro", "item": "XG-100 Pro 空气净化器（增强版）", "price": 1699, "stock": 12},
]

# 商品展示与导购用的元数据（参数取自知识库《商品型号与规格手册》）
PRODUCT_META = {
    "XR-200S": {
        "image": "/static/images/xr200s.png", "tag": "入门热销",
        "desc": "1800Pa 吸力 / 2600mAh 电池 / 90 分钟续航，小户型够用",
    },
    "XR-300 Plus": {
        "image": "/static/images/xr300plus.png", "tag": "大容量",
        "desc": "5200mAh 电池 / 0.55L 尘盒 / 五爪边刷，大户型更省心",
    },
    "XR-500 Max": {
        "image": "/static/images/xr500max.png", "tag": "旗舰新品",
        "desc": "4200Pa 吸力 / 6400mAh 电池 / 自动集尘基站，免手动倒尘",
    },
    "XG-100 Pro": {
        "image": "/static/images/xg100pro.png", "tag": "净化优选",
        "desc": "CADR 300m³/h / 适用 30-50㎡ / 专用滤芯 XG-F100P",
    },
}

DDL = [
    """
    CREATE TABLE IF NOT EXISTS products (
        sku        VARCHAR(64)  NOT NULL PRIMARY KEY,
        sku_key    VARCHAR(64)  NOT NULL UNIQUE,
        item       VARCHAR(128) NOT NULL,
        price      INT          NOT NULL,
        stock      INT          NOT NULL DEFAULT 0,
        updated_at DATETIME     NOT NULL
    ) DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_id   VARCHAR(40)  NOT NULL PRIMARY KEY,
        session_id VARCHAR(64)  NOT NULL,
        sku        VARCHAR(64)  NOT NULL,
        item       VARCHAR(128) NOT NULL,
        quantity   INT          NOT NULL,
        amount     INT          NOT NULL,
        address    VARCHAR(255) NOT NULL,
        status     VARCHAR(20)  NOT NULL,
        source     VARCHAR(10)  NOT NULL,
        created_at DATETIME     NOT NULL,
        KEY idx_orders_session (session_id, created_at)
    ) DEFAULT CHARSET=utf8mb4
    """,
]

_ready = False
_lock = threading.Lock()


def sku_key(raw: str) -> str:
    """型号归一化：XR-300 Plus / xr300plus / XR300 PLUS 视为同一商品。"""
    return re.sub(r"[\s\-_]", "", raw).upper()


def _connect(database: str | None = None) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=database,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
        connect_timeout=5,
    )


def ensure_ready() -> None:
    """幂等初始化：建库 → 建表 → 写入演示商品。"""
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        db_name = settings.mysql_db
        if not re.fullmatch(r"[A-Za-z0-9_]+", db_name):
            raise ValueError(f"非法的数据库名：{db_name}")

        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARSET=utf8mb4")
                cur.execute(f"USE `{db_name}`")
                for sql in DDL:
                    cur.execute(sql)
                for p in SEED_PRODUCTS:
                    cur.execute(
                        "INSERT IGNORE INTO products(sku, sku_key, item, price, stock, updated_at) "
                        "VALUES(%s,%s,%s,%s,%s,%s)",
                        (p["sku"], sku_key(p["sku"]), p["item"], p["price"], p["stock"], datetime.now()),
                    )
            conn.commit()
        finally:
            conn.close()
        _ready = True


def _rows(rows) -> list[dict]:
    """DATETIME 转成字符串，方便直接 JSON 序列化。"""
    out = []
    for row in rows:
        item = dict(row)
        for key, value in item.items():
            if isinstance(value, datetime):
                item[key] = value.strftime("%Y-%m-%d %H:%M:%S")
        out.append(item)
    return out


# ---------------------------------------------------------------- 商品
def list_products() -> list[dict]:
    ensure_ready()
    conn = _connect(settings.mysql_db)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT sku, item, price, stock FROM products ORDER BY sku")
            return _rows(cur.fetchall())
    finally:
        conn.close()


def get_product(key: str) -> dict | None:
    ensure_ready()
    conn = _connect(settings.mysql_db)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM products WHERE sku_key=%s", (key,))
            rows = _rows(cur.fetchall())
            return rows[0] if rows else None
    finally:
        conn.close()


def list_sku_names() -> str:
    return "、".join(p["sku"] for p in SEED_PRODUCTS)


# ---------------------------------------------------------------- 订单
def create_order(order_id: str, session_id: str, sku: str, item: str,
                 quantity: int, amount: int, address: str, status: str,
                 source: str = "user") -> dict:
    """扣库存 + 写订单放在同一个事务里，用 stock>=quantity 条件防超卖。"""
    ensure_ready()
    conn = _connect(settings.mysql_db)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE products SET stock=stock-%s, updated_at=%s WHERE sku=%s AND stock>=%s",
                (quantity, datetime.now(), sku, quantity),
            )
            if cur.rowcount != 1:
                conn.rollback()
                raise RuntimeError("库存不足或商品不存在，订单未创建")
            cur.execute(
                "INSERT INTO orders(order_id, session_id, sku, item, quantity, amount, address, "
                "status, source, created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (order_id, session_id, sku, item, quantity, amount, address,
                 status, source, datetime.now()),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_order(order_id) or {}


def get_order(order_id: str) -> dict | None:
    ensure_ready()
    conn = _connect(settings.mysql_db)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM orders WHERE order_id=%s", (order_id,))
            rows = _rows(cur.fetchall())
            return rows[0] if rows else None
    finally:
        conn.close()


def find_recent_order(session_id: str, sku: str, quantity: int, address: str,
                      within_seconds: int = 300) -> dict | None:
    """演示用幂等：5 分钟内同会话、同参数的重复触发直接返回已有订单。"""
    ensure_ready()
    conn = _connect(settings.mysql_db)
    try:
        since = datetime.now() - timedelta(seconds=within_seconds)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM orders WHERE session_id=%s AND sku=%s AND quantity=%s "
                "AND address=%s AND created_at>=%s ORDER BY created_at DESC LIMIT 1",
                (session_id, sku, quantity, address, since),
            )
            rows = _rows(cur.fetchall())
            return rows[0] if rows else None
    finally:
        conn.close()


def list_orders(session_id: str | None = None, limit: int = 20) -> list[dict]:
    ensure_ready()
    conn = _connect(settings.mysql_db)
    try:
        with conn.cursor() as cur:
            if session_id:
                cur.execute(
                    "SELECT * FROM orders WHERE session_id=%s ORDER BY created_at DESC LIMIT %s",
                    (session_id, limit),
                )
            else:
                cur.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT %s", (limit,))
            return _rows(cur.fetchall())
    finally:
        conn.close()
