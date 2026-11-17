"""测试公共夹具：用内存商城存储替换 MySQL，测试不依赖外部数据库。

pytest 会自动加载本文件；store fixture 同时替换工具层与接口层引用的 shop_db。
"""
from __future__ import annotations

import re
from datetime import datetime

import pytest

SEED = [
    {"sku": "XR-200S", "item": "XR-200S 扫地机器人（简配版）", "price": 1299, "stock": 20},
    {"sku": "XR-300 Plus", "item": "XR-300 Plus 扫地机器人（大容量版）", "price": 2599, "stock": 8},
    {"sku": "XR-500 Max", "item": "XR-500 Max 扫地机器人（旗舰版）", "price": 3999, "stock": 5},
    {"sku": "XG-100 Pro", "item": "XG-100 Pro 空气净化器（增强版）", "price": 1699, "stock": 12},
]


def sku_key(raw: str) -> str:
    return re.sub(r"[\s\-_]", "", raw).upper()


class FakeStore:
    """与 app/common/shop_db.py 同接口的内存实现。"""

    SEED_PRODUCTS = SEED
    PRODUCT_META = {
        "XR-200S": {"image": "/static/images/xr200s.png", "tag": "入门热销", "desc": "1800Pa 吸力"},
        "XR-300 Plus": {"image": "/static/images/xr300plus.png", "tag": "大容量", "desc": "5200mAh 电池"},
        "XR-500 Max": {"image": "/static/images/xr500max.png", "tag": "旗舰新品", "desc": "自动集尘"},
        "XG-100 Pro": {"image": "/static/images/xg100pro.png", "tag": "净化优选", "desc": "CADR 300"},
    }

    def __init__(self) -> None:
        self.products = {sku_key(p["sku"]): dict(p) for p in SEED}
        self.orders: dict[str, dict] = {}

    @staticmethod
    def sku_key(raw: str) -> str:
        return sku_key(raw)

    @staticmethod
    def list_sku_names() -> str:
        return "、".join(p["sku"] for p in SEED)

    def list_products(self) -> list[dict]:
        return [
            {"sku": p["sku"], "item": p["item"], "price": p["price"], "stock": p["stock"]}
            for p in self.products.values()
        ]

    def get_product(self, key: str) -> dict | None:
        row = self.products.get(key)
        return dict(row) if row else None

    def create_order(self, **kw) -> dict:
        row = dict(kw)
        row.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.orders[row["order_id"]] = row
        product = self.products[sku_key(row["sku"])]
        product["stock"] -= row["quantity"]
        return dict(row)

    def get_order(self, order_id: str) -> dict | None:
        row = self.orders.get(order_id)
        return dict(row) if row else None

    def find_recent_order(self, session_id: str, sku: str, quantity: int, address: str,
                          within_seconds: int = 300) -> dict | None:
        for row in reversed(list(self.orders.values())):
            if (row["session_id"] == session_id and row["sku"] == sku
                    and row["quantity"] == quantity and row["address"] == address):
                return dict(row)
        return None

    def list_orders(self, session_id: str | None = None, limit: int = 20) -> list[dict]:
        rows = [dict(r) for r in self.orders.values() if not session_id or r["session_id"] == session_id]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return rows[:limit]


@pytest.fixture
def store(monkeypatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr("app.agent.tools.store", fake)
    monkeypatch.setattr("app.api.routes.shop_db", fake)
    monkeypatch.setattr("app.agent.shop_agent.shop_db", fake)
    return fake
