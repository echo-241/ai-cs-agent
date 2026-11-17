"""商城接口测试：页面、人工购买、助手代买；用内存 store，不依赖 MySQL / 模型服务。

运行：pytest tests/test_shop.py -q
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import routes
from app.main import app

client = TestClient(app)
SID = "pytest-shop"
ADDR = "上海市浦东新区世纪大道 1 号"


def test_shop_page():
    resp = client.get("/shop")
    assert resp.status_code == 200
    assert "觅淘" in resp.text


def test_products(store):
    items = client.get("/api/shop/products").json()["items"]
    assert len(items) == 4
    assert {i["sku"] for i in items} >= {"XR-200S", "XG-100 Pro"}


def test_manual_order(store):
    resp = client.post("/api/shop/orders", json={
        "session_id": SID, "sku": "XR-200S", "quantity": 2, "address": ADDR,
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ok"] and data["source"] == "user"
    assert data["data"]["amount"] == 2598

    orders = client.get("/api/shop/orders", params={"session_id": SID}).json()["items"]
    assert len(orders) == 1 and orders[0]["source"] == "user"
    assert store.get_product(store.sku_key("XR-200S"))["stock"] == 18  # 库存真实扣减


def test_manual_order_validation(store):
    resp = client.post("/api/shop/orders", json={
        "session_id": SID, "sku": "不存在的型号", "quantity": 1, "address": ADDR,
    })
    assert resp.status_code == 400
    assert "不在可售目录" in resp.json()["detail"]

    short = client.post("/api/shop/orders", json={
        "session_id": SID, "sku": "XR-200S", "quantity": 1, "address": "上海",
    })
    assert short.status_code == 400 and "收货地址" in short.json()["detail"]


def test_agent_buy(store, monkeypatch):
    """LLM 自主返回 tool_calls，工具层执行后订单落库，来源标记为 agent。"""
    captured: dict = {}

    def fake_decide(message: str, facts: dict | None = None):
        captured["message"], captured["facts"] = message, facts
        return {"name": "create_order", "args": {
            "sku": "XG-100 Pro", "quantity": 1, "address": (facts or {}).get("address", ""),
        }}

    monkeypatch.setattr(routes, "decide_tool_call", fake_decide)
    resp = client.post("/api/shop/agent-buy", json={
        "session_id": SID, "message": "帮我买一台 XG-100 Pro", "address": ADDR,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] and body["call"]["name"] == "create_order"
    assert body["outcome"]["data"]["source"] == "agent"
    assert captured["facts"]["address"] == ADDR, "页面地址应作为会话事实传给模型"


def test_agent_buy_no_tool(store, monkeypatch):
    """模型不调用工具时给明确提示，不静默失败。"""
    monkeypatch.setattr(routes, "decide_tool_call", lambda *a, **kw: None)
    resp = client.post("/api/shop/agent-buy", json={
        "session_id": SID, "message": "随便聊聊", "address": ADDR,
    })
    assert resp.status_code == 200 and resp.json()["ok"] is False
    assert "模型服务" in resp.json()["message"]


def test_agent_can_query_legacy_order(store, monkeypatch):
    """助手也可以选择查订单工具，而不是只能下单。"""
    monkeypatch.setattr(routes, "decide_tool_call", lambda *a, **kw: {
        "name": "query_order", "args": {"order_id": "06715421bjfab412"},
    })
    body = client.post("/api/shop/agent-buy", json={
        "session_id": SID, "message": "查一下 06715421bjfab412", "address": ADDR,
    }).json()
    assert body["ok"] and "运输中" in body["outcome"]["message"]
