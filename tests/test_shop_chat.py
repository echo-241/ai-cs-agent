"""商城 AI 导购多轮对话测试：不连模型服务、不连 MySQL。

覆盖：纯聊天、跨轮上下文、对话内下单、参数缺失追问、模型降级、接口与历史恢复。
运行：pytest tests/test_shop_chat.py -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.agent import shop_agent
from app.api import routes
from app.common import db
from app.main import app

client = TestClient(app)
SID = "pytest-chat"
ADDR = "上海市浦东新区世纪大道 1 号"


@pytest.fixture(autouse=True)
def temp_chat_db(tmp_path, monkeypatch, store):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "chat.db")
    db.init_db()
    return store


class FakeChatModel:
    """第一次 invoke 可返回 tool_calls，后续 invoke 返回自然语言。"""

    def __init__(self, replies=None, tool_call=None, fail=False):
        self.replies = replies or ["这是导购回复"]
        self.tool_call = tool_call
        self.fail = fail
        self.calls: list = []

    def bind_tools(self, specs):
        self.specs = specs
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("model down")
        if self.tool_call and len(self.calls) == 1:
            return AIMessage(content="", tool_calls=[{
                "name": self.tool_call["name"],
                "args": self.tool_call["args"],
                "id": "call_1",
            }])
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return AIMessage(content=self.replies[idx])


def _model(monkeypatch, **kw) -> FakeChatModel:
    fake = FakeChatModel(**kw)
    monkeypatch.setattr(shop_agent, "get_tool_model", lambda: fake)
    return fake


# ---------------------------------------------------------------- 纯聊天 / 多轮
def test_chat_without_tool(monkeypatch):
    _model(monkeypatch, replies=["预算 2000 的话推荐 XR-200S"])
    out = shop_agent.chat(SID, "预算 2000 买哪款？", ADDR)
    assert out["ok"] and out["tool"] is None
    assert "XR-200S" in out["reply"]
    history = db.recent_messages(SID, 10)
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_multiturn_context(monkeypatch):
    fake = _model(monkeypatch, replies=["你好呀", "刚才说的是 XR-200S"])
    shop_agent.chat(SID, "你好", ADDR)
    out = shop_agent.chat(SID, "刚推荐的是哪款来着", ADDR)
    assert out["reply"] == "刚才说的是 XR-200S"
    # 第二轮请求里应带上第一轮的完整对话
    second_round = [m.content for m in fake.calls[1]]
    assert "你好" in second_round and "你好呀" in second_round
    assert "刚推荐的是哪款来着" in second_round


# ---------------------------------------------------------------- 对话内下单
def test_chat_places_order(monkeypatch, store):
    fake = _model(monkeypatch, replies=["", "已经帮你下单啦"], tool_call={
        "name": "create_order",
        "args": {"sku": "XR-200S", "quantity": 1, "address": ADDR},
    })
    out = shop_agent.chat(SID, "帮我买一台 XR-200S", ADDR)
    assert out["ok"] and out["reply"] == "已经帮你下单啦"
    assert out["tool"]["ok"] and out["tool"]["name"] == "create_order"
    assert out["tool"]["data"]["source"] == "agent"
    assert len(store.list_orders(SID)) == 1
    assert "工具调用结果" in fake.calls[1][0].content, "工具结果必须回灌给模型"


def test_chat_asks_for_missing_address(monkeypatch, store):
    _model(monkeypatch, replies=["", "请告诉我收货地址"], tool_call={
        "name": "create_order",
        "args": {"sku": "XR-200S", "quantity": 1, "address": ""},
    })
    out = shop_agent.chat(SID, "帮我买一台 XR-200S", "")
    assert out["tool"]["ok"] is False
    assert "收货地址" in out["tool"]["message"]
    assert out["reply"] == "请告诉我收货地址"
    assert store.list_orders(SID) == [], "校验失败不应落库"


def test_model_unavailable_degrades(monkeypatch):
    _model(monkeypatch, fail=True)
    out = shop_agent.chat(SID, "你好", ADDR)
    assert out["ok"] is False and out["tool"] is None
    assert "模型服务暂时不可用" in out["reply"]


# ---------------------------------------------------------------- HTTP 接口
def test_chat_api_and_history(monkeypatch):
    _model(monkeypatch, replies=["你好，想看点什么？"])
    resp = client.post("/api/shop/chat", json={
        "session_id": SID, "message": "你好", "address": ADDR,
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["reply"] == "你好，想看点什么？"

    hist = client.get("/api/shop/chat/history", params={"session_id": SID}).json()["items"]
    assert len(hist) == 2 and hist[0]["role"] == "user"


def test_products_include_photos(store):
    items = client.get("/api/shop/products").json()["items"]
    assert len(items) == 4
    assert all(i["image"].startswith("/static/images/") for i in items)
    assert all(i["tag"] and i["desc"] for i in items)


def test_old_agent_buy_still_works(store, monkeypatch):
    """保留旧的一次性入口，不影响已有调用方。"""
    monkeypatch.setattr(routes, "decide_tool_call", lambda *a, **kw: {
        "name": "create_order",
        "args": {"sku": "XG-100 Pro", "quantity": 1, "address": ADDR},
    })
    body = client.post("/api/shop/agent-buy", json={
        "session_id": SID, "message": "买一台 XG-100 Pro", "address": ADDR,
    }).json()
    assert body["ok"] and body["outcome"]["data"]["source"] == "agent"
