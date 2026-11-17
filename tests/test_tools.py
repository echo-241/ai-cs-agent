"""Function Calling 工具层测试：不连 Milvus、不打真实模型、不污染项目数据库。

运行：pytest tests/test_tools.py -q
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.agent import tools
from app.agent import orchestrator as orch_module
from app.agent.intent import IntentResult
from app.agent.orchestrator import AgentOrchestrator
from app.common import db

SESSION = "pytest-tools"


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch, store):
    # SQLite 只承载会话消息；商品与订单由 conftest 的内存 store 提供
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "test.db")
    db.init_db()
    db.ensure_session(SESSION, "pytest")
    return store


class FakeToolModel:
    """模拟 bind_tools + invoke，返回带 tool_calls 的 AIMessage。"""

    def __init__(self, call: dict | None = None, exc: Exception | None = None):
        self.call, self.exc, self.specs = call, exc, None

    def bind_tools(self, specs):
        self.specs = specs
        return self

    def invoke(self, messages):
        if self.exc:
            raise self.exc
        if not self.call:
            return AIMessage(content="无需调用工具")
        return AIMessage(content="", tool_calls=[{
            "name": self.call["name"], "args": self.call["args"], "id": "call_1",
        }])


class FakeIntentRecognizer:
    intents = ["place_order"]

    def recognize(self, user_input, chat_history=""):
        return IntentResult(intents=list(self.intents), slots={}, confidence=0.95)


def _model(monkeypatch, call=None, exc=None) -> FakeToolModel:
    fake = FakeToolModel(call, exc)
    monkeypatch.setattr(tools, "get_tool_model", lambda: fake)
    return fake


# ---------------------------------------------------------------- schema 与模型自主决策
def test_tool_specs_generated_from_pydantic():
    names = [spec["function"]["name"] for spec in tools.TOOL_SPECS]
    assert names == ["create_order", "query_order"]
    params = tools.TOOL_SPECS[0]["function"]["parameters"]
    assert set(params["properties"]) == {"sku", "quantity", "address"}
    assert params["properties"]["quantity"]["maximum"] == 10


def test_decide_tool_call_autonomous(monkeypatch):
    call = {"name": "create_order", "args": {"sku": "XR-200S", "quantity": 2, "address": "上海市浦东新区世纪大道1号"}}
    fake = _model(monkeypatch, call=call)
    assert tools.decide_tool_call("买两台 XR-200S 寄到上海", {"sku": "XR-200S"}) == call
    assert [s["function"]["name"] for s in fake.specs] == ["create_order", "query_order"]


@pytest.mark.parametrize("call,exc", [(None, None), (None, RuntimeError("模型超时"))])
def test_decide_tool_call_fallback(monkeypatch, call, exc):
    _model(monkeypatch, call=call, exc=exc)
    assert tools.decide_tool_call("保修多久") is None


# ---------------------------------------------------------------- 下单：正常与幂等
def test_create_order_ok(store):
    outcome = tools.run_tool("create_order", {
        "sku": "xr200s", "quantity": 2, "address": "上海市浦东新区世纪大道 1 号",
    }, SESSION, source="user")
    assert outcome["ok"] is True
    assert outcome["data"]["sku"] == "XR-200S"       # 型号归一化
    assert outcome["data"]["amount"] == 2598
    assert outcome["data"]["order_id"].startswith("SO")
    assert outcome["source"] == "user"
    assert store.get_order(outcome["data"]["order_id"])["status"] == "待发货"


def test_create_order_idempotent():
    args = {"sku": "XR-300 Plus", "quantity": 1, "address": "北京市海淀区中关村大街 1 号"}
    first = tools.run_tool("create_order", args, SESSION)
    second = tools.run_tool("create_order", args, SESSION)
    assert second["ok"] and second["data"]["order_id"] == first["data"]["order_id"]
    assert "未重复创建" in second["message"]


# ---------------------------------------------------------------- 参数校验 / 异常处理
@pytest.mark.parametrize("args,error", [
    ({"sku": "XR-200S", "address": "上海市浦东新区世纪大道 1 号", "quantity": 0}, "参数校验失败"),
    ({"sku": "XR-200S", "address": "上海市浦东新区世纪大道 1 号", "quantity": "两台"}, "参数校验失败"),
    ({"sku": "XR-200S", "quantity": 1, "address": "上海"}, "业务校验失败"),   # 地址过短
    ({"sku": "不存在的型号", "quantity": 1, "address": "上海市浦东新区世纪大道 1 号"}, "业务校验失败"),
    ({"sku": "XR-500 Max", "quantity": 6, "address": "上海市浦东新区世纪大道 1 号"}, "库存不足"),
])
def test_create_order_validation(args, error):
    out = tools.run_tool("create_order", args, SESSION)
    assert out["ok"] is False
    assert error in out["error"] or error in out["detail"]


def test_unknown_tool_and_bad_args():
    assert tools.run_tool("refund_money", {}, SESSION)["error"] == "业务校验失败"
    assert tools.run_tool("create_order", "不是对象", SESSION)["ok"] is False


def test_unexpected_exception_is_contained(monkeypatch, store):
    def boom(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "create_order", boom)
    out = tools.run_tool("create_order", {
        "sku": "XR-200S", "quantity": 1, "address": "上海市浦东新区世纪大道 1 号",
    }, SESSION)
    assert out["ok"] is False and out["error"] == "工具执行异常"
    assert "db down" in out["detail"] and "未完成" in out["message"]


# ---------------------------------------------------------------- 订单查询
def test_query_created_order_and_mock_order():
    created = tools.run_tool("create_order", {
        "sku": "XG-100 Pro", "quantity": 1, "address": "广州市天河区珠江新城 1 号",
    }, SESSION)["data"]
    out = tools.run_tool("query_order", {"order_id": created["order_id"].lower()}, SESSION)
    assert out["ok"] and "待发货" in out["message"]

    mock = tools.run_tool("query_order", {"order_id": "06715421bjfab412"}, SESSION)
    assert mock["ok"] and "运输中" in mock["message"]


@pytest.mark.parametrize("order_id", ["", "abc", "不存在的订单号", "06715421bjfab999"])
def test_query_order_invalid(order_id):
    out = tools.run_tool("query_order", {"order_id": order_id}, SESSION)
    assert out["ok"] is False and out["message"]


# ---------------------------------------------------------------- 编排层：LLM 触发工具 / 降级兜底
def _stream(monkeypatch, intents, call):
    FakeIntentRecognizer.intents = intents
    monkeypatch.setattr(orch_module, "IntentRecognizer", lambda: FakeIntentRecognizer())
    monkeypatch.setattr(orch_module, "decide_tool_call", lambda *a, **kw: call)

    def fake_tokens(self, messages, session_id, memory, sources, primary):
        yield {"type": "token", "text": "已为你下单"}
        return "已为你下单"

    monkeypatch.setattr(AgentOrchestrator, "_stream_tokens", fake_tokens)
    return list(AgentOrchestrator(None).stream(SESSION, "买一台 XR-200S", "hybrid_rerank"))


def test_orchestrator_tool_call_events(monkeypatch):
    call = {"name": "create_order", "args": {
        "sku": "XR-200S", "quantity": 1, "address": "深圳市南山区科技园 1 号",
    }}
    events = _stream(monkeypatch, ["place_order"], call)
    kinds = [e["type"] for e in events]
    assert kinds[:3] == ["route", "tool", "tool_result"]
    assert events[1]["name"] == "create_order"
    assert events[2]["ok"] is True
    done = [e for e in events if e["type"] == "done"][0]
    assert done["tool"] == "create_order" and done["tool_ok"] is True
    assert done["answer"] == "已为你下单"


def test_orchestrator_fallback_when_model_skips_tool(monkeypatch):
    """模型不调用工具（或不可用）时，走原有追问逻辑，链路不中断。"""
    events = _stream(monkeypatch, ["order_query"], None)
    done = [e for e in events if e["type"] == "done"][0]
    assert "订单号" in done["answer"]
    assert not any(e["type"] == "tool" for e in events)
