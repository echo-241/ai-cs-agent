"""全链路验收测试。

覆盖任务一要求的四步：上传文档 → 普通问答 → 文档问答 → 多轮追问。
运行：pytest tests/test_e2e.py -s
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

client = TestClient(app)


def parse_sse(text: str) -> dict:
    """把 SSE 流解析成结构化结果。"""
    route, sources, tokens, done, error = None, [], [], None, None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "route":
            route = event
        elif kind == "sources":
            sources = event.get("items", [])
        elif kind == "token":
            tokens.append(event["text"])
        elif kind == "done":
            done = event
        elif kind == "error":
            error = event

    return {
        "route": route,
        "sources": sources,
        "answer": "".join(tokens).strip(),
        "done": done,
        "error": error,
    }


@pytest.fixture(scope="module")
def session_id() -> str:
    resp = client.post("/api/sessions", json={"user_id": "pytest", "title": "验收会话"})
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def chat(session_id: str, message: str, mode: str = "hybrid_rerank") -> dict:
    resp = client.post("/api/chat", json={"session_id": session_id, "message": message, "mode": mode})
    assert resp.status_code == 200, resp.text
    parsed = parse_sse(resp.text)
    assert parsed["error"] is None, f"对话返回错误：{parsed['error']}"
    return parsed


# ---------------------------------------------------------------- 0. 服务与索引
def test_health_and_index():
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert data["chunks"] > 0, "知识库为空，先运行 scripts/build_index.py"

    docs = client.get("/api/documents").json()
    assert docs["total"] >= 8


# ---------------------------------------------------------------- 1. 上传文档
def test_upload_document(tmp_path):
    before = client.get("/api/stats").json()["vectors"]

    sample = tmp_path / "临时测试文档.txt"
    sample.write_text(
        "测试用售后说明\n\n"
        "1.1 测试专用换货条款\n"
        "本条款仅用于自动化测试，商品自签收之日起 30 个自然日内可申请换货，"
        "换货产生的往返运费由平台承担，用户无需支付任何费用。\n\n"
        "1.2 测试专用错误码\n"
        "E9999 表示自动化测试专用故障码，常见成因为测试数据注入，"
        "处理步骤为删除测试文档后重建索引。\n",
        encoding="utf-8",
    )

    with sample.open("rb") as fh:
        resp = client.post("/api/documents/upload", files={"file": (sample.name, fh, "text/plain")})
    assert resp.status_code == 200, resp.text
    stats = resp.json()
    assert stats["chunks"] > 0
    assert client.get("/api/stats").json()["vectors"] > before

    # 上传后能立刻检索到新内容
    result = client.get("/api/retrieve", params={"q": "E9999 是什么", "mode": "hybrid_rerank"}).json()
    assert any("E9999" in hit["text"] for hit in result["hits"]), result

    # 清理，避免污染演示知识库
    deleted = client.delete(f"/api/documents/{stats['doc_id']}")
    assert deleted.status_code == 200


# ---------------------------------------------------------------- 2. 普通问答
def test_chitchat(session_id):
    result = chat(session_id, "你好呀")
    assert result["route"]["intents"] == ["chitchat"], result["route"]
    assert result["answer"], "闲聊应当有回复"
    assert result["sources"] == [], "闲聊不应触发检索"


# ---------------------------------------------------------------- 3. 文档问答
def test_document_qa(session_id):
    result = chat(session_id, "E1028 是什么故障，怎么处理")
    assert "doc_qa" in result["route"]["intents"], result["route"]
    assert result["sources"], "知识库问答必须带回引用"
    assert result["sources"][0]["heading"].startswith("2.2.3"), result["sources"][0]
    answer = result["answer"]
    assert "E1028" in answer
    assert any(word in answer for word in ("充电", "电流", "适配器")), answer


# ---------------------------------------------------------------- 4. 多轮追问
def test_multiturn_order(session_id):
    first = chat(session_id, "帮我看下我的快递到哪了")
    assert "order_query" in first["route"]["intents"], first["route"]
    assert "订单号" in first["answer"], "缺少订单号时应追问"

    second = chat(session_id, "06715421bjfab412")
    assert "运输中" in second["answer"], second["answer"]


def test_multiturn_doc_followup(session_id):
    first = chat(session_id, "XR-200S 的电池容量是多少")
    assert "doc_qa" in first["route"]["intents"]
    assert "2600" in first["answer"], first["answer"]

    second = chat(session_id, "那它吸力呢")
    combined = first["answer"] + second["answer"]
    assert "1800" in combined, f"追问应沿着上一轮型号继续：{second['answer']}"


def test_multi_intent(session_id):
    """一句话两个意图：回答主意图，同时为次要意图登记工单。"""
    result = chat(session_id, "机器老是报 E1024，我要退货")
    intents = result["route"]["intents"]
    assert "doc_qa" in intents and "return_refund" in intents, intents
    assert "E1028" not in result["answer"]  # 不应串到别的错误码
    assert "工单" in result["answer"], "次要意图应登记工单并回执"
    assert result["done"]["ticket_id"], "done 事件应带工单号"
